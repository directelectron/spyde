"""
Vector Virtual Imaging (Electron, reuse raw VI machinery).

After Find Diffraction Vectors builds a `diffraction_vectors` tree, the
`Add Vector Virtual Image` sub-toolbar action must:
  * create a colour-cycled detector ROI on the vectors diffraction pattern,
  * open an output window whose nav-space image is computed FROM THE VECTORS
    (`vecs.virtual_image_from_roi_gpu`, calibrated ROI), not the raw 4D data,
  * recompute live and support disk / annular / rectangle + intensity/count.

Reuses `VirtualImageAction` (RegionAction template); only `reduce` differs.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs


def _signal_plot(session, tree):
    return next((p for p in session._plots
                 if not p.is_navigator and getattr(p, "signal_tree", None) is tree
                 and p.plot_state is not None), None)


def _wait(pred, timeout=25.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _calibrated_diffraction_4d(nav=(4, 5), sig=(24, 24), scale=0.1):
    data = np.zeros(nav + sig, dtype=np.float32)
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - 12) ** 2 + (yy - 12) ** 2 <= 16).astype(np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * 100.0
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = -(ax.size / 2.0) * scale
        ax.units = "1/nm"
    return s


def _make_vectors_tree(session):
    """Run Find Vectors and return the resulting vectors-image tree."""
    session._add_signal(_calibrated_diffraction_4d(scale=0.1))
    time.sleep(0.4)
    src_plot = next((p for p in session._plots
                     if not p.is_navigator and p.plot_state is not None), None)
    session._dispatch_toolbar_action(
        src_plot, "Find Diffraction Vectors",
        {"sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
         "min_distance": 3, "subpixel": True},
    )
    assert _wait(lambda: getattr(session.signal_trees[-1], "diffraction_vectors", None) is not None,
                 timeout=40), "Find Vectors never attached diffraction_vectors"
    return session.signal_trees[-1]


class TestVectorVirtualImaging:
    def test_add_vector_vi_computes_from_vectors(self):
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            vtree = _make_vectors_tree(session)
            vplot = _signal_plot(session, vtree)
            assert vplot is not None

            n_before = len(session._plots)
            session._dispatch_toolbar_action(
                vplot, "add_vector_virtual_image",
                {"type": "disk", "calculation": "intensity"},
            )
            # A new output window appears + is tracked as a chip.
            assert _wait(lambda: len(getattr(vplot, "_vi_items", [])) == 1), \
                "vector VI chip never registered"
            item = vplot._vi_items[0]
            assert item["parent_action"] == "Vector Virtual Imaging"
            assert item["color"] == "red"

            # The output plot must show a NON-ZERO nav image computed from vectors
            # (the centred disk → every pattern contributes to the count map).
            art = session._action_artifacts[(vplot.window_id, item["name"])]
            out_wid = art["out_wids"][0]
            out_plot = session._plot_by_window_id(out_wid)
            assert _wait(lambda: out_plot.current_data is not None
                         and np.asarray(out_plot.current_data).sum() > 0,
                         timeout=15), "vector VI output stayed blank"
            img = np.asarray(out_plot.current_data)
            assert img.shape == tuple(vtree.diffraction_vectors.nav_shape)
        finally:
            session.shutdown()

    def test_count_vs_intensity_and_shape_change(self):
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            vtree = _make_vectors_tree(session)
            vplot = _signal_plot(session, vtree)

            session._dispatch_toolbar_action(
                vplot, "add_vector_virtual_image",
                {"type": "disk", "calculation": "count"},
            )
            assert _wait(lambda: len(getattr(vplot, "_vi_items", [])) == 1)
            name = vplot._vi_items[0]["name"]
            art = session._action_artifacts[(vplot.window_id, name)]
            act = art["action"]

            # Live caret edit: switch disk → rectangle rebuilds the selector.
            from spyde.drawing.selectors import RectangleSelector
            session._update_vi(vplot.window_id, name, {"type": "rectangle"})
            assert _wait(lambda: isinstance(getattr(act, "_selector", None), RectangleSelector)), \
                "shape change did not rebuild the selector as a rectangle"
        finally:
            session.shutdown()
