"""
Find Diffraction Vectors (Electron port) — end-to-end data flow.

Dispatching the action must:
  * open a new vectors-image tree (signal_type spyde_diffraction_vectors_image)
    with a "Vector count map" navigator override (NOT a recomputed navigator),
  * run the Qt-free compute core on a background thread, and
  * attach `tree.diffraction_vectors` + render the result when done.

The compute is memory-safe (map_overlap, never .compute() on the full dataset).
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=25.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _diffraction_4d():
    """A 4D-STEM scan with one bright disk per pattern (a findable peak)."""
    nav, sig = (4, 5), (24, 24)
    data = np.zeros(nav + sig, dtype=np.float32)
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - 12) ** 2 + (yy - 12) ** 2 <= 16).astype(np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * 100.0
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    return s


class TestFindVectorsPort:
    def test_find_vectors_creates_attached_vectors_tree(self):
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_diffraction_4d(), source_path=None)
            time.sleep(0.4)
            src_plot = _signal_plot(session)
            assert src_plot is not None
            trees_before = len(session.signal_trees)

            session._dispatch_toolbar_action(
                src_plot, "Find Diffraction Vectors",
                {"sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                 "min_distance": 3, "subpixel": True, "method": "nxcorr"},
            )

            # A new vectors tree appears immediately (placeholder), before compute.
            assert _wait(lambda: len(session.signal_trees) == trees_before + 1, timeout=5)
            vtree = session.signal_trees[-1]
            assert vtree.root._signal_type == "spyde_diffraction_vectors_image"

            # Background compute attaches the vectors container when done.
            assert _wait(lambda: getattr(vtree, "diffraction_vectors", None) is not None), \
                "diffraction_vectors was never attached"
            vecs = vtree.diffraction_vectors
            # count_map is nav-shaped and the bright disks were found.
            cm = vecs.count_map()
            assert cm.shape == (4, 5)
            assert int(cm.sum()) > 0, "no vectors found on a clear bright-disk scan"
        finally:
            session.shutdown()

    def test_result_window_renders_vectors_and_overlays(self):
        """Qt parity: once computed, the result window must (a) render the disk
        frames when navigated — NOT the stale placeholder zeros — and (b) carry a
        red found-vectors marker overlay tracking its count-map navigator."""
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_diffraction_4d(), source_path=None)
            time.sleep(0.4)
            src_plot = _signal_plot(session)
            assert src_plot is not None

            session._dispatch_toolbar_action(
                src_plot, "Find Diffraction Vectors",
                {"sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                 "min_distance": 3, "subpixel": True, "method": "nxcorr"},
            )
            assert _wait(lambda: getattr(session.signal_trees[-1],
                                         "diffraction_vectors", None) is not None)
            vtree = session.signal_trees[-1]

            cm = vtree.diffraction_vectors.count_map()
            iy, ix = map(int, np.argwhere(cm > 0)[0])

            # (a) The navigator now slices via an IN-PROCESS render_frame function
            # (Qt parity, no async lazy/shm path), so the signal plot paints the
            # rendered disks directly — NOT the placeholder zeros.
            sp = vtree.signal_plots[0]
            sel = next(s for s in vtree.navigator_plot_manager.all_navigation_selectors
                       if sp in s.children)
            deadline = time.time() + 30.0    # render wiring installs async
            frame = sel.children[sp](sel, sp, np.array([[ix, iy]]))
            while float(np.asarray(frame).max()) <= 0 and time.time() < deadline:
                time.sleep(0.1)
                frame = sel.children[sp](sel, sp, np.array([[ix, iy]]))
            assert float(np.asarray(frame).max()) > 0, "result window still renders zeros"

            # (b) A found-vectors overlay is attached to the result window and
            # yields markers at a position that actually has vectors.
            assert _wait(lambda: getattr(vtree, "_result_vector_overlay", None) is not None)
            ov = vtree._result_vector_overlay
            offs = ov._offsets_for(iy, ix)
            assert len(offs) > 0
            # (c) Every marker lands inside the detector — spurious out-of-frame
            # vectors (pixel coords like 24000) are filtered, so no giant/off arcs.
            W = int(ov.vecs.sig_axes[0].size); H = int(ov.vecs.sig_axes[1].size)
            assert offs[:, 0].max() <= W + 8 and offs[:, 1].max() <= H + 8
            assert offs[:, 0].min() >= -8 and offs[:, 1].min() >= -8
        finally:
            session.shutdown()

    def test_rejects_non_4d_dataset(self):
        from spyde.backend.session import Session
        from spyde.actions.context import ActionContext
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            s = hs.signals.Signal2D(np.random.RandomState(0).rand(16, 16).astype(np.float32))
            session._add_signal(s, source_path=None)
            time.sleep(0.3)
            plot = _signal_plot(session)
            before = len(session.signal_trees)
            session._dispatch_toolbar_action(plot, "Find Diffraction Vectors", {})
            time.sleep(0.3)
            # No vectors tree created for a 2-D image.
            assert len(session.signal_trees) == before
        finally:
            session.shutdown()
