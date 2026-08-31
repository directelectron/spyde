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

import numpy as np
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle, close_session, make_session
from spyde.tests.migrated._async import wait_until


def _wait(pred, timeout=25.0):
    return wait_until(pred, timeout)


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


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
    def test_find_vectors_creates_attached_vectors_tree(self, captured_messages):
        session = make_session()
        try:
            session._add_signal(_diffraction_4d(), source_path=None)
            _settle(session)
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

            # Qt parity, on the SAME computed result (this was a separate test
            # re-running the byte-identical dispatch): once computed, the
            # result window must (a) render the disk frames when navigated —
            # NOT the stale placeholder zeros — and (b) carry a red
            # found-vectors marker overlay tracking its count-map navigator.
            iy, ix = map(int, np.argwhere(cm > 0)[0])

            # (a) The navigator now slices via an IN-PROCESS render_frame function
            # (Qt parity, no async lazy/shm path), so the signal plot paints the
            # rendered disks directly — NOT the placeholder zeros.
            sp = vtree.signal_plots[0]
            sel = next(s for s in vtree.navigator_plot_manager.all_navigation_selectors
                       if sp in s.children)
            def _peak(rendered) -> float:
                """Brightest pixel, or -1 for anything that is not an array of
                numbers — `None`, or the length-1 object array hyperspy wraps an
                unresolved frame in, either of which would raise in `float()`
                and report as the wrong failure."""
                try:
                    return float(np.asarray(rendered, dtype=float).max())
                except (TypeError, ValueError):
                    return -1.0

            # Wait for the batch to say it FINISHED, not for the container to
            # appear. `_finalize` attaches the container and only ~75 lines
            # later installs the render wiring, emitting "Found N…" last of all
            # — so `diffraction_vectors is not None` is true well before the
            # result window can draw anything, and reading pixels off the back
            # of it reads across that gap. This used to be covered by polling
            # the pixels for 30 s, which on a loaded runner was not always long
            # enough: "result window still renders zeros", #149 ubuntu-py3.11.
            assert _wait(lambda: any(
                isinstance(m, dict)
                and "diffraction vectors" in str(m.get("text", ""))
                for m in captured_messages)), \
                "the batch never reported that it had finished"
            # Which makes the two halves separable: is the renderer installed,
            # and does it draw? A pixel poll could not tell those apart.
            assert sel.children.get(sp) is getattr(vtree, "_render_frame_fn", None), \
                "the result window's renderer was never installed"
            frame = sel.children[sp](sel, sp, np.array([[ix, iy]]))
            assert _peak(frame) > 0, "the installed renderer still draws zeros"

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
            close_session(session)

    def test_rejects_non_4d_dataset(self):
        from spyde.actions.context import ActionContext
        session = make_session()
        try:
            s = hs.signals.Signal2D(np.random.RandomState(0).rand(16, 16).astype(np.float32))
            session._add_signal(s, source_path=None)
            _settle(session)
            plot = _signal_plot(session)
            before = len(session.signal_trees)
            session._dispatch_toolbar_action(plot, "Find Diffraction Vectors", {})
            _settle(session)
            # No vectors tree created for a 2-D image.
            assert len(session.signal_trees) == before
        finally:
            close_session(session)
