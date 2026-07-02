"""
test_wizard_double_fire.py — the React StrictMode double-mount contract.

StrictMode mounts a wizard TWICE synchronously (mount → cleanup → remount),
firing open, close, open before any worker thread lands. Every wizard whose
mount fires a staged handler must guard with the run/stop generation pair
(lifecycle.bump_generation / is_current): the close bumps the generation
FIRST, so the superseded open's deferred build is dropped on arrival.

Strain's guard is covered by test_strain_mapping.py::
test_strict_mode_double_mount_builds_only_one_controller; this file covers the
find-vectors preview pair. (Center-Zero-Beam's mount path is fully synchronous
— no worker — so it cannot race and needs no guard.)
"""
from __future__ import annotations

import threading
import time


def _signal_plot(session):
    for p in session._plots:
        if not p.is_navigator and p.plot_state is not None:
            return p
    return None


def _join_threads(name, timeout=10.0):
    for t in threading.enumerate():
        if t.name == name:
            t.join(timeout=timeout)


class TestFindVectorsPreviewDoubleFire:
    def test_strict_mode_leaves_exactly_one_live_preview(self, stem_4d_dataset, monkeypatch):
        """fv_open, fv_close, fv_open in a row → exactly ONE preview overlay is
        ALIVE at the end. (The superseded worker may or may not get as far as
        attaching before the stop bumps the generation — either way it must
        remove what it attached, never stacking a second live overlay.)"""
        from spyde.actions import find_vectors_action as fva
        from spyde.actions import vector_overlay as vo

        session = stem_4d_dataset["window"]
        plot = _signal_plot(session)
        tree = plot.signal_tree

        returned = []
        orig = vo.attach_find_vectors_preview

        def _tracking(*a, **kw):
            ov = orig(*a, **kw)
            ov._test_removed = False
            orig_remove = ov.remove

            def _rm():
                ov._test_removed = True
                return orig_remove()

            ov.remove = _rm
            returned.append(ov)
            return ov

        monkeypatch.setattr(vo, "attach_find_vectors_preview", _tracking)

        # The exact StrictMode sequence, synchronous, before any worker lands.
        fva.fv_preview(session, plot, {})
        fva.fv_stop(session, plot, {})
        fva.fv_preview(session, plot, {})
        _join_threads("fv-preview")
        time.sleep(0.2)

        alive = [ov for ov in returned if not ov._test_removed]
        assert len(alive) == 1, (
            f"expected exactly 1 live preview overlay, got {len(alive)} "
            f"of {len(returned)} attached")
        assert getattr(tree, "_fv_preview", None) is alive[0]

        # And a final close cleans it up.
        fva.fv_stop(session, plot, {})
        assert getattr(tree, "_fv_preview", None) is None
        assert all(ov._test_removed for ov in returned)
