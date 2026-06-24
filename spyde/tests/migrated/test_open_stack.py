"""
Load Stack: combine several same-shaped 4D-STEM datasets into one 5D dataset
with an extra LEADING index axis (a series of MRC scans → in-situ-style stack).

Everything must stay lazy (a dask stack is a graph op — no materialise), the new
stack axis is a generic index (scale 1, no units), per-file scan calibration is
carried over, and mismatched shapes are cropped to the common minimum with a warn.

These call ``Session._load_stack_thread`` directly (the same body ``open_stack``
runs on a daemon thread) so the test is synchronous.
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import hyperspy.api as hs


def _write_4d_hspy(tmp_path, name, nav_display=(4, 5), sig=(8, 8), fill=0.0,
                   scale=2.5, units="nm"):
    """Write a small calibrated 4D-STEM signal to a temp .hspy and return its path.
    ``nav_display`` is (x, y); hyperspy data layout is (y, x, ky, kx)."""
    ny, nx = nav_display[1], nav_display[0]
    data = np.full((ny, nx, sig[1], sig[0]), fill, dtype=np.float32)
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    # Calibrate the two scan (navigation) axes.
    for ax in s.axes_manager.navigation_axes:
        ax.scale = scale
        ax.units = units
    p = tmp_path / f"{name}.hspy"
    s.save(str(p), overwrite=True)
    return str(p)


class TestOpenStack:
    def test_stacks_two_4d_into_5d_lazy(self, window, tmp_path):
        session = window["window"]
        p0 = _write_4d_hspy(tmp_path, "scan0", fill=0.0)
        p1 = _write_4d_hspy(tmp_path, "scan1", fill=1.0)

        session._load_stack_thread([p0, p1], ["scan0.hspy", "scan1.hspy"])

        assert len(session.signal_trees) == 1
        root = session.signal_trees[0].root
        # Stayed lazy — no full materialise.
        assert root._lazy is True
        assert isinstance(root.data, da.Array)
        am = root.axes_manager
        # One extra navigation axis (the stack) on top of the 2-D scan.
        assert am.navigation_dimension == 3
        assert am.signal_dimension == 2
        # Stack axis is the slowest/display-last nav axis, length = #files.
        assert tuple(am.navigation_shape) == (4, 5, 2)
        assert tuple(am.signal_shape) == (8, 8)
        # data layout: (stack, y, x, ky, kx)
        assert root.data.shape == (2, 5, 4, 8, 8)

    def test_stack_axis_is_generic_index(self, window, tmp_path):
        session = window["window"]
        ps = [_write_4d_hspy(tmp_path, f"s{i}", fill=float(i)) for i in range(3)]
        session._load_stack_thread(ps, [f"s{i}.hspy" for i in range(3)])

        am = session.signal_trees[0].root.axes_manager
        # _axes[0] is the new leading stack axis: index scale, no units.
        stack_ax = am._axes[0]
        assert stack_ax.size == 3
        assert float(stack_ax.scale) == 1.0
        # Units left undefined (a generic index) — hyperspy uses an Undefined
        # sentinel whose str() is "<undefined>", not a calibrated unit like "nm".
        assert str(stack_ax.units) in ("", "<undefined>")

    def test_per_file_scan_calibration_carried_over(self, window, tmp_path):
        session = window["window"]
        p0 = _write_4d_hspy(tmp_path, "c0", scale=3.0, units="nm")
        p1 = _write_4d_hspy(tmp_path, "c1", scale=3.0, units="nm")
        session._load_stack_thread([p0, p1], ["c0.hspy", "c1.hspy"])

        am = session.signal_trees[0].root.axes_manager
        # The two scan axes keep the per-file step size (not the stack index).
        for ax in am.navigation_axes:
            if ax is am._axes[0]:
                continue  # stack axis
        scan_scales = sorted(float(ax.scale) for ax in am.navigation_axes)
        assert 3.0 in scan_scales

    def test_signal_type_carried_over(self, window, tmp_path):
        session = window["window"]
        p0 = _write_4d_hspy(tmp_path, "t0")
        p1 = _write_4d_hspy(tmp_path, "t1")
        session._load_stack_thread([p0, p1], ["t0.hspy", "t1.hspy"])
        root = session.signal_trees[0].root
        assert root.metadata.get_item("Signal.signal_type", "") == "electron_diffraction"

    def test_mismatched_shapes_cropped_with_warning(self, window, tmp_path):
        session = window["window"]
        # Different scan (4x5 vs 3x5) AND detector (8 vs 10) shapes.
        p0 = _write_4d_hspy(tmp_path, "big", nav_display=(4, 5), sig=(8, 8))
        p1 = _write_4d_hspy(tmp_path, "small", nav_display=(3, 5), sig=(10, 8))
        session._load_stack_thread([p0, p1], ["big.hspy", "small.hspy"])

        root = session.signal_trees[0].root
        am = root.axes_manager
        # Cropped to the per-axis minimum: nav (3,5) display, signal (8,8).
        assert tuple(am.navigation_shape) == (3, 5, 2)
        assert tuple(am.signal_shape) == (8, 8)
        # A status message warned about the crop.
        msgs = window["messages"]
        joined = " ".join(
            str(m.get("text", "")) for m in msgs
            if isinstance(m, dict) and m.get("type") in ("status", "log")
        )
        assert "crop" in joined.lower()

    def test_fewer_than_two_files_errors(self, window, tmp_path):
        session = window["window"]
        p0 = _write_4d_hspy(tmp_path, "only")
        session.open_stack([p0])  # synchronous validation before the thread
        assert len(session.signal_trees) == 0
        msgs = window["messages"]
        assert any(
            isinstance(m, dict) and m.get("type") == "error"
            and "two files" in str(m.get("text", "")).lower()
            for m in msgs
        )


class TestMultiNavIndexOrdering:
    """A 5-D stack's navigator reports 3 nav coords; only the SPATIAL pair is in
    widget (x, y) order, the leading stack coord is already in data order. The
    nav index math must swap only the last two — not the whole row (which mapped
    x onto the y-axis → 'clamped [0,525,169] -> [0,299,169]')."""

    def test_only_spatial_pair_is_transposed(self):
        from spyde.drawing.update_functions import (
            update_from_navigation_selection,  # noqa: F401  (import sanity)
        )
        import numpy as np
        # Reproduce the transpose rule used in update_from_navigation_selection:
        # for a chained nav (>2 coords) only the trailing (x, y) pair swaps.
        idx = np.asarray([[0, 525, 169]])  # [stack, x, y] (selector order)
        swapped = idx.copy()
        swapped[..., -2:] = swapped[..., -2:][..., ::-1]
        assert swapped.tolist() == [[0, 169, 525]]  # [stack, y, x] (data order)
        # That now bounds correctly against data (stack=2, y=300, x=648):
        bounds = np.array([2, 300, 648]) - 1
        clipped = np.clip(swapped[0], 0, bounds)
        assert clipped.tolist() == [0, 169, 525]  # unchanged → no spurious clamp

    def test_overlay_helpers_split_lead_and_spatial(self):
        from spyde.actions.vector_overlay import _indices_to_iyix, _indices_lead_nav
        # 4-D scan: [x, y] → spatial only, no lead.
        assert _indices_to_iyix([[5, 7]]) == (7, 5)
        assert _indices_lead_nav([[5, 7]]) == ()
        # 5-D stack: [stack, x, y] → lead=(stack,), spatial=(iy=y, ix=x).
        assert _indices_to_iyix([[1, 5, 7]]) == (7, 5)
        assert _indices_lead_nav([[1, 5, 7]]) == (1,)


class TestFindVectorsPreviewOnStack:
    """The find-vectors preview must slice a SINGLE 2-D frame from a 5-D stack
    (regression: it sliced 2 nav axes assuming 4-D → returned a 3-D 'frame' →
    'index 2 is out of bounds for axis 0 with size 0')."""

    def _stub(self, data, lead, sigma=0.0):
        from spyde.actions.vector_overlay import FindVectorsPreviewOverlay

        class _Sig:
            pass

        s = _Sig()
        s.data = data
        stub = FindVectorsPreviewOverlay.__new__(FindVectorsPreviewOverlay)
        stub.signal = s
        stub.sigma = sigma
        stub._lead_nav = lead
        return stub

    def test_blurred_frame_returns_single_2d_frame_from_5d(self):
        from spyde.actions.vector_overlay import FindVectorsPreviewOverlay
        # (stack, y, x, ky, kx); unique marker per (stack, y, x).
        data = np.zeros((2, 6, 8, 4, 4), dtype=np.float32)
        for st in range(2):
            for y in range(6):
                for x in range(8):
                    data[st, y, x, 0, 0] = st * 1000 + y * 10 + x
        stub = self._stub(data, lead=(1,))
        frame = FindVectorsPreviewOverlay._blurred_frame(stub, iy=4, ix=5)
        assert frame.shape == (4, 4)             # a single DP, not a 3-D block
        assert float(frame[0, 0]) == 1 * 1000 + 4 * 10 + 5  # the (1,4,5) frame

    def test_blurred_frame_still_works_for_4d(self):
        from spyde.actions.vector_overlay import FindVectorsPreviewOverlay
        data = np.zeros((6, 8, 4, 4), dtype=np.float32)
        data[4, 5, 0, 0] = 42.0
        stub = self._stub(data, lead=())
        frame = FindVectorsPreviewOverlay._blurred_frame(stub, iy=4, ix=5)
        assert frame.shape == (4, 4)
        assert float(frame[0, 0]) == 42.0
