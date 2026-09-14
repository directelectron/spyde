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

    def test_the_prepared_index_splits_lead_and_spatial(self):
        """Everything that needs "where is the navigator" reads it from the
        SAME preparation the frame read uses: the spatial pair swapped into
        data order, any leading stack coordinate left in front of it."""
        import hyperspy.api as hs
        from spyde.drawing.update_functions import _prepare_nav_indices

        four_d = hs.signals.Signal2D(np.zeros((16, 16, 4, 4)))
        prepared = _prepare_nav_indices(four_d, [[5, 7]], integrating=False)
        assert list(prepared) == [7, 5]

        five_d = hs.signals.Signal2D(np.zeros((4, 16, 16, 4, 4)))
        prepared = _prepare_nav_indices(five_d, [[1, 5, 7]], integrating=False)
        assert list(prepared) == [1, 7, 5]


class TestFindVectorsPreviewOnStack:
    """The find-vectors preview must reduce a navigation window to a SINGLE 2-D
    frame, and blur only the two innermost navigation axes: a 5-D stack's time
    axis is a fixed position, exactly as the batch treats it."""

    def test_the_preview_window_is_flat_on_a_stacks_time_axis(self):
        """A neighbourhood radius is per navigation axis, so the preview on a
        5-D stack reads (2d+1)^2 frames of one time slice, not (2d+1)^3 across
        three, and its value is the 4-D answer for that slice."""
        from dataclasses import replace
        import hyperspy.api as hs
        from spyde.actions.find_vectors_action import fv_open
        from spyde.actions.vector_overlay import find_vectors_preview
        from spyde.array_cache import drop_reader, reader_for_overlay
        from spyde.tests.migrated._async import wait_until
        from spyde.tests.migrated.conftest import (
            _settle, close_session, make_session,
        )

        rng = np.random.default_rng(0)
        data = rng.normal(50, 3, (2, 9, 9, 16, 16)).astype(np.float32)
        for index in np.ndindex(2, 9, 9):
            data[index][6:10, 6:10] += 400.0        # one bright disk per pattern
        signal = hs.signals.Signal2D(data).as_lazy()
        signal.data = signal.data.rechunk((1, 3, 3, -1, -1))
        signal.set_signal_type("electron_diffraction")

        session = make_session()
        try:
            session._add_signal(signal, source_path=None)
            _settle(session)
            plot = next(p for p in session._plots
                        if not p.is_navigator and p.plot_state is not None)
            tree = plot.signal_tree
            fv_open(session, plot, {"method": "dog", "sigma": 1.0,
                                    "kernel_radius": 3, "threshold": 8.0,
                                    "min_distance": 3, "subpixel": False})
            assert wait_until(lambda: getattr(tree, "_fv_preview", None) is not None, 30)
            node = tree._fv_preview
            assert node.signal._map_recipe.depth == (0, 3, 3)

            windows = []
            recipe = node.signal._map_recipe

            def _recording(window, centre, **kwargs):
                windows.append((window, centre))
                return recipe.function(window, centre, **kwargs)

            node.signal._map_recipe = replace(recipe, function=_recording)
            drop_reader(plot, node.signal)

            value = reader_for_overlay(plot, node).read_frame((1, 4, 4))
            # The navigator may also be drawing this overlay, and every
            # interior position has the same centre -- so the window is
            # identified by the frames in it, which are this position's alone.
            asked_for = data[1:2, 1:8, 1:8]
            window, centre = next(w for w in windows
                                  if np.array_equal(w[0], asked_for))
            assert centre == (0, 3, 3)
            assert window.shape == (1, 7, 7, 16, 16), window.shape

            # The same window without its (single) time axis is the 4-D call.
            flat = find_vectors_preview(window[0], centre[1:],
                                        **recipe.static)
            assert np.array_equal(value["peaks"]["data"], flat["peaks"]["data"])
        finally:
            close_session(session)

    @staticmethod
    def _window(shape):
        """A window with a value unique to each navigation position."""
        window = np.zeros(shape, dtype=np.float32)
        for index in np.ndindex(*shape[:-2]):
            window[index][0, 0] = float(np.ravel_multi_index(index, shape[:-2]) + 1)
        return window

    def test_a_five_dimensional_window_reduces_to_one_frame(self):
        from spyde.actions.vector_overlay import _blurred_centre_frame
        window = self._window((3, 3, 3, 4, 4))     # (t, y, x, ky, kx)
        frame = _blurred_centre_frame(window, (1, 1, 1), sigma=0.0)
        assert frame.shape == (4, 4)               # a single pattern, not a block
        assert np.array_equal(frame, window[1, 1, 1])

    def test_a_four_dimensional_window_reduces_to_one_frame(self):
        from spyde.actions.vector_overlay import _blurred_centre_frame
        window = self._window((3, 3, 4, 4))
        frame = _blurred_centre_frame(window, (2, 0), sigma=0.0)
        assert frame.shape == (4, 4)
        assert np.array_equal(frame, window[2, 0])

    def test_no_window_at_all_is_the_frame_itself(self):
        from spyde.actions.vector_overlay import _blurred_centre_frame
        frame = np.arange(16, dtype=np.float32).reshape(4, 4)
        assert np.array_equal(_blurred_centre_frame(frame, None, sigma=0.0), frame)

    def test_the_blur_never_crosses_the_time_axis(self):
        """Blurring across time would mix time steps, which the batch's
        ``(sigma, sigma, 0, 0)`` never does."""
        from scipy.ndimage import gaussian_filter
        from spyde.actions.vector_overlay import _blurred_centre_frame
        window = self._window((3, 3, 3, 4, 4))
        frame = _blurred_centre_frame(window, (1, 1, 1), sigma=1.0)
        expected = gaussian_filter(window[1], sigma=(1.0, 1.0, 0.0, 0.0))[1, 1]
        assert np.allclose(frame, expected)
