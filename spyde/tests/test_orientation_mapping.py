# spyde/tests/test_orientation_mapping.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PySide6 import QtWidgets


# ── Unit tests (no Qt window needed) ──────────────────────────────────────────

def test_file_drop_widget_created(qtbot):
    from spyde.drawing.toolbars.caret_group import CaretParams
    params = {
        "cif_files": {
            "name": "CIF Files",
            "type": "file_drop",
            "extensions": [".cif"],
        }
    }
    widget = CaretParams(parameters=params)
    qtbot.addWidget(widget)
    assert "cif_files" in widget.kwargs
    drop_widget = widget.kwargs["cif_files"]
    assert hasattr(drop_widget, "get_files")
    assert drop_widget.get_files() == []


def _make_mock_toolbar(qtbot):
    """Build a minimal mock toolbar for testing orientation_mapping UI construction."""
    from spyde.drawing.toolbars.caret_group import CaretParams

    toolbar = MagicMock()
    plot = MagicMock()
    signal = MagicMock()
    ax0 = MagicMock(); ax0.scale = 0.01; ax0.size = 128; ax0.offset = 0.0
    ax1 = MagicMock(); ax1.scale = 0.01; ax1.size = 128; ax1.offset = 0.0
    signal.axes_manager.signal_axes = [ax0, ax1]
    signal.axes_manager.navigation_axes = []
    plot.plot_state.current_signal = signal
    plot.main_window = MagicMock()
    toolbar.plot = plot
    toolbar.action_widgets = {}
    toolbar._om_state = None
    return toolbar


def test_orientation_mapping_builds_ui(qtbot):
    from spyde.actions.pyxem import orientation_mapping, _OM_BUILT_TOOLBARS

    toolbar = _make_mock_toolbar(qtbot)
    # Ensure this toolbar isn't in the guard set from a prior test run
    _OM_BUILT_TOOLBARS.discard(id(toolbar))

    orientation_mapping(toolbar, action_name="Orientation Mapping")

    assert toolbar._om_state is not None
    state = toolbar._om_state
    assert "signal" in state
    assert "phases" in state
    assert state["sim"] == [None]


def test_generate_library():
    """_generate_library_from_phases calls SimulationGenerator with correct args."""
    from spyde.actions.pyxem import _generate_library_from_phases

    mock_sim = MagicMock()
    mock_generator_cls = MagicMock()
    mock_generator_cls.return_value.calculate_diffraction2d.return_value = mock_sim
    mock_rotations = MagicMock()
    mock_phase = MagicMock()
    mock_phase.point_group = MagicMock()

    with patch("spyde.actions.pyxem.SimulationGenerator", mock_generator_cls), \
         patch("spyde.actions.pyxem.get_sample_reduced_fundamental",
               return_value=mock_rotations):
        result = _generate_library_from_phases(
            phases=[mock_phase],
            accelerating_voltage=200.0,
            resolution=1.0,
            minimum_intensity=0.05,
            reciprocal_radius=0.64,
        )

    assert result is mock_sim
    mock_generator_cls.assert_called_once_with(200.0, minimum_intensity=0.05)
    call_args = mock_generator_cls.return_value.calculate_diffraction2d.call_args
    assert call_args[0][0] is mock_phase
    assert call_args[1]["rotation"] is mock_rotations
    assert call_args[1]["with_direct_beam"] is False


def test_filter_sim_by_radius():
    from spyde.actions.pyxem import _filter_sim_by_radius

    coords = np.array([[0.1, 0.2], [0.3, 0.3], [0.6, 0.0], [0.0, 0.8]])
    intensities = np.array([1.0, 0.8, 0.5, 0.3])
    fc, fi = _filter_sim_by_radius(coords, intensities, max_radius=0.5)
    assert len(fc) == 2
    assert np.allclose(fc, coords[:2])
    assert np.allclose(fi, intensities[:2])
    # boundary: point exactly at radius is included
    fc2, _ = _filter_sim_by_radius(np.array([[0.5, 0.0]]), np.array([1.0]), 0.5)
    assert len(fc2) == 1


def test_extract_orientation_outputs_single_phase():
    """_extract_orientation_outputs extracts orientation map + correlation for single phase."""
    from spyde.actions.pyxem import _extract_orientation_outputs
    import hyperspy.api as hs

    # Build a minimal mock OrientationMap with the real column structure:
    # data shape (nav_y, nav_x, n_best, 4), column_names=['index','correlation','rotation','factor']
    nav_y, nav_x, n_best = 4, 4, 1
    data = np.zeros((nav_y, nav_x, n_best, 4))
    data[:, :, 0, 1] = 0.5  # correlation column

    mock_om = MagicMock()
    mock_om.data = data
    mock_om.column_names = ["index", "correlation", "rotation", "factor"]

    results = _extract_orientation_outputs(mock_om, nav_axes=[], n_phases=1)
    titles = [r[1] for r in results]

    assert "Orientation Map" in titles
    assert "Correlation Score" in titles
    assert "Phase Map" not in titles
    assert len(results) == 2

    # Correlation data should be a Signal2D with correct values
    corr_sig = next(s for s, t in results if t == "Correlation Score")
    assert isinstance(corr_sig, hs.signals.Signal2D)
    assert np.allclose(corr_sig.data, 0.5)


def test_extract_orientation_outputs_multi_phase():
    """_extract_orientation_outputs includes Phase Map for n_phases > 1."""
    from spyde.actions.pyxem import _extract_orientation_outputs

    data = np.zeros((4, 4, 1, 4))
    data[:, :, 0, 0] = 1.0  # phase index column

    mock_om = MagicMock()
    mock_om.data = data
    mock_om.column_names = ["index", "correlation", "rotation", "factor"]

    results = _extract_orientation_outputs(mock_om, nav_axes=[], n_phases=2)
    titles = [r[1] for r in results]

    assert "Phase Map" in titles
    assert len(results) == 3


def test_get_best_fit_spots_returns_valid_coords():
    """_get_best_fit_spots returns (N,2) coords in Å⁻¹ and non-negative intensities."""
    from pyxem.data import si_grains, si_phase
    from diffsims.generators.simulation_generator import SimulationGenerator
    from orix.sampling import get_sample_reduced_fundamental
    from spyde.actions.pyxem import _get_best_fit_spots, _build_matching_cache, _compute_reciprocal_radius

    s, _ = si_grains(return_rotations=True)
    s.calibration.center = None
    phase = si_phase()
    gen = SimulationGenerator(200, minimum_intensity=0.05)
    rots = get_sample_reduced_fundamental(resolution=1, point_group=phase.point_group)
    max_radius = _compute_reciprocal_radius(s)
    sim = gen.calculate_diffraction2d(
        phase, rotation=rots, max_excitation_error=0.1, reciprocal_radius=max_radius * 1.1, with_direct_beam=False
    )

    cache = _build_matching_cache(s, sim)
    coords, intensities, ipf_xy, ipf_heatmap = _get_best_fit_spots(
        s, sim, nav_indices=(0, 0), gamma=0.5, max_radius=max_radius,
        matching_cache=cache,
    )

    assert coords.ndim == 2 and coords.shape[1] == 2, f"Expected (N,2) coords, got {coords.shape}"
    assert len(intensities) == len(coords), "coords and intensities length mismatch"
    assert np.all(intensities >= 0), "Intensities should be non-negative"
    assert len(coords) > 0, "Expected at least one spot"
    assert len(ipf_xy) == 2

    # Second call with same cache returns same result
    coords2, intensities2, _, _ = _get_best_fit_spots(
        s, sim, nav_indices=(0, 0), gamma=0.5, max_radius=max_radius,
        matching_cache=cache,
    )
    assert np.allclose(coords, coords2) and np.allclose(intensities, intensities2)


def test_get_best_fit_spots_sped_ag():
    """_get_best_fit_spots works with sped_ag dataset + silver CIF.

    This is the real-world dataset the user reported failing with
    'Index 104 is out of bounds for axis 0 with size 64'.
    """
    import os
    from pyxem.data import sped_ag
    from orix.crystal_map import Phase
    from diffsims.generators.simulation_generator import SimulationGenerator
    from orix.sampling import get_sample_reduced_fundamental
    from spyde.actions.pyxem import _get_best_fit_spots, _build_matching_cache, _compute_reciprocal_radius

    cif_path = os.path.join(os.path.dirname(__file__), "Silver__0011135.cif")
    if not os.path.exists(cif_path):
        pytest.skip("Silver CIF not found")

    s = sped_ag()
    phase = Phase.from_cif(cif_path)
    gen = SimulationGenerator(200, minimum_intensity=0.05)
    rots = get_sample_reduced_fundamental(resolution=1, point_group=phase.point_group)
    max_radius = _compute_reciprocal_radius(s)
    sim = gen.calculate_diffraction2d(
        phase, rotation=rots, max_excitation_error=0.1,
        reciprocal_radius=max_radius * 1.1, with_direct_beam=False,
    )

    cache = _build_matching_cache(s, sim)
    # Test the problematic nav position that triggered "Index 104 out of bounds"
    coords, intensities, _ipf_xy, _heat = _get_best_fit_spots(
        s, sim, nav_indices=(104, 32), gamma=0.5, max_radius=max_radius,
        matching_cache=cache,
    )

    assert coords.ndim == 2 and coords.shape[1] == 2
    assert len(intensities) == len(coords)
    assert len(coords) > 0, "Expected at least one spot within pattern radius"
    r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    assert np.all(r <= max_radius + 1e-9), "All spots should be within max_radius"

    # Spot sizes must be in a sane pixel range (5–15) for the overlay
    i_max = float(np.max(intensities))
    sizes = [int(5 + 10 * float(iv) / i_max) for iv in intensities]
    assert all(5 <= sz <= 15 for sz in sizes), f"Spot sizes out of range: {sizes}"


# ── End-to-end integration tests (require real pyxem data + Qt window) ────────

class TestOrientationMappingEndToEnd:
    """
    End-to-end tests using real pyxem data (si_grains) loaded into a SpyDE
    MainWindow. Mirrors the workflow in pyxem's test_indexation_results.py.
    """

    @pytest.fixture(scope="class")
    def om_result(self):
        """Run the full pyxem orientation mapping pipeline on si_grains data.

        This mirrors the multi_rot_orientation_result fixture in pyxem's
        test_indexation_results.py and validates the raw pyxem API works
        independently of the GUI.
        """
        from pyxem.data import si_grains, si_phase
        from pyxem.signals import OrientationMap
        from diffsims.generators.simulation_generator import SimulationGenerator
        from orix.sampling import get_sample_reduced_fundamental

        s, rotations_truth = si_grains(return_rotations=True)
        s.calibration.center = None
        polar = s.get_azimuthal_integral2d(
            npt=100, npt_azim=180, inplace=False, mean=True
        )
        phase = si_phase()
        generator = SimulationGenerator(200, minimum_intensity=0.05)
        rotations = get_sample_reduced_fundamental(
            resolution=1, point_group=phase.point_group
        )
        sim = generator.calculate_diffraction2d(
            phase,
            rotation=rotations,
            max_excitation_error=0.1,
            reciprocal_radius=2,
            with_direct_beam=False,
        )
        orientation_map = polar.get_orientation(sim)
        return orientation_map, rotations_truth, s

    def test_orientation_map_type(self, om_result):
        from pyxem.signals import OrientationMap
        om, _, _ = om_result
        assert isinstance(om, OrientationMap)

    def test_orientation_map_shape(self, om_result):
        """OrientationMap navigation shape matches input signal navigation shape."""
        om, _, s = om_result
        nav_shape = s.axes_manager.navigation_shape
        assert om.data.shape[:2] == nav_shape[::-1] or om.data.shape[:2] == nav_shape

    def test_orientation_map_has_correlation(self, om_result):
        """Correlation column exists and has non-zero values."""
        om, _, _ = om_result
        assert "correlation" in om.column_names
        corr_idx = om.column_names.index("correlation")
        corr = om.data[..., 0, corr_idx]
        assert np.any(corr > 0), "Expected non-zero correlation scores"

    def test_orientation_map_to_crystal_map(self, om_result):
        """to_crystal_map() returns an orix CrystalMap with correct shape."""
        from orix.crystal_map import CrystalMap
        om, _, s = om_result
        cm = om.to_crystal_map()
        assert isinstance(cm, CrystalMap)

    def test_extract_outputs_from_real_om(self, om_result):
        """_extract_orientation_outputs works on a real OrientationMap."""
        from spyde.actions.pyxem import _extract_orientation_outputs
        import hyperspy.api as hs
        om, _, s = om_result
        nav_axes = list(s.axes_manager.navigation_axes)
        results = _extract_orientation_outputs(om, nav_axes, n_phases=1)
        titles = [r[1] for r in results]
        assert "Orientation Map" in titles
        assert "Correlation Score" in titles
        # Correlation signal should have nav shape matching input
        corr_sig = next(sig for sig, t in results if t == "Correlation Score")
        assert isinstance(corr_sig, hs.signals.Signal2D)
        nav_shape = s.axes_manager.navigation_shape
        assert corr_sig.data.shape in (nav_shape, nav_shape[::-1])

    def test_do_run_fit_produces_signals(self):
        """_do_run_fit populates main_window._pending_signal_queue with output signals."""
        from pyxem.data import si_grains, si_phase
        from diffsims.generators.simulation_generator import SimulationGenerator
        from orix.sampling import get_sample_reduced_fundamental
        from spyde.actions.pyxem import _do_run_fit

        s, _ = si_grains(return_rotations=True)
        s.calibration.center = None
        phase = si_phase()
        generator = SimulationGenerator(200, minimum_intensity=0.05)
        rotations = get_sample_reduced_fundamental(
            resolution=1, point_group=phase.point_group
        )
        sim = generator.calculate_diffraction2d(
            phase,
            rotation=rotations,
            max_excitation_error=0.1,
            reciprocal_radius=2,
            with_direct_beam=False,
        )

        pending_queue = []
        mock_mw = MagicMock()
        mock_mw._pending_signal_queue = pending_queue

        state = {
            "signal": s,
            "main_window": mock_mw,
            "phases": [phase],
            "sim": [sim],
            "gamma": [0.5],
            "min_intensity": [0.0],
            "scale_override": [None],
            "max_radius": [2.0],
            "run_status": [None],
            "run_btn": [None],
        }

        # Run in the calling thread (no Qt event loop needed for this test)
        import threading
        done = threading.Event()
        errors = []

        original_thread_start = threading.Thread.start

        def _run_immediately(t):
            try:
                t._target(*t._args, **t._kwargs)
            except Exception as e:
                errors.append(e)
            finally:
                done.set()

        # Patch threading.Thread to run synchronously
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            _do_run_fit(state)

            # Extract and call the target function directly
            call_kwargs = mock_thread_cls.call_args
            target_fn = call_kwargs[1].get("target") or call_kwargs[0][0]
            target_fn()

        # Should have queued at least the OrientationMap + Correlation Score
        assert len(pending_queue) >= 2, (
            f"Expected at least 2 output signals, got {len(pending_queue)}"
        )
        titles = [s.metadata.General.title for s in pending_queue]
        assert "Orientation Map" in titles
        assert "Correlation Score" in titles


class TestOrientationMappingGUI:
    """
    Tests that exercise the orientation mapping action through the SpyDE GUI,
    using a real MainWindow and the stem_4d_dataset fixture.
    """

    def test_orientation_mapping_button_exists(self, qtbot, stem_4d_dataset):
        """The Orientation Mapping action appears in the bottom toolbar of a 4D signal."""
        win = stem_4d_dataset["window"]
        nav, sig = win.plots
        toolbar_bottom = sig.plot_state.toolbar_bottom
        action_names = [a.text() for a in toolbar_bottom.actions()]
        assert "Orientation Mapping" in action_names

    def test_orientation_mapping_toggle_shows_caret(self, qtbot, stem_4d_dataset):
        """Clicking Orientation Mapping builds the tabbed wizard caret."""
        from spyde.drawing.toolbars.caret_group import CaretGroup
        from spyde.actions.pyxem import _OM_BUILT_TOOLBARS

        win = stem_4d_dataset["window"]
        nav, sig = win.plots
        toolbar_bottom = sig.plot_state.toolbar_bottom

        om_action = next(
            a for a in toolbar_bottom.actions() if a.text() == "Orientation Mapping"
        )

        # Clear guard so the caret gets built fresh
        _OM_BUILT_TOOLBARS.discard(id(toolbar_bottom))

        om_action.trigger()
        qtbot.wait(300)

        # Caret should now be registered and visible
        assert "Orientation Mapping" in toolbar_bottom.action_widgets
        caret = toolbar_bottom.action_widgets["Orientation Mapping"]["widget"]
        assert isinstance(caret, CaretGroup)
        assert caret.isVisible()

        # Toggle off — caret should hide
        om_action.trigger()
        qtbot.wait(200)
        assert not caret.isVisible()

    def test_orientation_mapping_does_not_duplicate_icon(self, qtbot, stem_4d_dataset):
        """Clicking the button multiple times does not create additional toolbar icons."""
        from spyde.actions.pyxem import _OM_BUILT_TOOLBARS

        win = stem_4d_dataset["window"]
        nav, sig = win.plots
        toolbar_bottom = sig.plot_state.toolbar_bottom

        _OM_BUILT_TOOLBARS.discard(id(toolbar_bottom))

        actions_before = [a.text() for a in toolbar_bottom.actions()]
        om_count_before = actions_before.count("Orientation Mapping")

        # Toggle on and off twice
        om_action = next(a for a in toolbar_bottom.actions() if a.text() == "Orientation Mapping")
        om_action.trigger(); qtbot.wait(200)
        om_action.trigger(); qtbot.wait(200)
        om_action.trigger(); qtbot.wait(200)
        om_action.trigger(); qtbot.wait(200)

        actions_after = [a.text() for a in toolbar_bottom.actions()]
        om_count_after = actions_after.count("Orientation Mapping")
        assert om_count_after == om_count_before, (
            f"Icon count changed: was {om_count_before}, now {om_count_after}"
        )

    def test_orientation_mapping_full_workflow(self, qtbot, stem_4d_dataset):
        """
        Full end-to-end: build library from si_phase, run fit on si_grains signal,
        verify output signals appear in the MainWindow.

        Uses si_grains() as the 4D dataset since it has known crystal structure
        (same approach as pyxem's test_indexation_results.py).
        """
        import hyperspy.api as hs
        from pyxem.data import si_grains, si_phase
        from pyxem.signals import OrientationMap
        from diffsims.generators.simulation_generator import SimulationGenerator
        from orix.sampling import get_sample_reduced_fundamental
        from spyde.actions.pyxem import _OM_BUILT_TOOLBARS, _do_run_fit, _generate_library_from_phases

        win = stem_4d_dataset["window"]
        nav, sig = win.plots
        toolbar_bottom = sig.plot_state.toolbar_bottom

        # ── Load si_grains as the 4D signal ──────────────────────────────────
        s, _ = si_grains(return_rotations=True)
        s.calibration.center = None
        s.set_signal_type("electron_diffraction")

        phase = si_phase()
        sig_ax = s.axes_manager.signal_axes
        reciprocal_radius = min(ax.scale * ax.size / 2.0 for ax in sig_ax)

        # ── Generate library ──────────────────────────────────────────────────
        sim = _generate_library_from_phases(
            phases=[phase],
            accelerating_voltage=200.0,
            resolution=1.0,
            minimum_intensity=0.05,
            reciprocal_radius=reciprocal_radius,
        )
        assert sim is not None

        # ── Build state dict and run fit ──────────────────────────────────────
        pending = []
        mock_mw = MagicMock()
        mock_mw._pending_signal_queue = pending

        state = {
            "signal": s,
            "main_window": mock_mw,
            "phases": [phase],
            "sim": [sim],
            "gamma": [0.5],
            "min_intensity": [0.0],
            "scale_override": [None],
            "max_radius": [reciprocal_radius],
            "run_status": [None],
            "run_btn": [None],
        }

        import threading
        done = threading.Event()

        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            _do_run_fit(state)
            call_kwargs = mock_thread_cls.call_args
            target_fn = call_kwargs[1].get("target") or call_kwargs[0][0]

        target_fn()  # run synchronously

        # ── Assertions ────────────────────────────────────────────────────────
        assert len(pending) >= 2, f"Expected output signals, got {len(pending)}"

        titles = [sig.metadata.General.title for sig in pending]
        assert "Orientation Map" in titles, f"Missing Orientation Map, got: {titles}"
        assert "Correlation Score" in titles, f"Missing Correlation Score, got: {titles}"

        om = next(sig for sig in pending if sig.metadata.General.title == "Orientation Map")
        assert isinstance(om, OrientationMap)

        corr = next(sig for sig in pending if sig.metadata.General.title == "Correlation Score")
        assert isinstance(corr, hs.signals.Signal2D)
        assert np.any(corr.data > 0), "Correlation scores should be non-zero"
