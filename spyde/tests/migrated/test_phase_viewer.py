"""
test_phase_viewer.py — the phase viewer: master pattern, unit cell, detector,
and the microscope set-up they are simulated in.

The viewer draws the same bands as the EBSD wizard's dictionary, so the
contract that matters is that they agree: the preview at a crystal rotation
must be exactly ``simulate_patterns`` at the matching Bunge angles. Getting a
frame backwards would still draw plausible bands, just for the wrong
orientation, so this is pinned against the established function — with and
without a tilted sample — rather than eyeballed.
"""
import os
import sys
import types

import numpy as np
import pytest
import hyperspy.api as hs
from orix.crystal_map import Phase

from spyde.actions import composition as comp
from spyde.actions import phase_viewer as pv
from spyde.ebsd import microscope
from spyde.ebsd.bands import (
    band_intensity, band_sphere, cubic_reflectors, euler_to_matrix,
    matrix_to_euler, simulate_patterns, sphere_directions, wavelength,
)
from spyde.ebsd.spots import reflections_from_phase, spot_pixels
from spyde.ebsd.unit_cell import unit_cell

CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")
AG = {"id": "ag", "elements": ["Ag"], "cif_path": CIF, "label": "Ag"}


class _Tree:
    def __init__(self, signal):
        self.root = signal
        self.signal_plots = []


class _Session:
    """With no ``_dispatch_to_main`` the worker steps run inline."""


@pytest.fixture
def messages(monkeypatch):
    sent = []
    monkeypatch.setattr(pv, "emit", sent.append)
    monkeypatch.setattr(pv, "emit_status", lambda text: None)
    monkeypatch.setattr(pv, "emit_error", lambda text: sent.append(("error", text)))
    monkeypatch.setattr(pv, "SPHERE_TEXTURE", (32, 64))
    return sent


def _sem_tree(*phases, **metadata):
    signal = hs.signals.Signal2D(np.zeros((2, 2, 30, 40), np.float32))
    signal.metadata.set_item("Acquisition_instrument.SEM.beam_energy",
                             metadata.pop("beam_energy", 20.0))
    for key, value in metadata.items():
        signal.metadata.set_item(key, value)
    tree = _Tree(signal)
    comp.write_phases(tree, list(phases) or [AG])
    return tree


def _open(monkeypatch, tree, phase_id="ag"):
    monkeypatch.setattr(comp, "_src_plot_tree", lambda session, plot: (None, tree))
    pv.show_phase(_Session(), None, {"phase": phase_id})
    return tree._phase_viewers[phase_id]


def _setup(kind="SEM", **values):
    fields = microscope.FIELDS[kind]
    merged = {f.key: f.default for f in fields} | values
    return microscope.Setup(kind, merged, {k: "default" for k in merged})


class TestOneBandProfile:
    def test_simulate_patterns_is_band_intensity_on_the_detector(self):
        from spyde.ebsd.bands import detector_directions
        reflectors = cubic_reflectors()
        euler = np.array([[0.3, 0.7, 1.1]])
        old_style = simulate_patterns(euler, reflectors, (30, 40), (0.5, 0.5, 0.55))
        rot = euler_to_matrix(*euler[0])
        # rays into the sample frame: the default 70° sample tilt
        rays = detector_directions((30, 40), (0.5, 0.5, 0.55)).reshape(-1, 3) \
            @ microscope._sample_to_detector(70.0, 0.0, 0.0, 0.0)
        direct = band_intensity(rays, reflectors,
                                normals=reflectors.normals @ rot).reshape(30, 40)
        np.testing.assert_allclose(old_style[0], direct, rtol=1e-6)

    def test_the_sphere_is_band_intensity_over_all_directions(self):
        reflectors = cubic_reflectors()
        np.testing.assert_allclose(
            band_sphere(reflectors, (20, 40)).ravel(),
            band_intensity(sphere_directions(20, 40), reflectors))

    def test_euler_round_trip(self):
        rng = np.random.default_rng(1)
        for euler in rng.uniform([0, 0.05, 0], [2 * np.pi, np.pi - 0.05, 2 * np.pi],
                                 (200, 3)):
            m = euler_to_matrix(*euler)
            np.testing.assert_allclose(euler_to_matrix(*matrix_to_euler(m)), m,
                                       atol=1e-12)


class TestSetup:
    def test_the_microscope_is_read_off_the_signal(self):
        assert microscope.microscope_of(_sem_tree().root) == "SEM"
        assert microscope.microscope_of(
            hs.signals.Signal2D(np.zeros((8, 8)))) == "TEM"

    def test_values_come_from_metadata_and_say_so(self):
        setup = microscope.read_setup(_sem_tree(beam_energy=15.0).root)
        assert setup.microscope == "SEM" and setup["beam_energy"] == 15.0
        assert setup.sources["beam_energy"] == "metadata"
        assert setup.sources["azimuthal"] == "default"

    def test_a_recorded_detector_supplies_the_angles(self):
        """A kikuchipy EBSD signal carries its detector: its angles and PC
        count as recorded with the data."""
        class Detector:
            sample_tilt, tilt, azimuthal, twist = 68.0, 3.0, 1.5, 0.5
            pc_average = np.array([0.45, 0.6, 0.58])

        signal = _sem_tree().root
        signal.detector = Detector()
        setup = microscope.read_setup(signal)
        assert (setup["sample_tilt"], setup["detector_tilt"], setup["azimuthal"],
                setup["twist"], setup["pc_y"]) == (68.0, 3.0, 1.5, 0.5, 0.6)
        assert setup.sources["twist"] == "data"

    def test_writing_a_field_records_it(self):
        tree = _sem_tree()
        microscope.write_field(tree.root, "SEM", "twist", 1.25)
        assert tree.root.metadata.get_item(
            "Acquisition_instrument.SEM.Detector.EBSD.twist") == 1.25

    def test_the_sample_normal_sits_above_the_pattern_centre(self):
        to_sample = microscope.detector_to_sample(_setup("SEM"))
        normal_on_screen = to_sample.T @ np.array([0.0, 0.0, 1.0])
        assert normal_on_screen[1] > 0.3            # up, by sin(20°)
        np.testing.assert_allclose(np.degrees(np.arccos(normal_on_screen[2])), 20)

    def test_the_tilts_are_kikuchipys(self):
        """The four SEM angles follow kikuchipy, the reference for EBSD
        geometry, exactly."""
        kp = pytest.importorskip("kikuchipy")
        rng = np.random.default_rng(2)
        for angles in rng.uniform([0, -30, -30, -30], [90, 30, 30, 30], (50, 4)):
            detector = kp.detectors.EBSDDetector(
                (60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=angles[0],
                tilt=angles[1], azimuthal=angles[2], twist=angles[3])
            np.testing.assert_allclose(
                microscope._sample_to_detector(*angles),
                detector.sample_to_detector.to_matrix()[0], atol=1e-12)


class TestPreviewParity:
    @pytest.mark.parametrize("sample_tilt", [90.0, 70.0, 55.0])
    @pytest.mark.parametrize("euler", [(0.0, 0.0, 0.0), (0.4, 0.9, 1.3)])
    def test_preview_matches_simulate_patterns(self, monkeypatch, messages,
                                               sample_tilt, euler):
        """The viewer's detector pane is the EBSD wizard's simulation at the
        same orientation and the same set-up — one geometry, not two."""
        tree = _sem_tree(**{"Acquisition_instrument.SEM.Stage.tilt_alpha": sample_tilt})
        viewer = _open(monkeypatch, tree)
        viewer.rotation = euler_to_matrix(*euler).T     # crystal → sample
        expected = simulate_patterns(
            np.array([euler]), viewer.reflectors(), viewer.shape,
            (viewer.setup["pc_x"], viewer.setup["pc_y"], viewer.setup["pc_z"]),
            sample_tilt=viewer.setup["sample_tilt"],
            detector_tilt=viewer.setup["detector_tilt"],
            azimuthal=viewer.setup["azimuthal"], twist=viewer.setup["twist"])[0]
        np.testing.assert_allclose(viewer.pattern(), expected, rtol=1e-5, atol=1e-6)


class TestUnitCellAndSpots:
    def test_fcc_cell_has_its_face_and_corner_atoms(self):
        positions, elements, edges = unit_cell(Phase.from_cif(CIF))
        assert len(positions) == 14 and set(elements) == {"Ag"}
        lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
        np.testing.assert_allclose(lengths, 4.0862, atol=1e-3)

    def test_fcc_001_zone_spot_rings(self):
        """At [001] the spots are {200}, {220}, {400}, {420}: radii in the
        ratio 1 : √2 : 2 : √5."""
        setup = _setup("TEM")
        half = setup["detector_size"] / 2 / setup["camera_length"]
        pixels, _ = spot_pixels(reflections_from_phase(Phase.from_cif(CIF)),
                                np.eye(3), microscope.detector_to_sample(setup),
                                wavelength(setup["beam_energy"]),
                                (-half, half, -half, half), (160, 160))
        radii = np.unique(np.round(np.hypot(*(pixels[1:] - pixels[0]).T), 1))
        np.testing.assert_allclose(radii[:4] / radii[0],
                                   [1, np.sqrt(2), 2, np.sqrt(5)], rtol=0.03)


class TestViewer:
    def test_the_figure_goes_to_the_phase_with_its_setup(self, monkeypatch, messages):
        _open(monkeypatch, _sem_tree(beam_energy=15.0))
        figure = next(m for m in messages if m.get("type") == "figure")
        assert figure["host"] == "phase" and figure["phase_id"] == "ag"
        assert "window_id" not in figure
        setup = next(m for m in messages if m.get("type") == "phase_setup")
        fields = {f["key"]: f for f in setup["fields"]}
        assert setup["microscope"] == "SEM" and fields["beam_energy"]["value"] == 15.0
        assert fields["beam_energy"]["source"] == "metadata"

    def test_turning_one_pane_turns_the_other(self, monkeypatch, messages):
        viewer = _open(monkeypatch, _sem_tree())
        before = viewer.detector.data.copy()
        rotation = euler_to_matrix(0.4, 0.9, 1.3).T
        viewer.sphere._set_rotation_from_js(list(rotation.ravel()))

        class Event:
            rotation = list(np.ravel(euler_to_matrix(0.4, 0.9, 1.3).T))

        viewer._on_turn(Event(), viewer.sphere)
        np.testing.assert_allclose(viewer.cell.rotation, rotation)
        assert not np.array_equal(before, viewer.detector.data)

    def test_a_field_edit_writes_metadata_and_moves_the_detector(self, monkeypatch,
                                                                 messages):
        tree = _sem_tree()
        viewer = _open(monkeypatch, tree)
        before = viewer.detector.data.copy()
        pv.phase_setup_set(_Session(), None, {"phase": "ag", "field": "pc_y",
                                              "value": 0.3})
        assert tree.root.metadata.get_item(
            "Acquisition_instrument.SEM.Detector.EBSD.pc_y") == 0.3
        assert not np.array_equal(before, viewer.detector.data)

    def test_a_tem_signal_shows_spots(self, monkeypatch, messages):
        signal = hs.signals.Signal2D(np.zeros((2, 2, 64, 64), np.float32))
        tree = _Tree(signal)
        comp.write_phases(tree, [AG])
        viewer = _open(monkeypatch, tree)
        assert viewer.setup.microscope == "TEM" and viewer.show_spots
        pixels, _ = viewer.spots()
        assert len(pixels) > 5
        setup = [m for m in messages if m.get("type") == "phase_setup"][-1]
        assert setup["spots_apply"] and not setup["full"]["applies"]

    def test_moving_the_detector_keeps_the_crystal_in_the_sample(
            self, monkeypatch, messages):
        """A geometry edit changes how the detector sees the crystal, not the
        crystal's orientation in the sample."""
        viewer = _open(monkeypatch, _sem_tree())
        viewer.rotation = euler_to_matrix(0.4, 0.9, 1.3).T
        held = viewer.rotation.copy()
        pv.phase_setup_set(_Session(), None, {"phase": "ag", "field": "sample_tilt",
                                              "value": 60.0})
        np.testing.assert_allclose(viewer.rotation, held)
        np.testing.assert_allclose(viewer.sphere.rotation, viewer.displayed())

    def test_full_simulation_is_refused_without_de_ebsd(self, monkeypatch, messages):
        monkeypatch.setattr(pv, "full_simulation_available", lambda: False)
        viewer = _open(monkeypatch, _sem_tree())
        pv.phase_setup_set(_Session(), None, {"phase": "ag", "master": "full"})
        assert viewer.master == "bands"

    def test_reopening_keeps_the_viewer(self, monkeypatch, messages):
        tree = _sem_tree()
        first = _open(monkeypatch, tree)
        assert _open(monkeypatch, tree) is first

    def test_a_phase_without_a_structure_says_so(self, monkeypatch, messages):
        tree = _sem_tree({"id": "ag", "elements": ["Ag"]})
        monkeypatch.setattr(comp, "_src_plot_tree", lambda session, plot: (None, tree))
        pv.show_phase(_Session(), None, {"phase": "ag"})
        assert any(isinstance(m, tuple) and "no structure" in m[1] for m in messages)


class TestSolvePatternCentre:
    """The button's wiring. The fit itself is de_ebsd's and needs a GPU-sized
    simulation, so here it is replaced by a known answer."""

    def _solving(self, monkeypatch, fit):
        from spyde.ebsd import calibrate

        monkeypatch.setattr(pv, "full_simulation_available", lambda: True)
        viewer = _open(monkeypatch, _sem_tree())
        monkeypatch.setattr(viewer, "_simulate_full", lambda key: "master")
        turned = euler_to_matrix(0.4, 0.9, 1.3).T

        def solve(patterns, master, setup, point_group, *, device, positions):
            assert master == "master" and patterns.shape == (len(positions), 30, 40)
            return calibrate.Calibration(pc=(0.41, 0.22, 0.6), rotations=turned[None],
                                         fit=fit, fit_before=0.1, positions=positions)

        monkeypatch.setattr(calibrate, "solve_pattern_centre", solve)
        pv.phase_setup_set(_Session(), None, {"phase": "ag", "solve_pc": True})
        return viewer, turned

    def test_a_trusted_fit_writes_the_centre_and_turns_the_crystal(
            self, monkeypatch, messages):
        viewer, turned = self._solving(monkeypatch, fit=0.8)
        assert (viewer.setup["pc_x"], viewer.setup["pc_y"], viewer.setup["pc_z"]) == (0.41, 0.22, 0.6)
        assert viewer.tree.root.metadata.get_item(
            "Acquisition_instrument.SEM.Detector.EBSD.pc_z") == 0.6
        np.testing.assert_allclose(viewer.rotation, turned)
        setup = [m for m in messages if isinstance(m, dict) and m.get("type") == "phase_setup"][-1]
        assert setup["solve"]["state"] == "done" and "4 patterns" in setup["solve"]["note"]

    def test_a_weak_fit_leaves_the_centre_alone(self, monkeypatch, messages):
        viewer, _ = self._solving(monkeypatch, fit=0.08)
        assert viewer.solve_state == "error" and "left as it was" in viewer.solve_note
        assert viewer.setup["pc_x"] != 0.41
        assert viewer.tree.root.metadata.get_item(
            "Acquisition_instrument.SEM.Detector.EBSD.pc_x") is None

    def test_offered_only_on_an_sem_scan(self, monkeypatch, messages):
        monkeypatch.setattr(pv, "full_simulation_available", lambda: True)
        assert _open(monkeypatch, _sem_tree()).can_solve()
        tem = _Tree(hs.signals.Signal2D(np.zeros((2, 2, 32, 32), np.float32)))
        comp.write_phases(tem, [AG])
        assert not _open(monkeypatch, tem).can_solve()


class TestCalibrationInputs:
    def test_positions_spread_over_the_scan_centre_first(self):
        from spyde.ebsd.calibrate import calibration_positions

        positions = calibration_positions((40, 30))
        assert len(positions) == 25 and positions[0] == (20, 14)   # the grid point nearest (20, 15)
        assert {p[0] for p in positions} == {0, 10, 20, 29, 39}
        assert calibration_positions((3,)) == [(1,), (0,), (2,)]
        assert len(calibration_positions((100,))) == 25   # a line scan reads as many as a map

    def test_large_patterns_are_binned_to_the_same_screen(self):
        from spyde.ebsd.calibrate import MAX_SIDE, read_patterns

        signal = hs.signals.Signal2D(np.ones((2, 480, 640), np.float32))
        patterns = read_patterns(signal, [(0,), (1,)])
        assert patterns.shape == (2, 120, 160) and max(patterns.shape[1:]) <= MAX_SIDE

    def test_the_static_background_is_what_the_patterns_share(self):
        from spyde.ebsd.calibrate import MIN_PATTERNS_FOR_STATIC, static_background

        profile = np.linspace(1, 2, 16, dtype=np.float32)[None, :] * np.ones((16, 1), np.float32)
        bands = np.random.default_rng(0).normal(0, 0.1, (64, 16, 16)).astype(np.float32)
        background = static_background(profile + bands)
        assert np.abs(background - profile).max() < 0.1
        assert static_background((profile + bands)[:MIN_PATTERNS_FOR_STATIC - 1]) is None


class TestPatternCentreSearch:
    """The grid search that brings the centre within the fit's reach."""

    @staticmethod
    def _peak(centre, width=0.06):
        def score(pc):
            return float(np.exp(-np.sum((np.asarray(pc) - centre) ** 2) / (2 * width**2)))
        return score

    def test_a_centre_far_from_the_start_is_found_to_a_quarter_step(self):
        from spyde.ebsd.calibrate import SEARCH_REFINE_STEPS, grid_search

        for centre in ((0.42, 0.213, 0.499), (0.5, 0.3, 0.6), (0.47, 0.71, 0.83)):
            best = grid_search(self._peak(np.array(centre)), start_x=0.5)[0]
            assert np.abs(np.subtract(best, centre)).max() <= SEARCH_REFINE_STEPS[-1], centre

    def test_each_centre_is_scored_once(self):
        from spyde.ebsd.calibrate import grid_search

        seen = []
        score = self._peak(np.array((0.5, 0.3, 0.6)))
        grid_search(lambda pc: seen.append(pc) or score(pc), start_x=0.5)
        assert len(seen) == len(set(seen))

    def test_the_best_fit_of_the_candidates_is_kept(self, monkeypatch):
        from spyde.ebsd import calibrate

        fits = {}

        def solve(patterns, master, setup, point_group, *, device=None, positions=(),
                  orientations=None):
            pc = (setup["pc_x"], setup["pc_y"], setup["pc_z"])
            fits[pc] = fit = 0.4 if pc[1] < 0.35 else 0.1
            return calibrate.Calibration(pc=pc, rotations=np.eye(3)[None], fit=fit,
                                         fit_before=0.0, positions=list(positions))

        monkeypatch.setattr(calibrate, "solve_pattern_centre", solve)
        monkeypatch.setattr(calibrate, "grid_search",
                            lambda score, start_x: [(0.5, 0.5, 0.55), (0.5, 0.3, 0.6)])
        monkeypatch.setattr(calibrate, "_lead_scorer", lambda *args: lambda pc: 0.0)
        monkeypatch.setattr(calibrate, "seed_orientations", lambda point_group: None)
        monkeypatch.setitem(sys.modules, "de_ebsd", types.ModuleType("de_ebsd"))
        index = types.ModuleType("de_ebsd.index")
        index.remove_background = lambda patterns, **kw: patterns
        monkeypatch.setitem(sys.modules, "de_ebsd.index", index)
        master = types.SimpleNamespace(data=types.SimpleNamespace(device="cpu"), to=lambda device: master)
        setup = {"pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.55, "sample_tilt": 70.0,
                 "detector_tilt": 0.0, "azimuthal": 0.0, "twist": 0.0}
        found = calibrate.search_pattern_centre(np.zeros((9, 8, 8), np.float32), master, setup, None)
        assert found.pc == (0.5, 0.3, 0.6) and found.fit == 0.4 and len(fits) == 2
