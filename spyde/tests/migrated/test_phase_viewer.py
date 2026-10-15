"""
test_phase_viewer.py — a phase's band sphere and its detector preview.

The viewer draws the same bands three ways (the sphere, the preview, and — in
the EBSD wizard — the dictionary and live overlay), so the contract that
matters is that they agree: the preview at a sphere rotation must be exactly
``simulate_patterns`` at the matching Bunge angles. Getting the frame
convention backwards would still draw plausible bands, just for the INVERSE
orientation, so this is pinned against the established function rather than
eyeballed.
"""
import os

import numpy as np
import pytest
import hyperspy.api as hs

from spyde.actions import composition as comp
from spyde.actions import phase_viewer as viewer
from spyde.ebsd.bands import (
    band_intensity, band_sphere, cubic_reflectors, euler_to_matrix,
    matrix_to_euler, reflectors_from_phase, simulate_patterns,
    sphere_directions,
)

CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")


class _Tree:
    def __init__(self, signal):
        self.root = signal
        self.signal_plots = []


class _Session:
    """With no ``_dispatch_to_main`` the worker step runs inline."""


@pytest.fixture
def messages(monkeypatch):
    sent = []
    monkeypatch.setattr(viewer, "emit_status",
                        lambda text: sent.append(("status", text)))
    monkeypatch.setattr(viewer, "emit_error",
                        lambda text: sent.append(("error", text)))
    from de_shell import ipc
    monkeypatch.setattr(ipc, "emit", lambda message: sent.append(message))
    return sent


def _figures(messages):
    return [m for m in messages if isinstance(m, dict) and m.get("type") == "figure"]


def _ebsd_tree(*phases, beam_energy=None):
    signal = hs.signals.Signal2D(np.zeros((2, 2, 30, 40), np.float32))
    if beam_energy is not None:
        signal.metadata.set_item("Acquisition_instrument.SEM.beam_energy",
                                 beam_energy)
    tree = _Tree(signal)
    comp.write_phases(tree, list(phases))
    return tree


def _controller(reflectors=None, shape=(40, 50), pc=(0.45, 0.55, 0.6)):
    reflectors = reflectors or cubic_reflectors()
    return viewer.PhaseViewerController(
        {"id": "p", "elements": ["Ag"]}, reflectors,
        np.random.default_rng(0).random((16, 32)), voltage_kv=20, pc=pc,
        shape=shape)


class TestOneBandProfile:
    def test_simulate_patterns_is_band_intensity_on_the_detector(self):
        """The refactor that made the sphere possible changed no pattern."""
        reflectors = cubic_reflectors()
        euler = np.array([[0.3, 0.7, 1.1]])
        old_style = simulate_patterns(euler, reflectors, (30, 40), (0.5, 0.5, 0.55))
        rot = euler_to_matrix(*euler[0])
        from spyde.ebsd.bands import detector_directions
        rays = detector_directions((30, 40), (0.5, 0.5, 0.55)).reshape(-1, 3)
        direct = band_intensity(rays, reflectors,
                                normals=reflectors.normals @ rot).reshape(30, 40)
        np.testing.assert_allclose(old_style[0], direct, rtol=1e-6)

    def test_the_sphere_is_band_intensity_over_all_directions(self):
        reflectors = cubic_reflectors()
        sphere = band_sphere(reflectors, (20, 40))
        expected = band_intensity(sphere_directions(20, 40), reflectors)
        np.testing.assert_allclose(sphere.ravel(), expected)

    def test_blocks_do_not_change_the_answer(self):
        reflectors = cubic_reflectors()
        directions = sphere_directions(30, 60).reshape(-1, 3)
        np.testing.assert_allclose(
            band_intensity(directions, reflectors, block=7),
            band_intensity(directions, reflectors))

    def test_euler_round_trip(self):
        rng = np.random.default_rng(1)
        for euler in rng.uniform([0, 0.05, 0], [2 * np.pi, np.pi - 0.05, 2 * np.pi],
                                 (200, 3)):
            m = euler_to_matrix(*euler)
            np.testing.assert_allclose(euler_to_matrix(*matrix_to_euler(m)), m,
                                       atol=1e-12)


class TestPreviewParity:
    @pytest.mark.parametrize("euler", [(0.0, 0.0, 0.0), (0.4, 0.9, 1.3),
                                       (2.5, 0.3, 4.0)])
    def test_preview_matches_simulate_patterns(self, euler):
        """The sphere's model rotation carries crystal directions into the
        sample frame, i.e. the TRANSPOSE of the Bunge matrix."""
        controller = _controller()
        rot = euler_to_matrix(*euler)
        expected = simulate_patterns(np.array([euler]), controller.reflectors,
                                     controller.shape, controller.pc)[0]
        np.testing.assert_allclose(controller.pattern(rot.T), expected, rtol=1e-6)

    def test_a_drag_repaints_the_preview_and_title(self):
        controller = _controller()
        before = controller.detector.data.copy()
        rot = euler_to_matrix(0.4, 0.9, 1.3)
        controller.sphere._set_rotation_from_js(list(rot.T.ravel()))

        class Event:
            rotation = list(rot.T.ravel())

        controller._on_turn(Event())
        assert not np.array_equal(before, controller.detector.data)
        assert controller.detector._state["title"].startswith("φ1 23°  Φ 52°")

    def test_a_wheel_zoom_does_not_repaint(self):
        controller = _controller()
        before = controller.detector.data.copy()

        class Event:
            rotation = None

        controller._on_turn(Event())
        np.testing.assert_array_equal(before, controller.detector.data)

    def test_the_footprint_is_fixed_to_the_microscope(self):
        controller = _controller()
        (patch,) = controller.sphere._state["patches"]
        assert patch["frame"] == "world" and patch["closed"]


class TestGeometry:
    def test_voltage_comes_from_the_signal(self):
        assert viewer.beam_voltage_kv(_ebsd_tree(beam_energy=15).root) == 15
        assert viewer.beam_voltage_kv(_ebsd_tree().root) == viewer.DEFAULT_VOLTAGE_KV

    def test_preview_keeps_the_detector_aspect(self):
        rows, cols = viewer.preview_shape(_ebsd_tree().root)       # 30 × 40
        assert cols == viewer.PREVIEW_PIXELS
        assert rows == round(viewer.PREVIEW_PIXELS * 30 / 40)


def _show(monkeypatch, tree, phase_id="ag"):
    monkeypatch.setattr(comp, "_src_plot_tree", lambda session, plot: (None, tree))
    viewer.show_phase(_Session(), None, {"phase": phase_id})


AG = {"id": "ag", "elements": ["Ag"], "cif_path": CIF, "label": "Ag"}


class TestShowPhase:
    def test_the_figure_goes_to_the_phase_not_to_a_window(self, monkeypatch,
                                                         messages):
        monkeypatch.setattr(viewer, "SPHERE_TEXTURE", (64, 128))
        tree = _ebsd_tree(AG, beam_energy=15)
        _show(monkeypatch, tree)

        (figure,) = _figures(messages)
        assert figure["host"] == "phase" and figure["phase_id"] == "ag"
        assert "window_id" not in figure
        held = tree._phase_viewers["ag"]
        assert held.voltage_kv == 15
        from orix.crystal_map import Phase
        expected = reflectors_from_phase(Phase.from_cif(CIF), voltage_kv=15)
        np.testing.assert_allclose(held.reflectors.widths, expected.widths)

    def test_showing_again_resends_without_recomputing(self, monkeypatch,
                                                       messages):
        monkeypatch.setattr(viewer, "SPHERE_TEXTURE", (64, 128))
        tree = _ebsd_tree(AG)
        _show(monkeypatch, tree)
        first = tree._phase_viewers["ag"]
        monkeypatch.setattr(viewer, "run_on_worker",
                            lambda *a, **k: pytest.fail("recomputed the sphere"))
        _show(monkeypatch, tree)
        assert tree._phase_viewers["ag"] is first
        one, two = _figures(messages)
        assert one["fig_id"] == two["fig_id"]

    def test_a_new_voltage_recomputes(self, monkeypatch, messages):
        monkeypatch.setattr(viewer, "SPHERE_TEXTURE", (64, 128))
        tree = _ebsd_tree(AG, beam_energy=20)
        _show(monkeypatch, tree)
        first = tree._phase_viewers["ag"]
        tree.root.metadata.set_item("Acquisition_instrument.SEM.beam_energy", 10)
        _show(monkeypatch, tree)
        assert tree._phase_viewers["ag"] is not first
        assert first._closed and tree._phase_viewers["ag"].voltage_kv == 10

    def test_a_phase_without_a_structure_says_so(self, monkeypatch, messages):
        tree = _ebsd_tree({"id": "ag", "elements": ["Ag"]})
        _show(monkeypatch, tree)
        assert not _figures(messages)
        assert any(isinstance(m, tuple) and m[0] == "error" and "no structure" in m[1]
                   for m in messages)

    def test_close_stops_repaints(self):
        controller = _controller()
        controller.close()
        before = controller.detector.data.copy()

        class Event:
            rotation = list(np.eye(3).ravel())

        controller._on_turn(Event())
        np.testing.assert_array_equal(before, controller.detector.data)
