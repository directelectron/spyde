"""
GUI: the Vector Orientation Mapping live Refine caret.

Gating (appears once a tree has diffraction_vectors), caret builds the 3-tab
wizard, and the refit path overlays fitted template spots on the measured
vectors with a strain readout. Library generation is bypassed by injecting a
pre-built TemplateLibrary so the test stays fast and offline.
"""
import numpy as np
import pytest
import dask.array as da
import hyperspy.api as hs
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from spyde.actions import vector_orientation as vo


# A square-ring template (≥4 spots) so fit_pattern has something to fit.
_TEMPLATE = np.array([
    [0.05, 0.0], [-0.05, 0.0], [0.0, 0.05], [0.0, -0.05],
    [0.05, 0.05], [-0.05, -0.05], [0.05, -0.05], [-0.05, 0.05],
], dtype=np.float32)


def _make_vecs():
    """Build a SpyDEDiffractionVectors whose every position holds the square
    template's spots (so the OM caret has a multi-spot pattern to fit)."""
    from spyde.signals.diffraction_vectors import (
        SpyDEDiffractionVectors, _build_nav_offsets, N_COLS,
        COL_NAV_X, COL_NAV_Y, COL_KX, COL_KY, COL_TIME, COL_INTENSITY,
    )
    ny, nx = 4, 4
    rows = []
    for iy in range(ny):
        for ix in range(nx):
            for kx, ky in _TEMPLATE:
                r = np.zeros(N_COLS, np.float32)
                r[COL_NAV_X] = ix
                r[COL_NAV_Y] = iy
                r[COL_KX] = kx
                r[COL_KY] = ky
                r[COL_TIME] = -1.0
                r[COL_INTENSITY] = 1.0
                rows.append(r)
    flat = np.array(rows, np.float32)
    nav_offsets = _build_nav_offsets(flat, (ny, nx))

    class _Ax:
        scale = 0.01
        offset = -0.16
        size = 32
        units = "1/A"
        name = "k"
    return SpyDEDiffractionVectors(
        flat_buffer=flat, nav_offsets=nav_offsets, nav_shape=(ny, nx),
        full_nav_shape=(ny, nx), sig_shape=(32, 32),
        sig_axes=[_Ax(), _Ax()], kernel_radius_px=3.0, kernel_radius_data=0.03)


def _stub_library():
    """Single-template library matching the synthetic pattern (no coarse cache
    → fit_pattern seeds every template at angle 0)."""
    return vo.TemplateLibrary(
        spots_xy=[_TEMPLATE.copy()],
        spots_I=[np.ones(len(_TEMPLATE), np.float32)],
        template_quats=np.array([[1.0, 0, 0, 0]]),
        template_phase=np.array([0], np.int16),
        phases_meta=[{"name": "x", "point_group": "m-3m"}],
        cache={}, radial_range=(0.0, 0.16), r_max=0.16)


def _signal_plot(win):
    for pw in win.plot_subwindows:
        for plot in pw.plots:
            if plot.plot_state is not None and not plot.is_navigator:
                return plot
    return None


class TestVectorOMCaret:
    @pytest.fixture(autouse=True)
    def setup(self, stem_4d_dataset, qtbot):
        self.win = stem_4d_dataset["window"]
        self.qtbot = qtbot
        self.plot = _signal_plot(self.win)
        assert self.plot is not None
        self.plot.plot_state.current_signal.set_signal_type(
            "spyde_diffraction_vectors_image")
        self.plot.signal_tree.diffraction_vectors = _make_vecs()
        self.plot.plot_state.rebuild_toolbars()
        QApplication.processEvents()

    def _tb(self):
        return self.plot.plot_state.toolbar_bottom

    def test_action_present_after_attach(self):
        names = [a.text() for a in self._tb().actions()]
        assert "Vector Orientation Mapping" in names

    def test_caret_builds_wizard(self):
        from spyde.actions.vector_orientation_action import _VOM_BUILT_TOOLBARS
        tb = self._tb()
        _VOM_BUILT_TOOLBARS.discard(id(tb))
        act = tb._find_action("Vector Orientation Mapping")
        assert act is not None
        act.trigger()
        QApplication.processEvents()
        QTest.qWait(50)
        # caret + state registered
        assert hasattr(tb, "_vom_state")
        caret = tb.action_widgets["Vector Orientation Mapping"]["widget"]
        assert caret is not None
        from PySide6.QtWidgets import QPushButton
        labels = [b.text() for b in caret.findChildren(QPushButton)]
        assert any("Load" in l for l in labels)
        assert any("Library" in l for l in labels)
        assert any("Refine" in l for l in labels)
        assert any("Generate" in l for l in labels)

    def test_generate_builds_overlay_and_refit(self, monkeypatch):
        """Click Generate (with library generation stubbed) → the caret builds
        the red/green overlays and a refit populates the strain readout."""
        from spyde.actions import vector_orientation_action as voa
        from spyde.actions.vector_orientation_action import _VOM_BUILT_TOOLBARS
        from PySide6.QtWidgets import QPushButton

        # Stub the slow library path so Generate returns instantly with our lib.
        monkeypatch.setattr(voa, "_VOM_BUILT_TOOLBARS", set())
        import spyde.actions.orientation_compute as oc
        import spyde.actions.vector_orientation as vomod
        import spyde.actions.pyxem as pyx
        monkeypatch.setattr(oc, "generate_library_from_phases",
                            lambda *a, **k: object())
        # build_template_library is imported inside _on_generate from vomod
        monkeypatch.setattr(vomod, "build_template_library",
                            lambda *a, **k: _stub_library())
        # the fixture signal is 128×128 nav but our test vectors are 4×4; pin the
        # crosshair read to a valid position so vecs.at() returns the pattern.
        monkeypatch.setattr(pyx, "_get_current_nav_indices", lambda p: (0, 0))

        tb = self._tb()
        act = tb._find_action("Vector Orientation Mapping")
        act.trigger()
        QApplication.processEvents()
        state = tb._vom_state
        caret = tb.action_widgets["Vector Orientation Mapping"]["widget"]

        # pretend a phase is loaded so Generate is enabled
        from orix.crystal_map import Phase
        state["phases"].append(Phase(name="x", point_group="m-3m"))
        gen_btn = next(b for b in caret.findChildren(QPushButton)
                       if "Generate" in b.text())
        gen_btn.setEnabled(True)
        gen_btn.click()

        # generation runs on a worker thread → wait for lib + overlay
        self.qtbot.waitUntil(lambda: state["lib"][0] is not None, timeout=5000)
        QApplication.processEvents()
        self.qtbot.waitUntil(lambda: state["scatter"][0] is not None, timeout=5000)

        # fire a refit and let the worker emit back to the GUI
        state["active"][0] = True
        state["refit_timer"][0].start()
        for _ in range(30):
            QApplication.processEvents()
            QTest.qWait(20)
            if state["scatter"][0].getData()[0] is not None and \
                    len(state["scatter"][0].getData()[0]) > 0:
                break
        gx, gy = state["scatter"][0].getData()
        assert gx is not None and len(gx) > 0, "no green template overlay drawn"
        vx, vy = state["vec_scatter"][0].getData()
        assert vx is not None and len(vx) > 0, "no red vector overlay drawn"

    def test_run_computes_field_and_opens_maps(self, monkeypatch):
        """Generate (stubbed) → Run tab Compute Map fits the whole 4x4 field and
        opens orientation + strain map windows.

        Forces the CPU (serial) compute path: the batched-GPU torch backward
        segfaults *only* under the pytest session-window harness (it runs
        correctly in the real app and in every standalone repro — QApplication,
        Dask cluster, numba.cuda all coexist fine). GPU correctness is covered
        by test_vector_orientation_gpu.py. Here we just verify the Run-tab
        wiring opens the four map windows and lands a finite-strain result.
        """
        from spyde.actions import vector_orientation_action as voa
        from spyde.actions import vector_orientation_gpu as vgpu
        from PySide6.QtWidgets import QPushButton

        monkeypatch.setattr(voa, "_VOM_BUILT_TOOLBARS", set())
        # Force the serial CPU fit: the batched torch backward segfaults under
        # the pytest harness (runs fine in the real app). The action picks the
        # torch path via select_device()/torch_available(), so disable both.
        monkeypatch.setattr(vgpu, "select_device", lambda: None)
        monkeypatch.setattr(vgpu, "torch_available", lambda: False)
        monkeypatch.setattr(vgpu, "gpu_available", lambda: False)
        import spyde.actions.orientation_compute as oc
        import spyde.actions.vector_orientation as vomod
        monkeypatch.setattr(oc, "generate_library_from_phases",
                            lambda *a, **k: object())
        monkeypatch.setattr(vomod, "build_template_library",
                            lambda *a, **k: _stub_library())

        tb = self._tb()
        tb._find_action("Vector Orientation Mapping").trigger()
        QApplication.processEvents()
        state = tb._vom_state
        caret = tb.action_widgets["Vector Orientation Mapping"]["widget"]
        from orix.crystal_map import Phase
        state["phases"].append(Phase(name="x", point_group="m-3m"))
        gen = next(b for b in caret.findChildren(QPushButton)
                   if "Generate" in b.text())
        gen.setEnabled(True); gen.click()
        self.qtbot.waitUntil(lambda: state["lib"][0] is not None, timeout=5000)
        QApplication.processEvents()

        n_before = len(self.win.plot_subwindows)
        run = next(b for b in caret.findChildren(QPushButton)
                   if "Compute Map" in b.text())
        assert run.isEnabled()
        run.click()
        # compute runs on a worker; wait for the result to land on the tree
        self.qtbot.waitUntil(
            lambda: getattr(self.plot.signal_tree, "vector_orientation", None)
            is not None, timeout=15000)
        for _ in range(40):
            QApplication.processEvents(); QTest.qWait(20)
            if len(self.win.plot_subwindows) > n_before:
                break

        res = self.plot.signal_tree.vector_orientation
        assert res.nav_shape == (4, 4)
        assert res.strain.shape == (4, 4, 3)
        # uniform field → every position fits, strain finite everywhere
        assert np.isfinite(res.strain[..., 0]).all()
        # progressive map windows opened (IPF-Z + 3 strain panels)
        assert len(self.win.plot_subwindows) >= n_before + 4
