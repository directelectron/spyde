"""
The orientation wizards' IPF window follows the action, not the library.

It opens with the phase's empty triangle when the caret is selected, fills
while the library builds, hands its panels to the heat map for Refine, hides
on deselect or ✕ and comes back with the heat map still drawing into it.
"""
from __future__ import annotations

import os
import time
import types

import numpy as np
import pytest

CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")


class _Session:
    """What the panel asks of a session, and a record of what it did."""

    def __init__(self):
        self.controllers = {}
        self.forgotten = []
        self._next = 100

    def next_window_id(self):
        self._next += 1
        return self._next

    def register_window_controller(self, window_id, controller):
        self.controllers[window_id] = controller

    def _forget_window(self, window_id):
        self.forgotten.append(window_id)
        controller = self.controllers.pop(window_id, None)
        if controller is not None:
            controller.close()


class _Controller:
    """A heat map controller as the panel sees it."""

    def __init__(self):
        self.bound = []
        self.redraws = 0

    def bind_panels(self, panels):
        self.bound.append(list(panels))

    def redraw(self):
        self.redraws += 1


@pytest.fixture
def phase():
    from orix.crystal_map import Phase
    return Phase.from_cif(CIF)


@pytest.fixture
def emitted(monkeypatch):
    import de_shell.ipc as ipc
    out = []
    monkeypatch.setattr(ipc, "emit", lambda message: out.append(message))
    return out


def _tree():
    import hyperspy.api as hs
    signal = hs.signals.Signal2D(np.zeros((4, 4), np.float32))
    signal.metadata.General.title = "Scan"
    return types.SimpleNamespace(root=signal)


def _library_infos(phase, n=12):
    """A small library's geometry: n random orientations of the phase."""
    from orix.quaternion import Rotation
    from spyde.actions.ipf_refine import build_phase_ipf_for
    quats = Rotation.random(n).data
    return build_phase_ipf_for(quats, np.zeros(n, int), [phase])


class TestTheTriangle:
    def test_a_phase_opens_the_empty_triangle_when_the_caret_is_open(
            self, phase, emitted):
        from spyde.actions.ipf_panel import ensure_panel
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "om")
        panel.wanted = True
        panel.set_phases([phase])

        assert panel.window_id is not None
        figures = [m for m in emitted if m.get("type") == "figure"]
        assert len(figures) == 1 and figures[0]["title"] == "Scan — IPF Refine"
        assert len(panel.infos) == 1 and panel.infos[0]["xs"].size == 0
        assert panel.infos[0]["labels"], "the triangle carries its corner labels"
        assert panel.panels[0]["points_shown"] is False

    def test_a_phase_waits_for_the_caret(self, phase, emitted):
        from spyde.actions.ipf_panel import ensure_panel, set_visible
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "om")
        panel.set_phases([phase])
        assert panel.window_id is None, "nothing selected, nothing shown"

        set_visible(session, tree, "Orientation Mapping", True)
        assert panel.window_id is not None
        set_visible(session, tree, "Orientation Mapping", False)
        assert panel.window_id is None and panel.infos, \
            "deselecting hides the window and keeps the triangle"

    def test_the_same_phase_again_keeps_the_window(self, phase, emitted):
        """Generate hands the panel its phases once more before the fill
        starts; rebuilding the window then would close and reopen it and
        lose the first fill paints to a figure still loading."""
        from spyde.actions.ipf_panel import ensure_panel
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "vom")
        panel.wanted = True
        panel.set_phases([phase])
        window_id = panel.window_id
        panel.set_phases([phase])
        assert panel.window_id == window_id
        assert session.forgotten == []

    def test_no_phase_no_window(self, emitted):
        from spyde.actions.ipf_panel import ensure_panel
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "vom")
        panel.wanted = True
        panel.set_phases([])
        assert panel.window_id is None


class TestTheFill:
    def test_the_fill_advances_while_the_library_builds(
            self, phase, emitted, monkeypatch):
        from spyde.actions import ipf_panel
        painted = []
        monkeypatch.setattr(ipf_panel, "paint_fill",
                            lambda panels, fraction: painted.append(fraction))
        session, tree = _Session(), _tree()
        panel = ipf_panel.ensure_panel(session, tree, "om")
        panel.wanted = True
        panel.set_phases([phase])

        panel.start_filling()
        deadline = time.monotonic() + 3.0
        while len(painted) < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        panel.stop_filling()
        assert len(painted) >= 3, "the fill never painted"
        assert painted == sorted(painted), "the fill only ever grows"
        assert 0.0 < painted[-1] < 1.0, "it approaches full and never arrives"

    def test_the_fill_lights_that_fraction_of_the_triangle(self, phase, emitted):
        from spyde.actions.ipf_panel import ensure_panel, paint_fill, FILL_LEVEL
        from spyde.actions import ipf_refine_render
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "om")
        panel.wanted = True
        panel.set_phases([phase])
        seen = []
        real = ipf_refine_render._corr_rgba

        def spy(values, outside, lut):
            seen.append((np.asarray(values), np.asarray(outside)))
            return real(values, outside, lut)

        ipf_refine_render._corr_rgba = spy
        try:
            paint_fill(panel.panels, 0.3)
        finally:
            ipf_refine_render._corr_rgba = real
        values, outside = seen[-1]
        inside = ~outside.reshape(values.shape)
        lit = (values == FILL_LEVEL) & inside
        assert abs(lit.sum() / inside.sum() - 0.3) < 0.03,             "the lit area is the fraction, not the width"
        assert not (values[~inside] > 0).any(), "nothing lights outside the triangle"

    def test_the_library_replaces_the_fill_with_its_points(self, phase, emitted):
        from spyde.actions.ipf_panel import ensure_panel
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "om")
        panel.wanted = True
        panel.set_phases([phase])
        panel.start_filling()

        infos = _library_infos(phase)
        panel.set_library(infos)
        assert panel._fill is None, "the library stops the fill"
        assert panel.panels[0]["info"] is infos[0]
        assert panel.panels[0]["points_shown"] is True
        assert panel.panels[0]["outside"].shape == (infos[0]["grid_n"],) * 2

    def test_the_heat_map_hides_the_points(self, phase, emitted):
        from spyde.actions.ipf_panel import ensure_panel
        from spyde.actions.ipf_refine_render import update_panels
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "om")
        panel.wanted = True
        panel.set_phases([phase])
        infos = _library_infos(phase)
        panel.set_library(infos)

        correlations = np.linspace(0.0, 1.0, infos[0]["lib_idx"].size)
        update_panels(panel.panels, correlations, {0: []})
        assert panel.panels[0]["points_shown"] is False


class TestHideAndShow:
    def test_the_heat_map_survives_a_hide(self, phase, emitted):
        from spyde.actions.ipf_panel import ensure_panel
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "vom")
        panel.wanted = True
        panel.set_phases([phase])
        panel.set_library(_library_infos(phase))
        controller = _Controller()
        panel.controller = controller
        first = panel.window_id

        panel.hide()
        assert session.forgotten == [first]
        assert panel.window_id is None
        assert controller.bound[-1] == [], "nothing to draw into while hidden"

        panel.show()
        assert panel.window_id is not None and panel.window_id != first
        assert controller.bound[-1] == panel.panels and controller.redraws == 1
        assert panel.panels[0]["points_shown"] is True, \
            "the library's points are back until the heat map draws"

    def test_the_window_close_box_hides_it(self, phase, emitted):
        from spyde.actions.ipf_panel import ensure_panel
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "vom")
        panel.wanted = True
        panel.set_phases([phase])
        window_id = panel.window_id

        session._forget_window(window_id)       # the ✕
        assert panel.window_id is None and panel.infos
        panel.show()
        assert panel.window_id is not None

    def test_a_library_is_kept_through_a_phase_edit(self, phase, emitted):
        """The library was built for the phases it has; a phase edited later
        takes effect at the next Generate."""
        from spyde.actions.ipf_panel import ensure_panel
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "om")
        panel.wanted = True
        panel.set_phases([phase])
        infos = _library_infos(phase)
        panel.set_library(infos)
        panel.controller = _Controller()

        panel.set_phases([phase, phase])
        assert panel.infos is not None and panel.infos[0] is infos[0]

    def test_remove_drops_the_tree_attribute(self, phase, emitted):
        from spyde.actions.ipf_panel import ensure_panel, panel_for
        session, tree = _Session(), _tree()
        panel = ensure_panel(session, tree, "om")
        panel.wanted = True
        panel.set_phases([phase])
        panel.remove()
        assert panel.window_id is None
        assert panel_for(tree, "om") is None
