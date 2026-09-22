"""
ipf_panel.py — the orientation wizards' IPF window, through its three stages.

The window belongs to the ACTION, not to the library: it opens when the
wizard's caret is selected and closes when it is deselected, and what it shows
follows the wizard's tabs.

* **Load** — a phase with a structure gives its reduced triangle, the
  fundamental sector of its point group with the corner labels, and nothing
  inside. One panel per phase.
* **Library** — while the library builds, the triangle fills in. The fill is a
  clock, not the library's progress: the library builders report nothing until
  they return, and a triangle that visibly fills says the program is working
  where a static one says it has hung. When the library lands the fill gives
  way to the orientations it sampled, as points.
* **Refine** — the correlation heat map, as before, following the crosshair;
  the points step aside on its first draw.

Both wizards share it. The dense and the vector wizard build their libraries
differently and match differently, but the triangle, the fill and the panels
the heat map draws into are the same, so they are built once here.
"""
from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

#: The toolbar action each wizard is selected by → the wizard's key.
ACTION_KEYS = {"Orientation Mapping": "om", "Vector Orientation Mapping": "vom"}

#: Seconds for the dummy fill to reach 63 %. It approaches full and never
#: arrives, so a library that takes a minute still shows movement at the end.
FILL_TAU = 6.0
#: Where on the heat map's colour scale the fill sits — a dim red, so it reads
#: as "in progress" and not as a correlation.
FILL_LEVEL = 0.55
FILL_INTERVAL = 0.15


def _attr(key: str) -> str:
    return f"_{key}_ipf_panel"


def panel_for(tree, key: str):
    """The tree's panel for the wizard *key*, or None."""
    return getattr(tree, _attr(key), None) if tree is not None else None


def ensure_panel(session, tree, key: str, signal=None):
    """The tree's panel for the wizard *key*, created if it has none."""
    panel = panel_for(tree, key)
    if panel is None:
        panel = IpfPanel(session, tree, key, signal if signal is not None else tree.root)
        setattr(tree, _attr(key), panel)
    return panel


def set_visible(session, tree, action_name: str, visible: bool) -> None:
    """The caret of *action_name* was selected (or deselected) on *tree*."""
    key = ACTION_KEYS.get(action_name)
    if key is None or tree is None:
        return
    if not visible:
        panel = panel_for(tree, key)
        if panel is not None:
            panel.wanted = False
            panel.hide()
        return
    panel = ensure_panel(session, tree, key)
    panel.wanted = True
    if panel.infos:
        panel.show()
    else:
        _load_phases(session, tree, panel)


def phases_changed(session, tree) -> None:
    """The sample's phases changed: every panel on the tree re-reads them."""
    for key in ACTION_KEYS.values():
        panel = panel_for(tree, key)
        if panel is not None:
            _load_phases(session, tree, panel)


def _load_phases(session, tree, panel) -> None:
    """Read the phases with a structure off the tree and hand them to the
    panel. Reading a .cif imports the crystallography stack, so it runs on a
    worker; a panel with a library keeps it (see :meth:`IpfPanel.set_phases`)."""
    from spyde.actions.composition import read_phases
    from spyde.actions.lifecycle import run_on_worker

    paths = [p.get("cif_path") for p in read_phases(tree) if p.get("cif_path")]

    def _read():
        from orix.crystal_map import Phase
        return [Phase.from_cif(path) for path in paths]

    if getattr(session, "_dispatch_to_main", None) is None:
        try:
            panel.set_phases(_read())
        except Exception as e:
            log.debug("reading the sample's phases failed: %s", e)
        return
    run_on_worker(session, _read, name="ipf-panel-phases", on_done=panel.set_phases)


class _WindowHandle:
    """What the session holds for the panel's window: closing the window
    (its ✕, a tree close) hides the panel, it does not retire it."""

    def __init__(self, panel):
        self.panel = panel

    def close(self) -> None:
        self.panel._window_gone()


class IpfPanel:
    """One wizard's IPF window on one tree. See the module docstring."""

    def __init__(self, session, tree, key: str, signal):
        self.session = session
        self.tree = tree
        self.key = key
        self.signal = signal
        #: orix Phases with a structure, in the sample's order.
        self.phases: list = []
        #: The per-phase panel geometry: the triangle alone, or the library's
        #: sampled orientations once it has one.
        self.infos: list = []
        self.panels: list = []
        self.figure = None
        self.window_id = None
        #: The heat map controller drawing into the panels, once Refine has one.
        self.controller = None
        #: True while the wizard's caret is open, so a phase that lands then
        #: opens the window without another click.
        self.wanted = False
        self._fill: dict | None = None
        self._closed = False

    # ── stages ───────────────────────────────────────────────────────────────

    def set_phases(self, phases) -> None:
        """The sample's phases: each one's triangle, nothing inside.

        A panel showing a library keeps it — the library was built for the
        phases it has, and a phase edited after Generate takes effect at the
        next Generate, where the library replaces the geometry again."""
        self.phases = list(phases or [])
        if self.controller is not None:
            return
        if not self.phases:
            self.infos = []
            self.hide()
            return
        if self.window_id is not None and self._same_triangles(self.phases):
            # The same triangles are already up. Rebuilding the window here
            # would close and reopen it for nothing, and a fill that starts
            # right after (Generate) would paint into a figure still loading.
            return
        from spyde.actions.ipf_refine import build_phase_ipf_geometry
        self.infos = build_phase_ipf_geometry(self.phases)
        if self.window_id is not None:
            self._rebuild()
        elif self.wanted:
            self.show()

    def _same_triangles(self, phases) -> bool:
        """Whether *phases* draw the panels already up: one per phase, in
        order, with the same point group — which is all a triangle is."""
        def group_of(phase):
            group = getattr(phase, "point_group", None)
            return str(getattr(group, "name", group))
        shown = [info.get("point_group") for info in self.infos]
        return shown == [group_of(phase) for phase in phases]

    def start_filling(self) -> None:
        """Fill the triangles while a library builds (see the module docstring)."""
        self.stop_filling()
        state = {"stop": False, "started": time.monotonic()}
        self._fill = state
        threading.Thread(target=self._fill_loop, args=(state,), daemon=True,
                         name=f"{self.key}-ipf-fill").start()

    def stop_filling(self) -> None:
        if self._fill is not None:
            self._fill["stop"] = True
            self._fill = None

    def fill_fraction(self) -> float:
        """How full the dummy fill is right now, 0 when it is not running."""
        if self._fill is None:
            return 0.0
        return 1.0 - math.exp(-(time.monotonic() - self._fill["started"]) / FILL_TAU)

    def _fill_loop(self, state: dict) -> None:
        while not state["stop"] and not self._closed:
            panels = self.panels
            if panels:
                fraction = 1.0 - math.exp(
                    -(time.monotonic() - state["started"]) / FILL_TAU)
                # Painted from this thread, as the heat map is painted from
                # the overlay painter's: a figure update is a message, and
                # the main loop may be busy behind the library's own work.
                paint_fill(panels, fraction)
            time.sleep(FILL_INTERVAL)

    def set_library(self, infos) -> None:
        """The library landed: its sampled orientations replace the fill.

        The panels are updated in place rather than rebuilt: a panel's raster
        spans the triangle and its labels, which the library does not change,
        so the geometry-only panel and the library's are the same extent."""
        self.stop_filling()
        self.infos = list(infos)
        for panel, info in zip(self.panels, self.infos):
            panel["info"] = info
            grid_n = int(info["grid_n"])
            panel["outside"] = np.asarray(info["outside"], bool).reshape(grid_n, grid_n)
            panel.pop("fill_order", None)
        show_points(self.panels)

    # ── the window ───────────────────────────────────────────────────────────

    def show(self) -> None:
        """Open the window if it is not up. Nothing to show without a phase."""
        if self._closed or self.window_id is not None or not self.infos:
            return
        self._rebuild()

    def hide(self) -> None:
        """Close the window, keeping everything that would go back into it."""
        window_id = self.window_id
        if window_id is None:
            return
        try:
            self.session._forget_window(window_id)
        except Exception as e:
            log.debug("closing the IPF panel window failed: %s", e)
        # _forget_window reaches _window_gone through the handle; when it did
        # not (a bare test session), do the bookkeeping here.
        if self.window_id == window_id:
            self._window_gone()

    def remove(self) -> None:
        """Full teardown: the tree is closing."""
        if self._closed:
            return
        self._closed = True
        self.stop_filling()
        self.hide()
        self.controller = None
        if panel_for(self.tree, self.key) is self:
            setattr(self.tree, _attr(self.key), None)

    def _title(self) -> str:
        try:
            base = self.signal.metadata.get_item("General.title", "Signal")
        except Exception:
            base = "Signal"
        return f"{base} — IPF Refine"

    def _rebuild(self) -> None:
        from spyde.actions.ipf_refine_render import (
            build_refine_figure, emit_refine_window,
        )
        if self.window_id is not None:
            self.hide()
        figure, figure_id, html, panels = build_refine_figure(self.infos)
        self.figure, self.panels = figure, panels
        self.window_id = emit_refine_window(self.session, figure, figure_id, html,
                                            title=self._title())
        self.session.register_window_controller(self.window_id, _WindowHandle(self))
        show_points(panels)
        if self.controller is not None:
            self.controller.bind_panels(panels)
            self.controller.redraw()

    def _window_gone(self) -> None:
        self.window_id = None
        self.figure = None
        self.panels = []
        if self.controller is not None:
            self.controller.bind_panels([])



# ── painting ─────────────────────────────────────────────────────────────────

def paint_fill(panels, fraction: float) -> None:
    """Light *fraction* of each triangle's AREA, sweeping from its left corner.

    By area and not by width: the triangle comes to a point at its left, so a
    sweep by width shows almost nothing for the first seconds and then most of
    the triangle at once. By area the lit part grows as the clock does.
    """
    from spyde.actions.ipf_refine_render import _corr_rgba, _rgba_b64

    for panel in panels:
        grid_n = int(panel["grid_n"])
        outside = np.asarray(panel["outside"], bool).reshape(grid_n, grid_n)
        order = panel.get("fill_order")
        if order is None:
            # Interior cells ranked left to right; a cell's rank over the
            # interior count is the fraction at which it lights.
            rows, columns = np.nonzero(~outside)
            rank = np.empty(rows.size, float)
            rank[np.argsort(columns, kind="stable")] = np.arange(rows.size)
            order = np.ones((grid_n, grid_n), float)
            order[rows, columns] = rank / max(1, rows.size)
            panel["fill_order"] = order
        values = np.where(order < fraction, FILL_LEVEL, 0.0)
        try:
            panel["raster"].set(image_b64=_rgba_b64(
                _corr_rgba(values, outside, panel["lut"])))
        except Exception as e:
            log.debug("painting the IPF fill failed: %s", e)


def show_points(panels) -> None:
    """Show each panel's sampled orientations as points over a blank raster,
    or nothing inside the triangle when the panel has no library yet."""
    from spyde.actions.ipf_refine_render import _corr_rgba, _rgba_b64

    for panel in panels:
        info = panel["info"]
        grid_n = int(panel["grid_n"])
        xs, ys = info.get("xs"), info.get("ys")
        offsets = (np.column_stack([xs, ys]).tolist()
                   if xs is not None and len(xs) else [])
        try:
            panel["raster"].set(image_b64=_rgba_b64(_corr_rgba(
                np.zeros((grid_n, grid_n)), panel["outside"], panel["lut"])))
            points = panel.get("points")
            if points is not None:
                points.set(offsets=offsets)
            panel["points_shown"] = bool(offsets)
        except Exception as e:
            log.debug("showing the IPF panel points failed: %s", e)
