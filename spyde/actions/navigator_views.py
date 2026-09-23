"""
navigator_views.py — the navigator chip strip (multiple named navigators).

A signal tree can carry several NAMED navigators (``tree.navigator_signals``:
the base sum, a vector count map, a dropped-in compatible signal, …). The
navigator window lists them as chips at the top of the image (the same idiom
as strain's εxx/εyy/εxy strip):

  • click a chip        → switch the LIVE navigator display in place (the
                          selectors keep working — only the image changes),
  • shift-click chips   → compare the selected navigators in ONE anyplotlib
                          figure, all driving the tree's REAL navigation
                          selector (so the DP follows whichever panel you drag):

      – 2-D navigator (4D-STEM): TILE side by side (``subplots(1, N)``,
        sharex/sharey → linked pan/zoom) with a copy of every navigation
        selector's crosshair on every panel, each following its selector.
      – 1-D navigator (in-situ MOVIE / time series): STACK as rows
        (``subplots(N, 1, sharex=True)`` → one shared time axis) with a single
        logical time cursor — one draggable vertical line per row, all kept in
        sync, showing the current frame. Dragging any row's line drives the
        real 1-D navigation selector; and because the sync hangs off the
        selector's own update path (an ``index_hook``), a PROGRAMMATIC move
        (the playback clock stepping the selector, a 5-D chain re-fire) also
        moves every line — playback code is never touched.

This replaces the old dead "Select Navigator" toolbar action (which emitted
``navigation_options`` into the void — no renderer consumer, no handler).
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _sig_list(tree, name):
    """``navigator_signals[name]`` may be one signal or a per-level list."""
    sigs = tree.navigator_signals.get(name)
    if sigs is None:
        return []
    return list(sigs) if isinstance(sigs, (list, tuple)) else [sigs]


def _nav_plot_for_window(mgr, plot_window):
    plots = mgr.plots.get(plot_window) or []
    return plots[0] if plots else None


def _current_name(tree, plot) -> str | None:
    """Which named navigator the nav plot currently displays."""
    state = getattr(plot, "plot_state", None) if plot is not None else None
    cur = getattr(state, "current_signal", None) if state is not None else None
    if cur is None:
        return None
    for name in tree.navigator_signals:
        if any(s is cur for s in _sig_list(tree, name)):
            return name
    return None


def emit_navigator_options(tree) -> None:
    """Tell the renderer which named navigators this tree's navigator window(s)
    offer — the chip strip appears when there are two or more."""
    mgr = getattr(tree, "navigator_plot_manager", None)
    if mgr is None:
        return
    names = list(tree.navigator_signals.keys())
    try:
        from de_shell.ipc import emit
        for pw in mgr.plot_windows:
            plot = _nav_plot_for_window(mgr, pw)
            emit({
                "type": "navigator_options",
                "window_id": pw.window_id,
                "names": names,
                "current": _current_name(tree, plot),
            })
    except Exception as e:
        log.debug("emitting navigator options failed: %s", e)


def select_navigator(session, plot, payload) -> None:
    """Staged handler for the navigator chips.

    ``payload["names"]``: one name → switch the live navigator display in
    place; two or more (shift-click) → build the tiled comparison figure."""
    names = payload.get("names") or []
    if isinstance(names, str):
        names = [names]
    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    mgr = getattr(tree, "navigator_plot_manager", None) if tree is not None else None
    if not names or mgr is None:
        return
    if len(names) == 1:
        _tiled_navigators(session).pop(getattr(plot, "window_id", None), None)
        _teardown_view_cursors(session, plot)
        _switch_navigator(tree, plot, names[0])
        emit_navigator_options(tree)
    elif _tree_nav_is_1d(tree):
        # In-situ movie / time series: stack the 1-D traces with a shared,
        # linked time cursor (see module docstring).
        _stack_navigators(session, plot, tree, names)
    else:
        _tile_navigators(session, plot, tree, names)


def _tree_nav_is_1d(tree) -> bool:
    """True when this tree's navigation space is 1-D (a movie/time series) —
    the base navigator is a single 1-D trace, so ⇧-click STACKS rather than
    tiles."""
    try:
        return int(getattr(tree, "nav_dim", 0)) == 1
    except Exception:
        return False


def _switch_navigator(tree, plot, name: str) -> None:
    """Swap the live navigator figure to the named navigator signal — the
    selectors stay put (only the displayed image changes)."""
    sigs = _sig_list(tree, name)
    if not sigs:
        return
    for sig in list(getattr(plot, "plot_states", {}) or {}):
        if any(s is sig for s in sigs):
            try:
                plot.set_plot_state(sig)
                data = np.asarray(sig.data)
                plot.needs_auto_level = True
                plot.set_data(np.nan_to_num(data).astype(np.float32))
            except Exception as e:
                log.debug("switching navigator display failed: %s", e)
            return
    log.debug("navigator %r has no plot state on window %s", name,
              getattr(plot, "window_id", None))


def _tiled_navigators(session) -> dict:
    """window_id → the navigator names currently tiled in that window."""
    if not hasattr(session, "_tiled_navigators"):
        session._tiled_navigators = {}
    return session._tiled_navigators


def refresh_tiled_navigators(session, plot_window) -> None:
    """Rebuild the tiled figure of a navigator window that is currently tiled,
    so it carries one crosshair per selector after a selector is added or
    removed. A no-op for a window that is not tiled."""
    names = _tiled_navigators(session).get(getattr(plot_window, "window_id", None))
    if not names:
        return
    mgr = getattr(plot_window, "multiplot_manager", None)
    plot = _nav_plot_for_window(mgr, plot_window) if mgr is not None else None
    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    if tree is not None:
        _tile_navigators(session, plot, tree, names)


def _crosshair_widget(selector):
    """The crosshair widget of a 2-D navigation selector (the composite's point
    sub-selector, or the selector itself when it is a bare crosshair)."""
    return getattr(getattr(selector, "_crosshair_selector", selector), "_widget", None)


def _tile_navigators(session, plot, tree, names) -> None:
    """Build ONE figure with the selected navigators side by side: sharex/sharey
    (linked pan/zoom) and, on every panel, one crosshair per navigation selector
    of this window.

    Each selector's own crosshair on the live navigator is the MASTER: a panel's
    crosshair only forwards a drag to it, and every panel mirrors it back (see
    ``_LinkedNavCursor``), so all copies of one selector agree on one position."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html
    from de_shell.actions.figure_registry import keep_alive
    from spyde.actions.views import TILED_LABEL
    from de_shell.ipc import emit

    wid = getattr(plot, "window_id", None)
    if wid is None:
        return
    pairs = []
    for name in names:
        sig = next((s for s in _sig_list(tree, name)
                    if np.asarray(s.data).ndim == 2), None)
        if sig is None:
            continue
        pairs.append((name, np.nan_to_num(np.asarray(sig.data, np.float32))))
    if len(pairs) < 2:
        return
    _tiled_navigators(session)[int(wid)] = list(names)
    _teardown_view_cursors(session, plot)
    mgr = tree.navigator_plot_manager
    selectors = [
        sel for sel in mgr.navigation_selectors.get(getattr(plot, "plot_window", None), [])
        if _crosshair_widget(sel) is not None
    ]

    try:
        fig, axes = apl.subplots(1, len(pairs), sharex=True, sharey=True)
        arr_axes = np.array(axes, dtype=object).ravel()
        copies = {id(sel): [] for sel in selectors}
        for ax, (name, img) in zip(arr_axes, pairs):
            p = ax.imshow(img, cmap="gray")
            try:
                ax.set_title(name)
            except Exception as e:
                log.debug("set_title on tiled navigator failed: %s", e)
            for sel in selectors:
                master = _crosshair_widget(sel)
                copies[id(sel)].append(p.add_crosshair_widget(
                    cx=float(master.cx), cy=float(master.cy),
                    color=getattr(sel, "color", None) or "green"))
        _view_cursors(session)[int(wid)] = [
            _LinkedNavCursor(session, sel, _crosshair_widget(sel),
                             copies[id(sel)], ("cx", "cy"))
            for sel in selectors
        ]

        fig_id = _electron.register(fig)
        html = finalize_figure_html(fig, fig_id)
        keep_alive(int(wid), fig)
        emit({
            "type": "figure", "fig_id": fig_id, "window_id": wid,
            "html": html, "title": " / ".join(n for n, _ in pairs),
            "is_navigator": True,
            "view_label": TILED_LABEL, "view_kind": "tiled",
        })
    except Exception as e:
        log.exception("tiling navigators failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Stacked 1-D navigators — one shared, linked time cursor across the rows
# ─────────────────────────────────────────────────────────────────────────────

# Reserved chip label the frontend uses for the stacked figure (like TILED_LABEL
# for the 2-D side-by-side): shown while ≥2 chips are selected, excluded from the
# chip strip itself.
STACKED_LABEL = "__stacked__"


def _real_nav_selector(session, window_id: int):
    """The tree's REAL 1-D navigation selector for this navigator window (the one
    whose VLine widget the DP follows)."""
    return getattr(session, "_nav_selectors", {}).get(window_id)


def _selector_vline_widget(sel):
    """The draggable VLine widget backing the real 1-D navigation selector.

    A movie navigator's selector is an ``IntegratingSelector1D`` composite; its
    crosshair (point) sub-selector holds the VLine. Fall back to ``sel._widget``
    (which the composite's ``__getattr__`` delegates to the active sub-selector)
    so this also works if a bare ``InfiniteLineSelector`` is ever wired."""
    inner = getattr(sel, "_inf_line_selector", None)
    w = getattr(inner, "_widget", None) if inner is not None else None
    if w is None:
        w = getattr(sel, "_widget", None)
    return w


def _selector_axis(sel):
    """(scale, offset) of the 1-D navigation (time) axis the selector indexes,
    so we can map between a frame INDEX and the VLine's DATA-space ``x``."""
    try:
        plot = sel.current_plot
        ax = plot.plot_state.current_signal.axes_manager.signal_axes[0]
        return float(ax.scale), float(ax.offset)
    except Exception:
        return 1.0, 0.0


class _LinkedNavCursor:
    """Display copies of ONE navigation selector's widget, linked to it.

    The selector's own widget on the live navigator is the MASTER and the only
    holder of the position. Each copy (a stacked row's line, a tiled panel's
    crosshair):

      • forwards a drag to the master — the master's normal update path then
        drives the signal plot — and mirrors it onto the other copies;
      • follows the master whenever the selector commits a position, from any
        source (a drag on the live navigator, playback stepping, a 5-D chain
        re-fire): an ``index_hook`` on the selector copies the master's
        position onto every copy.

    Every write to a copy passes ``_notify=False`` so it cannot echo back as a
    drag. The hook runs on the navigation dispatcher thread, so the widget
    writes are marshalled onto the main thread, and they read the master's
    position at that moment rather than the committed index: the dispatcher
    lags a drag, and an older index would pull the dragged copy backwards."""

    def __init__(self, session, sel, master, widgets, fields):
        self.session = session
        self.sel = sel
        self.master = master
        self.widgets = list(widgets)
        self.fields = tuple(fields)
        self._handlers: list = []  # widget callbacks are held weakly
        self._closed = False
        for widget in self.widgets:
            handler = self._make_drag_handler(widget)
            self._handlers.append(handler)
            for event_type in ("pointer_move", "pointer_up"):
                try:
                    widget.add_event_handler(handler, event_type)
                except Exception as e:
                    log.debug("wiring linked cursor %s handler failed: %s", event_type, e)
        self._index_hook = self._on_selector_index
        try:
            self.sel.index_hooks.append(self._index_hook)
        except Exception as e:
            log.debug("installing linked cursor index hook failed: %s", e)

    def _position(self, widget) -> dict:
        return {field: float(widget.get(field)) for field in self.fields}

    def _make_drag_handler(self, source):
        def handler(_ev=None):
            if self._closed:
                return
            position = self._position(source)
            for widget in self.widgets:
                if widget is not source:
                    self._set_quietly(widget, position)
            try:
                self.master.set(**position)
                self.sel.delayed_update_data(force=True)
            except Exception as e:
                log.debug("driving the navigation selector from a copy failed: %s", e)
        return handler

    def _on_selector_index(self, _indices) -> None:
        if not self._closed:
            self.session._dispatch_to_main(self._follow_master)

    def _follow_master(self) -> None:
        if self._closed:
            return
        position = self._position(self.master)
        for widget in self.widgets:
            self._set_quietly(widget, position)

    @staticmethod
    def _set_quietly(widget, position: dict) -> None:
        if all(widget.get(field) == value for field, value in position.items()):
            return
        try:
            # A widget-only update never reaches the panel's stored state, which
            # the renderer restores after a drag in another panel — a tiled
            # copy snapped back to where it was built. So push the whole panel
            # when its pixels travel separately (an image), which keeps that
            # push small; a 1-D panel would resend its whole trace every frame.
            panel = widget._plot
            if getattr(panel, "_GEOM_KEYS", None):
                widget.set(_notify=False, _push=False, **position)
                panel._push()
            else:
                widget.set(_notify=False, **position)
        except Exception as e:
            log.debug("syncing a linked cursor copy failed: %s", e)

    def close(self) -> None:
        """Detach from the selector so a torn-down view stops following it."""
        self._closed = True
        try:
            if self._index_hook in self.sel.index_hooks:
                self.sel.index_hooks.remove(self._index_hook)
        except Exception as e:
            log.debug("removing linked cursor index hook failed: %s", e)
        self.widgets = []


def _view_cursors(session) -> dict:
    """window_id → the ``_LinkedNavCursor``s of that window's stacked or tiled
    navigator figure."""
    if not hasattr(session, "_nav_view_cursors"):
        session._nav_view_cursors = {}
    return session._nav_view_cursors


def close_view_cursors(session, window_id) -> None:
    """Close every linked cursor of a navigator window. Idempotent."""
    for cursor in _view_cursors(session).pop(window_id, []):
        try:
            cursor.close()
        except Exception as e:
            log.debug("closing linked nav cursor failed: %s", e)


def _teardown_view_cursors(session, plot) -> None:
    window_id = getattr(plot, "window_id", None)
    if window_id is not None:
        close_view_cursors(session, int(window_id))


def _stack_navigators(session, plot, tree, names) -> None:
    """Build ONE figure that STACKS the selected 1-D navigator traces as rows
    with a shared time (x) axis (``subplots(N, 1, sharex=True)``), a draggable
    vertical line on every row, all linked into ONE logical time cursor wired to
    the tree's real 1-D navigation selector (see module docstring)."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html
    from de_shell.actions.figure_registry import keep_alive
    from de_shell.ipc import emit

    wid = getattr(plot, "window_id", None)
    if wid is None:
        return

    pairs = []
    for name in names:
        sig = next((s for s in _sig_list(tree, name)
                    if np.asarray(s.data).ndim == 1), None)
        if sig is None:
            continue
        pairs.append((name, np.nan_to_num(np.asarray(sig.data, np.float32))))
    if len(pairs) < 2:
        return

    sel = _real_nav_selector(session, int(wid))
    master = _selector_vline_widget(sel) if sel is not None else None
    scale, offset = _selector_axis(sel) if sel is not None else (1.0, 0.0)
    # The rows start on the live selector position.
    cur_x = float(master.get("x")) if master is not None else offset

    _tiled_navigators(session).pop(int(wid), None)
    _teardown_view_cursors(session, plot)

    try:
        fig, axes = apl.subplots(len(pairs), 1, sharex=True)
        arr_axes = np.array(axes, dtype=object).ravel()
        widgets = []
        for ax, (name, trace) in zip(arr_axes, pairs):
            n = int(trace.shape[0])
            xax = np.arange(n) * scale + offset
            p = ax.plot(trace, axes=[xax], label=name)
            try:
                ax.set_title(name)
            except Exception as e:
                log.debug("set_title on stacked navigator failed: %s", e)
            try:
                widgets.append(p.add_vline_widget(x=cur_x, color="#ff9100"))
            except Exception as e:
                log.debug("adding vline to stacked navigator failed: %s", e)

        if len(widgets) >= 2 and master is not None:
            _view_cursors(session)[int(wid)] = [
                _LinkedNavCursor(session, sel, master, widgets, ("x",))]

        fig_id = _electron.register(fig)
        html = finalize_figure_html(fig, fig_id)
        keep_alive(int(wid), fig)
        emit({
            "type": "figure", "fig_id": fig_id, "window_id": wid,
            "html": html, "title": " / ".join(n for n, _ in pairs),
            "is_navigator": True,
            "view_label": STACKED_LABEL, "view_kind": "stacked",
        })
    except Exception as e:
        log.exception("stacking navigators failed: %s", e)


def add_navigator_from_window(session, plot, payload) -> None:
    """Drop a signal window onto a navigator's top bar → add its displayed
    signal as a NAMED navigator of the navigator's tree (must be nav-shaped:
    ``_preprocess_navigator`` enforces the shape contract)."""
    from de_shell.ipc import emit_error, emit_status

    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    src_wid = (payload or {}).get("source_window_id")
    if tree is None or src_wid is None:
        return
    src_plot = session._plot_by_window_id(int(src_wid))
    if src_plot is None:
        emit_error("Add navigator: the dragged window has no plot")
        return
    src_tree = getattr(src_plot, "signal_tree", None)
    state = getattr(src_plot, "plot_state", None)
    src_sig = getattr(state, "current_signal", None) if state is not None else None
    if src_sig is None and src_tree is not None:
        src_sig = src_tree.root
    if src_sig is None:
        emit_error("Add navigator: the dragged window has no signal")
        return

    title = src_sig.metadata.get_item("General.title", "") or "Navigator"
    name = title
    n = 2
    while name in tree.navigator_signals:
        name = f"{title} ({n})"
        n += 1
    try:
        # Deep-copy so the navigator entry doesn't alias the source tree's data.
        tree.add_navigator_signal(name, src_sig.deepcopy())
        emit_status(f"Added navigator: {name}")
    except ValueError as e:
        emit_error(f"Add navigator: {e}")
    except Exception as e:
        emit_error(f"Add navigator failed: {e}")
        log.exception("add_navigator_from_window failed")


def extract_navigator(session, plot, payload) -> None:
    """Drag a navigator chip out to the MDI area → the named navigator becomes
    its OWN signal tree (a standalone image dataset)."""
    from de_shell.ipc import emit_error, emit_status

    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    name = (payload or {}).get("name")
    if tree is None or not name:
        return
    sig = next((s for s in _sig_list(tree, name)
                if np.asarray(s.data).ndim == 2), None)
    if sig is None:
        emit_error(f"Extract navigator: no 2-D image for {name!r}")
        return
    data = np.nan_to_num(np.asarray(sig.data, np.float32))
    src_title = tree.root.metadata.get_item("General.title", "") or ""

    def _calibrate(new_tree):
        try:
            for ax, ref in zip(new_tree.root.axes_manager.signal_axes,
                               sig.axes_manager.signal_axes):
                ax.scale, ax.offset = ref.scale, ref.offset
                ax.units, ax.name = ref.units, ref.name
        except Exception as e:
            log.debug("calibrating extracted navigator failed: %s", e)

    from spyde.actions.commit import commit_result_tree
    commit_result_tree(
        session, title=name, primary=data, levels=None,
        provenance={"action": "Extract Navigator", "item": name,
                    "source_title": src_title},
        on_tree=_calibrate,
    )
    emit_status(f"Extracted navigator {name!r} to a new signal tree")
