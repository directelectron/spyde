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
        sharex/sharey → linked pan/zoom) with a crosshair on every panel.
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
        from spyde.backend.ipc import emit
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
        _teardown_stacked(session, plot)
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


def _tile_navigators(session, plot, tree, names) -> None:
    """Build ONE figure with the selected navigators side by side: sharex/sharey
    (linked pan/zoom) + a crosshair on every panel, linked together AND wired to
    the tree's real navigation selector so dragging any panel drives the DP."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html
    from spyde.actions.figure_registry import keep_alive
    from spyde.actions.views import _link_crosshairs, TILED_LABEL
    from spyde.backend.ipc import emit

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

    try:
        fig, axes = apl.subplots(1, len(pairs), sharex=True, sharey=True)
        arr_axes = np.array(axes, dtype=object).ravel()
        widgets = []
        for ax, (name, img) in zip(arr_axes, pairs):
            p = ax.imshow(img, cmap="gray")
            try:
                ax.set_title(name)
            except Exception as e:
                log.debug("set_title on tiled navigator failed: %s", e)
            h, w = img.shape[:2]
            widgets.append(p.add_crosshair_widget(cx=w / 2.0, cy=h / 2.0))
        _link_crosshairs(widgets)
        _wire_to_real_selector(session, wid, widgets)

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


class _StackedNavCursor:
    """One shared, linked time cursor across the stacked 1-D navigator rows.

    Owns the per-row VLine widgets, the reference to the tree's REAL 1-D
    navigation selector, and a single re-entrancy guard that covers BOTH
    directions of the loop:

      • a DRAG on any row's line → write the position onto the real selector's
        VLine and fire ``delayed_update_data(force=True)`` (the normal nav
        cascade repaints the signal/image plot), then mirror it to the other
        rows;
      • a PROGRAMMATIC selector move (playback clock, chain re-fire) → the real
        selector commits a new index and fires this cursor's ``index_hook``,
        which sets every row's line ``x`` to match.

    The guard (``_busy``) stops the ``set → pointer_move → handler → set`` echo
    (widget ``.set`` fires ``pointer_move``) AND the drive→hook→drive loop.
    Position writes to widgets are UI updates, so when the sync originates on the
    ``_NavDispatcher`` thread (the index_hook) they are marshalled onto the
    asyncio main thread via ``session._dispatch_to_main``."""

    def __init__(self, session, window_id: int, widgets, sel):
        self.session = session
        self.window_id = int(window_id)
        self.widgets = list(widgets)
        self.sel = sel
        self._busy = False
        self._handlers: list = []  # keep wrapper refs alive (weak registration)
        self._closed = False
        self._wire_drag()
        self._install_index_hook()

    # ── row line dragged → drive the real selector + mirror the other rows ──
    def _wire_drag(self) -> None:
        for w in self.widgets:
            h = self._make_drag_handler(w)
            self._handlers.append(h)
            for et in ("pointer_move", "pointer_up"):
                try:
                    w.add_event_handler(h, et)
                except Exception as e:
                    log.debug("wiring stacked cursor %s handler failed: %s", et, e)

    def _make_drag_handler(self, src):
        def handler(_ev=None):
            if self._busy:
                return
            self._busy = True
            try:
                x = float(src.get("x"))
                # Mirror to the other rows so the cursor is one logical line.
                for w in self.widgets:
                    if w is src:
                        continue
                    try:
                        w.set(x=x)
                    except Exception as e:
                        log.debug("mirroring stacked cursor line failed: %s", e)
                # Drive the tree's REAL 1-D navigation selector (same path a
                # normal drag on the live navigator takes).
                widget = _selector_vline_widget(self.sel)
                if widget is not None:
                    try:
                        widget.x = x
                    except Exception as e:
                        log.debug("writing real selector x failed: %s", e)
                try:
                    self.sel.delayed_update_data(force=True)
                except Exception as e:
                    log.debug("driving real selector from stacked cursor failed: %s", e)
            finally:
                self._busy = False
        return handler

    # ── selector moved (any source) → sync every row's line ──────────────
    def _install_index_hook(self) -> None:
        """Hang the sync off the SELECTOR's generic update path so ANY move —
        drag, playback (translate_pixels + delayed_update_data), 5-D chain
        re-fire — moves the lines. ``index_hooks`` fire in ``_run_update`` on
        the ``_NavDispatcher`` thread with the committed indices."""
        hook = self._on_selector_index
        self._index_hook = hook
        try:
            self.sel.index_hooks.append(hook)
        except Exception as e:
            log.debug("installing stacked cursor index hook failed: %s", e)

    def _on_selector_index(self, indices) -> None:
        if self._closed:
            return
        try:
            idx = int(np.asarray(indices).ravel()[0])
        except Exception:
            return
        scale, offset = _selector_axis(self.sel)
        x = idx * scale + offset
        # Marshal the widget writes onto the main thread (the hook runs on the
        # nav-dispatcher thread).
        self.session._dispatch_to_main(lambda: self._set_all_lines(x))

    def _set_all_lines(self, x: float) -> None:
        if self._closed or self._busy:
            return
        self._busy = True
        try:
            for w in self.widgets:
                try:
                    w.set(x=float(x))
                except Exception as e:
                    log.debug("syncing stacked cursor line failed: %s", e)
        finally:
            self._busy = False

    def close(self) -> None:
        """Tear down: detach the index hook so a torn-down stacked view can't
        keep syncing (and can be GC'd). Called on chip switch-back / window
        close."""
        self._closed = True
        try:
            if self._index_hook in self.sel.index_hooks:
                self.sel.index_hooks.remove(self._index_hook)
        except Exception as e:
            log.debug("removing stacked cursor index hook failed: %s", e)
        self.widgets = []


def _stacked_cursors(session) -> dict:
    if not hasattr(session, "_stacked_nav_cursors"):
        session._stacked_nav_cursors = {}
    return session._stacked_nav_cursors


def _teardown_stacked(session, plot) -> None:
    """Remove any stacked cursor previously built for this navigator window (its
    index hook detaches from the real selector). Idempotent."""
    wid = getattr(plot, "window_id", None)
    if wid is None:
        return
    cursor = _stacked_cursors(session).pop(int(wid), None)
    if cursor is not None:
        try:
            cursor.close()
        except Exception as e:
            log.debug("tearing down stacked cursor failed: %s", e)


def _stack_navigators(session, plot, tree, names) -> None:
    """Build ONE figure that STACKS the selected 1-D navigator traces as rows
    with a shared time (x) axis (``subplots(N, 1, sharex=True)``), a draggable
    vertical line on every row, all linked into ONE logical time cursor wired to
    the tree's real 1-D navigation selector (see module docstring)."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html
    from spyde.actions.figure_registry import keep_alive
    from spyde.backend.ipc import emit

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
    scale, offset = _selector_axis(sel) if sel is not None else (1.0, 0.0)
    # Current frame index → the cursor starts on the live selector position.
    cur_idx = 0
    if sel is not None and getattr(sel, "current_indices", None) is not None:
        try:
            cur_idx = int(np.asarray(sel.current_indices).ravel()[0])
        except Exception:
            cur_idx = 0
    cur_x = cur_idx * scale + offset

    # Tear down any previous stacked cursor on this window before rebuilding.
    _teardown_stacked(session, plot)

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
                widgets.append(p.add_vline_widget(x=float(cur_x), color="#ff9100"))
            except Exception as e:
                log.debug("adding vline to stacked navigator failed: %s", e)

        if len(widgets) >= 2 and sel is not None:
            cursor = _StackedNavCursor(session, int(wid), widgets, sel)
            _stacked_cursors(session)[int(wid)] = cursor

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
    from spyde.backend.ipc import emit_error, emit_status

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
    from spyde.backend.ipc import emit_error, emit_status

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


def _wire_to_real_selector(session, window_id: int, widgets) -> None:
    """Each tiled crosshair drives the tree's REAL navigation selector (the one
    on the live navigator plot), so navigation keeps working while tiled."""
    sel = getattr(session, "_nav_selectors", {}).get(window_id)
    if sel is None:
        return
    cross = getattr(sel, "_crosshair_selector", sel)
    widget = getattr(cross, "_widget", None)
    if widget is None:
        return

    def make(w):
        def handler(_ev=None):
            try:
                cx, cy = w.get("cx"), w.get("cy")
                widget.cx = float(cx)
                widget.cy = float(cy)
                sel.delayed_update_data(force=True)
            except Exception as e:
                log.debug("driving real selector from tiled navigator failed: %s", e)
        return handler

    for w in widgets:
        h = make(w)
        for et in ("pointer_move", "pointer_up"):
            try:
                w.add_event_handler(h, et)
            except Exception as e:
                log.debug("wiring tiled navigator %s handler failed: %s", et, e)
