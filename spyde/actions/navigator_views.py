"""
navigator_views.py — the navigator chip strip (multiple named navigators).

A signal tree can carry several NAMED navigators (``tree.navigator_signals``:
the base sum, a vector count map, a dropped-in compatible signal, …). The
navigator window lists them as chips at the top of the image (the same idiom
as strain's εxx/εyy/εxy strip):

  • click a chip        → switch the LIVE navigator display in place (the
                          selectors keep working — only the image changes),
  • shift-click chips   → TILE the selected navigators side by side in ONE
                          anyplotlib figure: pan/zoom linked (sharex/sharey)
                          and a crosshair duplicated on every panel, all
                          driving the tree's REAL navigation selector (so the
                          DP follows whichever panel you drag).

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
        _switch_navigator(tree, plot, names[0])
        emit_navigator_options(tree)
    else:
        _tile_navigators(session, plot, tree, names)


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
