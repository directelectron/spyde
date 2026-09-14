"""
commit.py — the two tree-spawning lifecycles for action results.

Every action result that becomes a standalone dataset goes through one of two
doors (both funnel into ``Session._add_signal`` and stamp provenance):

``open_result_tree``
    The EARLY-OPEN variant: the window appears immediately with a blank /
    placeholder signal and the compute fills it progressively (Find-Vectors
    count map, Orientation live IPF). The caller finalizes (attaches results,
    repaints) when the batch lands.

``commit_result_tree``
    The SNAPSHOT variant — **the Commit action**: freeze a live/finished
    result into a new SignalTree. The primary map becomes the signal plot;
    extra named maps ride along as chip-selectable views (``spyde.actions.
    views``: single-click shows one, ⌘-click tiles a comparison). Wizards
    expose this as their Commit/Submit button (``<key>_commit`` →
    ``WizardController.commit()`` → here).

Provenance: both stamp ``tree._commit_provenance`` and
``metadata.General.spyde_provenance`` with whatever dict the caller passes
(convention: ``{"action": <name>, "params": {...}, "source_title": <str>}``)
so a committed tree records where it came from — including through save/load.

Scale and meaning: a result map lives in the SCAN's navigation space, so a
commit that names its ``source_signal`` gets the scan's spatial calibration on
every node (ticks in nm, a scale bar), and ``value_units`` records what the
numbers are (``metadata.Signal.quantity`` — the label the colorbar draws).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np

log = logging.getLogger(__name__)


def _stamp_provenance(tree, signal, provenance: dict | None) -> None:
    if not provenance:
        return
    tree._commit_provenance = dict(provenance)
    try:
        signal.metadata.set_item("General.spyde_provenance", dict(provenance))
    except Exception as e:
        log.debug("stamping provenance metadata failed: %s", e)


def spatial_navigation_axes(source) -> list:
    """The two SPATIAL navigation axes of *source* — ``[x, y]`` in hyperspy's
    order — or ``[]`` when it has fewer than two.

    A time-resolved scan navigates ``(t, y, x)`` in memory, which hyperspy
    presents reversed as ``(x, y, t)``: the first two are always the spatial
    pair, so the clock never labels a spatial axis of the map."""
    try:
        axes = list(source.axes_manager.navigation_axes)
    except Exception:
        return []
    return axes[:2] if len(axes) >= 2 else []


def copy_navigation_calibration(source, target) -> bool:
    """Give a 2-D *target* map the spatial calibration of *source*'s scan.

    The map's signal axes ``(x, y)`` take the scale, offset, units and name of
    the scan's navigation axes ``(x, y)`` — so a strain map's ticks read in nm
    and it draws a scale bar. Copies ONLY when the grids match size for size:
    a rebinned or cropped fit is not on the scan's grid, and labelling it with
    the scan's calibration would be a confident lie, so it keeps pixel axes.
    Returns whether the calibration was applied."""
    if source is None or target is None:
        return False
    nav_axes = spatial_navigation_axes(source)
    try:
        sig_axes = list(target.axes_manager.signal_axes)[:2]
    except Exception:
        return False
    if len(nav_axes) != 2 or len(sig_axes) != 2:
        return False
    if any(int(a.size) != int(b.size) for a, b in zip(nav_axes, sig_axes)):
        return False
    for axis, reference in zip(sig_axes, nav_axes):
        axis.scale, axis.offset = float(reference.scale), float(reference.offset)
        axis.units = str(reference.units)
        name = str(getattr(reference, "name", "") or "")
        if name and name != "<undefined>":
            axis.name = name
    return True


def navigation_extent(source, shape):
    """``(x_coordinates, y_coordinates, units)`` for a map of *shape* ``(ny, nx)``
    drawn over *source*'s scan — what a bare figure (the live strain window, a
    chip view) hands to ``set_extent`` so it draws calibrated ticks and a scale
    bar. ``None`` when the scan is uncalibrated or the grids differ."""
    nav_axes = spatial_navigation_axes(source)
    if len(nav_axes) != 2 or len(shape) < 2:
        return None
    x_axis, y_axis = nav_axes
    ny, nx = int(shape[0]), int(shape[1])
    if int(x_axis.size) != nx or int(y_axis.size) != ny:
        return None
    units = str(x_axis.units)
    if units in ("", "<undefined>", "px"):
        return None
    from spyde.drawing.plots.plot import _clean_units
    return (np.asarray(x_axis.axis, dtype=float),
            np.asarray(y_axis.axis, dtype=float), _clean_units(units))


def _quantity_for(label: str, value_units) -> str:
    """The ``Signal.quantity`` a node labelled *label* gets from *value_units*:
    one string for every node, or a per-label mapping (missing → none)."""
    if not value_units:
        return ""
    if isinstance(value_units, str):
        return value_units
    return str(value_units.get(label, "") or "")


def _stamp_quantity(signal, quantity: str) -> None:
    if not quantity:
        return
    try:
        signal.metadata.set_item("Signal.quantity", quantity)
    except Exception as e:
        log.debug("stamping Signal.quantity failed: %s", e)


def _stamp_display(signal, *, colormap: str | None, symmetric: bool) -> None:
    """Record how a node asked to be shown — ``metadata.Spyde.display`` — so
    the plot keeps the diverging map and the zero-centred range on every node
    switch, not only on the first paint after Commit."""
    try:
        if colormap:
            signal.metadata.set_item("Spyde.display.colormap", str(colormap))
        signal.metadata.set_item("Spyde.display.symmetric", bool(symmetric))
    except Exception as e:
        log.debug("stamping the display recipe failed: %s", e)


def open_result_tree(session, *, title: str, signal=None, data=None,
                     signal_type: str | None = None, navigator_override=None,
                     selector_type=None, provenance: dict | None = None):
    """Open a NEW SignalTree up front for progressive fill-in.

    Pass either a prepared hyperspy *signal* (e.g. the lazy zero placeholder a
    Find-Vectors batch builds) or a raw *data* array (wrapped in a Signal2D).
    Returns the tree.
    """
    import hyperspy.api as hs
    if signal is None:
        signal = hs.signals.Signal2D(np.asarray(data))
    if signal_type:
        try:
            signal.set_signal_type(signal_type)
        except Exception as e:
            log.debug("set_signal_type(%s) failed: %s", signal_type, e)
    signal.metadata.General.title = title
    kwargs: dict[str, Any] = {}
    if navigator_override is not None:
        kwargs["navigator_override"] = navigator_override
    if selector_type is not None:
        kwargs["selector_type"] = selector_type
    tree = session._add_signal(signal, **kwargs)
    _stamp_provenance(tree, signal, provenance)
    return tree


def commit_result_tree(session, *, title: str, primary, primary_label: str | None = None,
                       views: Sequence[tuple[str, Any]] = (),
                       levels: tuple[float, float] | str | None = "auto_sym",
                       cmap: str = "gray", attrs: dict[str, Any] | None = None,
                       provenance: dict | None = None,
                       on_tree: Callable[[Any], None] | None = None,
                       source_signal=None, value_units=None, signed=None):
    """Commit a finished result as a NEW SignalTree — the Commit action.

    *primary* is the map shown as the tree's signal plot: a 2-D scalar array,
    or an (H, W, 3) RGB image (e.g. an IPF color map — displayed as-is, no
    contrast lock). *views* are extra ``(label, 2-D array)`` maps registered
    as chip-selectable views on the same window.

    *levels* sets the scalar contrast: an explicit ``(lo, hi)`` shared by
    every view; ``None`` → each view gets its own robust range; ``"auto_sym"``
    (default) → each view gets its own robust range centred on zero, the
    right choice for signed strain components. Per view rather than one
    shared scale: εxx at ±0.5 % beside ω at ±3° on one scale is a flat εxx.

    *signed* names the views whose zero means something (a field component,
    divergence, curl) — ``True`` for all, or a set of labels; ``"auto_sym"``
    implies all. A signed node keeps a zero-centred range through every later
    contrast change (``Spyde.display.symmetric``), and every scalar node keeps
    *cmap* (``Spyde.display.colormap``) — so a diverging map is still diverging
    after a node switch, not only on the first paint.

    *attrs* are set on the tree (e.g. ``{"vector_orientation": result}`` so
    signal-type gates and downstream actions find the result object).
    *on_tree* runs after the tree is built (attach IPF explorers etc.).

    *source_signal* is the scan the maps were computed over: every node takes
    its spatial navigation calibration (``copy_navigation_calibration``), so
    the maps draw nm ticks and a scale bar instead of pixels. *value_units*
    says what the numbers mean — one string for every node, or ``{label:
    quantity}`` per view (``"εxx (%)"``, ``"Bx (mrad)"``) — stamped as
    ``metadata.Signal.quantity``, which the plot draws as its colorbar label.
    An RGB primary is a picture, not a measurement, and gets none.
    Returns the tree.
    """
    import hyperspy.api as hs
    from spyde.actions.views import emit_view_figure, register_views

    from spyde.actions._common import robust_map_limits

    primary = np.asarray(primary)
    rgb = primary.ndim == 3
    label = primary_label or title
    mats: list[tuple[str, np.ndarray]] = []
    if not rgb:
        mats.append((label, np.nan_to_num(primary.astype(np.float32))))
    mats += [(lbl, np.nan_to_num(np.asarray(m, np.float32))) for lbl, m in views]

    all_signed = levels == "auto_sym" or signed is True
    signed_labels = set() if (signed is None or signed is True) else {str(s) for s in signed}

    def is_signed(view_label: str) -> bool:
        return all_signed or view_label in signed_labels

    # One range per view. Failed positions are NaN in the maps; the robust band
    # ignores them, so a few bad fits cannot flatten the whole map.
    if levels is None or levels == "auto_sym":
        view_levels = {lbl: robust_map_limits(m, symmetric=is_signed(lbl))
                       for lbl, m in mats}
        lock_primary = levels == "auto_sym"
    else:
        shared = (float(levels[0]), float(levels[1]))
        view_levels = {lbl: shared for lbl, _ in mats}
        lock_primary = True

    # The root signal carries the primary scalar map (so a saved committed tree
    # holds real data); an RGB primary keeps a zeros root and is painted onto
    # the plot only (hyperspy roots stay scalar).
    root_data = np.zeros(primary.shape[:2], np.float32) if rgb else mats[0][1].copy()
    new_sig = hs.signals.Signal2D(root_data)
    new_sig.metadata.General.title = title
    copy_navigation_calibration(source_signal, new_sig)
    if not rgb:
        _stamp_quantity(new_sig, _quantity_for(label, value_units))
        _stamp_display(new_sig, colormap=cmap, symmetric=is_signed(label))
    tree = session._add_signal(new_sig)
    _stamp_provenance(tree, new_sig, provenance)
    for k, v in (attrs or {}).items():
        setattr(tree, k, v)

    # The views are ALSO committed as REAL child signal nodes — not just
    # chip-selectable display figures. A saved committed tree then carries every
    # component (a committed Strain tree used to hold εxx alone: the εyy/εxy/ω
    # chips were figures, so saving / downstream processing lost them), and the
    # Workflow panel can switch between the nodes.
    view_mats = mats[1:] if not rgb else mats
    for lbl, m in view_mats:
        try:
            child = hs.signals.Signal2D(m.copy())
            child.metadata.General.title = f"{title} {lbl}"
            copy_navigation_calibration(source_signal, child)
            _stamp_quantity(child, _quantity_for(lbl, value_units))
            _stamp_display(child, colormap=cmap, symmetric=is_signed(lbl))
            tree.add_node(new_sig, child, lbl)
            tree.update_plot_states(child)
        except Exception as e:
            log.debug("committing view node %r failed: %s", lbl, e)
    if view_mats:
        try:
            session._reemit_signal_tree(tree)
        except Exception as e:
            log.debug("re-emitting committed tree failed: %s", e)

    sp = next(iter(getattr(tree, "signal_plots", []) or []), None)
    if sp is not None:
        try:
            if not rgb:
                # The primary's colours are the tree's: the plot keeps them
                # across node switches through the display recipe above.
                sp.set_colormap(cmap)
            if not rgb and lock_primary:
                sp.needs_auto_level = False
                sp.set_clim(*view_levels[label])
            else:
                sp.needs_auto_level = True
            sp.set_data(primary if rgb else mats[0][1])
        except Exception as e:
            log.debug("painting committed signal plot failed: %s", e)
        if primary_label or views:
            try:
                sp.set_view_tag(label, "2d")
            except Exception as e:
                log.debug("tagging committed view failed: %s", e)
        wid = getattr(sp, "window_id", None)
        if wid is not None and mats and views:
            axes = navigation_extent(source_signal, primary.shape[:2])
            value_labels = {lbl: _quantity_for(lbl, value_units) for lbl, _ in mats}
            register_views(wid, mats, cmap=cmap, levels=view_levels, axes=axes,
                           value_labels=value_labels)
            first_extra = 1 if not rgb else 0
            for lbl, m in mats[first_extra:]:
                emit_view_figure(wid, m, lbl, kind="2d", cmap=cmap,
                                 levels=view_levels.get(lbl), axes=axes,
                                 value_label=value_labels.get(lbl, ""))

    if on_tree is not None:
        try:
            on_tree(tree)
        except Exception as e:
            log.debug("commit on_tree hook failed: %s", e)
    return tree
