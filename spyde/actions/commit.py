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
                       on_tree: Callable[[Any], None] | None = None):
    """Commit a finished result as a NEW SignalTree — the Commit action.

    *primary* is the map shown as the tree's signal plot: a 2-D scalar array,
    or an (H, W, 3) RGB image (e.g. an IPF color map — displayed as-is, no
    contrast lock). *views* are extra ``(label, 2-D array)`` maps registered
    as chip-selectable views on the same window.

    *levels* locks the scalar contrast: an explicit ``(lo, hi)``, ``None``
    (auto-level), or ``"auto_sym"`` (default) → symmetric ±max|value| across
    the primary and all views — the right choice for signed strain components,
    keeping every view on one comparable scale.

    *attrs* are set on the tree (e.g. ``{"vector_orientation": result}`` so
    signal-type gates and downstream actions find the result object).
    *on_tree* runs after the tree is built (attach IPF explorers etc.).
    Returns the tree.
    """
    import hyperspy.api as hs
    from spyde.actions.views import emit_view_figure, register_views

    primary = np.asarray(primary)
    rgb = primary.ndim == 3
    label = primary_label or title
    mats: list[tuple[str, np.ndarray]] = []
    if not rgb:
        mats.append((label, np.nan_to_num(primary.astype(np.float32))))
    mats += [(lbl, np.nan_to_num(np.asarray(m, np.float32))) for lbl, m in views]

    if levels == "auto_sym":
        finite = [float(np.nanmax(np.abs(m))) for _, m in mats if np.isfinite(m).any()]
        lim = (max(finite) if finite else 1.0) or 1.0
        levels = (-lim, lim)
    if rgb:
        levels = None

    # The root signal carries the primary scalar map (so a saved committed tree
    # holds real data); an RGB primary keeps a zeros root and is painted onto
    # the plot only (hyperspy roots stay scalar).
    root_data = np.zeros(primary.shape[:2], np.float32) if rgb else mats[0][1].copy()
    new_sig = hs.signals.Signal2D(root_data)
    new_sig.metadata.General.title = title
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
            if levels is not None:
                sp.needs_auto_level = False
                sp.set_clim(float(levels[0]), float(levels[1]))
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
            register_views(wid, mats, cmap=cmap, levels=levels)
            first_extra = 1 if not rgb else 0
            for lbl, m in mats[first_extra:]:
                emit_view_figure(wid, m, lbl, kind="2d", cmap=cmap, levels=levels)

    if on_tree is not None:
        try:
            on_tree(tree)
        except Exception as e:
            log.debug("commit on_tree hook failed: %s", e)
    return tree
