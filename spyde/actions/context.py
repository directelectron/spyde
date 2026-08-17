"""
context.py — SpyDE's action-context helpers.

:class:`ActionContext` itself is app-agnostic and lives in the shell
(``de_shell.actions.context``); it is re-exported here so action modules keep
importing one module. What stays is the pair of resolvers that know what SpyDE's
plots are made of — signal trees and plot states.
"""
from __future__ import annotations

from de_shell.actions.context import ActionContext  # noqa: F401  (re-exported API)


def src_plot_tree(session, plot):
    """Resolve the source signal plot + its tree for a staged-wizard handler:
    the given *plot*, or the first non-navigator signal plot in the session.
    Returns ``(plot, tree)`` (either may be ``None``)."""
    src = plot or next(
        (p for p in session._plots if not p.is_navigator and p.plot_state is not None),
        None,
    )
    tree = getattr(src, "signal_tree", None) if src is not None else None
    return src, tree


def current_signal(src):
    """The signal currently displayed by *src* (its plot_state's current_signal)."""
    ps = getattr(src, "plot_state", None) if src is not None else None
    return getattr(ps, "current_signal", None) if ps is not None else None
