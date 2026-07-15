"""
toolbar_actions.py — the Report Builder's YAML toolbar-action wrappers (Phase 3).

A YAML toolbar action (``spyde/toolbars.yaml``) resolves to a plain function
called as ``fn(ctx, action_name=name, **params)`` with an
:class:`~spyde.actions.context.ActionContext`. The report handlers, by contrast,
are staged actions with the ``fn(session, plot, payload)`` signature. This module
bridges the two: ``copy_to_report`` is the "Copy to Report" toolbar button — it
snapshots the clicked window into a report figure cell by delegating to the
existing ``report_cell_from_window`` handler (which auto-opens a report via
``_ensure_open``).
"""
from __future__ import annotations

import logging

from spyde.backend import ipc

log = logging.getLogger(__name__)


def copy_to_report(ctx, action_name=None, **params):
    """Copy the clicked plot's current image into the report as a figure cell.

    Delegates to ``report_cell_from_window`` (which snapshots the source ``Plot``
    NOW into a single-panel FigureSpec + appends a figure cell, opening a fresh
    report first if none exists). Non-toggling: adds one cell per click."""
    plot = getattr(ctx, "plot", None)
    session = getattr(ctx, "session", None)
    window_id = getattr(plot, "window_id", None) if plot is not None else None
    if session is None or window_id is None:
        ipc.emit_error("Copy to Report: no active plot window.")
        return None
    from spyde.actions.report.handlers import report_cell_from_window
    report_cell_from_window(session, plot, {"source_window_id": int(window_id)})
    return None
