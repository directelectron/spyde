"""
report — the SpyDE Report Builder backend (Phase 1).

A *report* is a portable, human-readable document (markdown + embedded figure
snapshots) the user composes from live SpyDE plots and exports to share. The
on-disk container is a ``.spyde-report`` zip whose contents are plain markdown +
YAML (NO JSON anywhere) — see :mod:`spyde.actions.report.model`.

Public surface:

* :mod:`~spyde.actions.report.model` — the data model + ``.spyde-report``
  (de)serialization (``ReportDoc``, ``Cell``, ``FigureSpec``, ``SignalRef``).
* :mod:`~spyde.actions.report.figure_builder` — build a live anyplotlib figure
  for a report figure cell.
* :mod:`~spyde.actions.report.handlers` — the staged action handlers
  (``report_new`` / ``report_open`` / ``report_save`` / …), registered in
  :data:`spyde.actions.registry.STAGED_HANDLERS`.
"""
from __future__ import annotations
