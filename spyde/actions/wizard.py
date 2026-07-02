"""
wizard.py — WizardController, the base class for staged-wizard actions.

A *wizard* is the staged action shape (see ``spyde/actions/README.md``): the
renderer caret mounts → ``<key>_open`` starts a live preview / controller;
parameter edits stream in (``<key>_tune`` / ``<key>_set_<param>``); a heavy
stage runs (``<key>_run``); an optional Commit snapshots the live result into
a new SignalTree (``<key>_commit`` → :meth:`commit` →
``spyde.actions.commit.commit_result_tree``); unmount → ``<key>_close`` tears
everything down.

The controller owns the wizard's state (library / overlay / field / windows)
instead of a bare dict on the tree, and provides the lifecycle plumbing every
wizard needs:

* the run/stop **generation guard** (:meth:`guard` / :meth:`still` /
  :meth:`cancel_inflight`) — see ``lifecycle.bump_generation`` for the React
  StrictMode contract (open, close, open fired synchronously before any
  worker lands must leave exactly ONE live controller);
* **window registration** (:meth:`own_window`) so bare-figure windows the
  wizard opens are reachable by dispatch and torn down by
  ``Session._forget_window`` (which calls :meth:`close`);
* the **worker marshal** (:meth:`run_on_worker`) bound to the session;
* **overlay replacement** (:meth:`replace_overlay`).

Subclasses override :meth:`remove` (full teardown — MUST be idempotent; guard
with ``self._closed``) and optionally :meth:`commit`.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from spyde.actions.lifecycle import (
    bump_generation, is_current, replace_tree_attr, run_on_worker as _run_on_worker,
)

log = logging.getLogger(__name__)


class WizardController:
    #: the wizard's short prefix — "strain" → ``tree._strain_run_gen`` and the
    #: ``strain_*`` staged-action names.
    key: str = ""

    #: Declared parameter schema — REQUIRED for every wizard. The same dict
    #: spec as ``toolbars.yaml parameters:`` / ``Action.parameters`` (``type``
    #: int/float/bool/enum/file, ``name``, ``default``, ``min``/``max``/
    #: ``step``, ``choices``, ``tab``, ``extensions``), so any host — the
    #: Electron caret or an auto-generated notebook form — can render the
    #: wizard's controls from one source of truth. Resolved host-agnostically
    #: via ``registry.wizard_parameters(key)``; completeness is enforced by
    #: ``test_wizard_schemas.py``. (Three-host parity contract:
    #: NOTEBOOK_PARITY_PLAN.md §6.)
    parameters: dict = {}

    def __init__(self, session, tree):
        self.session = session
        self.tree = tree
        self._closed = False

    # ── generation guard ──────────────────────────────────────────────────────

    @property
    def _gen_key(self) -> str:
        return f"_{self.key}_run_gen"

    def guard(self) -> int:
        """Open a new run generation (call synchronously in the open handler,
        BEFORE spawning any worker). Deferred builds check :meth:`still`."""
        return bump_generation(self.tree, self._gen_key)

    def still(self, gen: int) -> bool:
        """True if *gen* is still the current run generation."""
        return is_current(self.tree, self._gen_key, gen)

    def cancel_inflight(self) -> None:
        """Invalidate any in-flight open (call FIRST in the close handler)."""
        bump_generation(self.tree, self._gen_key)

    # ── plumbing ──────────────────────────────────────────────────────────────

    def own_window(self, window_id) -> None:
        """Register this controller for a bare-figure window it opened."""
        if window_id is None or self.session is None:
            return
        reg = getattr(self.session, "register_window_controller", None)
        if reg is not None:
            reg(int(window_id), self)

    def run_on_worker(self, work: Callable[[], Any], *, name: str | None = None,
                      on_done=None, on_error=None) -> None:
        _run_on_worker(self.session, work, name=name or f"{self.key}-worker",
                       on_done=on_done, on_error=on_error)

    def replace_overlay(self, attr: str, factory):
        """Swap ``tree.<attr>`` for a fresh overlay, removing the prior one."""
        return replace_tree_attr(self.tree, attr, factory)

    # ── lifecycle hooks ───────────────────────────────────────────────────────

    def close(self) -> None:
        """WindowController protocol — ``Session._forget_window`` calls this
        when an owned window goes away for any reason. Default: full teardown."""
        self.remove()

    def remove(self) -> None:
        """Full teardown of everything the wizard added. MUST be idempotent
        (guard with ``self._closed``)."""
        raise NotImplementedError

    def commit(self):
        """Snapshot the live result into a new SignalTree (the ``<key>_commit``
        stage) — implement with ``spyde.actions.commit.commit_result_tree``.
        Returns the new tree."""
        raise NotImplementedError(f"{type(self).__name__} has no commit stage")
