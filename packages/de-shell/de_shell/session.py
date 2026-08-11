"""
session.py — the app-agnostic half of the backend coordinator.

Every shell app has exactly one Session: the object the frontend talks to
through IPC and that talks back through ``de_shell.ipc.emit``. Most of what such
an object does is the same in all three apps — hand out window ids, keep the
registry of open plots and window controllers, marshal worker results onto the
asyncio main thread, and persist a little JSON of user settings.

None of that knows what the data is. ``SessionBase`` owns that half; an app
subclasses it and adds its own (SpyDE: signal trees, the Dask cluster, file
I/O, the action mixins).

Two things deliberately stay the app's job:

* **Settings location.** Passed in as ``settings_dir``, not derived here. SpyDE
  reads ``SPYDE_SETTINGS_DIR`` and falls back to ``~/.spyde``; another app has
  its own directory and its own override variable, and baking a guess into the
  shell would silently write one app's preferences into another's file.
* **Shutdown order.** ``shutdown()`` here tears down only what it owns and is
  safe to call twice. Subclasses override, do their own work, and call
  ``super().shutdown()`` — ordering between an app's cluster, workers and
  caches is app knowledge.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)


class SessionBase:
    """Window registry, main-loop marshalling and the settings store."""

    #: Cap on the persisted recent-files list.
    MAX_RECENT = 20

    def __init__(self, settings_dir: str) -> None:
        # ── Window / plot registry ────────────────────────────────────────────
        self._plots: list[Any] = []           # every open Plot
        self._next_window_id = 0
        self._active_window_id: int | None = None   # focused window
        # window_id -> controller for windows that are NOT registered Plots
        # (bare `figure` emits). See the WindowController protocol in
        # de_shell.actions.registry; _forget_window closes + evicts.
        self._window_controllers: dict[int, object] = {}

        # ── Main-thread marshalling ───────────────────────────────────────────
        # Set by set_main_loop once the asyncio loop is running. Until then
        # _dispatch_to_main runs inline, which is what makes a Session usable in
        # a plain test with no loop at all.
        self._main_loop = None

        # Set by shutdown() so late work draining on a background thread can't
        # resurrect anything after teardown.
        self._closed = False

        # ── Settings ──────────────────────────────────────────────────────────
        self._settings_path = os.path.join(settings_dir, "settings.json")
        self._settings: dict[str, Any] = self._load_settings()
        self._recent_files: list[str] = []
        try:
            self._recent_files = list(
                self._settings.get("recent_files", []))[:self.MAX_RECENT]
        except Exception as e:
            log.debug("restoring recent files from settings failed: %s", e)
        self._update_channel: str = (
            self._settings.get("update_channel")
            if self._settings.get("update_channel") in ("stable", "beta")
            else "stable"
        )

    # ── Main-thread marshalling ───────────────────────────────────────────────

    def set_main_loop(self, loop) -> None:
        """Register the main asyncio loop so background workers can marshal their
        result-apply onto this (main) thread. Call once the loop is running.

        NB the process's frozen-timer pathology (waits only wake on process I/O
        — see the backend tick in @de/shell-main's backendProcess) is healed by
        Electron's 0.5 Hz stdin tick; an in-process wake ticker was tried and is
        useless here because its own sleep freezes the same way.
        """
        self._main_loop = loop

    def _dispatch_to_main(self, fn) -> None:
        """Schedule ``fn()`` on the main asyncio thread.

        Falls back to running inline when no loop is registered yet (early
        startup, and tests that never start one) — so a worker callback is never
        silently dropped just because the loop isn't up.
        """
        loop = self._main_loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(fn)
                return
            except Exception as e:
                log.debug("dispatch_to_main failed, running inline: %s", e)
        fn()

    # ── Window / plot registry ────────────────────────────────────────────────

    def next_window_id(self) -> int:
        wid = self._next_window_id
        self._next_window_id += 1
        return wid

    def register_plot(self, plot) -> None:
        self._plots.append(plot)

    def unregister_plot(self, plot) -> None:
        # Identity-based, and removes every occurrence. Kept exactly as it was
        # when lifted out of SpyDE: `register_plot` does not dedupe, so `remove`
        # (first match, __eq__-based) would not be equivalent.
        self._plots = [p for p in self._plots if p is not plot]

    def _plot_by_window_id(self, window_id: int):
        for p in self._plots:
            if getattr(p, "window_id", None) == window_id:
                return p
        return None

    def register_window_controller(self, window_id: int, controller) -> None:
        """Give a non-Plot window (a bare `figure` emit) a dispatch + teardown
        identity. See the WindowController protocol in de_shell.actions.registry;
        the app's `_forget_window` pops the controller and calls its close()."""
        self._window_controllers[window_id] = controller

    def controller_by_window_id(self, window_id: int | None):
        if window_id is None:
            return None
        return self._window_controllers.get(window_id)

    # ── Settings & recent files ───────────────────────────────────────────────

    def _load_settings(self) -> dict:
        try:
            with open(self._settings_path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_settings(self) -> None:
        os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
        with open(self._settings_path, "w", encoding="utf-8") as fh:
            json.dump(self._settings, fh, indent=2)

    def _add_recent(self, path: str) -> None:
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._settings["recent_files"] = self._recent_files[:self.MAX_RECENT]
        try:
            self._save_settings()
        except Exception as e:
            log.debug("saving recent-files settings failed: %s", e)

    def get_recent_files(self) -> list[str]:
        return list(self._recent_files[:self.MAX_RECENT])

    def set_update_channel(self, channel: str) -> None:
        """Persist the update channel ('stable' or 'beta') to settings.json.

        Mirrors the choice the Electron main process's autoUpdater actually acts
        on — kept here too so the preference is visible from the Python side and
        survives a settings.json inspection independent of Electron's storage.
        """
        if channel not in ("stable", "beta"):
            log.warning("ignoring invalid update_channel %r", channel)
            return
        self._update_channel = channel
        self._settings["update_channel"] = channel
        try:
            self._save_settings()
        except Exception as e:
            log.debug("saving update_channel setting failed: %s", e)

    # ── First-run welcome tour ────────────────────────────────────────────────

    @property
    def first_run(self) -> bool:
        """True until the welcome tour has been opened/dismissed once. Mirrors
        the ``tutorial_seen`` settings key: absent (never set) => first run."""
        return not bool(self._settings.get("tutorial_seen", False))

    def mark_tutorial_seen(self) -> None:
        """Persist that the welcome tour has been shown, so it never auto-opens
        again. Idempotent — the renderer calls it every time the tour opens."""
        if self._settings.get("tutorial_seen") is True:
            return
        self._settings["tutorial_seen"] = True
        try:
            self._save_settings()
        except Exception as e:
            log.debug("saving tutorial_seen setting failed: %s", e)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Tear down what the base owns. Idempotent.

        Subclasses override, shut their own things down first, then call
        ``super().shutdown()`` — the ordering between an app's cluster, worker
        threads and caches is the app's knowledge, not the shell's.
        """
        self._closed = True
        self._window_controllers.clear()
