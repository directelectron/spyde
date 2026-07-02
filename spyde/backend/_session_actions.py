"""
_session_actions.py — ActionRouterMixin extracted from session.py.

The renderer→backend action router (``dispatch_action``), the YAML toolbar-action
invoker (``_dispatch_toolbar_action``), action-artifact tracking, overlay
visibility, action (de)activation, and per-VI caret edits.

The staged-action table lives in ``spyde.actions.registry`` (STAGED_HANDLERS);
the ``_TEST_ACTIONS`` / ``_TEST_ACTIONS_ENABLED`` packaged-build gate lives
here.

The mixin only USES ``self.<attr>`` (``self._action_artifacts``, ``self._plots``
…) and ``self.<method>`` (``self._plot_by_window_id``, ``self._close_plot``,
``self._load_test_*`` …) provided by the final Session.
"""
from __future__ import annotations

import logging
import os
import threading

from spyde.backend import ipc
from spyde.backend.ipc import emit_error
from spyde.actions.registry import STAGED_HANDLERS, resolve_staged

log = logging.getLogger(__name__)

# Test-only actions (load synthetic/example data, scripted nav-drag, headless
# orientation) are reachable from the renderer and download real datasets. They
# back the Playwright e2e suite, which runs the UNPACKED app, but must NOT be
# exposed in a shipped (packaged) build. The Electron main process sets
# SPYDE_PACKAGED=1 only when app.isPackaged (index.ts) and the backend inherits
# it (runner.ts), so: enabled in dev + e2e, disabled in production.
_TEST_ACTIONS_ENABLED = os.environ.get("SPYDE_PACKAGED") != "1"
_TEST_ACTIONS = frozenset({
    "load_test_data", "load_test_data_lazy", "load_test_data_lazy_chunked",
    "load_test_data_si_grains", "load_test_data_sped_ag", "test_nav_drag",
    "load_test_vectors", "run_test_orientation",
})

# The staged-action table (STAGED_HANDLERS) lives in spyde.actions.registry so
# that adding an action only touches the actions package (+ toolbars.yaml).


class ActionRouterMixin:
    def dispatch_action(self, msg: dict) -> None:
        """Route an action message from Electron to the appropriate handler."""
        action = msg.get("action")
        payload = msg.get("payload", {})
        window_id = msg.get("window_id")

        plot = self._plot_by_window_id(window_id) if window_id is not None else None

        if action in _TEST_ACTIONS and not _TEST_ACTIONS_ENABLED:
            log.warning("ignoring test-only action %r in a packaged build", action)
            return

        if action == "load_test_data":
            self._load_test_data()
        elif action == "load_test_data_lazy":
            self._load_test_data_lazy()
        elif action == "load_test_data_lazy_chunked":
            self._load_test_data_lazy_chunked()
        elif action == "load_test_data_si_grains":
            self._load_test_data_si_grains()
        elif action == "load_test_data_sped_ag":
            self._load_test_data_sped_ag()
        elif action == "test_nav_drag":
            # Run on a BACKGROUND thread: the drag loop sleeps/polls, and if it ran
            # on the main asyncio thread it would block loop.call_soon_threadsafe —
            # i.e. the very main-thread applies it's trying to observe.
            threading.Thread(
                target=self._test_nav_drag, args=(payload.get("targets") or [],),
                daemon=True, name="test-nav-drag",
            ).start()
        elif action == "load_test_vectors":
            self._load_test_vectors()
        elif action in STAGED_HANDLERS:
            # Staged-wizard handlers (Orientation / Find-Vectors / Vector-OM /
            # Center-Zero-Beam) share the (session, plot, payload) signature and
            # are imported lazily so their heavy deps load only on first use.
            # Ensure window_id is always reachable via payload too: some staged
            # windows (e.g. the Strain map) are bare `figure` messages, not a
            # registered Plot, so `plot` above is None and a handler can only
            # resolve its target via payload["window_id"] — the renderer sends
            # windowId as the dispatch's own top-level field, not nested in the
            # payload object, so without this a caret button that only carries
            # e.g. {component: "eyy"} silently resolved to nothing.
            if "window_id" not in payload and window_id is not None:
                payload = {**payload, "window_id": window_id}
            resolve_staged(action)(self, plot, payload)
        elif action == "run_test_orientation":
            # Test-only: run Orientation Mapping with a built-in phase (no CIF
            # dialog) on the active signal, so the E2E workflow can be driven
            # headlessly / in Playwright. payload={"phase":"si"|"ag"} (default si).
            self._run_test_orientation(plot, payload)
        elif action == "set_selector_mode":
            self.set_selector_mode(window_id, bool(payload.get("integrate")))
        elif action == "select_signal_node":
            self._select_signal_node(plot, payload.get("signal_id"))
        elif action == "set_axis":
            self._set_axis(plot, payload)
        elif action == "set_offset_crosshair":
            self._set_offset_crosshair(plot, payload)
        elif action == "set_overlay":
            self._set_overlay(plot, payload.get("name"),
                              bool(payload.get("visible", True)))
        elif action == "set_action_active":
            self._set_action_active(
                window_id, payload.get("name"), bool(payload.get("active"))
            )
        elif action == "update_vi":
            self._update_vi(window_id, payload.get("name"), payload.get("params", {}))
        elif action == "open_file":
            self.open_file(payload["path"])
        elif action == "open_stack":
            self.open_stack(payload.get("paths") or [])
        elif action == "confirm_nav_shape":
            self._confirm_nav_shape(payload)
        elif action == "set_signal_type":
            self._set_signal_type(plot, payload.get("signal_type", ""))
        elif action == "load_example":
            self.load_example_data(payload["name"])
        elif action == "set_active":
            wid = payload.get("window_id", window_id)
            if wid is not None:
                self._active_window_id = wid
        elif action == "save_signal":
            self._save_signal(payload.get("path"), plot)
        elif action == "set_colormap":
            self._set_colormap(plot, payload.get("name"))
        elif action == "set_clim":
            self._set_clim(plot, payload.get("vmin"), payload.get("vmax"))
        elif action == "close_window":
            self._close_window(window_id)
        elif action == "resize_figure":
            self._resize_figure(window_id, payload.get("width"), payload.get("height"))
        elif action == "figure_event":
            self._dispatch_figure_event(window_id, payload.get("event_json"))
        elif action == "toolbar_action":
            self._dispatch_toolbar_action(
                plot, payload.get("name"), payload.get("params", {})
            )
        else:
            log.warning("Unknown action: %s", action)

    def _dispatch_toolbar_action(self, plot, name: str, params: dict) -> None:
        """Invoke a YAML-configured toolbar action by name on *plot*.

        The action function is resolved from TOOLBAR_ACTIONS and called with an
        ActionContext, so the same functions that ran under the Qt toolbar run
        here unchanged.  Parameter values collected by the Electron parameter
        panel arrive in *params* and are forwarded as kwargs.
        """
        if plot is None or not name:
            emit_error("Toolbar action: no active plot or action name")
            return

        # Actions whose modules still carry the Qt/interactive implementation and
        # haven't been ported to the host-agnostic template yet. Clicking them
        # gives a clear message instead of a confusing Qt-without-QApplication
        # traceback. (Virtual Imaging / FFT / Line Profile / Rebin ARE ported.)
        NOT_YET_PORTED: set = set()
        if name in NOT_YET_PORTED:
            emit_error(f"'{name}' is not yet available in the Electron build.")
            return

        try:
            import importlib
            from spyde import TOOLBAR_ACTIONS
            from spyde.actions.context import ActionContext

            meta = TOOLBAR_ACTIONS["functions"].get(name)
            if meta is None:
                # Sub-toolbar action (e.g. "add_virtual_image") — search the
                # subfunctions of every top-level action.
                for parent in TOOLBAR_ACTIONS["functions"].values():
                    subs = parent.get("subfunctions", {}) or {}
                    if name in subs:
                        meta = subs[name]
                        break
            if meta is None:
                emit_error(f"Unknown toolbar action: {name}")
                return
            module_path, _, attr = meta["function"].rpartition(".")
            target = getattr(importlib.import_module(module_path), attr)
            ctx = ActionContext(plot=plot, params=params, action_name=name)

            # A target may be either an Action subclass (template style) or a
            # plain function (legacy style). Both receive the same ActionContext.
            from spyde.actions.action import Action
            if isinstance(target, type) and issubclass(target, Action):
                result = target(ctx).run(**params)
            else:
                result = target(ctx, action_name=name, **params)
            self._track_action_artifacts(plot, name, result)
        except Exception as e:
            emit_error(f"Action '{name}' failed: {e}")
            log.exception("Action '%s' failed", name)

    def _track_action_artifacts(self, src_plot, name: str, result) -> None:
        """Remember the selector + output windows a RegionAction created so the
        toolbar can mark the action 'active' and hide them again on deselect."""
        if result is None or not hasattr(result, "active_children"):
            return
        src_wid = getattr(src_plot, "window_id", None)
        if src_wid is None:
            return
        out_wids = sorted({
            c.window_id for c in getattr(result, "active_children", [])
            if getattr(c, "window_id", None) is not None
        })
        self._action_artifacts[(src_wid, name)] = {"selector": result, "out_wids": out_wids}
        ipc.emit({"type": "action_active", "window_id": src_wid, "name": name, "active": True})

    def _set_overlay(self, plot, name: str, visible: bool) -> None:
        """Show/hide the live DP overlay(s) tied to a toolbar action — the marker
        overlay is only drawn while its action (caret) is SELECTED. The overlay
        still tracks the navigator while hidden, so re-selecting redraws the
        current frame."""
        tree = getattr(plot, "signal_tree", None) if plot is not None else None
        if tree is None or not name:
            return
        overlays = []
        if name == "Find Diffraction Vectors":
            # Two overlays: the SOURCE-DP one (_vector_overlay) and the one on the
            # RESULT vectors-image window (_result_vector_overlay). The user clicks
            # the action on EITHER window, so toggle both.
            overlays.append(getattr(tree, "_vector_overlay", None))
            overlays.append(getattr(tree, "_result_vector_overlay", None))
        elif name == "Orientation Mapping":
            overlays.append(getattr(tree, "_orientation_overlay", None))
            wiz = getattr(tree, "_om_wizard", None)
            if wiz:
                overlays.append(wiz.get("overlay"))
        elif name == "Vector Orientation Mapping":
            wiz = getattr(tree, "_vom_wizard", None)
            if wiz:
                overlays.append(wiz.get("overlay"))
        for ov in overlays:
            if ov is not None and hasattr(ov, "set_visible"):
                try:
                    ov.set_visible(visible)
                except Exception as e:
                    log.debug("toggling overlay visibility failed: %s", e)

    def _set_action_active(self, window_id: int, name: str, active: bool) -> None:
        """Deselecting an action hides the output window + ROI selector it made
        (Qt parity: an unchecked toolbar action removes its artifacts)."""
        key = (window_id, name)
        art = self._action_artifacts.get(key)
        if active or art is None:
            return
        # Closing each output plot also cleans its source ROI (parent_selector).
        for wid in art.get("out_wids", []):
            p = self._plot_by_window_id(wid)
            if p is not None:
                self._close_plot(p)
        try:
            art["selector"].close()
        except Exception as e:
            log.debug("closing action selector failed: %s", e)
        self._action_artifacts.pop(key, None)
        ipc.emit({"type": "action_active", "window_id": window_id, "name": name, "active": False})
        # If this was a virtual-image chip, drop it from the source plot's list
        # and tell the sub-toolbar to remove the chip.
        src = self._plot_by_window_id(window_id)
        if src is not None and hasattr(src, "_vi_items"):
            src._vi_items = [it for it in src._vi_items if it.get("name") != name]
        ipc.emit({"type": "sub_item", "window_id": window_id,
                  "action": "Virtual Imaging", "name": name, "active": False})

    def _update_vi(self, window_id: int, name: str, params: dict) -> None:
        """A per-VI caret edit — apply new detector params and recompute that
        virtual image live."""
        art = self._action_artifacts.get((window_id, name))
        if not art:
            return
        act = art.get("action")
        if act is not None and hasattr(act, "update_live_params"):
            act.update_live_params(params)
            # A detector-type change rebuilds the selector — refresh the ref so
            # removal closes the current ROI.
            new_sel = getattr(act, "_selector", None)
            if new_sel is not None:
                art["selector"] = new_sel
        # Keep the source plot's VI list + the renderer chip in sync.
        src = self._plot_by_window_id(window_id)
        item = None
        for it in getattr(src, "_vi_items", []) or []:
            if it.get("name") == name:
                it.update({k: v for k, v in params.items()})
                item = it
                break
        if item is not None:
            ipc.emit({
                "type": "sub_item", "window_id": window_id,
                "action": item.get("parent_action", "Virtual Imaging"),
                "name": name, "color": item.get("color"),
                "vtype": item.get("type"), "calculation": item.get("calculation"),
                "active": True,
            })
