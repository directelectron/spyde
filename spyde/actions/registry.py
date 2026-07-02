"""
registry.py — the staged-action registry and the window-controller protocol.

SpyDE has exactly TWO dispatch paths for renderer→backend actions (see
``spyde/actions/README.md``; do not invent a third):

1. **YAML toolbar actions** — declared in ``spyde/toolbars.yaml``, resolved and
   invoked by ``Session._dispatch_toolbar_action`` with an ``ActionContext``.
2. **Staged actions** — the wizard/caret protocol: each entry below maps an
   action name to a ``"module.function"`` dotted path with the uniform
   ``fn(session, plot, payload)`` signature. Modules are imported lazily so
   heavy dependencies load on first use, not at startup.

Staged-action NAMING CONVENTION (``<key>`` is the wizard's short prefix):

    <key>_open           wizard mounted → start live preview / controller
    <key>_close          wizard unmounted → tear everything down
    <key>_tune           debounced live re-tune of preview params
    <key>_set_<param>    discrete parameter change
    <key>_run            heavy compute stage (may open a result tree)
    <key>_commit         snapshot the live result into a NEW SignalTree

Wizard-specific extra stages are allowed but must keep the ``<key>_`` prefix
(e.g. ``om_generate_library``).

WindowController protocol
-------------------------
Windows that are NOT registered ``Plot``s (bare ``figure`` emits: the strain
map, IPF views, tiled comparisons) must register a *controller* with
``session.register_window_controller(window_id, controller)`` so dispatch and
teardown can reach them. A controller is duck-typed:

    window_id: int                     # the window it drives
    close() -> None                    # full teardown; called by
                                       # Session._forget_window when the window
                                       # goes away for ANY reason
    handle_action(name, payload) -> bool   # optional: consume an action aimed
                                           # at this window; return True if
                                           # handled

``spyde.actions.wizard.WizardController`` provides a base implementation.
"""
from __future__ import annotations

import importlib
from typing import Callable

STAGED_HANDLERS: dict[str, str] = {
    "om_generate_library": "spyde.actions.orientation_action.om_generate_library",
    "om_refine":           "spyde.actions.orientation_action.om_refine",
    "om_run":              "spyde.actions.orientation_action.om_run",
    "fv_open":             "spyde.actions.find_vectors_action.fv_open",
    "fv_tune":             "spyde.actions.find_vectors_action.fv_tune",
    "fv_run":              "spyde.actions.find_vectors_action.fv_run",
    "fv_close":            "spyde.actions.find_vectors_action.fv_close",
    "vom_generate_library": "spyde.actions.vector_orientation_om.vom_generate_library",
    "vom_refine":          "spyde.actions.vector_orientation_om.vom_refine",
    "vom_run":             "spyde.actions.vector_orientation_om.vom_run",
    "strain_open":         "spyde.actions.strain_action.strain_open",
    "strain_set_component": "spyde.actions.strain_action.strain_set_component",
    "strain_set_method":   "spyde.actions.strain_action.strain_set_method",
    "strain_set_match_radius": "spyde.actions.strain_action.strain_set_match_radius",
    "strain_set_overlay":  "spyde.actions.strain_action.strain_set_overlay",
    "strain_close":        "spyde.actions.strain_action.strain_close",
    "strain_commit":       "spyde.actions.strain_action.strain_commit",
    "ipf_set_direction":   "spyde.actions.ipf_view.ipf_set_direction",
    "tile_views":          "spyde.actions.views.tile_views",
    "set_composition":     "spyde.actions.composition.set_composition",
    "cod_search":          "spyde.actions.composition.cod_search",
    "cod_pick":            "spyde.actions.composition.cod_pick",
    "czb_run":             "spyde.actions.center_zero_beam.czb_run",
    "czb_open":            "spyde.actions.center_zero_beam.czb_open",
    "czb_pick":            "spyde.actions.center_zero_beam.czb_pick",
    "czb_close":           "spyde.actions.center_zero_beam.czb_close",
    "set_log_level":       "spyde.backend.log_stream.set_log_level",
    "get_gpu_status":      "spyde.actions.gpu_status.get_gpu_status",
    "set_update_channel":  "spyde.backend.session.dispatch_set_update_channel",
}


def resolve_staged(name: str) -> Callable | None:
    """Lazily import and return the handler for a staged action name."""
    dotted = STAGED_HANDLERS.get(name)
    if dotted is None:
        return None
    mod, fn = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), fn)


def register_staged(name: str, dotted_path: str) -> None:
    """Register a staged action (``fn(session, plot, payload)``) by dotted path."""
    STAGED_HANDLERS[name] = dotted_path
