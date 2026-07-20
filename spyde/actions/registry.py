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
    "fv_models":           "spyde.actions.find_vectors_action.fv_models",
    "fv_refresh_models":   "spyde.actions.find_vectors_action.fv_refresh_models",
    "vom_generate_library": "spyde.actions.vector_orientation_om.vom_generate_library",
    "vom_refine":          "spyde.actions.vector_orientation_om.vom_refine",
    "vom_run":             "spyde.actions.vector_orientation_om.vom_run",
    "strain_open":         "spyde.actions.strain_action.strain_open",
    "strain_set_component": "spyde.actions.strain_action.strain_set_component",
    "strain_set_method":   "spyde.actions.strain_action.strain_set_method",
    "strain_set_match_radius": "spyde.actions.strain_action.strain_set_match_radius",
    "strain_set_fit":      "spyde.actions.strain_action.strain_set_fit",
    "strain_set_overlay":  "spyde.actions.strain_action.strain_set_overlay",
    "strain_close":        "spyde.actions.strain_action.strain_close",
    "strain_commit":       "spyde.actions.strain_action.strain_commit",
    "vi_commit":           "spyde.actions.virtual_image.vi_commit",
    "ipf_set_direction":   "spyde.actions.ipf_view.ipf_set_direction",
    "tile_views":          "spyde.actions.views.tile_views",
    "select_navigator":    "spyde.actions.navigator_views.select_navigator",
    "add_navigator_from_window": "spyde.actions.navigator_views.add_navigator_from_window",
    "extract_navigator":   "spyde.actions.navigator_views.extract_navigator",
    "set_composition":     "spyde.actions.composition.set_composition",
    "cod_search":          "spyde.actions.composition.cod_search",
    "cod_pick":            "spyde.actions.composition.cod_pick",
    "czb_run":             "spyde.actions.center_zero_beam.czb_run",
    "czb_open":            "spyde.actions.center_zero_beam.czb_open",
    "czb_pick":            "spyde.actions.center_zero_beam.czb_pick",
    "czb_set_region":      "spyde.actions.center_zero_beam.czb_set_region",
    "czb_close":           "spyde.actions.center_zero_beam.czb_close",
    "download_cancel":     "spyde.backend.example_download.download_cancel",
    "compute_configure":   "spyde.backend.compute_config.compute_configure",
    "set_log_level":       "spyde.backend.log_stream.set_log_level",
    "set_debug_flag":      "spyde.backend.debug_flags.set_debug_flag",
    "get_gpu_status":      "spyde.actions.gpu_status.get_gpu_status",
    "set_update_channel":  "spyde.backend.session.dispatch_set_update_channel",
    "get_first_run":       "spyde.backend.session.get_first_run",
    "mark_tutorial_seen":  "spyde.backend.session.dispatch_mark_tutorial_seen",
    # Report Builder (spyde/actions/report/) — the report sidebar's staged actions.
    "report_new":              "spyde.actions.report.handlers.report_new",
    "report_open":             "spyde.actions.report.handlers.report_open",
    "report_save":             "spyde.actions.report.handlers.report_save",
    "report_save_as_template": "spyde.actions.report.handlers.report_save_as_template",
    "report_close":            "spyde.actions.report.handlers.report_close",
    "report_add_cell":         "spyde.actions.report.handlers.report_add_cell",
    "report_add_image_cell":   "spyde.actions.report.handlers.report_add_image_cell",
    # Report/Presentation redesign Wave A — the split-block primitive (text side
    # BESIDE a figure/photo side, one atomic cell).
    "report_add_split_cell":   "spyde.actions.report.handlers.report_add_split_cell",
    "report_set_split_layout": "spyde.actions.report.handlers.report_set_split_layout",
    "report_set_split_figure": "spyde.actions.report.handlers.report_set_split_figure",
    "report_update_cell":      "spyde.actions.report.handlers.report_update_cell",
    "report_remove_cell":      "spyde.actions.report.handlers.report_remove_cell",
    "report_move_cell":        "spyde.actions.report.handlers.report_move_cell",
    "report_move_slide":       "spyde.actions.report.handlers.report_move_slide",
    "report_set_caption":      "spyde.actions.report.handlers.report_set_caption",
    "report_set_title":        "spyde.actions.report.handlers.report_set_title",
    # Report Builder Phase 6 — Present mode (slide grouping + go-live excursion)
    "report_toggle_slide_break": "spyde.actions.report.handlers.report_toggle_slide_break",
    "report_set_live_action":  "spyde.actions.report.handlers.report_set_live_action",
    "report_set_slide_kind":   "spyde.actions.report.handlers.report_set_slide_kind",
    "report_set_slide_style":  "spyde.actions.report.handlers.report_set_slide_style",
    "report_set_slide_notes":  "spyde.actions.report.handlers.report_set_slide_notes",
    "report_add_figure":       "spyde.actions.report.handlers.report_add_figure",
    "report_refresh_figure":   "spyde.actions.report.handlers.report_refresh_figure",
    "repfig_refresh_panel":    "spyde.actions.report.handlers.repfig_refresh_panel",
    "report_snapshots":        "spyde.actions.report.handlers.report_snapshots",
    "report_cell_from_window": "spyde.actions.report.handlers.report_cell_from_window",
    # Report Builder Phase 3 — export + copy/paste (spyde/actions/report/export_html.py)
    "report_export_html":      "spyde.actions.report.export_html.report_export_html",
    "report_export_markdown":  "spyde.actions.report.export_html.report_export_markdown",
    "report_paste_cell":       "spyde.actions.report.export_html.report_paste_cell",
    # Report Builder Phase 2 — combined report figures (spyde/actions/report/compose.py)
    "repfig_query_compose":    "spyde.actions.report.compose.repfig_query_compose",
    "repfig_compose":          "spyde.actions.report.compose.repfig_compose",
    "repfig_set_layer":        "spyde.actions.report.compose.repfig_set_layer",
    "repfig_set_text_size":    "spyde.actions.report.compose.repfig_set_text_size",
    "repfig_remove_layer":     "spyde.actions.report.compose.repfig_remove_layer",
    "repfig_remove_panel":     "spyde.actions.report.compose.repfig_remove_panel",
    "repfig_add_annotation":   "spyde.actions.report.compose.repfig_add_annotation",
    "repfig_update_annotation": "spyde.actions.report.compose.repfig_update_annotation",
    "repfig_remove_annotation": "spyde.actions.report.compose.repfig_remove_annotation",
    "repfig_set_edit_mode":    "spyde.actions.report.compose.repfig_set_edit_mode",
    # Selection-driven edit + figure-level layout / annotations.
    "repfig_select_panel":     "spyde.actions.report.compose.repfig_select_panel",
    "repfig_set_layout":       "spyde.actions.report.compose.repfig_set_layout",
    "repfig_apply_layout_preset": "spyde.actions.report.compose.repfig_apply_layout_preset",
    "repfig_add_fig_annotation": "spyde.actions.report.compose.repfig_add_fig_annotation",
    "repfig_update_fig_annotation": "spyde.actions.report.compose.repfig_update_fig_annotation",
    "repfig_remove_fig_annotation": "spyde.actions.report.compose.repfig_remove_fig_annotation",
    # Report Builder Phase 3 — fresh-slice zoom-inset callouts.
    "repfig_add_callout":      "spyde.actions.report.compose.repfig_add_callout",
    "repfig_add_time_callouts": "spyde.actions.report.compose.repfig_add_time_callouts",
    "repfig_add_zoom_callout": "spyde.actions.report.compose.repfig_add_zoom_callout",
    # Report Builder Phase 2 — MDI live image layering (spyde/actions/overlay.py)
    "overlay_add":             "spyde.actions.overlay.overlay_add",
    "overlay_set":             "spyde.actions.overlay.overlay_set",
    "overlay_remove":          "spyde.actions.overlay.overlay_remove",
    "overlay_query":           "spyde.actions.overlay.overlay_query",
    # Movie BLOCK (spyde/actions/report/movie.py) — an editable, persistent in-situ
    # movie cell in the report/presentation doc + its full-screen editor. Replaces
    # the mvx caret wizard (removed in Phase 2); reuses the movie_export render engine.
    "report_add_movie_cell":   "spyde.actions.report.movie.report_add_movie_cell",
    "report_set_movie_source": "spyde.actions.report.movie.report_set_movie_source",
    "movie_open":              "spyde.actions.report.movie.movie_open",
    "movie_close":             "spyde.actions.report.movie.movie_close",
    "movie_scrub":             "spyde.actions.report.movie.movie_scrub",
    "movie_play":              "spyde.actions.report.movie.movie_play",
    "movie_stop":              "spyde.actions.report.movie.movie_stop",
    "movie_tune":              "spyde.actions.report.movie.movie_tune",
    "movie_crop_mode":         "spyde.actions.report.movie.movie_crop_mode",
    "movie_add_text_overlay":  "spyde.actions.report.movie.movie_add_text_overlay",
    "movie_add_overlay_image": "spyde.actions.report.movie.movie_add_overlay_image",
    "movie_drop_window":       "spyde.actions.report.movie.movie_drop_window",
    "movie_export":            "spyde.actions.report.movie.movie_export",
    "movie_cancel":            "spyde.actions.report.movie.movie_cancel",
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


# ─────────────────────────────────────────────────────────────────────────────
# Wizard parameter schemas — the single host-agnostic lookup
# ─────────────────────────────────────────────────────────────────────────────

# Every wizard declares its parameter schema in Python (a `parameters`
# classattr on its WizardController, or a module-level PARAMETERS for
# controller-less wizards), in the SAME dict spec as toolbars.yaml
# `parameters:`. FV and OM keep their schema in toolbars.yaml (their carets
# already render from it); this table maps each wizard key to wherever its
# schema lives, so any host (an Electron panel, a notebook form generator, a
# doc generator) resolves them uniformly. Completeness is enforced by
# test_wizard_schemas.py. (Three-host parity: NOTEBOOK_PARITY_PLAN.md §6.)
_WIZARD_SCHEMAS: dict[str, tuple[str, str]] = {
    # key: (module, attribute) — attribute is a controller class (its
    # `parameters`) or a dict.
    "strain": ("spyde.actions.strain_action", "StrainController"),
    "vom":    ("spyde.actions.vector_orientation_om", "VomWizard"),
    "czb":    ("spyde.actions.center_zero_beam", "PARAMETERS"),
    # YAML-declared (resolved from spyde.TOOLBAR_ACTIONS):
    "fv":     ("__yaml__", "Find Diffraction Vectors"),
    "om":     ("__yaml__", "Orientation Mapping"),
}


def _yaml_parameters(action_title: str) -> dict:
    import spyde
    for group in spyde.TOOLBAR_ACTIONS.values():
        if isinstance(group, dict) and action_title in group:
            return dict(group[action_title].get("parameters") or {})
    return {}


def wizard_parameters(key: str) -> dict:
    """Return wizard ``key``'s declared parameter schema (a copy).

    The uniform entry point for rendering a wizard's controls in ANY host —
    same spec as toolbars.yaml ``parameters:`` (type/name/default/min/max/
    step/choices/tab/extensions). Raises ``KeyError`` for unknown keys.
    """
    module, attr = _WIZARD_SCHEMAS[key]
    if module == "__yaml__":
        return _yaml_parameters(attr)
    obj = getattr(importlib.import_module(module), attr)
    schema = obj if isinstance(obj, dict) else getattr(obj, "parameters", {})
    return dict(schema)


def wizard_keys() -> list[str]:
    """All wizard keys with a declared schema."""
    return list(_WIZARD_SCHEMAS)
