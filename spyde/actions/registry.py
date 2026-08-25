"""
registry.py — SpyDE's staged-action table and wizard schema map.

The registry MECHANISM (lazy dotted-path resolution, the wizard-schema lookup,
the WindowController protocol) is app-agnostic and lives in the shell —
``de_shell.actions.registry``, which also carries the full protocol
documentation. What is here is SpyDE's CONTENT: which action names exist and
where each wizard declares its parameters.

Both tables are registered with the shell at import, and the shell's resolver
functions are re-exported so callers keep importing one module.
"""
from __future__ import annotations

from de_shell.actions.registry import (  # noqa: F401  (re-exported API)
    YAML_SCHEMA, register_staged, register_staged_table, resolve_staged,
    register_wizard_schema, register_wizard_schemas, set_yaml_schema_resolver,
    wizard_parameters, wizard_keys,
)

# SpyDE's own table. Registered into the shell below, after which the name
# STAGED_HANDLERS is REBOUND to the shell's dict — one authority, so a later
# register_staged() is visible to everything that reads the table.
_SPYDE_STAGED_HANDLERS: dict[str, str] = {
    "fit_open":             "spyde.actions.fit_action.fit_open",
    "fit_close":            "spyde.actions.fit_action.fit_close",
    "fit_add_component":    "spyde.actions.fit_action.fit_add_component",
    "fit_remove_component": "spyde.actions.fit_action.fit_remove_component",
    "fit_set_param":        "spyde.actions.fit_action.fit_set_param",
    "fit_tune":             "spyde.actions.fit_action.fit_tune",
    "fit_run":              "spyde.actions.fit_action.fit_run",
    "fit_commit":           "spyde.actions.fit_action.fit_commit",
    "fit_from_composition": "spyde.actions.fit_action.fit_from_composition",
    "fit_current":          "spyde.actions.fit_action.fit_current",
    "fit_navigated":        "spyde.actions.fit_action.fit_navigated",
    "fit_refit_poor":       "spyde.actions.fit_action.fit_refit_poor",
    "fit_save_model":       "spyde.actions.fit_action.fit_save_model",
    "fit_load_model":       "spyde.actions.fit_action.fit_load_model",
    "bg_open":              "spyde.actions.background_action.bg_open",
    "bg_close":             "spyde.actions.background_action.bg_close",
    "bg_set_model":         "spyde.actions.background_action.bg_set_model",
    "bg_set_region":        "spyde.actions.background_action.bg_set_region",
    "bg_apply":             "spyde.actions.background_action.bg_apply",
    "om_generate_library": "spyde.actions.orientation_action.om_generate_library",
    "om_refine":           "spyde.actions.orientation_action.om_refine",
    "om_run":              "spyde.actions.orientation_action.om_run",
    "ebsd_build_dictionary": "spyde.actions.ebsd_action.ebsd_build_dictionary",
    "ebsd_refine":         "spyde.actions.ebsd_action.ebsd_refine",
    "ebsd_run":            "spyde.actions.ebsd_action.ebsd_run",
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
    # DPC / field mapping (spyde/actions/dpc_action.py).
    "dpc_open":            "spyde.actions.dpc_action.dpc_open",
    "dpc_close":           "spyde.actions.dpc_action.dpc_close",
    "dpc_set_center":      "spyde.actions.dpc_action.dpc_set_center",
    "dpc_pick_center":     "spyde.actions.dpc_action.dpc_pick_center",
    "dpc_set_beam":        "spyde.actions.dpc_action.dpc_set_beam",
    "dpc_load_vacuum":     "spyde.actions.dpc_action.dpc_load_vacuum",
    "dpc_auto_rotation":   "spyde.actions.dpc_action.dpc_auto_rotation",
    "dpc_tune":            "spyde.actions.dpc_action.dpc_tune",
    "dpc_set_view":        "spyde.actions.dpc_action.dpc_set_view",
    "dpc_run":             "spyde.actions.dpc_action.dpc_run",
    "dpc_commit":          "spyde.actions.dpc_action.dpc_commit",
    "crop_open":           "spyde.actions.base.crop_open",
    "crop_close":          "spyde.actions.base.crop_close",
    "crop_set_region":     "spyde.actions.base.crop_set_region",
    # Drift Correction (spyde/actions/drift_action.py).
    "drift_open":          "spyde.actions.drift_action.drift_open",
    "drift_close":         "spyde.actions.drift_action.drift_close",
    "drift_set_method":    "spyde.actions.drift_action.drift_set_method",
    "drift_tune":          "spyde.actions.drift_action.drift_tune",
    "drift_run":           "spyde.actions.drift_action.drift_run",
    "drift_discard":       "spyde.actions.drift_action.drift_discard",
    "drift_commit":        "spyde.actions.drift_action.drift_commit",
    "download_cancel":     "spyde.backend.example_download.download_cancel",
    "compute_configure":   "spyde.backend.compute_config.compute_configure",
    "set_log_level":       "de_shell.log_stream.set_log_level",
    "set_debug_flag":      "de_shell.debug_flags.set_debug_flag",
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
    "report_add_figure_placeholder": "spyde.actions.report.handlers.report_add_figure_placeholder",
    "report_set_split_layout": "spyde.actions.report.handlers.report_set_split_layout",
    "report_set_split_figure": "spyde.actions.report.handlers.report_set_split_figure",
    "report_split_remove_figure": "spyde.actions.report.handlers.report_split_remove_figure",
    "report_update_cell":      "spyde.actions.report.handlers.report_update_cell",
    "report_remove_cell":      "spyde.actions.report.handlers.report_remove_cell",
    # Test-only: release a SPYDE_TEST_HOLD pause point (backend/test_hold.py).
    # Inert in production — with the env var unset there is no hold to release.
    "test_hold_release":       "spyde.actions.find_vectors_action.test_hold_release",
    "report_undo":             "spyde.actions.report.handlers.report_undo",
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
    # Deck THEME — colours / type / footer bar / logo. Per-document, with a
    # "set as default" that seeds every new deck (settings.json).
    "report_set_theme":        "spyde.actions.report.handlers.report_set_theme",
    "report_theme_set_default": "spyde.actions.report.handlers.report_theme_set_default",
    "report_theme_use_default": "spyde.actions.report.handlers.report_theme_use_default",
    "report_theme_reset":      "spyde.actions.report.handlers.report_theme_reset",
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

_SPYDE_WIZARD_SCHEMAS: dict[str, tuple[str, str]] = {
    # key: (module, attribute) — attribute is a controller class (its
    # `parameters`) or a dict.
    "fit":    ("spyde.actions.fit_action", "FitWizard"),
    "bg":     ("spyde.actions.background_action", "PARAMETERS"),
    "strain": ("spyde.actions.strain_action", "StrainController"),
    "vom":    ("spyde.actions.vector_orientation_om", "VomWizard"),
    "ebsd":   ("spyde.actions.ebsd_action", "EbsdWizard"),
    "czb":    ("spyde.actions.center_zero_beam", "PARAMETERS"),
    "drift":  ("spyde.actions.drift_action", "DriftWizard"),
    "dpc":    ("spyde.actions.dpc_action", "DpcWizard"),
    # YAML-declared (resolved from spyde.TOOLBAR_ACTIONS):
    "fv":     (YAML_SCHEMA, "Find Diffraction Vectors"),
    "om":     (YAML_SCHEMA, "Orientation Mapping"),
}


def _yaml_parameters(action_title: str) -> dict:
    """Resolve a wizard schema declared in toolbars.yaml (FV and OM keep theirs
    there, because their carets already render from it)."""
    import spyde
    for group in spyde.TOOLBAR_ACTIONS.values():
        if isinstance(group, dict) and action_title in group:
            return dict(group[action_title].get("parameters") or {})
    return {}


# Hand SpyDE's content to the shell. At import, so anything that resolves an
# action or a wizard schema sees a populated registry regardless of import order.
register_staged_table(_SPYDE_STAGED_HANDLERS)
register_wizard_schemas(_SPYDE_WIZARD_SCHEMAS)
set_yaml_schema_resolver(_yaml_parameters)

# Rebind to the shell's live dicts so `spyde.actions.registry.STAGED_HANDLERS`
# and the shell's are the SAME object — otherwise a register_staged() call
# would land in one and be read from the other.
from de_shell.actions.registry import (  # noqa: E402
    STAGED_HANDLERS, _WIZARD_SCHEMAS,  # noqa: F401  (re-exported API)
)
