"""
strain_action.py — the interactive strain-mapping action.

Runs strain on a Find-Vectors result (``tree.diffraction_vectors``), opens the
component-map + ellipse-glyph window, and attaches a **draggable reference
crosshair**: moving it picks the unstrained region and recomputes the whole
strain field live. A component toggle (εxx / εyy / εxy / ω) swaps the displayed
map in place. No Qt.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions._common import STRAIN_COMPONENTS as _COMPONENTS
from spyde.actions.lifecycle import bump_generation, is_current
from spyde.actions.wizard import WizardController

log = logging.getLogger(__name__)


def _default_reference(vecs) -> tuple:
    """A sensible unstrained reference: the pixel with the most vectors (the
    best-determined local lattice)."""
    try:
        cm = np.asarray(vecs.count_map())
        iy, ix = np.unravel_index(int(np.argmax(cm)), cm.shape)
        return int(iy), int(ix)
    except Exception:
        return 0, 0


def _zero_beam_filtered(g_ref) -> np.ndarray:
    """Reference spots with the central/direct (zero) beam removed.

    The zero beam (|g|≈0) carries no lattice information and would pin the fit's
    translation/centroid, so it's excluded from every strain reference. Threshold
    = 25% of the median nonzero |g| (well below the first ring, above numerical
    noise at the centre)."""
    g = np.asarray(g_ref, dtype=float).reshape(-1, 2)
    if len(g) == 0:
        return g
    mag = np.linalg.norm(g, axis=1)
    nz = mag[mag > 0]
    thresh = 0.25 * float(np.median(nz)) if nz.size else 0.0
    return g[mag > thresh]


class StrainController(WizardController):
    """Owns the live strain window: recomputes the field when the reference
    crosshair moves, swaps the shown component on toggle, switches the reference
    method (Region/CIF), and commits the current field as a new SignalTree.

    The reference pixel is picked with its OWN dedicated navigator crosshair —
    added the same way the "Add Selector" toolbar action does
    (``MultiplotManager.add_navigation_selector_and_signal_plot``): a brand-new
    selector on the navigator plus a linked diffraction-pattern window, additive
    to (never stealing) the user's main navigation crosshair. That reference DP
    is where the green/grey spot-selection overlay lives; only the SELECTED
    (green) spots feed ``compute_strain_field``."""

    key = "strain"

    def __init__(self, vecs, plot2d, *, window_id=None,
                 component="exx", ref_yx=(0, 0), session=None,
                 src_tree=None, src_dp_plot=None, match_radius_px=6.0):
        super().__init__(session, src_tree)
        self.vecs = vecs
        self.p = plot2d                  # the strain MAP figure (output)
        self.window_id = window_id
        self.component = component
        self.ref_yx = (int(ref_yx[0]), int(ref_yx[1]))
        self.field = None
        self.cif_mode = False
        self._g_ref_full = None          # reference spots (zero-beam removed)
        self._recompute_gen = 0          # latest-wins guard for async recomputes
        # Source-DP selection overlay (the interactive part) + the dedicated
        # reference selector/window this controller owns (see _attach_reference_selector).
        self.src_tree = src_tree
        self.src_dp_plot = src_dp_plot
        self.match_radius_px = float(match_radius_px)
        self.overlay = None
        self._ref_selector = None
        self._ref_plot = None

    def attach(self):
        self._set_full_reference(self.vecs.kxy_at(*self.ref_yx))
        # The strain window is a bare `figure` (not a registered Plot) — give it
        # a dispatch + teardown identity in the session's controller registry.
        self.own_window(self.window_id)
        self._attach_reference_selector()
        self._attach_selection_overlay()
        # Skip the initial recompute when the field was already computed by the
        # caller (_build_window pre-populates ctrl.field and the figure already
        # shows it) — recomputing here would run the full per-pixel fit a second
        # time on window open. Later ref/CIF/method interactions still call
        # _recompute() directly.
        if self.field is None:
            self._recompute()
        return self

    # Cyan — distinct from the main navigator's green crosshair and from the
    # selection overlay's green/grey/orange (selected/excluded/displacement).
    _REF_CROSSHAIR_COLOR = "#00e5ff"

    def _attach_reference_selector(self) -> None:
        """Add a NEW, independent navigator crosshair (+ linked DP window) for
        picking the reference pixel — same mechanism as the "Add Selector"
        toolbar action. This is additive: the user's main navigation crosshair
        is untouched, and this new one drives ONLY the strain reference."""
        tree = self.src_tree
        npm = getattr(tree, "navigator_plot_manager", None) if tree is not None else None
        if npm is None:
            log.debug("[strain-ref] no navigator_plot_manager on src_tree=%r — "
                      "dedicated reference crosshair NOT created", tree)
            return
        nav_window = next(iter(npm.plot_windows), None)
        if nav_window is None:
            log.debug("[strain-ref] navigator_plot_manager has no plot_windows — "
                      "dedicated reference crosshair NOT created")
            return
        try:
            ref_window = npm.add_navigation_selector_and_signal_plot(
                nav_window, color=self._REF_CROSSHAIR_COLOR)
        except Exception as e:
            log.exception("strain reference selector attach failed: %s", e)
            return
        self._ref_plot = getattr(ref_window, "current_plot_item", None)
        self._ref_selector = npm.navigation_selectors[nav_window][-1]
        log.debug("[strain-ref] created ref_window=%r ref_plot=%r window_id=%s "
                  "ref_selector=%r color=%s", ref_window, self._ref_plot,
                  getattr(self._ref_plot, "window_id", None), self._ref_selector,
                  self._REF_CROSSHAIR_COLOR)
        # Park the new crosshair on the default reference pixel and adopt its
        # linked DP as the overlay target (the reference spots are drawn there).
        try:
            self._ref_selector._widget.cx = float(self.ref_yx[1])
            self._ref_selector._widget.cy = float(self.ref_yx[0])
        except Exception as e:
            log.debug("positioning strain reference crosshair failed: %s", e)
        ref_dp = next(iter(self._ref_selector.active_children), None)
        log.debug("[strain-ref] ref_dp (overlay target)=%r ref_yx=%s", ref_dp, self.ref_yx)
        if ref_dp is not None:
            self.src_dp_plot = ref_dp
        if self._on_ref_selector not in self._ref_selector.index_hooks:
            self._ref_selector.index_hooks.append(self._on_ref_selector)
        self._ref_selector.update_data()
        # This window exists solely to host the reference crosshair + the
        # green/grey selection overlay — it has no actions of its own (Find
        # Vectors, Strain Mapping, etc. would be meaningless here and just
        # clutter a window the user isn't meant to drive actions from).
        self._suppress_toolbar(self._ref_plot)
        # Give it a title distinct from the tree's other windows (it otherwise
        # inherits the tree's root title, e.g. "— Vectors" — IDENTICAL to the
        # Find-Vectors result window it sits on top of, freshly opened and
        # focused, so the found-vectors red circles look like they vanished
        # when they're just hidden underneath this same-titled window).
        self._rename_ref_window("Strain Reference")

    def _rename_ref_window(self, title: str) -> None:
        plot = self._ref_plot
        fig_id = getattr(plot, "fig_id", None)
        html = getattr(plot, "_figure_html", None)
        if plot is None or fig_id is None or html is None:
            return
        try:
            from spyde.backend.ipc import emit
            emit({"type": "figure", "fig_id": fig_id, "window_id": plot.window_id,
                  "html": html, "title": title, "is_navigator": False})
        except Exception as e:
            log.debug("renaming strain reference window failed: %s", e)

    @staticmethod
    def _suppress_toolbar(plot) -> None:
        """Clear the reference window's toolbar (it hosts no actions)."""
        if plot is None:
            return
        try:
            from spyde.backend.ipc import emit
            emit({"type": "toolbar_config", "window_id": plot.window_id,
                  "plot_id": id(plot), "toolbar_actions": []})
            log.debug("[strain-ref] toolbar suppressed for window_id=%s", plot.window_id)
        except Exception as e:
            log.debug("suppressing strain reference toolbar failed: %s", e)

    def _attach_selection_overlay(self) -> None:
        """Attach the green/grey reference-spot selection overlay to the
        DEDICATED reference-pixel diffraction-pattern plot (added in
        _attach_reference_selector) — double-clicking a spot there toggles it
        in/out of the fit. Displacement arrows (main navigator vs. this
        reference) are drawn separately on the source tree's own DP."""
        dp = self.src_dp_plot
        if dp is None or self.src_tree is None:
            log.debug("[strain-ref] _attach_selection_overlay SKIPPED: dp=%r src_tree=%r",
                      dp, self.src_tree)
            return
        from spyde.actions.vector_overlay import attach_strain_selection_overlay
        try:
            self.overlay = attach_strain_selection_overlay(
                dp, self.vecs, self.src_tree, ref_yx=self.ref_yx,
                ref_spots=self._g_ref_full, match_radius_px=self.match_radius_px,
                on_toggle=self._on_selection_toggle)
            log.debug("[strain-ref] selection overlay attached on dp=%r overlay=%r",
                      dp, self.overlay)
        except Exception as e:
            log.exception("strain selection overlay attach failed: %s", e)

    def _on_ref_selector(self, indices) -> None:
        """The DEDICATED reference crosshair moved → adopt its position as the
        new reference pixel (Region mode) and re-fit."""
        from spyde.actions.vector_overlay import _indices_to_iyix
        iy, ix = _indices_to_iyix(indices)
        log.debug("[strain-ref] reference crosshair moved -> (%d,%d) (was %s)",
                  iy, ix, self.ref_yx)
        if (iy, ix) == self.ref_yx:
            return
        self.set_reference(iy, ix)

    def _on_selection_toggle(self) -> None:
        """A reference spot was double-clicked green↔grey → re-fit from the new set."""
        log.debug("[strain-ref] selection toggled -> recomputing strain field "
                  "(n_selected=%d)", len(self._selected_reference()))
        self._recompute()

    # ── reference (Region crosshair pixel, or CIF-snapped absolute) ───────────
    def _set_full_reference(self, g_ref) -> None:
        # Use every reference spot EXCEPT the zero beam (see _zero_beam_filtered).
        self._g_ref_full = _zero_beam_filtered(g_ref)

    def _selected_reference(self):
        # The overlay's green selection is the source of truth once it exists;
        # before/without it, use the full (zero-beam-filtered) reference.
        if self.overlay is not None:
            sel = self.overlay.selected_reference()
            if len(sel) >= 2:
                return sel
        return self._g_ref_full

    def _recompute(self) -> None:
        """Recompute the full strain field OFF the asyncio main thread.

        compute_strain_field is a per-pixel scipy fit (seconds on a real scan);
        running it inline on the main loop froze the UI on every reference / ring
        / CIF interaction. run_on_worker offloads it and marshals the figure
        update back via Session._dispatch_to_main (inline when no session — e.g.
        bare handler tests). A generation counter drops superseded results
        (latest-wins) so a slow compute can't clobber a newer one."""
        from spyde.actions.strain_mapping import compute_strain_field
        from spyde.actions.strain_display import update_strain_view
        ref = self._selected_reference()
        if ref is None or len(ref) < 2:
            return

        gen = bump_generation(self, "_recompute_gen")
        log.debug("[strain-ref] _recompute gen=%d n_ref_vectors=%d component=%s",
                  gen, len(ref), self.component)

        def _apply(field):
            # Drop a stale result superseded by a newer interaction.
            if not is_current(self, "_recompute_gen", gen):
                log.debug("[strain-ref] _recompute gen=%d SUPERSEDED (current=%d) — dropped",
                          gen, self._recompute_gen)
                return
            self.field = field
            update_strain_view(self.p, self.field, self.component)
            log.debug("[strain-ref] _recompute gen=%d applied to plot", gen)

        self.run_on_worker(
            lambda: compute_strain_field(self.vecs, ref_vectors=ref),
            name="strain-recompute", on_done=_apply,
        )

    def set_reference(self, ry: int, rx: int) -> None:
        """Region mode: a new reference pixel (relative strain). Updates the
        selection overlay's reference spots (all re-selected) and re-fits."""
        log.debug("[strain-ref] set_reference (%d,%d) (was %s)", ry, rx, self.ref_yx)
        self.ref_yx = (int(ry), int(rx))
        self.cif_mode = False
        self._set_full_reference(self.vecs.kxy_at(*self.ref_yx))
        if self.overlay is not None:
            self.overlay.set_reference(self.ref_yx, self._g_ref_full)
        self._recompute()

    def set_cif_reference(self, phase) -> None:
        """CIF mode: snap the reference pixel's vectors to the phase's ideal |g|
        families → absolute strain (no unstrained region needed)."""
        from spyde.actions.strain_mapping import cif_g_families, snap_reference_to_cif
        snapped = snap_reference_to_cif(self.vecs.kxy_at(*self.ref_yx), cif_g_families(phase))
        snapped = _zero_beam_filtered(snapped)
        if len(snapped) < 2:
            return
        self.cif_mode = True
        self._g_ref_full = snapped
        if self.overlay is not None:
            self.overlay.set_reference(self.ref_yx, self._g_ref_full)
        self._recompute()

    def set_match_radius(self, r_px: float) -> None:
        self.match_radius_px = max(1.0, float(r_px))
        if self.overlay is not None:
            self.overlay.set_match_radius(self.match_radius_px)

    def set_component(self, component: str) -> None:
        from spyde.actions.strain_display import update_strain_view
        if component not in _COMPONENTS or self.field is None:
            return
        self.component = component
        update_strain_view(self.p, self.field, component)

    def commit(self):
        """Freeze the current strain field as a NEW SignalTree — εxx is the signal
        plot, εyy / εxy / ω ride along as chip-selectable view figures (same shape
        as the Vector-OM result window). The live window stays open for tuning."""
        if self.field is None or self.session is None:
            return None
        from spyde.actions._common import STRAIN_TITLES as titles
        from spyde.actions.commit import commit_result_tree
        f = self.field
        return commit_result_tree(
            self.session, title="Strain",
            primary=f.exx, primary_label=titles["exx"],
            views=[(titles["eyy"], f.eyy), (titles["exy"], f.exy),
                   (titles["omega"], f.omega)],
            levels="auto_sym",
            provenance={
                "action": "Strain Mapping",
                "params": {"ref_yx": list(self.ref_yx), "cif_mode": self.cif_mode,
                           "match_radius_px": self.match_radius_px},
            },
        )

    def remove(self) -> None:
        """Tear down EVERYTHING the action added: the selection overlay, the
        dedicated reference crosshair + its DP window, and the strain-map
        window itself. Called when the action is toggled off (strain_close)
        and by close() when the window is torn down externally. Idempotent —
        re-entry (remove → _forget_window → close → remove) is a no-op.
        The user's main navigator/DP are left untouched — the reference
        selector was always a SEPARATE, additive crosshair."""
        from spyde.backend.ipc import emit
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self.overlay is not None:
            try:
                self.overlay.remove()
            except Exception as e:
                log.debug("removing strain selection overlay failed: %s", e)
            self.overlay = None
        # Close the dedicated reference crosshair + its linked DP window (added
        # in _attach_reference_selector) — this also unhooks its index_hooks and
        # the navigator selector itself via the session's plot-close teardown.
        ref_plot = getattr(self, "_ref_plot", None)
        if ref_plot is not None and self.session is not None:
            try:
                self.session._close_signal_plot(ref_plot, self.src_tree)
            except Exception as e:
                log.debug("closing strain reference window failed: %s", e)
        self._ref_plot = None
        self._ref_selector = None
        # Close the strain-map window: route through the session's teardown so
        # the controller registry + kept-alive figures are evicted with it
        # (re-entry into remove() is stopped by the _closed flag above).
        if self.window_id is not None:
            forget = (getattr(self.session, "_forget_window", None)
                      if self.session is not None else None)
            if forget is not None:
                try:
                    forget(int(self.window_id))
                except Exception as e:
                    log.debug("forgetting strain window failed: %s", e)
            else:
                # Stand-alone / stub session: emit + unregister directly.
                try:
                    emit({"type": "window_closed", "window_id": int(self.window_id)})
                except Exception as e:
                    log.debug("closing strain window failed: %s", e)
                reg = (getattr(self.session, "_window_controllers", None)
                       if self.session is not None else None)
                if isinstance(reg, dict):
                    reg.pop(int(self.window_id), None)
        # Drop the back-reference so a later open rebuilds cleanly.
        if self.src_tree is not None and getattr(self.src_tree, "_strain_controller", None) is self:
            self.src_tree._strain_controller = None


# ── toolbar entry (ActionContext convention: fn(ctx, ...)) ────────────────────

def strain_mapping(ctx, action_name: str = "Strain Mapping", **params) -> None:
    """Toolbar entry on a Find-Vectors result — open the interactive strain
    window (one-shot; the live reference crosshair + component toggle take over)."""
    strain_run(ctx.session, ctx.plot, params or {})


# ── staged handlers (session.py dispatch: fn(session, plot, payload)) ──────────

def _ctrl_for(session, plot, payload):
    """Resolve the live StrainController for an action message.

    The strain window is a bare `figure` (not a registered Plot), so
    _plot_by_window_id → None and the plot-based lookup fails. Resolve by
    window_id from the session's window-controller registry first; fall back
    to the source tree's back-reference."""
    wid = payload.get("window_id")
    if wid is None and plot is not None:
        wid = getattr(plot, "window_id", None)
    if wid is not None:
        lookup = getattr(session, "controller_by_window_id", None)
        ctrl = lookup(int(wid)) if lookup is not None else None
        if isinstance(ctrl, StrainController):
            return ctrl
    tree = getattr(plot, "signal_tree", None)
    return getattr(tree, "_strain_controller", None) if tree is not None else None


def strain_run(session, plot, payload) -> None:
    """Open the interactive strain window for the active Find-Vectors result.

    The initial full-field fit (compute_strain_field) is a per-pixel scipy loop
    — run it on a worker thread and build/emit the window back on the main
    thread, so opening the action doesn't freeze the UI."""
    from spyde.backend.ipc import emit, emit_error, emit_status
    from spyde.actions.lifecycle import resolve_vectors, wait_for_vectors
    from spyde.actions.strain_mapping import compute_strain_field
    from spyde.actions.strain_display import build_strain_figure

    tree, vecs = resolve_vectors(session, plot)
    if vecs is None:
        # Find Vectors finalizes (attaches diffraction_vectors) on a worker
        # thread; the user can open Strain in the brief window before it lands,
        # or while a slow batch is still running — self-wait, then re-dispatch.
        # Without an event loop (bare handler tests) error immediately.
        if wait_for_vectors(session, plot,
                            lambda: strain_run(session, plot, payload),
                            what="Strain mapping"):
            return
        emit_error("Strain mapping needs a Find Vectors result (no diffraction vectors).")
        return

    # Idempotent: opening the caret again must NOT spawn a second strain window.
    # Re-show the existing controller's overlay and bail.
    existing = getattr(tree, "_strain_controller", None)
    if existing is not None and not getattr(existing, "_closed", False):
        if existing.overlay is not None:
            existing.overlay.set_visible(True)
        return

    # Re-entrancy guard: React StrictMode mounts the wizard TWICE synchronously
    # (mount → cleanup → remount) on every open, firing strain_run then
    # strain_stop then strain_run again before either's worker thread has had a
    # chance to set tree._strain_controller — so the "existing" check above
    # can't see the first call in flight and BOTH proceed, building two
    # StrainControllers (two reference crosshairs, two overlays, two windows).
    # The run/stop generation guard (lifecycle.bump_generation) — bumped
    # synchronously here, BEFORE spawning the compute thread — closes that
    # race: strain_stop bumps it too (cancelling any in-flight run), and a
    # stale generation's _build_window is dropped on arrival instead of
    # building a second live controller.
    from spyde.actions.lifecycle import bump_generation, is_current, run_on_worker
    gen = bump_generation(tree, "_strain_run_gen")

    ref_yx = _default_reference(vecs)

    def _build_window(field):
        if not is_current(tree, "_strain_run_gen", gen):
            return   # superseded by a strain_stop or a newer strain_run
        _fig, fig_id, html, p = build_strain_figure(field, component="exx")
        wid = session.next_window_id()
        from spyde.actions.figure_registry import keep_alive
        keep_alive(int(wid), _fig)
        emit({"type": "figure", "fig_id": fig_id, "window_id": int(wid),
              "html": html, "title": "Strain (εxx)", "is_navigator": False,
              "strain_components": list(_COMPONENTS)})
        src_dp = next(iter(getattr(tree, "signal_plots", [])), None)
        ctrl = StrainController(vecs, p, window_id=wid, component="exx",
                                ref_yx=ref_yx, session=session,
                                src_tree=tree, src_dp_plot=src_dp,
                                match_radius_px=float(payload.get("match_radius_px", 6.0)))
        ctrl.field = field
        ctrl.attach()
        tree._strain_controller = ctrl
        emit_status("Strain field ready.")

    if getattr(session, "_dispatch_to_main", None) is not None:
        emit_status("Computing strain field…")
    run_on_worker(
        session, lambda: compute_strain_field(vecs, ref_yx),
        name="strain-run", on_done=_build_window,
        on_error=lambda e: emit_error(f"Strain mapping failed: {e}"),
    )


def strain_set_component(session, plot, payload) -> None:
    """Component toggle (εxx / εyy / εxy / ω) → swap the shown map in place."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is not None:
        ctrl.set_component(str(payload.get("component", "exx")))


def strain_set_method(session, plot, payload) -> None:
    """Reference-method caret: 'region' (relative, crosshair pixel) or 'cif'
    (absolute, from a crystal's ideal spacings; needs payload['cif_path'])."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    method = str(payload.get("method", "region"))
    if method == "cif":
        cif_path = payload.get("cif_path")
        if not cif_path:
            return                                  # CIF chosen but no file yet
        try:
            from orix.crystal_map import Phase
            ctrl.set_cif_reference(Phase.from_cif(cif_path))
        except Exception as e:
            from spyde.backend.ipc import emit_error
            emit_error(f"Strain CIF reference failed: {e}")
    else:
        ry, rx = ctrl.ref_yx
        ctrl.set_reference(ry, rx)                  # region (relative) mode


def strain_set_match_radius(session, plot, payload) -> None:
    """Match-radius slider: how far (px) a frame peak can be from a reference spot
    to count as matched (drives the displacement arrows)."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is not None:
        ctrl.set_match_radius(float(payload.get("match_radius_px", 6.0)))


def strain_commit(session, plot, payload) -> None:
    """Submit: freeze the current strain field as a new SignalTree."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        from spyde.backend.ipc import emit_error
        emit_error("No live strain field to submit.")
        return
    ctrl.commit()


def strain_set_overlay(session, plot, payload) -> None:
    """Show/hide the reference-spot selection + displacement overlay on the source
    DP (fired by the renderer when the Strain caret opens/closes)."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None or ctrl.overlay is None:
        return
    try:
        ctrl.overlay.set_visible(bool(payload.get("visible", True)))
    except Exception as e:
        log.debug("strain set_overlay failed: %s", e)


def strain_stop(session, plot, payload) -> None:
    """Toggle the action OFF: remove EVERYTHING strain_run added — the strain-map
    window, the selection overlay, the nav hooks. The source DP/navigator stay."""
    # Bump the tree's run-generation FIRST, unconditionally — this invalidates
    # any strain_run still in flight (its compute thread hasn't finished, so
    # tree._strain_controller doesn't exist yet and _ctrl_for below finds
    # nothing to remove). Without this, React StrictMode's synchronous
    # mount→cleanup→remount race (strain_run, strain_stop, strain_run, all
    # before either worker thread lands) let BOTH strain_run calls build a
    # live controller — two reference crosshairs, two overlays, two windows.
    tree = getattr(plot, "signal_tree", None)
    if tree is None:
        for cand in getattr(session, "signal_trees", []):
            if getattr(cand, "_strain_controller", None) is not None:
                tree = cand
                break
    if tree is not None:
        bump_generation(tree, "_strain_run_gen")
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is not None:
        ctrl.remove()
