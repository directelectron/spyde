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
import threading

import numpy as np

from spyde.actions._common import STRAIN_COMPONENTS as _COMPONENTS

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


# window_id → live StrainController. The strain window is created from a bare
# `figure` message (not a registered Plot in session._plots), so the staged
# strain_* handlers can't resolve it via _plot_by_window_id — they look it up
# here instead. Registered in strain_run, cleared is harmless (small + leaks
# nothing heavy beyond the controller, which the source tree also references).
_CONTROLLERS: "dict[int, StrainController]" = {}


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


class StrainController:
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

    def __init__(self, vecs, plot2d, *, window_id=None,
                 component="exx", ref_yx=(0, 0), session=None,
                 src_tree=None, src_dp_plot=None, match_radius_px=6.0):
        self.vecs = vecs
        self.p = plot2d                  # the strain MAP figure (output)
        self.window_id = window_id
        self.session = session
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
        if self.window_id is not None:
            _CONTROLLERS[int(self.window_id)] = self
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
        / CIF interaction. Offload it to a worker thread like every other heavy
        action and marshal the figure update back via Session._dispatch_to_main.
        A generation counter drops superseded results (latest-wins) so a slow
        compute can't clobber a newer one. Falls back to inline when no session
        is available (tests)."""
        from spyde.actions.strain_mapping import compute_strain_field
        from spyde.actions.strain_display import update_strain_view
        ref = self._selected_reference()
        if ref is None or len(ref) < 2:
            return

        self._recompute_gen += 1
        gen = self._recompute_gen
        log.debug("[strain-ref] _recompute gen=%d n_ref_vectors=%d component=%s",
                  gen, len(ref), self.component)

        session = self.session
        if session is None or getattr(session, "_dispatch_to_main", None) is None:
            # No event loop to marshal back to (e.g. tests) — run inline.
            self.field = compute_strain_field(self.vecs, ref_vectors=ref)
            update_strain_view(self.p, self.field, self.component)
            return

        def _work():
            try:
                field = compute_strain_field(self.vecs, ref_vectors=ref)
            except Exception as e:
                log.exception("strain field compute failed: %s", e)
                return

            def _apply():
                # Drop a stale result superseded by a newer interaction.
                if gen != self._recompute_gen:
                    log.debug("[strain-ref] _recompute gen=%d SUPERSEDED (current=%d) — dropped",
                              gen, self._recompute_gen)
                    return
                self.field = field
                update_strain_view(self.p, self.field, self.component)
                log.debug("[strain-ref] _recompute gen=%d applied to plot", gen)

            session._dispatch_to_main(_apply)

        threading.Thread(target=_work, daemon=True, name="strain-recompute").start()

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

    def commit(self) -> None:
        """Freeze the current strain field as a NEW SignalTree — εxx is the signal
        plot, εyy / εxy / ω ride along as chip-selectable view figures (same shape
        as the Vector-OM result window). The live window stays open for tuning."""
        if self.field is None or self.session is None:
            return
        from spyde.actions._common import STRAIN_TITLES
        _commit_strain_tree(self.session, self.vecs, self.field, STRAIN_TITLES)

    def remove(self) -> None:
        """Tear down EVERYTHING the action added: the selection overlay, the
        dedicated reference crosshair + its DP window, and the strain-map
        window itself. Called when the action is toggled off (strain_stop).
        The user's main navigator/DP are left untouched — the reference
        selector was always a SEPARATE, additive crosshair."""
        from spyde.backend.ipc import emit
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
        # Close the strain-map window in the renderer.
        if self.window_id is not None:
            try:
                emit({"type": "window_closed", "window_id": int(self.window_id)})
            except Exception as e:
                log.debug("closing strain window failed: %s", e)
            _CONTROLLERS.pop(int(self.window_id), None)
        # Drop the back-reference so a later open rebuilds cleanly.
        if self.src_tree is not None and getattr(self.src_tree, "_strain_controller", None) is self:
            self.src_tree._strain_controller = None


# ── commit: freeze the live field as a new SignalTree ─────────────────────────

def _commit_strain_tree(session, vecs, field, titles) -> None:
    """Add a new SignalTree carrying the strain field — εxx is the signal plot,
    εyy / εxy / ω are chip-selectable view figures (mirrors the Vector-OM result
    window via spyde.actions.views)."""
    import hyperspy.api as hs
    from spyde.actions.views import emit_view_figure, register_views

    ny, nx = field.nav_shape
    base = "Strain"
    comps = [(titles["exx"], field.exx), (titles["eyy"], field.eyy),
             (titles["exy"], field.exy), (titles["omega"], field.omega)]
    mats = [(lbl, np.nan_to_num(np.asarray(m, np.float32))) for lbl, m in comps]
    finite = [float(np.nanmax(np.abs(m))) for _, m in mats if np.isfinite(m).any()]
    lim = (max(finite) if finite else 1.0) or 1.0

    new_sig = hs.signals.Signal2D(mats[0][1].copy())
    new_sig.metadata.General.title = base
    tree = session._add_signal(new_sig)
    sp = next(iter(getattr(tree, "signal_plots", [])), None)
    if sp is not None:
        sp.needs_auto_level = False
        try:
            sp.set_clim(-lim, lim)
            sp.set_data(mats[0][1])
        except Exception as e:
            log.debug("painting committed strain signal plot failed: %s", e)
        try:
            sp.set_view_tag(titles["exx"], "2d")
        except Exception as e:
            log.debug("tagging committed strain view failed: %s", e)
        wid = getattr(sp, "window_id", None)
        if wid is not None:
            register_views(wid, mats, levels=(-lim, lim))
            for lbl, m in mats[1:]:
                emit_view_figure(wid, m, lbl, kind="2d", levels=(-lim, lim))
    return tree


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
    window_id from the module registry first; fall back to the source tree."""
    wid = payload.get("window_id")
    if wid is None and plot is not None:
        wid = getattr(plot, "window_id", None)
    if wid is not None:
        ctrl = _CONTROLLERS.get(int(wid))
        if ctrl is not None:
            return ctrl
    tree = getattr(plot, "signal_tree", None)
    return getattr(tree, "_strain_controller", None) if tree is not None else None


def strain_run(session, plot, payload) -> None:
    """Open the interactive strain window for the active Find-Vectors result.

    The initial full-field fit (compute_strain_field) is a per-pixel scipy loop
    — run it on a worker thread and build/emit the window back on the main
    thread, so opening the action doesn't freeze the UI."""
    from spyde.backend.ipc import emit, emit_error, emit_status
    from spyde.actions.strain_mapping import compute_strain_field
    from spyde.actions.strain_display import build_strain_figure

    def _resolve_vecs():
        t = getattr(plot, "signal_tree", None)
        v = getattr(t, "diffraction_vectors", None) if t is not None else None
        if v is None:
            # The caret's window_id may resolve to a sibling plot of the vectors
            # tree (e.g. the count-map navigator); fall back to any tree that has
            # diffraction_vectors.
            for cand in getattr(session, "signal_trees", []):
                if getattr(cand, "diffraction_vectors", None) is not None:
                    return cand, cand.diffraction_vectors
        return t, v

    def _batch_running() -> bool:
        for cand in getattr(session, "signal_trees", []):
            if getattr(cand, "_fv_batch_running", False):
                return True
        return False

    tree, vecs = _resolve_vecs()
    if vecs is None:
        # Find Vectors finalizes (attaches diffraction_vectors) on a worker thread;
        # the user can open Strain in the brief window before it lands, or while a
        # slow batch is still running. In the real app (event loop present) wait
        # for it on a worker, then resume via _dispatch_to_main. Without a loop
        # (tests) error immediately.
        if getattr(session, "_dispatch_to_main", None) is None:
            emit_error("Strain mapping needs a Find Vectors result (no diffraction vectors).")
            return

        def _wait_then_run():
            import time
            waited = 0.0
            status_at = 0.0
            while True:
                _, v = _resolve_vecs()
                if v is not None:
                    session._dispatch_to_main(lambda: strain_run(session, plot, payload))
                    return
                running = _batch_running()
                # No result and nothing in flight: give the brief post-attach
                # window (~6s) a chance, then give up. While a batch IS running,
                # keep waiting — it can legitimately take well over a minute on a
                # slow cluster — with a periodic status ping so it doesn't look
                # stuck, up to a generous hard cap.
                if not running and waited >= 6.0:
                    emit_error("Strain mapping needs a Find Vectors result (no diffraction vectors).")
                    return
                if running and waited - status_at >= 5.0:
                    emit_status("Waiting for diffraction vectors to finish computing…")
                    status_at = waited
                if waited >= 300.0:
                    emit_error("Strain mapping timed out waiting for diffraction vectors.")
                    return
                time.sleep(0.1)
                waited += 0.1
        threading.Thread(target=_wait_then_run, daemon=True, name="strain-wait-vecs").start()
        return

    # Idempotent: opening the caret again must NOT spawn a second strain window.
    # Re-show the existing controller's overlay and bail.
    existing = getattr(tree, "_strain_controller", None)
    if existing is not None and getattr(existing, "window_id", None) in _CONTROLLERS:
        if existing.overlay is not None:
            existing.overlay.set_visible(True)
        return

    # Re-entrancy guard: React StrictMode mounts the wizard TWICE synchronously
    # (mount → cleanup → remount) on every open, firing strain_run then
    # strain_stop then strain_run again before either's worker thread has had a
    # chance to set tree._strain_controller — so the "existing" check above
    # can't see the first call in flight and BOTH proceed, building two
    # StrainControllers (two reference crosshairs, two overlays, two windows).
    # A generation counter bumped synchronously here — BEFORE spawning the
    # compute thread — closes that race: strain_stop bumps it too (cancelling
    # any in-flight run), and a stale generation's _build_window is dropped
    # on arrival instead of building a second live controller.
    gen = getattr(tree, "_strain_run_gen", 0) + 1
    tree._strain_run_gen = gen

    ref_yx = _default_reference(vecs)

    def _build_window(field):
        if getattr(tree, "_strain_run_gen", None) != gen:
            return   # superseded by a strain_stop or a newer strain_run
        _fig, fig_id, html, p = build_strain_figure(field, component="exx")
        wid = session.next_window_id()
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

    if getattr(session, "_dispatch_to_main", None) is None:
        # No event loop to marshal back to (e.g. tests) — run inline.
        _build_window(compute_strain_field(vecs, ref_yx))
        return

    emit_status("Computing strain field…")

    def _work():
        try:
            field = compute_strain_field(vecs, ref_yx)
        except Exception as e:
            log.exception("initial strain field compute failed: %s", e)
            emit_error(f"Strain mapping failed: {e}")
            return
        session._dispatch_to_main(lambda: _build_window(field))

    threading.Thread(target=_work, daemon=True, name="strain-run").start()


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
        tree._strain_run_gen = getattr(tree, "_strain_run_gen", 0) + 1
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is not None:
        ctrl.remove()
