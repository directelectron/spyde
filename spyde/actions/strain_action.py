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
    method (Region/CIF), and commits the current field as a new SignalTree."""

    def __init__(self, vecs, plot2d, *, window_id=None,
                 component="exx", ref_yx=(0, 0), session=None):
        self.vecs = vecs
        self.p = plot2d
        self.window_id = window_id
        self.session = session
        self.component = component
        self.ref_yx = (int(ref_yx[0]), int(ref_yx[1]))
        self.field = None
        self._crosshair = None
        self.cif_mode = False
        self._g_ref_full = None          # current reference vector set (zero-beam removed)
        self._recompute_gen = 0          # latest-wins guard for async recomputes

    def attach(self):
        self._set_full_reference(self.vecs.kxy_at(*self.ref_yx))
        if self.window_id is not None:
            _CONTROLLERS[int(self.window_id)] = self
        # Skip the initial recompute when the field was already computed by the
        # caller (_build_window pre-populates ctrl.field and the figure already
        # shows it) — recomputing here would run the full per-pixel fit a second
        # time on window open. Later ref/CIF/method interactions still call
        # _recompute() directly.
        if self.field is None:
            self._recompute()
        try:
            ry, rx = self.ref_yx
            self._crosshair = self.p.add_crosshair_widget(cx=float(rx), cy=float(ry),
                                                          color="#00e5ff")
            from spyde.drawing.selectors.base_selector import event_handler_fn
            self._pick_cb = event_handler_fn(self._on_pick)
            self._crosshair.add_event_handler(self._pick_cb, "pointer_up")
        except Exception as e:
            log.debug("strain crosshair attach failed: %s", e)
        return self

    # ── reference (Region crosshair pixel, or CIF-snapped absolute) ───────────
    def _set_full_reference(self, g_ref) -> None:
        # Use every reference spot EXCEPT the zero beam (see _zero_beam_filtered).
        self._g_ref_full = _zero_beam_filtered(g_ref)

    def _selected_reference(self):
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
                    return
                self.field = field
                update_strain_view(self.p, self.field, self.component)

            session._dispatch_to_main(_apply)

        threading.Thread(target=_work, daemon=True, name="strain-recompute").start()

    def _on_pick(self, event=None):
        try:
            rx = int(round(float(self._crosshair.cx)))
            ry = int(round(float(self._crosshair.cy)))
        except Exception:
            return
        ny, nx = self.vecs.nav_shape
        if 0 <= ry < ny and 0 <= rx < nx and (ry, rx) != self.ref_yx:
            self.set_reference(ry, rx)

    def set_reference(self, ry: int, rx: int) -> None:
        """Region mode: the crosshair picks an unstrained pixel (relative strain)."""
        self.ref_yx = (int(ry), int(rx))
        self.cif_mode = False
        self._set_full_reference(self.vecs.kxy_at(*self.ref_yx))
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
        self._recompute()

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


# ── toolbar entry (ActionContext convention: fn(ctx, ...)) ────────────────────

def strain_mapping(ctx, action_name: str = "Strain Mapping", **params) -> None:
    """Toolbar entry on a Find-Vectors result — open the interactive strain
    window (one-shot; the live reference crosshair + component toggle take over)."""
    strain_run(ctx.session, ctx.plot, {})


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

    tree = getattr(plot, "signal_tree", None)
    vecs = getattr(tree, "diffraction_vectors", None) if tree is not None else None
    if vecs is None:
        emit_error("Strain mapping needs a Find Vectors result (no diffraction vectors).")
        return

    ref_yx = _default_reference(vecs)

    def _build_window(field):
        _fig, fig_id, html, p = build_strain_figure(field, component="exx")
        wid = session.next_window_id()
        emit({"type": "figure", "fig_id": fig_id, "window_id": int(wid),
              "html": html, "title": "Strain (εxx)", "is_navigator": False,
              "strain_components": list(_COMPONENTS)})
        ctrl = StrainController(vecs, p, window_id=wid, component="exx",
                                ref_yx=ref_yx, session=session)
        ctrl.field = field
        ctrl.attach()
        tree._strain_controller = ctrl

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


def strain_commit(session, plot, payload) -> None:
    """Submit: freeze the current strain field as a new SignalTree."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        from spyde.backend.ipc import emit_error
        emit_error("No live strain field to submit.")
        return
    ctrl.commit()
