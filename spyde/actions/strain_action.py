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


class StrainController:
    """Owns the live strain window: recomputes the field when the reference
    crosshair moves and swaps the shown component on toggle."""

    def __init__(self, vecs, plot2d, glyph_group, *, window_id=None,
                 component="exx", ref_yx=(0, 0), session=None):
        self.vecs = vecs
        self.p = plot2d
        self.glyph = glyph_group
        self.window_id = window_id
        self.session = session
        self.component = component
        self.ref_yx = (int(ref_yx[0]), int(ref_yx[1]))
        self.field = None
        self._crosshair = None
        self.cif_mode = False
        self._g_ref_full = None          # current full reference vector set
        self.rings = []                  # ring |g| (ascending)
        self._ring_idx = None            # per-ref-vector ring index
        self.selected_rings = set()      # enabled ring indices (use all by default)
        self._recompute_gen = 0          # latest-wins guard for async recomputes

    def attach(self):
        self._set_full_reference(self.vecs.kxy_at(*self.ref_yx))
        # Skip the initial recompute when the field was already computed by the
        # caller (_build_window pre-populates ctrl.field and the figure already
        # shows it) — recomputing here would run the full per-pixel fit a second
        # time on window open. Later ref/ring/CIF interactions still call
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

    # ── reference + ring selection (both modes flow through ref_vectors) ──────
    def _set_full_reference(self, g_ref) -> None:
        from spyde.actions.strain_mapping import group_rings
        self._g_ref_full = np.asarray(g_ref, dtype=float).reshape(-1, 2)
        self.rings, self._ring_idx = group_rings(self._g_ref_full)
        self.selected_rings = set(range(len(self.rings)))      # use all
        self._emit_rings()

    def _selected_reference(self):
        if self._ring_idx is None or len(self.selected_rings) >= len(self.rings):
            return self._g_ref_full
        keep = np.array([i in self.selected_rings for i in self._ring_idx], dtype=bool)
        return self._g_ref_full[keep]

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
            update_strain_view(self.p, self.field, self.component, self.glyph)
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
                update_strain_view(self.p, self.field, self.component, self.glyph)

            session._dispatch_to_main(_apply)

        threading.Thread(target=_work, daemon=True, name="strain-recompute").start()

    def _emit_rings(self) -> None:
        if self.window_id is None:
            return
        try:
            from spyde.backend.ipc import emit
            emit({"type": "strain_rings", "window_id": int(self.window_id),
                  "rings": [round(float(g), 4) for g in self.rings],
                  "selected": sorted(self.selected_rings), "cif": bool(self.cif_mode)})
        except Exception as e:
            log.debug("emitting strain_rings failed: %s", e)

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
        if len(snapped) < 2:
            return
        self.cif_mode = True
        self._set_full_reference(snapped)
        self._recompute()

    def set_rings(self, selected) -> None:
        """Peak selection: keep only the chosen reflection rings in the fit."""
        sel = set(int(i) for i in selected)
        self.selected_rings = sel or set(range(len(self.rings)))
        self._recompute()

    def set_component(self, component: str) -> None:
        from spyde.actions.strain_display import update_strain_view
        if component not in _COMPONENTS or self.field is None:
            return
        self.component = component
        update_strain_view(self.p, self.field, component, self.glyph)


# ── toolbar entry (ActionContext convention: fn(ctx, ...)) ────────────────────

def strain_mapping(ctx, action_name: str = "Strain Mapping", **params) -> None:
    """Toolbar entry on a Find-Vectors result — open the interactive strain
    window (one-shot; the live reference crosshair + component toggle take over)."""
    strain_run(ctx.session, ctx.plot, {})


# ── staged handlers (session.py dispatch: fn(session, plot, payload)) ──────────

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
        _fig, fig_id, html, p, glyph = build_strain_figure(field, component="exx")
        wid = session.next_window_id()
        emit({"type": "figure", "fig_id": fig_id, "window_id": int(wid),
              "html": html, "title": "Strain (εxx)", "is_navigator": False,
              "strain_components": list(_COMPONENTS)})
        ctrl = StrainController(vecs, p, glyph, window_id=wid, component="exx",
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
    tree = getattr(plot, "signal_tree", None)
    ctrl = getattr(tree, "_strain_controller", None) if tree is not None else None
    if ctrl is not None:
        ctrl.set_component(str(payload.get("component", "exx")))


def strain_set_cif(session, plot, payload) -> None:
    """Set the absolute reference from a CIF (``payload['cif_path']``) — the
    reflection family is guessed from the measured |g| vs the CIF's ideal spacings.
    Passing no path reverts to the region (crosshair) reference."""
    tree = getattr(plot, "signal_tree", None)
    ctrl = getattr(tree, "_strain_controller", None) if tree is not None else None
    if ctrl is None:
        return
    cif_path = payload.get("cif_path")
    if not cif_path:
        ry, rx = ctrl.ref_yx
        ctrl.set_reference(ry, rx)                  # back to region mode
        return
    try:
        from orix.crystal_map import Phase
        ctrl.set_cif_reference(Phase.from_cif(cif_path))
    except Exception as e:
        from spyde.backend.ipc import emit_error
        emit_error(f"Strain CIF reference failed: {e}")


def strain_set_rings(session, plot, payload) -> None:
    """Peak selection: ``payload['selected']`` = the reflection-ring indices to
    keep in the fit (empty/missing → use all)."""
    tree = getattr(plot, "signal_tree", None)
    ctrl = getattr(tree, "_strain_controller", None) if tree is not None else None
    if ctrl is not None:
        ctrl.set_rings(payload.get("selected", []))
