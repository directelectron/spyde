"""
strain_action.py — the interactive strain-mapping action.

Runs strain on a Find-Vectors result (``tree.diffraction_vectors``), opens the
component-map + ellipse-glyph window, and attaches a **draggable reference
crosshair**: moving it picks the unstrained region and recomputes the whole
strain field live. A component toggle (εxx / εyy / εxy / ω) swaps the displayed
map in place. No Qt.
"""
from __future__ import annotations

import numpy as np

_COMPONENTS = ("exx", "eyy", "exy", "omega")
_TITLES = {"exx": "εxx", "eyy": "εyy", "exy": "εxy", "omega": "ω"}


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

    def __init__(self, vecs, plot2d, glyph_group, *, component="exx", ref_yx=(0, 0)):
        self.vecs = vecs
        self.p = plot2d
        self.glyph = glyph_group
        self.component = component
        self.ref_yx = (int(ref_yx[0]), int(ref_yx[1]))
        self.field = None
        self._crosshair = None
        self._ref_vectors = None        # set → CIF absolute reference; None → region

    def attach(self):
        from spyde.actions.strain_mapping import compute_strain_field
        self.field = compute_strain_field(self.vecs, self.ref_yx)
        try:
            ry, rx = self.ref_yx
            self._crosshair = self.p.add_crosshair_widget(cx=float(rx), cy=float(ry),
                                                          color="#00e5ff")
            self._crosshair.add_event_handler(self._on_pick, "pointer_up")
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("strain crosshair attach failed: %s", e)
        return self

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
        from spyde.actions.strain_mapping import compute_strain_field
        from spyde.actions.strain_display import update_strain_view
        self.ref_yx = (int(ry), int(rx))
        self._ref_vectors = None                    # crosshair → back to region mode
        self.field = compute_strain_field(self.vecs, self.ref_yx)
        update_strain_view(self.p, self.field, self.component, self.glyph)

    def set_cif_reference(self, phase) -> None:
        """CIF mode: snap the reference pixel's vectors to the phase's ideal |g|
        families → absolute strain (no unstrained region needed)."""
        from spyde.actions.strain_mapping import (
            cif_g_families, snap_reference_to_cif, compute_strain_field)
        from spyde.actions.strain_display import update_strain_view
        fam = cif_g_families(phase)
        sample = self.vecs.kxy_at(*self.ref_yx)
        ref_vectors = snap_reference_to_cif(sample, fam)
        if len(ref_vectors) < 2:
            return
        self._ref_vectors = ref_vectors
        self.field = compute_strain_field(self.vecs, ref_vectors=ref_vectors)
        update_strain_view(self.p, self.field, self.component, self.glyph)

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
    """Open the interactive strain window for the active Find-Vectors result."""
    from spyde.backend.ipc import emit, emit_error
    from spyde.actions.strain_mapping import compute_strain_field
    from spyde.actions.strain_display import build_strain_figure

    tree = getattr(plot, "signal_tree", None)
    vecs = getattr(tree, "diffraction_vectors", None) if tree is not None else None
    if vecs is None:
        emit_error("Strain mapping needs a Find Vectors result (no diffraction vectors).")
        return

    ref_yx = _default_reference(vecs)
    field = compute_strain_field(vecs, ref_yx)
    _fig, fig_id, html, p, glyph = build_strain_figure(field, component="exx")

    wid = session.next_window_id()
    emit({"type": "figure", "fig_id": fig_id, "window_id": int(wid),
          "html": html, "title": "Strain (εxx)", "is_navigator": False,
          "strain_components": list(_COMPONENTS)})

    ctrl = StrainController(vecs, p, glyph, component="exx", ref_yx=ref_yx)
    ctrl.attach()
    tree._strain_controller = ctrl


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
