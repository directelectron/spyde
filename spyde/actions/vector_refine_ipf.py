"""The live IPF correlation heat map for the vector orientation matcher.

The pattern under the crosshair is correlated against every sampled
orientation, and each phase's inverse-pole-figure triangle is coloured by the
result. That surface is what the single best number cannot show: a confident
position is one bright spot, an ambiguous one has several, and a phase the
pattern does not belong to is uniformly dim.

The dense orientation mapper has had this for a while and it is the same
picture, so the geometry and the panels are the same code
(``ipf_refine``/``ipf_refine_render``). Only the source of the correlations
differs — zone axes scored by the correlation matcher rather than diffsims
templates scored by pyxem — and only that lives here.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def zone_correlations_overlay(*, rows, fitter):
    """One position's correlation against every sampled orientation.

    Returned in the shape the shared panel renderer expects: the correlations
    of every phase end to end, and the index of the best, so the panels can
    mark where the match landed.
    """
    correlations = fitter.zone_correlations(rows)
    if not correlations:
        return None
    flat = np.concatenate([np.asarray(c, float) for c in correlations])
    if flat.size == 0:
        return None
    return flat, [int(np.argmax(flat))]


def build_zone_ipf(fitter, phases) -> list[dict]:
    """Triangle geometry for the orientations each phase's plan samples.

    The matcher samples zone axes rather than building diffsims templates, so
    the orientation table comes off the plan; it is projected onto the same
    triangle as the dense path's, which is what lets both render through the
    same panels.
    """
    from spyde.actions.ipf_refine import build_phase_ipf_for
    from spyde.actions.vector_orientation_quantem import quantem_quats_to_orix

    quats, phase_of = [], []
    for index, orientation_map in enumerate(fitter._maps):
        zone = quantem_quats_to_orix(orientation_map.zone_quats.cpu().numpy())
        quats.append(np.asarray(zone, float))
        phase_of.append(np.full(len(zone), index, dtype=int))
    if not quats:
        return []
    return build_phase_ipf_for(np.vstack(quats), np.concatenate(phase_of), phases)


class VectorRefineIpfController:
    """Repaints the phase triangles as the navigator moves.

    The correlation is an overlay node on the vectors, so it is re-evaluated at
    every committed position and its value recolours the panels on the painter
    thread — the same arrangement the dense refine heat map uses.

    Runs inline rather than on the overlay lane: this is the correlation only,
    without the refinement or the strain solve that make the matched-pattern
    overlay expensive.

    """

    def __init__(self, vectors, fitter, infos, panels):
        self.vectors = vectors
        self.fitter = fitter
        self.infos = infos
        self.panels = panels
        self.circles = {info["phase_index"]: [] for info in infos}
        self.tree = None
        self.node = None
        self._closed = False

    def attach(self, tree):
        from spyde.actions.vector_overlay import VectorRows, _add_overlay

        self.tree = tree
        self.node = _add_overlay(
            tree, tree.root, zone_correlations_overlay, name="vom_refine_ipf",
            groups={}, source=False, expensive=False,
            iterating={"rows": VectorRows(self.vectors)},
            static={"fitter": self.fitter},
            on_value=self.draw,
        )
        return self

    def draw(self, value) -> None:
        """Recolour every phase panel from one position's correlations."""
        from spyde.actions.ipf_refine_render import best_xy_for, update_panels

        if not self.panels or value is None:
            return
        try:
            correlations, best = value
            update_panels(self.panels, correlations, self.circles,
                          best_xy_for(self.infos, int(best[0])))
        except Exception as e:
            log.debug("repainting the vector refine triangles failed: %s", e)

    def close(self) -> None:
        self.remove()

    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        from spyde.actions.vector_overlay import remove_overlay_node

        try:
            remove_overlay_node(self.tree, self.node)
        except Exception as e:
            log.debug("removing the vector refine IPF overlay failed: %s", e)
        self.node = None


def open_refine_ipf(session, signal, fitter, phases, vectors, tree):
    """Open the heat-map window and wire it to the navigator.

    Returns the controller, or None — a failure here costs the heat map and
    must not cost the library that was just built.
    """
    try:
        from spyde.actions.ipf_refine_render import (
            build_refine_figure, emit_refine_window,
        )

        infos = build_zone_ipf(fitter, phases)
        if not infos:
            return None
        figure, figure_id, html, panels = build_refine_figure(infos)
        base = signal.metadata.get_item("General.title", "Signal")
        window_id = emit_refine_window(session, figure, figure_id, html,
                                       title=f"{base} — IPF Refine")
        controller = VectorRefineIpfController(
            vectors, fitter, infos, panels).attach(tree)
        # Give the bare-figure window a teardown identity, so closing it with ✕
        # unhooks the navigator overlay instead of leaving it evaluating.
        if controller is not None:
            session.register_window_controller(window_id, controller)
        return controller
    except Exception as e:
        log.debug("vector refine IPF window failed: %s", e)
        return None
