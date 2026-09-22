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

    Runs on the overlay lane, like the matched-pattern overlay. It is only the
    correlation, about 26 ms, but inline that was 26 ms added to every
    navigator step before the pattern under the crosshair could be read and
    painted — the drag felt heavy the moment the heat map opened. On the lane
    a superseded position is dropped and the surface is drawn for the position
    the user rests on, which is the one worth looking at.

    Double-clicking a triangle adds or removes a circle that RESTRICTS the
    match to the orientations inside it, the same gesture and the same meaning
    as the dense heat map's. It is how a user resolves an ambiguous pattern:
    the surface shows several bright spots, and confining the match to one of
    them says which is the right one.

    The restriction is on the LIVE fit — this heat map and the matched-pattern
    overlay, which share one matcher — and not on Compute Maps, which builds
    its own plans per run. That is the dense heat map's scope too. It is a
    thing you do while looking at one pattern, not a setting the scan is
    indexed under.
    """

    def __init__(self, vectors, fitter, infos, panels, fit_overlay=None):
        self.vectors = vectors
        self.fitter = fitter
        self.infos = infos
        self.panels = panels
        #: The matched-pattern overlay, which fits through the SAME matcher and
        #: so answers under the same restriction. It has its own cached reader,
        #: so a mask change has to invalidate it too or the green pattern keeps
        #: showing the orientation the user just ruled out.
        self.fit_overlay = fit_overlay
        self.circles = {info["phase_index"]: [] for info in infos}
        self.tree = None
        self.node = None
        self._closed = False

    def attach(self, tree):
        from spyde.actions.vector_overlay import VectorRows, _add_overlay

        self.tree = tree
        self.node = _add_overlay(
            tree, tree.root, zone_correlations_overlay, name="vom_refine_ipf",
            groups={}, source=False, expensive=True,
            iterating={"rows": VectorRows(self.vectors)},
            static={"fitter": self.fitter},
            on_value=self.draw,
        )
        self.bind_panels(self.panels)
        return self

    def bind_panels(self, panels) -> None:
        """Draw into *panels* from now on — the IPF window was rebuilt (shown
        again after a hide), or closed (an empty list)."""
        self.panels = list(panels)
        for panel in self.panels:
            self._wire_double_click(panel)

    def redraw(self) -> None:
        """Re-evaluate the current position, so a window shown again is not
        blank until the next navigator move."""
        self._reevaluate(self.node)

    # ── restricting the match to part of the triangle ────────────────────────
    def toggle_circle(self, phase_index: int, x: float, y: float) -> None:
        """Remove the circle the click lands in, else add one centred there.

        Then push the resulting restriction into the matcher, so the live
        overlay's fit and this heat map agree about which orientations are in
        play.
        """
        info = next((i for i in self.infos
                     if i["phase_index"] == phase_index), None)
        if info is None:
            return
        circles = self.circles[phase_index]
        for index, (cx, cy, radius) in enumerate(circles):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                circles.pop(index)
                break
        else:
            circles.append((float(x), float(y),
                            0.09 * float((info["maxs"] - info["mins"]).mean())))
        self.apply_mask()

    def apply_mask(self) -> None:
        """Hand the circles to the matcher as one keep-mask per phase."""
        from spyde.actions.ipf_refine import rot_mask_from_circles

        counts = self.fitter.zone_counts
        mask = rot_mask_from_circles(self.infos, self.circles, int(sum(counts)))
        if mask is None:
            self.fitter.set_zone_mask(None)
        else:
            # build_zone_ipf stacks the phases' zone axes in order, so each
            # phase's block is the contiguous run the mask has to be cut into.
            offsets = np.cumsum([0] + list(counts))
            self.fitter.set_zone_mask(
                [mask[offsets[i]:offsets[i + 1]] for i in range(len(counts))])
        self.refresh()

    def refresh(self) -> None:
        """Re-evaluate the current position, so a mask change shows at once
        rather than at the next navigator move.

        The mask lives on the matcher's plans, not in the overlay's recipe —
        it has to, because the matched-pattern overlay fits through the same
        matcher and must honour the same restriction. So the statics being
        re-declared here are unchanged by identity; what they buy is the tree's
        "this overlay's inputs moved, drop the cached reader and re-run" path,
        once per node that reads the matcher.
        """
        self._reevaluate(self.node)
        self._reevaluate(self.fit_overlay)

    def _reevaluate(self, node) -> None:
        if self.tree is None or node is None:
            return
        try:
            self.tree.replace_overlay_static(node, fitter=self.fitter)
        except Exception as e:
            log.debug("re-evaluating a vector refine overlay failed: %s", e)

    def _wire_double_click(self, panel):
        phase_index = panel["info"]["phase_index"]

        def _on_double_click(event=None):
            x = getattr(event, "xdata", None)
            y = getattr(event, "ydata", None)
            if x is None and isinstance(event, dict):
                x, y = event.get("xdata"), event.get("ydata")
            if x is not None and y is not None:
                self.toggle_circle(phase_index, float(x), float(y))

        try:
            panel["xy"].add_event_handler(_on_double_click, "double_click")
        except Exception as e:
            log.debug("wiring a vector refine panel double-click failed: %s", e)

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

        # The restriction outlives this window otherwise: it lives on the
        # matcher, which the matched-pattern overlay goes on using, and the
        # circles that would lift it are going away with the panels. A mask
        # nobody can see and nobody can undo is worse than none.
        if any(self.circles.values()):
            self.circles = {key: [] for key in self.circles}
            try:
                self.fitter.set_zone_mask(None)
            except Exception as e:
                log.debug("clearing the zone mask on close failed: %s", e)
            # Only when the tree is staying: this same close runs on tree
            # teardown, where re-evaluating would spend a fit on a navigator
            # that is going away. Only the pattern overlay, too — this
            # window's own overlay is about to be removed.
            if not getattr(self.tree, "_spyde_closed", False):
                self._reevaluate(self.fit_overlay)

        try:
            remove_overlay_node(self.tree, self.node)
        except Exception as e:
            log.debug("removing the vector refine IPF overlay failed: %s", e)
        self.node = None


def open_refine_ipf(session, signal, fitter, phases, vectors, tree,
                    fit_overlay=None):
    """Put the heat map into the wizard's IPF window and wire it to the
    navigator.

    The window is the action's (see :mod:`spyde.actions.ipf_panel`): it was
    showing the phase's triangle, then the library filling in, and from here
    on it shows the correlation. Closing it hides it; re-selecting the action
    brings it back with this controller still drawing into it.

    ``fit_overlay`` is the matched-pattern overlay node, so a mask drawn here
    also redraws the pattern it restricts.

    Returns the controller, or None — a failure here costs the heat map and
    must not cost the library that was just built.
    """
    try:
        from spyde.actions.ipf_panel import ensure_panel

        infos = build_zone_ipf(fitter, phases)
        if not infos:
            return None
        panel = ensure_panel(session, tree, "vom", signal)
        panel.phases = list(phases)
        panel.set_library(infos)
        panel.show()
        controller = VectorRefineIpfController(
            vectors, fitter, infos, panel.panels, fit_overlay=fit_overlay).attach(tree)
        panel.controller = controller
        return controller
    except Exception as e:
        log.debug("vector refine IPF window failed: %s", e)
        return None
