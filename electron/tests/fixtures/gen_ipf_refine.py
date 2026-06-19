"""
gen_ipf_refine.py — render the OM refine IPF correlation-heatmap figure (real
Silver library + a real pattern + a mask circle + best marker) to a standalone
HTML, for a screenshot. Run: uv run electron/tests/fixtures/gen_ipf_refine.py
"""
import os

import numpy as np
import hyperspy.api as hs

HERE = os.path.dirname(__file__)
CIF = os.path.join(HERE, "..", "..", "..", "spyde", "tests", "Silver__0011135.cif")


def _sig(nav=(4, 4), sig=(96, 96), scale=0.0134):
    rng = np.random.RandomState(2)
    s = hs.signals.Signal2D(rng.rand(*nav, *sig).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = -(ax.size / 2.0) * scale
        ax.units = "$A^{-1}$"
    return s


def main():
    from orix.crystal_map import Phase
    from spyde.actions.orientation_compute import (
        generate_library_from_phases, build_matching_cache,
    )
    from spyde.actions.orientation_action import _reciprocal_radius
    from spyde.actions import ipf_refine
    from spyde.actions.ipf_refine_render import (
        build_refine_figure, update_panels, best_xy_for,
    )
    from spyde.drawing.plots.plot import finalize_figure_html

    s = _sig()
    sim = generate_library_from_phases(
        phases=[Phase.from_cif(CIF)], accelerating_voltage=200.0,
        resolution=6.0, minimum_intensity=1e-4, reciprocal_radius=_reciprocal_radius(s))
    cache = build_matching_cache(s, sim)

    infos = ipf_refine.build_phase_ipf(sim)
    corr, best = ipf_refine.match_correlations(np.asarray(s.data[1, 1], float),
                                               sim, cache, gamma=0.7)
    fig, fig_id, _html, panels = build_refine_figure(infos)

    # a mask circle near the lower-left + the best-match marker
    info = infos[0]
    cx = float(info["mins"][0] + 0.35 * (info["maxs"][0] - info["mins"][0]))
    cy = float(info["mins"][1] + 0.30 * (info["maxs"][1] - info["mins"][1]))
    circles = {0: [(cx, cy, 0.10)]}
    update_panels(panels, corr, circles, best_xy_for(infos, int(best[0])))

    html = finalize_figure_html(fig, fig_id)      # re-finalize → embeds the heatmap
    out = os.path.join(HERE, "ipf_refine.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"phases={len(infos)} templates={len(corr)} corrmax={corr.max():.3f} -> {out}")


if __name__ == "__main__":
    main()
