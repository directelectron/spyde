"""
gen_ipf.py — render a colourful IPF orientation map three ways to compare how
anyplotlib shows the reduced-IPF RGB vs matplotlib (the reference):

  ipf_ref.png         matplotlib imshow(rgb)            — ground truth
  ipf_direct.html     anyplotlib ax.imshow(rgb)         — embedded RGB
  ipf_live_scalar.html + ipf_live_updates.json
                      anyplotlib imshow(scalar,gray) then a LIVE set_data(rgb)
                      state push — exactly what the SpyDE IPF window does.

Run:  uv run electron/tests/fixtures/gen_ipf.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)


def ipf_rgb(ny=44, nx=56):
    """A smoothly-varying cubic IPF colour map (orix IPFColorKeyTSL)."""
    from orix.quaternion import Orientation, Rotation
    from orix.quaternion.symmetry import Oh
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d

    eul = np.zeros((ny, nx, 3))
    eul[..., 0] = np.linspace(0, np.pi, nx)[None, :]       # phi1 across x
    eul[..., 1] = np.linspace(0, np.pi / 2, ny)[:, None]   # Phi across y
    rots = Rotation.from_euler(eul.reshape(-1, 3))
    ori = Orientation(rots, symmetry=Oh)
    key = IPFColorKeyTSL(Oh, direction=Vector3d.zvector())
    colors = key.orientation2color(ori).reshape(ny, nx, 3)
    return np.clip(colors * 255.0, 0, 255).astype(np.uint8)


def main():
    rgb = ipf_rgb()

    # 1) matplotlib reference
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.imshow(rgb)
    ax.set_title("matplotlib imshow(rgb)")
    fig.savefig(os.path.join(HERE, "ipf_ref.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    import anyplotlib as apl
    import anyplotlib._electron as el
    from spyde.drawing.plots.plot import finalize_figure_html

    # 2) anyplotlib DIRECT imshow(rgb)
    f1, axes1 = apl.subplots(1, 1)
    a1 = np.array(axes1, dtype=object).ravel()[0]
    a1.imshow(rgb)
    html1 = finalize_figure_html(f1, "ipfdirect")
    with open(os.path.join(HERE, "ipf_direct.html"), "w") as fh:
        fh.write(html1)

    # 3) anyplotlib LIVE path (what the IPF window does): scalar+gray imshow,
    #    finalize the SCALAR html, then capture the set_data(rgb) state push.
    updates = []
    orig = el.emit
    el.emit = lambda m: updates.append(m)
    try:
        f2, axes2 = apl.subplots(1, 1)
        a2 = np.array(axes2, dtype=object).ravel()[0]
        p2 = a2.imshow(np.zeros((10, 10), dtype=np.float32), cmap="gray")
        fid = el.register(f2)
        html2 = finalize_figure_html(f2, fid)       # SCALAR html (pre set_data)
        updates.clear()                              # only keep the set_data push
        p2.set_data(rgb)
    finally:
        el.emit = orig
    with open(os.path.join(HERE, "ipf_live_scalar.html"), "w") as fh:
        fh.write(html2)
    with open(os.path.join(HERE, "ipf_live_updates.json"), "w") as fh:
        json.dump([{"key": m.get("key"), "value": m.get("value")}
                   for m in updates if m.get("type") == "state_update"], fh)

    keys = sorted({m.get("key") for m in updates if m.get("type") == "state_update"})
    print(f"rgb shape {rgb.shape}, set_data pushed {len(updates)} updates; keys={keys}")


if __name__ == "__main__":
    main()
