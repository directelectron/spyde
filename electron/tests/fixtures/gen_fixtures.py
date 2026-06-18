"""Generate real anyplotlib figure HTML fixtures for Playwright visual tests.

Run from the repo root:  uv run python electron/tests/fixtures/gen_fixtures.py

Produces:
  real_bright.html      — a figure with a bright diagonal gradient baked in
                          (Playwright screenshots this; must NOT be all black)
  black_placeholder.html + replay_state.json
                          — a figure with a 10x10 zero placeholder, plus the
                          panel state for a bright image. The Playwright replay
                          test injects the figure, then the state as a delayed
                          awi_state, and asserts the canvas becomes non-black.
"""
import json
import os

import numpy as np
import anyplotlib as apl
from anyplotlib.embed import build_standalone_html

HERE = os.path.dirname(os.path.abspath(__file__))

# Mirror the runtime dark-mode injection from spyde/drawing/plots/plot.py so the
# fixtures faithfully represent what the app produces.
_DARK_STYLE = "<style>html,body{background:#1e1e2e !important;color-scheme:dark}</style>"


def _dark(html: str) -> str:
    return html.replace("<body>", _DARK_STYLE + "<body>", 1)


def bright_image(n=64):
    x = np.linspace(0, 1, n)
    return (np.add.outer(x, x) * 200).astype(np.float32)  # diagonal gradient


# ── real_bright.html: bright data baked in ───────────────────────────────────
fig, ax = apl.subplots(1, 1)
ax.imshow(bright_image(), cmap="viridis")
html = _dark(build_standalone_html(fig, fig_id="bright"))
with open(os.path.join(HERE, "real_bright.html"), "w") as f:
    f.write(html)

# ── black placeholder + replay state ─────────────────────────────────────────
fig2, ax2 = apl.subplots(1, 1)
plot2 = ax2.imshow(np.zeros((10, 10), dtype=np.float32), cmap="viridis")
placeholder_html = _dark(build_standalone_html(fig2, fig_id="replay"))
with open(os.path.join(HERE, "black_placeholder.html"), "w") as f:
    f.write(placeholder_html)

# Now set bright data and capture the panel traits as awi_state messages.
plot2.set_data(bright_image())
plot2.set_clim(0, 200)
panel_states = []
for tname in fig2.trait_names():
    if tname.startswith("panel_") and (tname.endswith("_json") or tname.endswith("_geom")):
        panel_states.append({"key": tname, "value": getattr(fig2, tname)})

with open(os.path.join(HERE, "replay_state.json"), "w") as f:
    json.dump({"fig_id": "replay", "states": panel_states}, f)

print("wrote fixtures:", os.listdir(HERE))
