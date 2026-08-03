"""orientation_embed.py — embed a FULL orientation-mapping result in an HTML report.

The sibling of :mod:`vectors_embed`, and the same bargain: an orientation result
is small enough to inline into a self-contained page, and rich enough to drive
the whole IPF explorer client-side. Per nav position it is one best-match
orientation — a stereographic ``(x, y)`` in the fundamental sector, a reduced
crystal direction on the unit sphere, and an IPF colour. 20 bytes per position
per sample direction.

Measured at sped_ag scale (208x64 = 13,312 positions): packing 0.38 s for a
1.24 MB base64 blob, 0.30 s more to build the page, 3.04 MB of HTML all told —
against the vectors explorer's ~21 MB. Most of the difference between the blob
and the page is the drawn cloud's scatter offsets, which ride the figure's own
state as JSON text; that is what ``CLOUD_MAX`` bounds.

ONE anyplotlib figure with THREE panels, mirroring the app's two windows:

- **map** — the IPF colour map, with a draggable **crosshair**. This is the
  window-1 projection map.
- **triangle** — every position's orientation as a point in the fundamental
  sector, in its own IPF colour, plus a white **marker** on the picked one.
  Window 2's ``[2D] · [Points]``.
- **sphere** — the same directions on the unit sphere (``scatter3d``), with a
  **highlight** on the picked one and the camera **rotated to face it**.
  Window 2's ``[3D] · [Points]``.

All three re-colour together when the **X / Y / Z** sample direction changes.
Three panels rather than the app's 2D⇄3D toggle because a toggle would have to
swap panel KINDS inside one mounted figure, and the report needs a single
figure/single ``mount()`` — that is what renders in the report SIDEBAR rather
than degrading to a plain snapshot (see :mod:`vectors_embed`).

Nothing recomputes: a pick is an index into the packed arrays, so the page needs
no orix, no backend and no network, and still works years later.

Packing (little-endian, one base64 blob) — see :func:`pack_orientation`:
    uint8   phase[M]                          per-position best-match phase
    per direction d in (x, y, z):
      uint8   rgb[M*3]                        IPF colour  → the map image
      float32 xy[M*2]                         sector (x, y) → the triangle marker
      float32 xyz[M*3]                        unit-sphere direction → the highlight
M = ny*nx. 20 bytes per position per direction (+ M for the phase map).

Hooked into ``export_html._render_body`` (interactive mode) and the report
sidebar's live figure build, exactly like the vectors explorer.
"""
from __future__ import annotations

import base64
import html as _html
import json
import logging

import numpy as np

log = logging.getLogger(__name__)

DIRECTIONS = ("x", "y", "z")

# The embed refuses a scan bigger than this. 20 bytes/position/direction × 3
# directions = 60 B/position, so 2 M positions ≈ 120 MB packed — already far
# past what belongs in an HTML file, and the base64 inflation makes it 160 MB.
MAX_EMBED_POSITIONS = 2_000_000

# Cloud points actually DRAWN in the triangle / sphere panels. The full
# per-position arrays stay in the packed blob (a pick indexes them), but the
# scatter offsets ride in the figure's own state as JSON text — at ~30 chars a
# point that is the single biggest thing in the page, and it is re-serialised on
# every direction switch. Same ceiling the live 2-D view uses
# (``ipf_window.POINTS2D_MAX``), so the embed shows the same cloud.
CLOUD_MAX = 30_000

POINT_SIZE_2D = 5
POINT_SIZE_3D = 6.0
FIG_PX = 300                  # each panel's CSS height

_ESM_TEXT: "str | None" = None

# Memoized page per cell id, holding a STRONG reference to the result it was
# built from (identity-compared with `is`) — the vectors explorer's cache
# contract, and for the same reason: an int `id()` is reused once the original
# is collected, which would serve a stale page for a different result.
_EXPLORER_CACHE: "dict[str, tuple[object, str]]" = {}


def _esm_text() -> str:
    """anyplotlib's ESM, read once per process (it is ~400 KB and never
    changes)."""
    global _ESM_TEXT
    if _ESM_TEXT is None:
        from anyplotlib import embed as apl_embed
        _ESM_TEXT = apl_embed.esm_path().read_text(encoding="utf-8")
    return _ESM_TEXT


def clear_explorer_cache(cell_id: "str | None" = None) -> None:
    """Drop the memoized page(s) — one cell's, or all of them."""
    if cell_id is None:
        _EXPLORER_CACHE.clear()
    else:
        _EXPLORER_CACHE.pop(cell_id, None)


def _as_orientation_map(result):
    """Normalise a raw-OM or vector-OM result to a ``SpyDEOrientationMap``.

    Delegates to the live IPF window's own normaliser rather than repeating the
    rule, so a third producer only has to be taught to one of them. Lazy: this
    module is imported from the report handlers, and ``ipf_view`` pulls the
    figure registry."""
    from spyde.actions.ipf_view import _as_orientation_map as _norm
    return _norm(result)


# ── packing ───────────────────────────────────────────────────────────────────

def _per_position_direction(om, direction: str):
    """``(rgb (M,3) uint8, xy (M,2) float32, xyz (M,3) float32)`` for every nav
    position, in nav raster order.

    Deliberately NOT ``om.ipf_sphere_points`` / ``om.ipf_color_map``: the former
    drops non-finite rows and subsamples, so its row i is not nav position i —
    and an index that does not mean "nav position" is exactly what a crosshair
    pick needs. Everything here stays full length and nav-aligned; a position
    with no valid orientation keeps a NaN row and the page skips it.
    """
    from orix.plot import IPFColorKeyTSL
    from orix.projections import StereographicProjection
    from orix.quaternion import Orientation, Rotation
    from spyde.signals.orientation_map import _direction_vector

    best_phase = np.asarray(om.phase_idx)[..., 0].reshape(-1)
    best_q = np.asarray(om.quats)[..., 0, :].reshape(-1, 4)
    m = best_q.shape[0]
    d = _direction_vector(direction)

    rgb = np.zeros((m, 3), dtype=np.uint8)
    xy = np.full((m, 2), np.nan, dtype=np.float32)
    xyz = np.full((m, 3), np.nan, dtype=np.float32)
    proj = StereographicProjection()
    for i in range(om.n_phases):
        mask = best_phase == i
        if not mask.any():
            continue
        pg = om.orix_phase(i).point_group
        rot = Rotation(best_q[mask])
        v = (rot * d).in_fundamental_sector(pg)
        xyz[mask] = np.asarray(v.data, dtype=np.float32)
        vx, vy = proj.vector2xy(v)
        xy[mask] = np.column_stack([np.atleast_1d(vx),
                                    np.atleast_1d(vy)]).astype(np.float32)
        key = IPFColorKeyTSL(pg.laue, direction=d)
        colors = key.orientation2color(Orientation(rot, symmetry=pg))
        rgb[mask] = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return rgb, xy, xyz


def _phase_geometry(om) -> list:
    """One record per phase PRESENT in the map: its sector outline, corner
    labels and axis limits — the static furniture the triangle panel draws."""
    from spyde.signals.orientation_map import ipf_triangle_xy

    present = sorted({int(p) for p in np.asarray(om.phase_map()).reshape(-1)})
    out = []
    for pidx in present:
        try:
            phase = om.orix_phase(pidx)
            edges, label_xy, labels = ipf_triangle_xy(phase)
        except Exception as e:
            log.debug("[report] sector geometry for phase %s failed: %s", pidx, e)
            continue
        e = np.asarray(edges, dtype=float)
        if e.size == 0:
            continue
        pad = 0.05 * max(1e-6, float(np.ptp(e[:, 0]) + np.ptp(e[:, 1])) / 2.0)
        out.append({
            "idx": int(pidx),
            "name": str(getattr(phase, "name", "") or f"phase {pidx}"),
            "edges": e.tolist(),
            "labels": [{"x": float(p[0]), "y": float(p[1]), "text": str(t)}
                       for p, t in zip(np.asarray(label_xy, dtype=float), labels)],
            "xlim": [float(e[:, 0].min()) - pad, float(e[:, 0].max()) + pad],
            "ylim": [float(e[:, 1].min()) - pad, float(e[:, 1].max()) + pad],
        })
    return out


def pack_orientation(result) -> "dict | None":
    """Pack the whole result into ``{"header": …, "b64": …, "arrays": …}``, or
    None when it is empty / over :data:`MAX_EMBED_POSITIONS`.

    ``arrays`` is kept for the figure builder (which needs the same numbers to
    draw the initial cloud); only ``header`` and ``b64`` reach the page.
    """
    try:
        om = _as_orientation_map(result)
        ny, nx = (int(v) for v in om.nav_shape)
    except Exception as e:
        log.debug("[report] orientation result not packable: %s", e)
        return None
    m = ny * nx
    if m <= 0:
        return None
    if m > MAX_EMBED_POSITIONS:
        log.debug("[report] orientation embed refused: %d positions > cap %d",
                  m, MAX_EMBED_POSITIONS)
        return None

    phases = _phase_geometry(om)
    if not phases:
        return None
    try:
        phase = np.ascontiguousarray(
            np.clip(np.asarray(om.phase_map()).reshape(-1), 0, 255), dtype=np.uint8)
    except Exception as e:
        log.debug("[report] orientation phase map failed: %s", e)
        return None

    arrays: dict[str, tuple] = {}
    blocks = [phase.tobytes()]
    for d in DIRECTIONS:
        try:
            rgb, xy, xyz = _per_position_direction(om, d)
        except Exception as e:
            log.debug("[report] orientation direction %s failed: %s", d, e)
            return None
        arrays[d] = (rgb, xy, xyz)
        blocks.append(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        blocks.append(np.ascontiguousarray(xy, dtype=np.float32).tobytes())
        blocks.append(np.ascontiguousarray(xyz, dtype=np.float32).tobytes())

    # The cloud is uniformly strided, not sliced: a head-slice of a raster-order
    # scan is its top rows, which is a crop of the sample rather than a sample of
    # the orientations.
    stride = max(1, int(np.ceil(m / CLOUD_MAX)))
    header = {
        "nav": [ny, nx],
        "m": m,
        "dirs": list(DIRECTIONS),
        "phases": phases,
        "stride": stride,
    }
    return {
        "header": header,
        "b64": base64.b64encode(b"".join(blocks)).decode("ascii"),
        "arrays": arrays,
        "phase": phase,
    }


def _hex_colors(rgb: np.ndarray) -> list:
    """(N,3) uint8 → ``#rrggbb`` strings (anyplotlib's per-point colour form)."""
    a = np.asarray(rgb, dtype=np.uint8)
    return ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b in a]


# ── figure ────────────────────────────────────────────────────────────────────

def _build_figure(payload) -> "tuple[dict, str, str, str, str] | None":
    """Build the map | triangle | sphere figure → ``(state, map_id, xy_id,
    sphere_id, esm_text)``.

    Panel ids are read back out of the serialized state by KIND plus which
    widgets a panel carries, never by ordering — the state dict does not promise
    the order the axes were created in.
    """
    try:
        import anyplotlib as apl
        from anyplotlib import embed as apl_embed
    except Exception as e:
        log.debug("[report] anyplotlib unavailable for orientation embed: %s", e)
        return None

    hdr = payload["header"]
    ny, nx = hdr["nav"]
    stride = int(hdr["stride"])
    rgb, xy, xyz = payload["arrays"]["z"]        # the page opens on Z
    phase0 = hdr["phases"][0]

    # Panel widths follow content aspect (the vectors explorer's rule): the
    # panels share a height, so a width ratio IS an aspect. The triangle and the
    # sphere are both square; only the map varies, and it is clamped so a line
    # scan cannot demand an absurd figure.
    map_aspect = min(3.0, max(1 / 3.0, (nx / ny) if ny else 1.0))
    fig_w = int(round(FIG_PX * (map_aspect + 2.0))) + 24
    fig, axes = apl.subplots(1, 3, figsize=(fig_w, FIG_PX),
                             width_ratios=[map_aspect, 1.0, 1.0])
    arr = np.array(axes, dtype=object).ravel()
    ax_map, ax_xy, ax_sph = arr[0], arr[1], arr[2]

    # MAP — the IPF colour image + the crosshair that drives everything.
    p_map = ax_map.imshow(np.asarray(rgb, dtype=np.uint8).reshape(ny, nx, 3))
    p_map.add_widget("crosshair", color="#ffffff", cx=nx / 2, cy=ny / 2)

    # TRIANGLE — the cloud, the sector outline, then the pick marker. The marker
    # is added LAST so it draws on top of the cloud.
    finite = np.isfinite(xy).all(axis=1)
    cloud = np.flatnonzero(finite)[::stride]
    xyp = ax_xy.axes2d(xlim=tuple(phase0["xlim"]), ylim=tuple(phase0["ylim"]),
                       aspect="equal")
    xyp.scatter(xy[cloud, 0], xy[cloud, 1], s=POINT_SIZE_2D,
                c=_hex_colors(rgb[cloud]), edgecolors="rgba(0,0,0,0)")
    edges = np.asarray(phase0["edges"], dtype=float)
    xyp.plot(edges[:, 0], edges[:, 1], color="#cdd6f4", linewidth=1.0)
    for lbl in phase0["labels"]:
        try:
            xyp.text(lbl["x"], lbl["y"], lbl["text"], color="#a6adc8")
        except Exception as e:
            log.debug("[report] sector label failed: %s", e)
    seed = cloud[0] if len(cloud) else 0
    xyp.scatter([float(xy[seed, 0])], [float(xy[seed, 1])], s=13,
                c=["#ffffff"], edgecolors="#000000")

    # SPHERE — the same directions in 3-D, with the pick highlighted. Opened
    # AIMED at the cloud (``ipf_window._aim_at``): scatter3d's default camera
    # looks at a spot a cubic IPF cloud never reaches, so the panel would open
    # showing the sphere's blank back.
    sph = np.flatnonzero(np.isfinite(xyz).all(axis=1))[::stride]
    from spyde.actions.ipf_window import _aim_at
    az, el = _aim_at(xyz[sph])
    p3d = ax_sph.scatter3d(
        xyz[sph, 0], xyz[sph, 1], xyz[sph, 2],
        colors=np.clip(np.asarray(rgb[sph], dtype=np.float32) / 255.0, 0.0, 1.0),
        point_size=POINT_SIZE_3D, x_label="[100]", y_label="[010]",
        z_label="[001]", bounds=((-1.0, 1.0),) * 3, zoom=1.4, gpu=True,
        azimuth=az, elevation=el,
    )
    try:
        p3d.set_sphere(1.0)
    except Exception as e:
        log.debug("[report] IPF reference sphere failed: %s", e)
    if len(sph):
        p3d.set_highlight(float(xyz[seed, 0]), float(xyz[seed, 1]),
                          float(xyz[seed, 2]), color="#ffffff", size=11)

    state = apl_embed.figure_state(fig)

    map_id = xy_id = sph_id = None
    for k in state:
        if not (k.startswith("panel_") and k.endswith("_json")):
            continue
        pid = k[len("panel_"):-len("_json")]
        try:
            pj = json.loads(state[k])
        except Exception:
            continue
        kind = pj.get("kind")
        if kind == "3d":
            sph_id = pid
        elif kind == "2d":
            map_id = pid
        else:                       # PlotXY serialises as the 1-D panel kind
            xy_id = pid
    if map_id is None or xy_id is None or sph_id is None:
        log.debug("[report] could not identify orientation embed panels "
                  "(map=%s xy=%s sphere=%s)", map_id, xy_id, sph_id)
        return None
    return state, map_id, xy_id, sph_id, _esm_text()


# Dark theme, matching vectors_embed (Catppuccin Mocha) so the two explorers
# read as the same product inside one report.
_EXPLORER_CSS = """
:root { color-scheme: dark; }
body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       background: #1e1e2e; color: #cdd6f4; font-size: 12px; }
.ox-wrap { display: flex; flex-direction: column; gap: 8px; padding: 8px;
           align-items: flex-start; }
.ox-wrap h4 { margin: 0; font-size: 12px; font-weight: 600; color: #bac2de; }
.ox-controls { display: flex; gap: 12px; align-items: center; font-size: 11px; }
.ox-seg { display: inline-flex; align-items: center; gap: 6px; }
.ox-seg-label { color: #6c7086; }
.ox-seg-group { display: inline-flex; border: 1px solid #313244;
                border-radius: 999px; overflow: hidden; }
.ox-seg-btn { appearance: none; border: 0; padding: 3px 11px; cursor: pointer;
              background: #1e1e2e; color: #a6adc8; font: inherit; }
.ox-seg-btn[aria-pressed="true"] { background: #89b4fa; color: #11111b;
                                   font-weight: 600; }
.ox-status-dot { width: 7px; height: 7px; border-radius: 50%;
                 background: #a6e3a1; display: inline-block; }
.ox-meta { color: #6c7086; }
#ox-fig { background: #181825; border-radius: 6px; padding: 4px; }
"""

_EXPLORER_JS_TMPL = r"""
const HDR = JSON.parse(document.getElementById('ox-header').textContent);
const B64 = document.getElementById('ox-data').textContent.trim();
const STATE = JSON.parse(document.getElementById('ox-state').textContent);
const MAP_ID = document.getElementById('ox-mapid').textContent.trim();
const XY_ID = document.getElementById('ox-xyid').textContent.trim();
const SPH_ID = document.getElementById('ox-sphid').textContent.trim();
const ESM = document.getElementById('ox-esm').textContent;

const esmUrl = URL.createObjectURL(new Blob([ESM], { type: 'text/javascript' }));
const { mount } = await import(esmUrl);

const NY = HDR.nav[0], NX = HDR.nav[1], M = HDR.m, STRIDE = HDR.stride;
const DIRS = HDR.dirs;

const ab = await (await fetch('data:application/octet-stream;base64,' + B64))
  .arrayBuffer();
let off = 0;
const PHASE = new Uint8Array(ab, off, M); off += M;
// Per direction: rgb(uint8 M*3) | xy(float32 M*2) | xyz(float32 M*3). Float
// views need their byte offset 4-aligned; M*3 uint8 bytes need not be, so the
// float blocks are COPIED out rather than viewed in place. At ~20 B/position
// that is one allocation of the same size as the blob, once, at load.
const D = {};
for (const d of DIRS) {
  const rgb = new Uint8Array(ab, off, M * 3); off += M * 3;
  const xy = new Float32Array(ab.slice(off, off + 4 * M * 2)); off += 4 * M * 2;
  const xyz = new Float32Array(ab.slice(off, off + 4 * M * 3)); off += 4 * M * 3;
  D[d] = { rgb, xy, xyz };
}

let dir = 'z';                       // the direction the page was built for
let pick = 0;                        // flat nav index of the marked position
let H = null;

const mapKey = 'panel_' + MAP_ID + '_json';
const xyKey = 'panel_' + XY_ID + '_json';
const sphKey = 'panel_' + SPH_ID + '_json';

// Patch through the LIVE model state (handle.get), never the page-load parse:
// re-applying the stale load-time json resets the fitted view and collapses the
// panel to zero width (the vectors explorer learned this the hard way).
function live(key) {
  try { return JSON.parse(H.get(key)); }
  catch (e) { return JSON.parse(STATE[key]); }
}

// The cloud's flat nav indices, in the order the scatter was built: finite rows
// only, then strided. Rebuilt per direction because which rows are finite is a
// property of the direction's fold.
function cloudIndices(d) {
  const xy = D[d].xy, out = [];
  for (let i = 0; i < M; i++) {
    if (Number.isFinite(xy[2 * i]) && Number.isFinite(xy[2 * i + 1])) out.push(i);
  }
  const strided = [];
  for (let i = 0; i < out.length; i += STRIDE) strided.push(out[i]);
  return strided;
}

function hex(rgb, i) {
  const s = (v) => v.toString(16).padStart(2, '0');
  return '#' + s(rgb[3 * i]) + s(rgb[3 * i + 1]) + s(rgb[3 * i + 2]);
}

// The triangle panel's markers, in creation order: [0] the cloud, [1] the
// sector outline, then the corner labels, and LAST the pick marker. Identified
// by type + size rather than index so an added label cannot shift the lookup.
function markerGroups(pj) {
  const ms = pj.markers || [];
  const points = ms.filter((g) => g.type === 'points');
  return { cloud: points[0] || null, pick: points[points.length - 1] || null,
           all: ms };
}

function setPick(i) {
  pick = i;
  const { xy, xyz } = D[dir];
  const x = xy[2 * i], y = xy[2 * i + 1];
  if (Number.isFinite(x) && Number.isFinite(y)) {
    const pj = live(xyKey);
    const g = markerGroups(pj).pick;
    if (g) { g.offsets = [[x, y]]; H.applyUpdate(xyKey, JSON.stringify(pj)); }
  }
  const vx = xyz[3 * i], vy = xyz[3 * i + 1], vz = xyz[3 * i + 2];
  if (Number.isFinite(vx) && Number.isFinite(vy) && Number.isFinite(vz)) {
    const pj = live(sphKey);
    pj.highlight = { x: vx, y: vy, z: vz, color: '#ffffff', size: 11 };
    // Rotate the sphere to bring the picked orientation to the centre. This is
    // `ipf_window.face_camera` verbatim: el = asin(vz), az = atan2(vx, -vy).
    // Not "some angle that points that way" — atan2(vy, vx) - 90° is the same
    // direction 180° out, and lands the highlight on the sphere's far edge.
    // `_view_from_python` is what tells the renderer this camera is INTENDED;
    // without it the panel preserves the camera the viewer orbited to and the
    // rotation is silently dropped.
    const r = Math.hypot(vx, vy, vz) || 1;
    pj.azimuth = Math.atan2(vx, -vy) * 180 / Math.PI;
    pj.elevation = Math.asin(Math.max(-1, Math.min(1, vz / r))) * 180 / Math.PI;
    pj._view_from_python = true;
    H.applyUpdate(sphKey, JSON.stringify(pj));
  }
  readout();
}

function readout() {
  const iy = Math.floor(pick / NX), ix = pick % NX;
  const { xy } = D[dir];
  const x = xy[2 * pick], y = xy[2 * pick + 1];
  const el = document.getElementById('ox-readout');
  if (!el) return;
  el.textContent = Number.isFinite(x)
    ? `IPF-${dir.toUpperCase()} · scan (${ix}, ${iy}) · sector (${x.toFixed(3)}, `
      + `${y.toFixed(3)}) · phase ${PHASE[pick]}`
    : `IPF-${dir.toUpperCase()} · scan (${ix}, ${iy}) · no indexed orientation`;
}

function b64(bytes) {
  let s = '';
  const CH = 0x8000;                 // chunked: apply() blows the arg limit
  for (let i = 0; i < bytes.length; i += CH) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
  }
  return btoa(s);
}

// The 3-D panel's point cloud does NOT live in `panel_<id>_json` — anyplotlib
// hoists vertices and per-point colours into a separate `panel_<id>_geom` trait
// (see _applyGeom / _loadGeom in figure_esm.js) so a camera nudge never
// re-transmits the geometry. Writing `vertices` into the view json therefore
// changes nothing at all; it has to go through the geom channel, which has its
// own change observer that reloads and redraws.
const sphGeomKey = 'panel_' + SPH_ID + '_geom';
function liveGeom() {
  try { return JSON.parse(H.get(sphGeomKey)); }
  catch (e) { return JSON.parse(STATE[sphGeomKey] || '{}'); }
}

// Switching the sample direction re-colours all three panels from the packed
// arrays — the same "re-colour every view" the app's X/Y/Z buttons do, with no
// backend to ask.
function setDirection(d) {
  if (!D[d]) return;
  dir = d;
  const { rgb, xy, xyz } = D[d];
  const idx = cloudIndices(d);

  // TRIANGLE: point markers keep their offsets inline in the view json.
  const xj = live(xyKey);
  const g = markerGroups(xj).cloud;
  if (g) {
    g.offsets = idx.map((i) => [xy[2 * i], xy[2 * i + 1]]);
    g.facecolors = idx.map((i) => hex(rgb, i));
  }
  H.applyUpdate(xyKey, JSON.stringify(xj));

  // SPHERE: vertices float32 (N*3), point colours uint8 (N*3), z float32 (N) —
  // the exact wire dtypes the Python side encodes.
  const n = idx.length;
  const v = new Float32Array(n * 3);
  const c = new Uint8Array(n * 3);
  const z = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    v[3 * k] = xyz[3 * i]; v[3 * k + 1] = xyz[3 * i + 1]; v[3 * k + 2] = xyz[3 * i + 2];
    c[3 * k] = rgb[3 * i]; c[3 * k + 1] = rgb[3 * i + 1]; c[3 * k + 2] = rgb[3 * i + 2];
    z[k] = xyz[3 * i + 2];
  }
  const geom = liveGeom();
  geom.vertices_b64 = b64(new Uint8Array(v.buffer));
  geom.point_colors_b64 = b64(c);
  geom.z_values_b64 = b64(new Uint8Array(z.buffer));
  H.applyUpdate(sphGeomKey, JSON.stringify(geom));
  const sj = live(sphKey);
  sj.vertices_count = n;
  sj._geom_rev = (sj._geom_rev || 0) + 1;
  H.applyUpdate(sphKey, JSON.stringify(sj));

  pushMap(rgb);
  syncSeg();
  setPick(pick);
}

// The map image is repainted onto a PASS-THROUGH OVERLAY canvas over the map
// panel's image rect — the same primitive the vectors explorer's DP repaint
// uses, and for the same reason: pushing pixels back through the figure's own
// state keys is not debuggable in the standalone shim, while a drawImage on top
// is. anyplotlib keeps the crosshair, the axes and the theme underneath.
let ovMap = null;
function makeOverlay(panel, iw, ih) {
  const host = panel.plotCanvas.parentElement;
  const scale = Math.min(panel.imgW / iw, panel.imgH / ih);
  const w = iw * scale, h = ih * scale;
  const x = (panel.imgW - w) / 2, y = (panel.imgH - h) / 2;
  const c = document.createElement('canvas');
  c.width = iw; c.height = ih;
  c.style.cssText = 'position:absolute;pointer-events:none;z-index:2;'
    + 'image-rendering:pixelated;'
    + `left:${panel.plotCanvas.offsetLeft + x}px;`
    + `top:${panel.plotCanvas.offsetTop + y}px;`
    + `width:${w}px;height:${h}px;`;
  if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
  host.insertBefore(c, panel.plotCanvas.nextSibling);
  return c.getContext('2d');
}

function pushMap(rgb) {
  if (!ovMap) return;
  const img = ovMap.createImageData(NX, NY);
  const d = img.data;
  for (let i = 0; i < M; i++) {
    d[4 * i] = rgb[3 * i]; d[4 * i + 1] = rgb[3 * i + 1];
    d[4 * i + 2] = rgb[3 * i + 2]; d[4 * i + 3] = 255;
  }
  ovMap.putImageData(img, 0, 0);
}

// The crosshair on the MAP panel is the only interaction that changes the pick.
// Route by panel id: a stray `cx` from any other panel must not move it.
function onEvent(ev) {
  if (ev.event_type !== 'pointer_move' && ev.event_type !== 'pointer_up') return;
  if (String(ev.panel_id == null ? '' : ev.panel_id) !== MAP_ID) return;
  if (typeof ev.cx !== 'number') return;
  const ix = Math.max(0, Math.min(NX - 1, Math.round(ev.cx)));
  const iy = Math.max(0, Math.min(NY - 1, Math.round(ev.cy)));
  setPick(iy * NX + ix);
}

function syncSeg() {
  document.querySelectorAll('.ox-seg-btn').forEach((b) => {
    b.setAttribute('aria-pressed', String(b.dataset.dir === dir));
  });
}

H = mount(document.getElementById('ox-fig'), STATE, { onEvent: onEvent });
await new Promise((res) => requestAnimationFrame(() => setTimeout(res, 30)));
const mapPanel = H.api.panels.get(MAP_ID);
if (mapPanel) { ovMap = makeOverlay(mapPanel, NX, NY); pushMap(D[dir].rgb); }

document.querySelectorAll('.ox-seg-btn').forEach((b) => {
  b.addEventListener('click', () => setDirection(b.dataset.dir));
});
syncSeg();
setPick(Math.floor(M / 2));

// Test hook: the SAME code paths the crosshair and the direction buttons call,
// reachable without synthesising widget drags. `state()` is what the browser
// spec asserts on — where the pick is, where its marker landed, and where the
// camera is now pointing.
window.__ox = {
  setPick(iy, ix) { setPick(iy * NX + ix); },
  setDirection,
  state() {
    const sj = live(sphKey);
    const g = markerGroups(live(xyKey)).pick;
    return {
      dir, pick,
      iy: Math.floor(pick / NX), ix: pick % NX,
      marker: g ? g.offsets[0] : null,
      highlight: sj.highlight || null,
      azimuth: sj.azimuth, elevation: sj.elevation,
    };
  },
  // Where nav (iy, ix) is on SCREEN, so a test can press exactly on the
  // crosshair instead of guessing at a fraction of the figure. Mirrors
  // anyplotlib's own image→canvas mapping: the overlay canvas IS the image
  // rect, and image coordinates are pixel CENTRES.
  navToPage(iy, ix) {
    const p = H.api.panels.get(MAP_ID);
    if (!p || !p.overlayCanvas) return null;
    const r = p.overlayCanvas.getBoundingClientRect();
    return {
      x: r.left + ((ix + 0.5) / NX) * r.width,
      y: r.top + ((iy + 0.5) / NY) * r.height,
    };
  },
  _h: () => ({ H, MAP_ID, XY_ID, SPH_ID }),
};
document.getElementById('ox-root').dataset.ready = '1';

// AUTO-FIT: the figure is built at a fixed native width (~3 panels wide). The
// HTML export's article iframe fits it; the report SIDEBAR cell (~390 px) does
// not, and anyplotlib's own cell-scaling does not engage in a standalone embed.
// So scale the mounted figure down with a CSS transform (never up past 1). The
// overlay canvas lives inside that element and scales in lockstep.
function fitFigure() {
  const host = document.getElementById('ox-fig');
  const outer = host && host.firstElementChild;
  if (!outer) return;
  outer.style.transformOrigin = 'top left';
  const natural = outer.offsetWidth || 1;
  const avail = (document.documentElement.clientWidth || natural) - 24;
  const s = Math.min(1, avail / natural);
  outer.style.transform = s < 1 ? `scale(${s})` : '';
  outer.style.marginBottom = s < 1 ? `${-(1 - s) * outer.offsetHeight}px` : '';
}
fitFigure();
window.addEventListener('resize', fitFigure);
"""


def orientation_explorer_html(result, caption: str = "",
                              cache_key: "str | None" = None) -> "str | None":
    """The self-contained IPF explorer page for one orientation result, or None
    when it is empty / over the embed cap.

    ``cache_key`` (a cell id) memoizes the built page per ``(cell_id, result
    identity)``: a rebuild for the same cell and the same result reuses the
    packed blob and the serialized figure instead of re-running orix over every
    position three times, which is the expensive part here.
    """
    if cache_key is not None:
        hit = _EXPLORER_CACHE.get(cache_key)
        if hit is not None and hit[0] is result:
            return hit[1]

    payload = pack_orientation(result)
    if payload is None:
        return None
    built = _build_figure(payload)
    if built is None:
        return None
    state, map_id, xy_id, sph_id, esm = built
    cap = _html.escape(caption or "")

    def _json_script(el_id: str, obj) -> str:
        # </script> cannot appear inside a script element — escape the slash.
        txt = json.dumps(obj).replace("</", "<\\/")
        return f"<script type=\"application/json\" id=\"{el_id}\">{txt}</script>"

    esm_safe = esm.replace("</script>", "<\\/script>")
    seg = "".join(
        f"<button type=\"button\" class=\"ox-seg-btn\" data-dir=\"{d}\" "
        f"data-testid=\"ox-dir-{d}\" aria-pressed=\"{str(d == 'z').lower()}\">"
        f"{d.upper()}</button>"
        for d in DIRECTIONS
    )
    page = (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<style>{_EXPLORER_CSS}</style></head><body>"
        "<div id=\"ox-root\" class=\"ox-wrap\">"
        "<h4>Orientation map — the crosshair marks that pixel's orientation "
        "on the triangle and turns the sphere to face it</h4>"
        "<div id=\"ox-fig\"></div>"
        "<div class=\"ox-controls\">"
        "<span class=\"ox-status-dot\" title=\"live\"></span>"
        "<span class=\"ox-seg\"><span class=\"ox-seg-label\">direction</span>"
        f"<span class=\"ox-seg-group\" role=\"group\">{seg}</span></span>"
        "</div>"
        "<div id=\"ox-readout\" class=\"ox-meta\"></div>"
        f"<div class=\"ox-meta\">{cap}</div>"
        "</div>"
        f"{_json_script('ox-header', payload['header'])}"
        f"<script type=\"application/json\" id=\"ox-data\">{payload['b64']}</script>"
        f"{_json_script('ox-state', state)}"
        f"<script type=\"text/plain\" id=\"ox-mapid\">{map_id}</script>"
        f"<script type=\"text/plain\" id=\"ox-xyid\">{xy_id}</script>"
        f"<script type=\"text/plain\" id=\"ox-sphid\">{sph_id}</script>"
        f"<script type=\"text/plain\" id=\"ox-esm\">{esm_safe}</script>"
        f"<script type=\"module\">{_EXPLORER_JS_TMPL}</script>"
        "</body></html>"
    )
    if cache_key is not None:
        _EXPLORER_CACHE[cache_key] = (result, page)
    return page


def orientation_for_cell(session, cell) -> "object | None":
    """The orientation result behind a figure cell's base layer, or None.

    Resolution goes through the cell spec's SignalRef → live plot → tree, then
    the SAME chain the live IPF window uses (``ipf_view.tree_orientation_result``)
    so a cell dragged from a raw-OM, vector-OM or EBSD result all resolve.
    """
    spec = getattr(cell, "spec", None)
    if spec is None or not spec.panels:
        return None
    try:
        layers = spec.panels[0].layers
        if not layers:
            return None
        plot = layers[0].source.resolve(session)
        if plot is None:
            return None
        from spyde.actions.ipf_view import tree_orientation_result
        return tree_orientation_result(getattr(plot, "signal_tree", None))
    except Exception as e:
        log.debug("[report] orientation resolve for cell failed: %s", e)
        return None
