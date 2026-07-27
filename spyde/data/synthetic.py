"""synthetic.py — bundled synthetic datasets for EELS, EDS and EBSD.

These exist so Waves 1-4 have something loadable per modality from day one,
before the real/downloadable datasets land (Wave 5, #80). They are also the
right thing to test against: the ground truth is *known exactly*, so a fit or
an indexing run is checked against the numbers the data was built from rather
than against a golden file that silently encodes yesterday's bug.

Design rules, inherited from the ``si_grains`` / ``movie`` test data:

* **Asymmetric.** Every spatial map is distinguishable from its own transpose
  and from both mirrors — a ramp along one axis, a block off-centre, a wedge.
  Symmetric test data hides exactly the bugs this data exists to catch.
* **Crisp.** Real peaks at real energies, real Kikuchi geometry. A fit that
  works here should work on an experiment; a detector that finds nothing here
  is broken.
* **Small and eager by default.** A Playwright spec must load one in well under
  a second. Callers that want scale pass bigger shapes explicitly.
* **Ground truth in metadata** under ``Spyde.synthetic``.

Nothing here imports torch, exspy or kikuchipy — generators are plain numpy so
they work in every environment, including CI without the optional extras.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _concentration_maps(ny: int, nx: int) -> dict[str, np.ndarray]:
    """Three deliberately asymmetric composition maps in [0, 1].

    Each is distinguishable from its transpose and from both mirrors, so a
    quantification map that comes out flipped is obvious on screen:

    ``a`` left-to-right ramp plus a block in the TOP-LEFT quadrant
    ``b`` a wedge occupying the lower-right triangle, fading downward
    ``c`` whatever is left over, so the three always sum to 1
    """
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    fy, fx = yy / max(ny - 1, 1), xx / max(nx - 1, 1)

    a = 0.15 + 0.55 * fx                      # ramps along x only
    a[(yy < ny // 3) & (xx < nx // 3)] = 0.85  # TOP-LEFT block

    b = np.where(fx + fy > 1.0, 0.20 + 0.50 * fy, 0.05).astype(np.float32)

    total = a + b
    over = total > 0.95                        # keep room for the third element
    a[over] *= 0.95 / total[over]
    b[over] *= 0.95 / total[over]
    return {"a": a, "b": b, "c": 1.0 - a - b}


def _stamp(sig, **truth) -> None:
    """Record the parameters the data was generated from, so tests assert
    against them instead of a golden file.

    Values stay as numpy arrays rather than nested lists: a 256x256
    concentration map is 65k floats, and metadata holds an array far more
    cheaply than a list-of-lists. Read it back with :func:`ground_truth`.
    """
    try:
        sig.metadata.set_item("Spyde.synthetic", dict(truth))
    except Exception as e:                                    # pragma: no cover
        log.debug("stamping synthetic ground truth failed: %s", e)


def ground_truth(sig) -> dict:
    """The generator parameters stamped on a synthetic signal, as plain Python.

    HyperSpy turns nested metadata dicts into ``DictionaryTreeBrowser`` nodes,
    which have no ``.keys()``/``.values()`` and compare unhelpfully — so read
    ground truth through here, not straight off ``metadata.Spyde.synthetic``.
    """
    def _plain(v):
        if hasattr(v, "as_dictionary"):
            return {k: _plain(x) for k, x in v.as_dictionary().items()}
        if isinstance(v, dict):
            return {k: _plain(x) for k, x in v.items()}
        return v

    node = sig.metadata.get_item("Spyde.synthetic", None)
    if node is None:
        raise ValueError(f"{sig!r} carries no synthetic ground truth — it did "
                         "not come from spyde.data.synthetic")
    return _plain(node)


def _try_signal_type(sig, signal_type: str) -> None:
    """``set_signal_type`` needs the matching HyperSpy extension (exspy for
    EELS/EDS, kikuchipy for EBSD). Those are OPTIONAL extras, so a plain
    install must still get usable data — it just stays a generic Signal1D/2D.

    HyperSpy logs a multi-line WARNING pointing at the extensions list when the
    type is unknown. That is expected here and alarming to a user who simply
    has not installed an extra, so it is demoted to a debug line.
    """
    import logging as _logging
    hs_io = _logging.getLogger("hyperspy.io")
    prev = hs_io.level
    hs_io.setLevel(_logging.ERROR)
    try:
        sig.set_signal_type(signal_type)
        if type(sig).__name__ in ("Signal1D", "Signal2D"):
            log.debug("signal_type %r unavailable (optional extra not "
                      "installed); left as %s", signal_type, type(sig).__name__)
    except Exception as e:
        log.debug("set_signal_type(%s) failed: %s", signal_type, e)
    finally:
        hs_io.setLevel(prev)


# ---------------------------------------------------------------------------
# EELS
# ---------------------------------------------------------------------------

# Real core-loss onsets (eV). C-K and O-K sit in the same window with a decade
# of background between them, which is what makes background modelling matter.
EELS_EDGES = {"C_K": 284.0, "N_K": 401.0, "O_K": 532.0}


def _hydrogenic_edge(e: np.ndarray, onset: float, resolution: float) -> np.ndarray:
    """A core-loss edge: sharp rise at the onset, then a ``E**-4`` style decay,
    smeared by the spectrometer resolution.

    Not a GOS table — the real shapes come from exspy in Wave 2 (#63). This is
    the right *shape* for exercising a fit: a step the background cannot mimic,
    followed by a tail that overlaps the next edge.
    """
    above = e >= onset
    shape = np.zeros_like(e)
    ratio = np.divide(e, onset, out=np.ones_like(e), where=above)
    shape[above] = ratio[above] ** -4.0
    # Spectrometer resolution: gaussian smear, width in channels.
    step = float(e[1] - e[0])
    sigma_ch = max(resolution / step, 0.8)
    half = int(np.ceil(3 * sigma_ch))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_ch) ** 2)
    k /= k.sum()
    return np.convolve(shape, k, mode="same")


def eels_si(nav=(16, 16), n_channels: int = 1024, *, e_start: float = 200.0,
            e_stop: float = 800.0, resolution: float = 1.2,
            counts: float = 3.0e5, seed: int = 0, noise: bool = True):
    """A synthetic EELS spectrum image: power-law background + three core-loss
    edges whose intensities follow the asymmetric composition maps.

    Returns a ``Signal1D`` with a calibrated eV signal axis. Ground truth (the
    per-element concentration maps and the onsets) is in
    ``metadata.Spyde.synthetic``, so a quantification result can be scored
    directly against the maps it should recover.

    The default 16x16 x 1024 is Playwright-sized. Pass a bigger *nav* for
    benchmarking — ``eels_si(nav=(256, 256))`` is the 11.6-minute ``multifit``
    case that Wave 1 exists to fix.
    """
    import hyperspy.api as hs

    ny, nx = int(nav[0]), int(nav[1])
    rng = np.random.default_rng(seed)
    e = np.linspace(e_start, e_stop, n_channels, dtype=np.float64)

    conc = _concentration_maps(ny, nx)
    names = list(EELS_EDGES)
    # Background thickness also varies across the scan (and along y, so it is
    # independent of every concentration map) — a background model that ignores
    # position gets this wrong in a way a flat background would hide.
    yy = np.mgrid[0:ny, 0:nx][0].astype(np.float32) / max(ny - 1, 1)
    bg_scale = (0.7 + 0.6 * yy).astype(np.float64)

    edges = {n: _hydrogenic_edge(e, EELS_EDGES[n], resolution) for n in names}
    background = (e / e_start) ** -3.0

    data = np.empty((ny, nx, n_channels), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            spec = bg_scale[iy, ix] * background
            for key, name in zip(("a", "b", "c"), names):
                spec = spec + 0.35 * conc[key][iy, ix] * edges[name]
            data[iy, ix] = spec
    data *= counts / data.max()
    if noise:
        data = rng.poisson(np.clip(data, 0, None)).astype(np.float32)

    s = hs.signals.Signal1D(data)
    ax = s.axes_manager.signal_axes[0]
    ax.name, ax.units = "Energy loss", "eV"
    ax.offset, ax.scale = e_start, float(e[1] - e[0])
    for i, a in enumerate(s.axes_manager.navigation_axes):
        a.name, a.units, a.scale = ("x", "y")[i], "nm", 2.0
    s.metadata.General.title = "Synthetic EELS SI"
    # The fits are meaningless without these (#61) — ship them with the data.
    s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200.0)
    s.metadata.set_item("Acquisition_instrument.TEM.convergence_angle", 10.0)
    s.metadata.set_item("Acquisition_instrument.TEM.Detector.EELS."
                        "collection_angle", 20.0)
    _try_signal_type(s, "EELS")
    _stamp(s, kind="eels", edges={n: EELS_EDGES[n] for n in names},
           elements=names,
           concentration={n: conc[k]
                          for k, n in zip(("a", "b", "c"), names)},
           background_exponent=3.0, resolution_ev=resolution)
    return s


# ---------------------------------------------------------------------------
# EDS
# ---------------------------------------------------------------------------

# Real characteristic line energies (keV) with in-family intensity ratios.
EDS_LINES = {
    "Fe": [("Ka", 6.404, 1.00), ("Kb", 7.058, 0.13)],
    "Ni": [("Ka", 7.478, 1.00), ("Kb", 8.265, 0.13)],
    "Cu": [("Ka", 8.048, 1.00), ("Kb", 8.905, 0.13)],
}


def _detector_sigma(e_kev: np.ndarray | float) -> np.ndarray | float:
    """Si(Li)/SDD resolution: ``sigma**2`` grows linearly with energy (Fano).
    Tuned to ~130 eV FWHM at Mn-Ka, i.e. a normal EDS detector."""
    return np.sqrt(1.4e-4 + 1.1e-3 * np.asarray(e_kev, float)) / 2.3548 * 2.3548


def eds_si(nav=(16, 16), n_channels: int = 2048, *, e_stop: float = 20.0,
           counts: float = 4.0e4, seed: int = 0, noise: bool = True):
    """A synthetic EDS spectrum image: bremsstrahlung background + K-family
    lines for Fe / Ni / Cu at their real energies.

    Fe-Kb (7.058) and Ni-Ka (7.478) overlap, and Ni-Kb (8.265) sits between
    Cu-Ka (8.048) and Cu-Kb (8.905) — deliberately, because resolving
    overlapping families is the thing EDS fitting has to get right. A peak
    finder that treats every maximum as one element fails here, as it should.

    Ground truth (concentrations, line energies, family ratios) is in
    ``metadata.Spyde.synthetic``.
    """
    import hyperspy.api as hs

    ny, nx = int(nav[0]), int(nav[1])
    rng = np.random.default_rng(seed)
    e = np.linspace(0.0, e_stop, n_channels, dtype=np.float64)
    scale = float(e[1] - e[0])

    conc = _concentration_maps(ny, nx)
    elements = list(EDS_LINES)

    # Bremsstrahlung: Kramers' law with detector absorption killing the low end.
    with np.errstate(divide="ignore", invalid="ignore"):
        brem = np.where(e > 0.15, (20.0 - e) / np.maximum(e, 1e-3), 0.0)
    brem = np.clip(brem, 0, None) * (1.0 - np.exp(-np.maximum(e, 0) / 1.5))
    brem /= brem.max()

    peaks = {}
    for el in elements:
        acc = np.zeros_like(e)
        for _name, energy, ratio in EDS_LINES[el]:
            sig = _detector_sigma(energy)
            acc += ratio * np.exp(-0.5 * ((e - energy) / sig) ** 2)
        peaks[el] = acc

    data = np.empty((ny, nx, n_channels), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            spec = 0.25 * brem
            for key, el in zip(("a", "b", "c"), elements):
                spec = spec + conc[key][iy, ix] * peaks[el]
            data[iy, ix] = spec
    data *= counts / data.max()
    if noise:
        data = rng.poisson(np.clip(data, 0, None)).astype(np.float32)

    s = hs.signals.Signal1D(data)
    ax = s.axes_manager.signal_axes[0]
    ax.name, ax.units = "Energy", "keV"
    ax.offset, ax.scale = 0.0, scale
    for i, a in enumerate(s.axes_manager.navigation_axes):
        a.name, a.units, a.scale = ("x", "y")[i], "nm", 5.0
    s.metadata.General.title = "Synthetic EDS SI"
    s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200.0)
    s.metadata.set_item("Acquisition_instrument.TEM.Detector.EDS."
                        "energy_resolution_MnKa", 130.0)
    s.metadata.set_item("Acquisition_instrument.TEM.Detector.EDS.live_time", 10.0)
    _try_signal_type(s, "EDS_TEM")
    _stamp(s, kind="eds", elements=elements,
           lines={el: EDS_LINES[el] for el in elements},
           concentration={el: conc[k]
                          for k, el in zip(("a", "b", "c"), elements)})
    return s


# ---------------------------------------------------------------------------
# EBSD
# ---------------------------------------------------------------------------

def _euler_to_matrix(phi1, Phi, phi2) -> np.ndarray:
    """Bunge ZXZ Euler angles -> rotation matrices, batched over the leading
    axes. Returns ``(..., 3, 3)``."""
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    m = np.empty(np.shape(phi1) + (3, 3), float)
    m[..., 0, 0] = c1 * c2 - s1 * s2 * c
    m[..., 0, 1] = s1 * c2 + c1 * s2 * c
    m[..., 0, 2] = s2 * s
    m[..., 1, 0] = -c1 * s2 - s1 * c2 * c
    m[..., 1, 1] = -s1 * s2 + c1 * c2 * c
    m[..., 1, 2] = c2 * s
    m[..., 2, 0] = s1 * s
    m[..., 2, 1] = -c1 * s
    m[..., 2, 2] = c
    return m


def _cubic_plane_normals() -> tuple[np.ndarray, np.ndarray]:
    """{111}, {200} and {220} plane normals for a cubic crystal, with rough
    structure-factor weights. One normal per Friedel pair (a band and its
    inverse are the same band)."""
    fams = [((1, 1, 1), 1.00), ((2, 0, 0), 0.70), ((2, 2, 0), 0.45)]
    normals, weights = [], []
    for hkl, w in fams:
        seen = set()
        h, k, l = hkl
        for perm in {(h, k, l), (h, l, k), (k, h, l), (k, l, h), (l, h, k), (l, k, h)}:
            for sx in (1, -1):
                for sy in (1, -1):
                    for sz in (1, -1):
                        v = (perm[0] * sx, perm[1] * sy, perm[2] * sz)
                        if v == (0, 0, 0) or tuple(-x for x in v) in seen:
                            continue
                        seen.add(v)
        for v in seen:
            n = np.array(v, float)
            normals.append(n / np.linalg.norm(n))
            # Band width goes as 1/|g|: bigger d-spacing -> wider band.
            weights.append(w / np.linalg.norm(np.array(v, float)))
    return np.array(normals), np.array(weights)


def detector_directions(detector=(60, 60), pc=(0.5, 0.5, 0.55)) -> np.ndarray:
    """Unit vectors from the sample to each detector pixel (gnomonic).

    ``(dy, dx, 3)``. Shared by the pattern generator and by dictionary
    simulation so an indexing test cannot pass by having both sides make the
    same geometry mistake in the same place — they use the identical geometry
    on purpose, and the thing under test is the MATCHING, not the projection.
    """
    pcx, pcy, L = float(pc[0]), float(pc[1]), float(pc[2])
    dy, dx = int(detector[0]), int(detector[1])
    gy, gx = np.mgrid[0:dy, 0:dx].astype(float)
    rx = (gx + 0.5) / dx - pcx
    ry = pcy - (gy + 0.5) / dy                    # detector y is flipped
    r = np.stack([rx, ry, np.full_like(rx, L)], -1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def simulate_patterns(euler, detector=(60, 60), pc=(0.5, 0.5, 0.55),
                      *, background: bool = False) -> np.ndarray:
    """Kikuchi patterns for a list of orientations -> ``(N, dy, dx)`` float32.

    *euler* is ``(N, 3)`` Bunge angles in radians. This is what builds an
    indexing DICTIONARY: sample orientation space, simulate each, match against
    it (#71).

    ``background=False`` by default because a dictionary is matched by
    normalised cross-correlation, which is invariant to the smooth gradient a
    real detector adds — and the experimental side has background removal
    applied before matching anyway (#70).
    """
    euler = np.atleast_2d(np.asarray(euler, float))
    rot = _euler_to_matrix(euler[:, 0], euler[:, 1], euler[:, 2])   # (N, 3, 3)
    r = detector_directions(detector, pc)
    dy, dx = r.shape[:2]
    flat_r = r.reshape(-1, 3)

    normals, weights = _cubic_plane_normals()
    widths = 0.055 * (weights / weights.max()) + 0.012

    out = np.empty((len(euler), dy, dx), np.float32)
    for i in range(len(euler)):
        n_rot = normals @ rot[i].T
        d = flat_r @ n_rot.T
        band = np.exp(-0.5 * (d / widths) ** 2) * weights
        out[i] = band.sum(1).reshape(dy, dx).astype(np.float32)
    if background:
        gy, gx = np.mgrid[0:dy, 0:dx].astype(np.float32)
        out = out / max(out.max(), 1e-9) + (
            0.35 + 0.30 * ((gy / dy) * 0.6 + (gx / dx) * 0.4))
    return out


def ebsd_patterns(nav=(16, 16), detector=(60, 60), *, pc=(0.5, 0.5, 0.55),
                  seed: int = 0, noise: float = 0.06):
    """Synthetic EBSD patterns with **exact known orientations**.

    Each nav pixel gets an orientation; the pattern is built by gnomonic
    projection of a cubic crystal's {111}/{200}/{220} Kikuchi bands onto a flat
    detector. A band appears where the detector direction is near-perpendicular
    to a plane normal, which is the real geometry — so a dictionary-indexing
    run (#71) or a refinement (#72) that recovers the stamped Euler angles is
    genuinely working, not matching a fixture to itself.

    The orientation field is **two grains plus a gradient**, arranged
    asymmetrically: a rotated wedge in the lower-right against a background
    grain whose orientation drifts along x. That gives an IPF map with a real
    boundary and an intra-grain gradient — the two things an orientation map
    has to show correctly.

    Returns a ``Signal2D`` (nav | detector). Ground-truth Euler angles are in
    ``metadata.Spyde.synthetic["euler"]`` as ``(ny, nx, 3)`` radians.
    """
    import hyperspy.api as hs

    ny, nx = int(nav[0]), int(nav[1])
    dy, dx = int(detector[0]), int(detector[1])
    rng = np.random.default_rng(seed)

    # --- orientation field ------------------------------------------------
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    fy, fx = yy / max(ny - 1, 1), xx / max(nx - 1, 1)
    phi1 = np.deg2rad(10.0 + 35.0 * fx)          # drifts along x only
    Phi = np.full((ny, nx), np.deg2rad(30.0))
    phi2 = np.full((ny, nx), np.deg2rad(15.0))
    grain2 = (fx + fy) > 1.15                     # lower-right wedge
    phi1[grain2], Phi[grain2], phi2[grain2] = (np.deg2rad(70.0),
                                               np.deg2rad(55.0),
                                               np.deg2rad(40.0))
    rot = _euler_to_matrix(phi1, Phi, phi2)       # (ny, nx, 3, 3)

    # --- detector directions (gnomonic) -----------------------------------
    pcx, pcy, L = float(pc[0]), float(pc[1]), float(pc[2])
    gy, gx = np.mgrid[0:dy, 0:dx].astype(float)
    rx = (gx + 0.5) / dx - pcx
    ry = pcy - (gy + 0.5) / dy                    # detector y is flipped
    r = np.stack([rx, ry, np.full_like(rx, L)], -1)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)      # (dy, dx, 3)

    normals, weights = _cubic_plane_normals()
    # Band half-width in units of cos(angle to the normal); scaled by |g| so
    # low-index planes give the wide bands, as in a real pattern.
    widths = 0.055 * (weights / weights.max()) + 0.012

    data = np.empty((ny, nx, dy, dx), dtype=np.float32)
    flat_r = r.reshape(-1, 3)
    for iy in range(ny):
        for ix in range(nx):
            # Rotate the plane normals into the sample frame for this pixel.
            n_rot = normals @ rot[iy, ix].T                 # (B, 3)
            d = flat_r @ n_rot.T                            # (dy*dx, B)
            band = np.exp(-0.5 * (d / widths) ** 2) * weights
            data[iy, ix] = band.sum(1).reshape(dy, dx).astype(np.float32)

    # Detector background: real patterns sit on a smooth, off-centre gradient
    # that background correction (#70) has to remove.
    bg = 0.35 + 0.30 * ((gy / dy) * 0.6 + (gx / dx) * 0.4)
    data = data / max(data.max(), 1e-9) + bg.astype(np.float32)
    if noise:
        data += rng.normal(0.0, noise, data.shape).astype(np.float32)
    data = np.clip(data, 0, None)
    data = (data / data.max() * 255.0).astype(np.uint8)

    s = hs.signals.Signal2D(data)
    for i, a in enumerate(s.axes_manager.navigation_axes):
        a.name, a.units, a.scale = ("x", "y")[i], "um", 0.5
    for i, a in enumerate(s.axes_manager.signal_axes):
        a.name = ("dx", "dy")[i]
    s.metadata.General.title = "Synthetic EBSD"
    _try_signal_type(s, "EBSD")
    _stamp(s, kind="ebsd", euler=np.stack([phi1, Phi, phi2], -1),
           pc=np.array([pcx, pcy, L]), n_bands=int(len(normals)),
           grain2_mask=grain2)
    return s
