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
# Atoms
# ---------------------------------------------------------------------------

def atom_lattice(grid=(8, 10), spacing: float = 16.0, *, sigma: float = 2.6,
                 dumbbell: float = 0.0, displacement: float = 0.0,
                 ellipticity: float = 0.0, noise: float = 0.01,
                 seed: int = 0):
    """A HAADF-like atomic-resolution image with **exactly known positions**.

    Atoms sit on a rectangular lattice of gaussians. Every optional feature
    exists so a downstream measurement has something real to recover:

    ``dumbbell``
        Split each site into a PAIR separated by this many pixels along x, for
        the dumbbell workflow. 0 leaves single atoms.
    ``displacement``
        Amplitude of a smooth sinusoidal displacement field, so a
        displacement/strain map is non-trivial rather than uniformly zero.
    ``ellipticity``
        Stretches sigma_x against sigma_y across the field, so an ellipticity
        map has a known gradient to reproduce.

    The grid is deliberately NON-SQUARE and the spacing along x and y is the
    same, so a transposed result is obvious rather than plausible.

    Ground truth (``metadata.Spyde.synthetic``, read with :func:`ground_truth`)
    carries ``positions`` as ``(N, 2)`` in ``(x, y)`` pixel order — atomap's
    convention, so a comparison needs no reindexing.
    """
    import hyperspy.api as hs

    ny, nx = int(grid[0]), int(grid[1])
    rng = np.random.default_rng(seed)
    margin = spacing
    h = int(round(margin * 2 + spacing * (ny - 1)))
    w = int(round(margin * 2 + spacing * (nx - 1)))

    gy, gx = np.mgrid[0:ny, 0:nx].astype(float)
    xs = margin + gx * spacing
    ys = margin + gy * spacing
    if displacement:
        xs = xs + displacement * np.sin(2 * np.pi * gy / max(ny - 1, 1))
        ys = ys + displacement * np.cos(2 * np.pi * gx / max(nx - 1, 1))

    sx = sigma * (1.0 + ellipticity * gx / max(nx - 1, 1))
    sy = np.full_like(sx, sigma)

    centres = []
    for iy in range(ny):
        for ix in range(nx):
            if dumbbell:
                centres.append((xs[iy, ix] - dumbbell / 2, ys[iy, ix],
                                sx[iy, ix], sy[iy, ix]))
                centres.append((xs[iy, ix] + dumbbell / 2, ys[iy, ix],
                                sx[iy, ix], sy[iy, ix]))
            else:
                centres.append((xs[iy, ix], ys[iy, ix], sx[iy, ix], sy[iy, ix]))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w), np.float32)
    for cx, cy, s_x, s_y in centres:
        img += np.exp(-0.5 * (((xx - cx) / s_x) ** 2 + ((yy - cy) / s_y) ** 2))
    img /= max(img.max(), 1e-9)
    if noise:
        img = img + rng.normal(0.0, noise, img.shape).astype(np.float32)

    s = hs.signals.Signal2D(np.clip(img, 0, None).astype(np.float32))
    for a in s.axes_manager.signal_axes:
        a.units, a.scale = "nm", 0.01
    s.metadata.General.title = "Synthetic atom lattice"
    _stamp(s, kind="atoms",
           positions=np.array([[c[0], c[1]] for c in centres], float),
           grid=(ny, nx), spacing=float(spacing), sigma=float(sigma),
           dumbbell=float(dumbbell), displacement=float(displacement),
           ellipticity=float(ellipticity))
    return s


# ---------------------------------------------------------------------------
# EBSD
# ---------------------------------------------------------------------------

# The projection, the band set and the pattern renderer live in
# `spyde.ebsd.bands` — the ONE place that geometry is written down. The
# generator, the dictionary simulator and the live band overlay all import it
# from there, so an indexing test cannot pass by having two sides make the same
# mistake in the same place, and the overlay's lines cannot drift away from the
# bands they are drawn on. Re-exported here under the names this module has
# always used. (`spyde.ebsd.bands` is numpy-only and imports nothing from
# spyde, so this cannot cycle.)
from spyde.ebsd.bands import (                                      # noqa: E402
    cubic_reflectors as _cubic_reflectors,
    detector_directions,
    euler_to_matrix as _euler_to_matrix,
    normals_to_sample as _normals_to_sample,
    simulate_patterns,
)


def _cubic_plane_normals() -> tuple[np.ndarray, np.ndarray]:
    """{111}, {200} and {220} plane normals for a cubic crystal, with rough
    structure-factor weights. One normal per Friedel pair (a band and its
    inverse are the same band)."""
    refl = _cubic_reflectors()
    return refl.normals, refl.weights


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
    gy, gx = np.mgrid[0:dy, 0:dx].astype(float)
    r = detector_directions(detector, pc)               # (dy, dx, 3)

    refl = _cubic_reflectors()
    normals, weights, widths = refl.normals, refl.weights, refl.widths

    data = np.empty((ny, nx, dy, dx), dtype=np.float32)
    flat_r = r.reshape(-1, 3)
    for iy in range(ny):
        for ix in range(nx):
            # Rotate the plane normals into the sample frame for this pixel
            # (bands.normals_to_sample owns the direction and why it matters).
            n_rot = _normals_to_sample(normals, rot[iy, ix])   # (B, 3)
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
           pc=np.asarray(pc, float), n_bands=int(len(normals)),
           grain2_mask=grain2)
    return s


# ---------------------------------------------------------------------------
# In-situ particle movie
# ---------------------------------------------------------------------------

# The particle table. Hand-written rather than randomised so every number in
# the ground truth is exact and readable: a test that says "particle 4
# dissolves at frame 16" can be checked against this table by eye.
#
# Columns: y0, x0, radius, amp, birth, death, vy, vx, faint
#   y0/x0   position at t=0 in the SAMPLE frame, pixels
#   birth   first frame the particle exists (0 = present from the start)
#   death   first frame it is GONE (-1 = never dissolves)
#   vy/vx   sample-frame velocity, px/frame
#   faint   amplitude is scaled down to `faint_amplitude` -- the Section 0.9 probes
_PARTICLES: tuple[dict, ...] = (
    # two bright anchors, static and persistent
    dict(y0=24.0, x0=22.0, radius=7.0, amp=1.00, birth=0, death=-1, vy=0.0, vx=0.0, faint=False),
    dict(y0=70.0, x0=30.0, radius=9.0, amp=0.90, birth=0, death=-1, vy=0.0, vx=0.0, faint=False),
    # a mover, for trails and displacement
    dict(y0=30.0, x0=80.0, radius=6.0, amp=0.85, birth=0, death=-1, vy=1.10, vx=-0.70, faint=False),
    # nucleation
    dict(y0=60.0, x0=94.0, radius=5.0, amp=0.80, birth=8, death=-1, vy=0.0, vx=0.0, faint=False),
    # dissolution
    dict(y0=16.0, x0=58.0, radius=6.0, amp=0.75, birth=0, death=16, vy=0.0, vx=0.0, faint=False),
    # the merge pair: converge along x until they overlap
    dict(y0=84.0, x0=60.0, radius=5.0, amp=0.80, birth=0, death=-1, vy=0.0, vx=0.45, faint=False),
    dict(y0=84.0, x0=82.0, radius=5.0, amp=0.80, birth=0, death=-1, vy=0.0, vx=-0.45, faint=False),
    # faint, low-contrast -- these are what plan Section 0.9 is about
    dict(y0=46.0, x0=102.0, radius=4.0, amp=1.00, birth=0, death=-1, vy=0.0, vx=0.0, faint=True),
    dict(y0=88.0, x0=14.0, radius=3.0, amp=0.85, birth=0, death=-1, vy=0.0, vx=0.0, faint=True),
)

#: The two particles that merge, plus the nucleating and dissolving ones.
MERGE_PAIR = (5, 6)
NUCLEATION_INDEX = 3
DISSOLUTION_INDEX = 4


def _drift_curve(n_frames: int, amplitude: float) -> np.ndarray:
    """``(n_frames, 2)`` per-frame CORRECTION, matching ``DriftModel``'s sign.

    ``drift[t]`` is what you ADD to frame *t* to align it, so the sample appears
    displaced by ``-drift[t]`` in the raw frame.

    The two axes get deliberately DIFFERENT shapes -- y grows monotonically with
    a slight curve, x swings negative and comes back. A swapped or negated axis
    therefore shows up as a wrong-shaped curve rather than merely a wrong
    number, which is the whole point of an asymmetric fixture.
    """
    if n_frames < 2:
        return np.zeros((max(1, n_frames), 2), np.float64)
    t = np.arange(n_frames, dtype=np.float64) / (n_frames - 1)
    dy = amplitude * t ** 1.3
    dx = -0.6 * amplitude * np.sin(1.7 * np.pi * t)
    return np.stack([dy, dx], axis=-1)


def _soft_disc(yy, xx, cy, cx, radius, edge=0.9):
    """A disc with a smooth ~1 px edge, so sub-pixel centroids are meaningful.

    A hard-edged disc quantises its own centroid to the pixel grid, which would
    make any sub-pixel tracking assertion against this fixture untestable.
    """
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return 0.5 * (1.0 - np.tanh((r - radius) / edge))


def _particle_arrays(drift: np.ndarray, faint_amplitude: float) -> dict:
    """The per-particle arrays, in ONE place so the renderer and the stamped
    ground truth cannot disagree about the motion model."""
    return dict(
        drift=drift,
        p_y0=np.array([p["y0"] for p in _PARTICLES]),
        p_x0=np.array([p["x0"] for p in _PARTICLES]),
        p_radius=np.array([p["radius"] for p in _PARTICLES]),
        p_amp=np.array([(faint_amplitude * p["amp"]) if p["faint"] else p["amp"]
                        for p in _PARTICLES]),
        p_birth=np.array([p["birth"] for p in _PARTICLES]),
        p_death=np.array([p["death"] for p in _PARTICLES]),
        p_vy=np.array([p["vy"] for p in _PARTICLES]),
        p_vx=np.array([p["vx"] for p in _PARTICLES]),
        p_faint=np.array([p["faint"] for p in _PARTICLES]),
        n_particles=len(_PARTICLES),
    )


def particle_truth_at(truth: dict, t: int):
    """Expected ``(positions, radii, present)`` at frame *t*.

    ``positions`` is ``(N, 2)`` ``(y, x)`` in **pixels, in the LAB frame** --
    what a segmenter looking at the RAW frame should find, drift included.
    ``present`` is a boolean mask of which particles exist in that frame.

    Every consumer computes its expectations through here rather than
    re-deriving the motion model, so a test cannot pass by repeating the same
    mistake the generator made.
    """
    t = int(t)
    y0 = np.asarray(truth["p_y0"], float)
    x0 = np.asarray(truth["p_x0"], float)
    vy = np.asarray(truth["p_vy"], float)
    vx = np.asarray(truth["p_vx"], float)
    birth = np.asarray(truth["p_birth"], int)
    death = np.asarray(truth["p_death"], int)
    drift = np.asarray(truth["drift"], float)

    sample = np.stack([y0 + vy * t, x0 + vx * t], axis=-1)
    lab = sample - drift[t]                    # see _drift_curve on the sign
    present = (t >= birth) & ((death < 0) | (t < death))
    return lab, np.asarray(truth["p_radius"], float), present


def _merge_frame(arrays: dict, n_frames: int) -> int:
    """First frame where the merge pair's discs overlap.

    Derived from the same motion model that draws them rather than hard-coded,
    so the stamped truth stays correct if the table or the frame count changes.
    Returns -1 if they never touch within the movie.
    """
    a, b = MERGE_PAIR
    ra, rb = _PARTICLES[a]["radius"], _PARTICLES[b]["radius"]
    for t in range(n_frames):
        pos, _, present = particle_truth_at(arrays, t)
        if present[a] and present[b] and np.hypot(*(pos[a] - pos[b])) <= ra + rb:
            return t
    return -1


def particle_movie(n_frames: int = 24, shape=(96, 112), *,
                   drift_amplitude: float = 6.0,
                   faint_amplitude: float = 0.11,
                   scale: float = 0.5, seed: int = 0, noise: float = 0.015):
    """An in-situ particle movie whose every event is known exactly.

    A 1-D navigation (time) axis over 2-D images: nine particles on a drifting
    support film, with one nucleation, one dissolution, one merge, one mover and
    two deliberately faint low-contrast probes.

    Everything a downstream test needs to assert is stamped as ground truth --
    read it with :func:`ground_truth` and evaluate the motion model with
    :func:`particle_truth_at`.

    Parameters
    ----------
    n_frames
        Number of time points. The nucleation (frame 8) and dissolution
        (frame 16) frames are fixed, so keep this above ~20 for both to occur.
    shape
        ``(ny, nx)``. **Deliberately non-square** so a transposed frame is
        obvious at a glance.
    drift_amplitude
        Peak per-frame drift correction, pixels. The support film drifts
        rigidly; particles move relative to it.
    faint_amplitude
        Peak amplitude of the two faint probes. The default puts them around
        7x the noise sigma -- findable, but not by a threshold tuned for the
        bright particles. This is the plan's Section 0.9 gate.
    scale
        Pixel size in nm, written onto both signal axes.
    seed
        Everything random here (film speckle, noise) comes from this.
    noise
        Gaussian sigma added last. 0 disables.

    Notes
    -----
    **Why there is a speckled support film.** Drift is only recoverable if
    something static dominates the correlation. Particles move, appear and
    vanish, so they cannot serve; a smooth gradient has too little
    high-frequency content to correlate sharply. A rigidly-drifting speckle
    field is both physically right and what lets
    :func:`spyde.drift.solve_translation` recover ``drift`` from this movie.

    The film is generated once on a padded canvas and sampled per frame with
    bilinear interpolation, so there are no edge artifacts at any drift.
    Particles are drawn **analytically** at their lab positions, so their
    centroids stay exact regardless of how the film was resampled.
    """
    import hyperspy.api as hs
    from scipy.ndimage import gaussian_filter, map_coordinates

    ny, nx = int(shape[0]), int(shape[1])
    n_frames = int(n_frames)
    rng = np.random.default_rng(seed)
    drift = _drift_curve(n_frames, float(drift_amplitude))
    arrays = _particle_arrays(drift, float(faint_amplitude))

    # The support film, on a canvas padded to cover every drift.
    pad = int(np.ceil(np.abs(drift).max())) + 3
    fy, fx = ny + 2 * pad, nx + 2 * pad
    yy_p, _ = np.mgrid[0:fy, 0:fx]
    film = 0.10 + 0.12 * (yy_p / fy)           # ramp along y ONLY (asymmetric)
    film += 0.09 * gaussian_filter(rng.standard_normal((fy, fx)), 1.6)

    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    frames = np.empty((n_frames, ny, nx), dtype=np.float32)

    for t in range(n_frames):
        dy, dx = drift[t]
        # Sample the film at the lab-frame offset: content sits at -drift.
        frame = map_coordinates(film, [yy + pad - dy, xx + pad - dx],
                                order=1, mode="nearest")
        pos, radii, present = particle_truth_at(arrays, t)
        for i, p in enumerate(_PARTICLES):
            if not present[i]:
                continue
            amp = (faint_amplitude * p["amp"]) if p["faint"] else p["amp"]
            frame = frame + amp * _soft_disc(yy, xx, pos[i, 0], pos[i, 1], radii[i])
        frames[t] = frame

    if noise:
        frames += (noise * rng.standard_normal(frames.shape)).astype(np.float32)

    s = hs.signals.Signal2D(frames)
    for ax in s.axes_manager.signal_axes:
        ax.scale, ax.units = float(scale), "nm"
    tax = s.axes_manager.navigation_axes[0]
    tax.name, tax.units, tax.scale = "time", "s", 0.05
    s.metadata.General.title = "Synthetic particle movie"
    # `insitu` is registered by SpyDE's OWN hyperspy extension, so unlike the
    # EELS/EBSD generators this cannot silently fail for a missing extra.
    _try_signal_type(s, "insitu")
    _stamp(s, kind="particle_movie", **arrays,
           n_frames=n_frames, frame_shape=np.asarray([ny, nx]),
           scale=float(scale), noise=float(noise),
           faint_amplitude=float(faint_amplitude),
           merge_pair=np.asarray(MERGE_PAIR),
           nucleation_index=NUCLEATION_INDEX,
           dissolution_index=DISSOLUTION_INDEX,
           nucleation_frame=int(_PARTICLES[NUCLEATION_INDEX]["birth"]),
           dissolution_frame=int(_PARTICLES[DISSOLUTION_INDEX]["death"]),
           merge_frame=_merge_frame(arrays, n_frames))
    return s
