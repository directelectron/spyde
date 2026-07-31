"""The Kikuchi band overlay's detector convention, pinned against **kikuchipy**.

``test_ebsd_bands`` proves the overlay is self-consistent: the lines land on the
bands ``spyde.ebsd.bands`` itself renders. That is necessary and not sufficient
— the simulator, the dictionary and the overlay all share this one module, so a
mirrored, transposed or anisotropically sheared projection would make all three
agree with each other and disagree with the microscope. Nothing in that file can
see it, and on a near-symmetric cubic pattern the eye cannot either.

So this file pins the convention against an OUTSIDE authority: kikuchipy, the
reference implementation for EBSD detector geometry. The numbers below were
captured from it (see ``TestAgainstLiveKikuchipy`` for the regeneration recipe,
which re-derives them from scratch whenever kikuchipy is installed) with the
detector configured ``sample_tilt=90, tilt=0, azimuthal=0, twist=90`` — which
makes kikuchipy's sample-to-detector matrix exactly the identity, i.e. the same
untilted geometry ``spyde.ebsd.bands`` assumes. The plane normals are therefore
given directly in the DETECTOR frame and handed to :func:`band_lines` with the
identity orientation, which isolates the projection and the pixel convention
from the (separately pinned, see ``TestConventionAgreesWithOrix``) rotation.

The one convention difference that survives is pixel INDEXING, and it is
kikuchipy's: it maps the gnomonic bounds onto pixel indices ``0 … N-1``, i.e. it
reads an index as a detector EDGE, while SpyDE reads it as a pixel CENTRE
(``-0.5 … N-0.5``, which is what anyplotlib's ``transform="data"`` markers draw
in). That is the exact affine :func:`_to_kikuchipy_pixels` below, and once it is
applied the two agree to ~1e-6 px — not "close", identical. Which is why the
tolerance here is 1e-3 px: every wrong variant (x-flip, y-flip, transpose,
dropping the aspect ratio) misses by tens of pixels, four orders of magnitude
clear of the noise floor.

The **non-square** detector is the load-bearing case. On a square one an
aspect-ratio error is invisible, and a 60x60 detector is all the bundled
synthetic EBSD data has ever used.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.ebsd.bands import (
    Reflectors, band_lines, detector_directions, zone_axis_points,
)

# ── captured kikuchipy reference ─────────────────────────────────────────────
# Ni (Fm-3m, a=3.52 A), reflectors from min_dspacing=1.2 A at 20 kV, orientation
# Bunge (33.3, 30.0, 15.0) deg -- deliberately off any zone axis, so the band
# pattern is visibly handed and a mirror cannot hide in the symmetry.
EULER_DEG = (33.3, 30.0, 15.0)

NONSQUARE_SHAPE = (60, 80)
NONSQUARE_PC = (0.42, 0.63, 0.55)
NONSQUARE_NORMALS = np.array([
    [0.68426782, 0.71765641, 0.12940952],
    [-0.67558940, 0.55706892, 0.48296291],
    [0.27451141, -0.41790368, 0.86602540],
    [0.00613657, 0.90136693, 0.43301270],
    [-0.96156426, -0.11355250, 0.25000000],
    [-0.28974154, -0.80296224, 0.52086608],
    [0.67182273, -0.68940974, 0.27086608],
    [0.67795929, 0.21195719, 0.70387879],
    [0.30989498, 0.89102021, 0.33173498],
    [-0.91027940, 0.17731189, 0.37410146],
    [-0.48150030, 0.31136538, 0.81927350],
    [-0.48926251, -0.82878362, 0.27155094],
    [0.73479298, 0.45499920, 0.50304573],
    [-0.29825166, 0.81920330, 0.48984886],
])
NONSQUARE_KP_LINES = np.array([   # x0, y0, x1, y1 in kikuchipy detector pixels
    [72.907504, 80.740836, -12.417097, -0.270812],
    [82.864984, 5.301644, 11.229856, 91.811103],
    [-12.452639, -0.227851, 16.835424, -19.385336],
    [90.061142, 53.144454, -23.914291, 52.371775],
    [34.675549, -21.661826, 48.396438, 94.036611],
    [-12.421727, -0.265221, 92.279744, 37.355930],
    [-15.033473, 71.205829, 68.594260, -9.944681],
    [17.129181, 93.808700, -12.410840, -0.278367],
    [84.176569, 66.913095, -25.345274, 28.982349],
    [57.099614, -16.645163, 35.066489, 95.990683],
    [92.043504, 31.910006, 52.512879, 92.782848],
    [-12.419573, -0.267822, 88.128060, 58.838820],
    [46.521610, 94.501509, -12.414221, -0.274285],
    [92.245564, 35.160149, -13.357472, 73.445377],
])

SQUARE_SHAPE = (60, 60)
SQUARE_PC = (0.42, 0.63, 0.55)
SQUARE_NORMALS = np.array([
    [0.68426782, 0.71765641, 0.12940952],
    [-0.67558940, 0.55706892, 0.48296291],
    [0.00613657, 0.90136693, 0.43301270],
    [-0.96156426, -0.11355250, 0.25000000],
    [-0.28974154, -0.80296224, 0.52086608],
    [0.67182273, -0.68940974, 0.27086608],
    [0.67795929, 0.21195719, 0.70387879],
    [0.30989498, 0.89102021, 0.33173498],
    [-0.91027940, 0.17731189, 0.37410146],
    [-0.48150030, 0.31136538, 0.81927350],
    [-0.48926251, -0.82878362, 0.27155094],
    [0.73479298, 0.45499920, 0.50304573],
    [-0.29825166, 0.81920330, 0.48984886],
    [-0.91416051, -0.39276261, 0.10024019],
])
SQUARE_KP_LINES = np.array([
    [58.294843, 74.977044, -14.579671, 5.492980],
    [68.646726, 12.103535, 8.531089, 85.009212],
    [72.731233, 53.085292, -23.383483, 52.430937],
    [27.257567, -13.292649, 38.943905, 85.667434],
    [-12.222049, 2.768464, 75.223112, 34.322245],
    [-17.180785, 65.310892, 53.995577, -4.049744],
    [5.717602, 83.959340, -17.539170, 9.570993],
    [67.518137, 64.115667, -25.455075, 31.779777],
    [46.958584, -8.225239, 28.298633, 87.570759],
    [74.613718, 45.489727, 52.812723, 79.203127],
    [-13.352941, 4.026432, 72.221983, 54.544566],
    [33.391789, 86.954078, -15.948313, 7.273147],
    [75.293275, 38.183091, -13.258044, 70.422436],
    [7.881387, -10.443592, 47.686129, 82.202505],
])

CASES = [
    pytest.param(SQUARE_SHAPE, SQUARE_PC, SQUARE_NORMALS, SQUARE_KP_LINES,
                 id="square-60x60"),
    pytest.param(NONSQUARE_SHAPE, NONSQUARE_PC, NONSQUARE_NORMALS,
                 NONSQUARE_KP_LINES, id="nonsquare-60x80"),
]

#: How close SpyDE and kikuchipy have to be, in detector pixels. They agree to
#: ~1e-6 px; a flip / transpose / dropped aspect ratio misses by 10-100 px.
TOL_PX = 1e-3


# ── helpers ──────────────────────────────────────────────────────────────────
def _to_kikuchipy_pixels(seg, shape):
    """SpyDE pixel-CENTRE coordinates -> kikuchipy's pixel-INDEX ones.

    SpyDE spans the detector over ``-0.5 … N-0.5`` (pixel centres on integers,
    the convention anyplotlib's ``transform="data"`` markers draw in and the one
    ``simulate_patterns`` renders in). kikuchipy divides the gnomonic bounds by
    ``x_scale = (x_max-x_min)/(N-1)``, so its indices run ``0 … N-1`` over the
    same physical span. Hence the ``(N-1)/N`` shrink on top of the half-pixel
    shift. This is a pure relabelling of the axis — it cannot hide a flip, a
    transpose or a scale error, all of which are odd or anisotropic.
    """
    dy, dx = int(shape[0]), int(shape[1])
    return (np.asarray(seg, float) + 0.5) * np.array([(dx - 1) / dx,
                                                      (dy - 1) / dy])


def _line_through(p0, p1):
    """Normalised ``(n, c)`` with ``n·p + c = 0``."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    n = np.array([-d[1], d[0]])
    n = n / max(np.linalg.norm(n), 1e-30)
    return n, -float(n @ np.asarray(p0, float))


def _spyde_segment(normal, shape, pc):
    """The band-centre segment SpyDE draws for one DETECTOR-frame plane normal.

    Identity Euler angles, so ``normals_to_sample`` is a no-op and the normal
    reaches the projection untouched.
    """
    refl = Reflectors(np.asarray(normal, float).reshape(1, 3),
                      np.array([1.0]), np.array([0.05]))
    segs, _w = band_lines(np.zeros(3), refl, shape, pc)
    return None if len(segs) == 0 else np.asarray(segs[0], float)


def _max_distance_to_kp_line(seg_kp_px, kp_line):
    """Largest perpendicular distance from a segment's endpoints to kikuchipy's
    line. Compared as LINES, not endpoint-to-endpoint: kikuchipy draws its
    plane traces out to a fixed gnomonic radius while SpyDE clips to the
    detector, so the segments coincide but their ends do not.
    """
    n, c = _line_through(kp_line[:2], kp_line[2:])
    return float(np.abs(np.asarray(seg_kp_px) @ n + c).max())


def _pairs(shape, pc, normals, kp_lines):
    """(normal, spyde segment, kikuchipy line) for every band SpyDE draws.

    kikuchipy's in-bounds test is looser than a clip to the detector box, so a
    few of its lines miss the frame entirely; those have no SpyDE counterpart.
    """
    out = []
    for n, kl in zip(normals, kp_lines):
        seg = _spyde_segment(n, shape, pc)
        if seg is not None:
            out.append((n, seg, kl))
    return out


# ── the tests ────────────────────────────────────────────────────────────────
class TestMatchesKikuchipy:
    """SpyDE's drawn band lines ARE kikuchipy's, to a millionth of a pixel."""

    @pytest.mark.parametrize("shape,pc,normals,kp_lines", CASES)
    def test_band_lines_match(self, shape, pc, normals, kp_lines):
        pairs = _pairs(shape, pc, normals, kp_lines)
        assert len(pairs) >= 8, "too few comparable bands to prove anything"
        worst = max(_max_distance_to_kp_line(_to_kikuchipy_pixels(seg, shape), kl)
                    for _n, seg, kl in pairs)
        assert worst < TOL_PX, (
            f"SpyDE's band lines are up to {worst:.4f} px from kikuchipy's on a "
            f"{shape} detector — the overlay's projection disagrees with the "
            f"reference implementation")

    @pytest.mark.parametrize("shape,pc,normals,kp_lines", CASES)
    def test_line_directions_match(self, shape, pc, normals, kp_lines):
        """Position and DIRECTION separately: a sheared projection can keep a
        line's distance from the centre while rotating it (that is exactly what
        dropping the aspect ratio does — up to 8 degrees on a 4:3 detector)."""
        worst = 0.0
        for _n, seg, kl in _pairs(shape, pc, normals, kp_lines):
            q = _to_kikuchipy_pixels(seg, shape)
            d_s = q[1] - q[0]
            d_k = np.array([kl[2] - kl[0], kl[3] - kl[1]])
            d_s = d_s / np.linalg.norm(d_s)
            d_k = d_k / np.linalg.norm(d_k)
            worst = max(worst, float(np.degrees(
                np.arccos(np.clip(abs(d_s @ d_k), 0, 1)))))
        assert worst < 0.01, (
            f"band lines are rotated by up to {worst:.4f} deg relative to "
            f"kikuchipy's on a {shape} detector")


class TestTheTestHasTeeth:
    """Every way the overlay could plausibly be wrong, and the margin by which
    :class:`TestMatchesKikuchipy` rejects it. Without this the parity test could
    be passing on a symmetry rather than on the geometry."""

    #: name -> transform of a SpyDE segment already in kikuchipy pixels
    VARIANTS = {
        "flip-x": lambda p, dy, dx: np.column_stack([dx - 1 - p[:, 0], p[:, 1]]),
        "flip-y": lambda p, dy, dx: np.column_stack([p[:, 0], dy - 1 - p[:, 1]]),
        "flip-both": lambda p, dy, dx: np.column_stack([dx - 1 - p[:, 0],
                                                        dy - 1 - p[:, 1]]),
        "transpose": lambda p, dy, dx: np.column_stack([p[:, 1], p[:, 0]]),
    }

    @pytest.mark.parametrize("name", list(VARIANTS))
    @pytest.mark.parametrize("shape,pc,normals,kp_lines", CASES)
    def test_a_flip_or_transpose_would_be_rejected(self, name, shape, pc,
                                                   normals, kp_lines):
        dy, dx = shape
        fn = self.VARIANTS[name]
        worst = max(
            _max_distance_to_kp_line(fn(_to_kikuchipy_pixels(seg, shape), dy, dx), kl)
            for _n, seg, kl in _pairs(shape, pc, normals, kp_lines))
        assert worst > 5.0, (
            f"a {name} of the overlay is only {worst:.3f} px from the correct "
            f"answer on a {shape} detector — this parity test cannot see it")

    def test_dropping_the_aspect_ratio_would_be_rejected(self):
        """The bug this file was written for: normalising x by the detector
        WIDTH and y by its HEIGHT (rather than both by the height) assumes
        non-square pixels and shears every band. Exactly zero effect on a square
        detector, which is why it survived — so it is checked on the 4:3 one.
        """
        dy, dx = NONSQUARE_SHAPE
        pcx, pcy, L = NONSQUARE_PC
        aspect = dx / dy
        worst_px, worst_deg = 0.0, 0.0
        for n, _seg, kl in _pairs(NONSQUARE_SHAPE, NONSQUARE_PC,
                                  NONSQUARE_NORMALS, NONSQUARE_KP_LINES):
            # the pre-fix closed form: a = nx/dx (not nx/dy), no aspect on pcx
            a, b = n[0] / dx, -n[1] / dy
            c = L * n[2] - pcx * n[0] + pcy * n[1]
            # two points on that line, in SpyDE pixel-centre coordinates
            if abs(b) > abs(a):
                us = np.array([0.0, float(dx)])
                pts = np.column_stack([us, -(c + a * us) / b]) - 0.5
            else:
                vs = np.array([0.0, float(dy)])
                pts = np.column_stack([-(c + b * vs) / a, vs]) - 0.5
            q = _to_kikuchipy_pixels(pts, NONSQUARE_SHAPE)
            worst_px = max(worst_px, _max_distance_to_kp_line(q, kl))
            d_s = q[1] - q[0]
            d_k = np.array([kl[2] - kl[0], kl[3] - kl[1]])
            d_s = d_s / np.linalg.norm(d_s)
            d_k = d_k / np.linalg.norm(d_k)
            worst_deg = max(worst_deg, float(np.degrees(
                np.arccos(np.clip(abs(d_s @ d_k), 0, 1)))))
        assert worst_px > 5.0 and worst_deg > 1.0, (
            f"the width/height-normalised projection is only {worst_px:.3f} px "
            f"/ {worst_deg:.3f} deg from kikuchipy's — this test cannot see the "
            f"regression it exists to catch")
        assert aspect != 1.0


class TestBrukerConventionHoldsEverywhere:
    """The same convention, stated as an equation, applied to the other two
    entry points. ``band_lines`` is what the overlay draws, but the DICTIONARY
    is rendered through :func:`detector_directions` and the zone-axis markers
    through :func:`zone_axis_points`; all three have to mean the same detector
    or the overlay drifts off the very bands it was matched against.

    For a detector pixel whose CENTRE is at plot coordinates ``(x, y)``, the
    Bruker gnomonic coordinates are::

        x_g = ((x + 0.5)/dx - pcx) * (dx/dy) / pcz
        y_g = (pcy - (y + 0.5)/dy) / pcz

    and a band of plane normal ``n`` passes through it when
    ``nx*x_g + ny*y_g + nz = 0``.
    """

    @staticmethod
    def _gnomonic(x, y, shape, pc):
        dy, dx = int(shape[0]), int(shape[1])
        pcx, pcy, L = (float(v) for v in pc)
        x_g = ((np.asarray(x, float) + 0.5) / dx - pcx) * (dx / dy) / L
        y_g = (pcy - (np.asarray(y, float) + 0.5) / dy) / L
        return x_g, y_g

    @pytest.mark.parametrize("shape,pc,normals,kp_lines", CASES)
    def test_band_line_pixels_satisfy_the_relation(self, shape, pc, normals,
                                                   kp_lines):
        for n, seg, _kl in _pairs(shape, pc, normals, kp_lines):
            x_g, y_g = self._gnomonic(seg[:, 0], seg[:, 1], shape, pc)
            resid = np.abs(n[0] * x_g + n[1] * y_g + n[2])
            assert resid.max() < 1e-5, (
                f"a drawn band pixel is not on the plane {np.round(n, 4)} "
                f"(residual {resid.max():.2e})")

    @pytest.mark.parametrize("shape,pc", [(SQUARE_SHAPE, SQUARE_PC),
                                          (NONSQUARE_SHAPE, NONSQUARE_PC)])
    def test_detector_directions_satisfy_the_relation(self, shape, pc):
        """The simulator's rays and the overlay's lines must be the same
        detector — this is the link that makes the overlay meaningful."""
        r = detector_directions(shape, pc)
        dy, dx = shape
        gy, gx = np.mgrid[0:dy, 0:dx]
        x_g, y_g = self._gnomonic(gx, gy, shape, pc)
        # r is the unit ray, so (x_g, y_g, 1) is parallel to it
        want = np.stack([x_g, y_g, np.ones_like(x_g)], -1)
        want = want / np.linalg.norm(want, axis=-1, keepdims=True)
        assert np.allclose(r, want, atol=1e-12)

    @pytest.mark.parametrize("shape,pc", [(SQUARE_SHAPE, SQUARE_PC),
                                          (NONSQUARE_SHAPE, NONSQUARE_PC)])
    def test_zone_axes_satisfy_the_relation(self, shape, pc):
        """A zone axis is drawn where the crystal direction pierces the
        detector, so its gnomonic coordinates must be ``(tx/tz, ty/tz)``."""
        euler = np.deg2rad(EULER_DEG)
        za = zone_axis_points(euler, detector=shape, pc=pc, max_axes=12)
        assert len(za) >= 1, "no zone axes drawn — nothing tested"
        segs, _w = band_lines(euler, detector=shape, pc=pc)
        assert len(segs) >= 3
        # Every drawn zone axis lies on at least two drawn band lines, and it
        # does so in the SAME gnomonic frame the bands use.
        for p in za:
            d = sorted(abs(float((p - s[0]) @ _line_through(s[0], s[1])[0]))
                       for s in segs)
            assert d[1] < 1.0, f"zone axis {p} lies on fewer than 2 bands"

    def test_a_square_detector_is_unaffected_by_the_aspect_ratio(self):
        """Carrying the aspect ratio must be a NO-OP where ``dx == dy`` — the
        bundled synthetic EBSD scan, every existing test and every PC stamped on
        stored data live on a square detector, and none of them may move."""
        shape, pc = SQUARE_SHAPE, SQUARE_PC
        dy, dx = shape
        pcx, pcy, L = pc
        assert dx == dy, "this test is only meaningful on a square detector"
        for n, seg, _kl in _pairs(shape, pc, SQUARE_NORMALS, SQUARE_KP_LINES):
            # the PRE-FIX closed form: a = nx/dx, and no aspect factor on pcx
            a, b = n[0] / dx, -n[1] / dy
            c = L * n[2] - pcx * n[0] + pcy * n[1]
            # every pixel the current code draws still satisfies it exactly
            u, v = seg[:, 0] + 0.5, seg[:, 1] + 0.5
            assert np.abs(a * u + b * v + c).max() < 1e-6, (
                "the aspect-ratio fix moved a band line on a SQUARE detector")


class TestAgainstLiveKikuchipy:
    """Regenerate the reference from kikuchipy itself, when it is installed.

    kikuchipy is an OPTIONAL extra (``spyde[ebsd]``), so the captured numbers
    above are what runs in CI. This is how they were produced and how to check
    they are still what kikuchipy says.
    """

    def test_reference_still_matches_kikuchipy(self):
        kp = pytest.importorskip("kikuchipy")
        pytest.importorskip("diffsims")
        from diffpy.structure import Atom, Lattice, Structure
        from diffsims.crystallography import ReciprocalLatticeVector
        from orix.crystal_map import Phase
        from orix.quaternion import Rotation

        phase = Phase(name="ni", space_group=225,
                      structure=Structure(atoms=[Atom("Ni", [0, 0, 0])],
                                          lattice=Lattice(3.52, 3.52, 3.52,
                                                          90, 90, 90)))
        rlv = ReciprocalLatticeVector.from_min_dspacing(phase, 1.2)
        rlv = rlv.unique(use_symmetry=True).symmetrise()
        rlv.sanitise_phase()
        rlv.calculate_structure_factor()
        rlv.calculate_theta(20e3)
        sim = kp.simulations.KikuchiPatternSimulator(rlv)
        rot = Rotation.from_euler(np.deg2rad([EULER_DEG]))

        for shape, pc, normals, kp_lines in [
                (SQUARE_SHAPE, SQUARE_PC, SQUARE_NORMALS, SQUARE_KP_LINES),
                (NONSQUARE_SHAPE, NONSQUARE_PC, NONSQUARE_NORMALS,
                 NONSQUARE_KP_LINES)]:
            # sample_tilt=90 + twist=90 makes sample_to_detector the identity,
            # i.e. kikuchipy's detector frame IS spyde.ebsd.bands' frame.
            det = kp.detectors.EBSDDetector(shape=shape, pc=pc, sample_tilt=90.0,
                                            tilt=0.0, azimuthal=0.0, twist=90.0)
            assert np.allclose(det.sample_to_detector.to_matrix().squeeze(),
                               np.eye(3)), "kikuchipy detector frame moved"
            geo = sim.on_detector(det, rot)
            n_live = np.asarray(geo._lines.vector_detector.data).reshape(-1, 3)
            n_live = n_live / np.linalg.norm(n_live, axis=1, keepdims=True)
            c_live = geo.lines_coordinates(coordinates="pixel", exclude_nan=False)

            for n, kl in zip(normals, kp_lines):
                j = int(np.argmax(np.abs(n_live @ n)))
                assert abs(abs(float(n_live[j] @ n)) - 1) < 1e-6, \
                    "captured normal is no longer in kikuchipy's reflector set"
                assert np.allclose(c_live[j], kl, atol=1e-4), (
                    f"kikuchipy's own line for {np.round(n, 4)} on {shape} has "
                    f"moved: {np.round(c_live[j], 4)} vs captured "
                    f"{np.round(kl, 4)}")
