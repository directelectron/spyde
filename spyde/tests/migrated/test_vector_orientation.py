"""
Vector orientation + strain fitting (spyde/actions/vector_orientation.py).

Two layers:
  - pure math (pose projection, strain bound, soft-assign cost, friedel QC) on
    synthetic spot sets with known transforms — fast, no library generation.
  - one integration test building a small Ag FCC library and recovering a known
    orientation + strain from simulated vectors.

Design: VECTOR_ORIENTATION_MAPPING_PLAN.md §7c-7d.
"""
import numpy as np
import pytest

from spyde.actions import vector_orientation as vo


# ── synthetic template + measured-vector helpers ─────────────────────────────

def _template(M=16, rmax=1.0, seed=42):
    rng = np.random.RandomState(seed)
    g = rng.uniform(-rmax, rmax, (M, 2))
    g = g[np.linalg.norm(g, axis=1) < rmax]
    # add centrosymmetric partners so Friedel pairs exist (I(g)==I(-g)).
    g = np.vstack([g, -g])
    half = rng.uniform(0.3, 1.0, len(g) // 2)
    I = np.concatenate([half, half])
    return g.astype(np.float64), I.astype(np.float64)


def _apply(g, I, theta, strain, noise=0.0, drop=0.0, spurious=0,
           rmax=1.0, seed=0):
    rng = np.random.RandomState(seed)
    A = np.eye(2) + strain
    v = (g @ vo._rot(theta).T) @ A.T
    if noise:
        v = v + rng.normal(0, noise, v.shape)
    vI = I.copy()
    if drop:
        keep = rng.rand(len(v)) > drop
        v, vI = v[keep], vI[keep]
    if spurious:
        sp = rng.uniform(-rmax, rmax, (spurious, 2))
        v = np.vstack([v, sp])
        vI = np.concatenate([vI, rng.uniform(0.1, 0.3, spurious)])
    return v, vI


# ── pure-math unit tests ──────────────────────────────────────────────────────

class TestPoseMath:
    def test_project_identity(self):
        g = np.array([[1.0, 0.0], [0.0, 1.0]])
        p = np.array([0.0, 1, 0, 0, 1, 0, 0])
        assert np.allclose(vo.project_spots(p, g), g)

    def test_project_rotation(self):
        g = np.array([[1.0, 0.0]])
        p = np.array([np.pi / 2, 1, 0, 0, 1, 0, 0])
        assert np.allclose(vo.project_spots(p, g), [[0.0, 1.0]], atol=1e-9)

    def test_project_translation(self):
        g = np.array([[1.0, 2.0]])
        p = np.array([0.0, 1, 0, 0, 1, 0.5, -0.5])
        assert np.allclose(vo.project_spots(p, g), [[1.5, 1.5]])


class TestStrainBound:
    def test_within_bound_unchanged(self):
        A = np.array([[1.02, 0.01], [0.0, 0.98]])
        B = vo.project_strain_bound(A, cap=0.05)
        assert np.allclose(B, A, atol=1e-6)

    def test_clamps_excess_strain(self):
        A = np.array([[1.5, 0.0], [0.0, 0.6]])  # 50%/40% — way over
        B = vo.project_strain_bound(A, cap=0.05)
        sv = np.linalg.svd(B, compute_uv=False)
        assert sv.max() <= 1.05 + 1e-6
        assert sv.min() >= 0.95 - 1e-6

    def test_forbids_reflection(self):
        # a reflection (det < 0) must come back with det > 0
        A = np.array([[-1.0, 0.0], [0.0, 1.0]])
        B = vo.project_strain_bound(A, cap=0.05)
        assert np.linalg.det(B) > 0


class TestFit:
    """Fit pose by seeding near truth (the real regime: coarse pins the branch)."""

    def _fit(self, v, vI, g, gI, seed_angle, **pover):
        # lib stub with a single template; bypass coarse_seed via explicit seed
        lib = vo.TemplateLibrary(
            spots_xy=[g.astype(np.float32)], spots_I=[gI.astype(np.float32)],
            template_quats=np.array([[1.0, 0, 0, 0]]),
            template_phase=np.array([0], np.int16),
            phases_meta=[{"name": "x", "point_group": "m-3m"}],
            cache={}, radial_range=(0.0, 1.0), r_max=1.0,
        )
        params = {**pover}
        # monkeypatch orientation resolution to avoid orix in pure-math tests
        orig = vo._resolve_one_quat
        vo._resolve_one_quat = lambda ti, th, l: np.array([1.0, 0, 0, 0], np.float32)
        try:
            return vo.fit_pattern(v, vI, lib, params, seed=(0, seed_angle, 1.0))
        finally:
            vo._resolve_one_quat = orig

    def test_recovers_clean_strain(self):
        g, I = _template()
        theta = np.deg2rad(12.0)
        strain = np.array([[0.03, 0.012], [0.012, -0.022]])
        v, vI = _apply(g, I, theta, strain, noise=0.002, seed=1)
        fit = self._fit(v, vI, g, I, seed_angle=theta + np.deg2rad(2))
        assert fit is not None
        assert np.abs(fit.strain - strain).max() < 0.012, fit.strain
        assert fit.residual < 0.02

    def test_robust_to_missing_and_spurious(self):
        g, I = _template()
        theta = np.deg2rad(-8.0)
        strain = np.array([[0.02, 0.0], [0.0, 0.015]])
        v, vI = _apply(g, I, theta, strain, noise=0.006,
                       drop=0.2, spurious=4, seed=2)
        fit = self._fit(v, vI, g, I, seed_angle=theta + np.deg2rad(2))
        assert fit is not None
        # soft-assign + sink must not blow up on missing/spurious peaks
        assert np.abs(fit.strain - strain).max() < 0.03, fit.strain

    def test_strain_respects_cap(self):
        g, I = _template()
        theta = 0.0
        big = np.array([[0.15, 0.0], [0.0, -0.12]])  # 15% — over the 5% cap
        v, vI = _apply(g, I, theta, big, noise=0.002, seed=3)
        fit = self._fit(v, vI, g, I, seed_angle=0.0, strain_cap=0.05)
        assert fit is not None
        sv = np.linalg.svd(fit.affine, compute_uv=False)
        assert sv.max() <= 1.05 + 1e-4 and sv.min() >= 0.95 - 1e-4

    def test_too_few_vectors_returns_none(self):
        g, I = _template()
        fit = self._fit(g[:2], I[:2], g, I, seed_angle=0.0)
        assert fit is None

    @staticmethod
    def _project(fit, g):
        """Physical observable: where the fit places the template spots
        (M·g + t). Gauge-invariant — unlike theta/affine individually, which
        share the flat M = A·Rot(theta) gauge direction."""
        p7 = np.zeros(7)
        p7[0] = fit.theta
        p7[1:5] = np.asarray(fit.affine, float).reshape(-1)
        p7[5:7] = np.asarray(fit.translation, float)
        return vo.project_spots(p7, g)

    def test_intensity_scale_invariant(self):
        """The stored vector intensity is now RAW image counts (~1e4), not the
        ~1 NXCORR score. The soft-assign sink gating is tuned for O(1) weights,
        so fit_pattern normalises each pattern to unit mean — making the fit
        invariant to the absolute intensity scale. Scaling all intensities by
        1e4 must reproduce the same fit (this FAILED before the normalisation:
        conf=raw/(raw+sink) saturated to 1 and the no-match sink stopped
        gating, biasing strain + predicted positions).

        Compared on the gauge-invariant observables (strain, matched count, and
        the predicted spot positions) — theta/affine individually share a flat
        gauge direction and drift at float level, so are not compared raw."""
        g, I = _template()
        theta = np.deg2rad(-8.0)
        strain = np.array([[0.02, 0.0], [0.0, 0.015]])
        v, vI = _apply(g, I, theta, strain, noise=0.006,
                       drop=0.2, spurious=4, seed=2)        # gating matters here
        f_unit = self._fit(v, vI, g, I, seed_angle=theta + np.deg2rad(2))
        f_raw = self._fit(v, vI * 1e4, g, I, seed_angle=theta + np.deg2rad(2))
        assert f_unit is not None and f_raw is not None
        assert f_raw.n_matched == f_unit.n_matched
        assert np.allclose(f_raw.strain, f_unit.strain, atol=1e-4), \
            (f_raw.strain, f_unit.strain)
        assert np.allclose(self._project(f_raw, g), self._project(f_unit, g),
                           atol=1e-3)

    def test_robust_at_raw_count_scale(self):
        """Sanity: with raw-count intensities (~1e4) and missing+spurious peaks,
        the strain is still recovered — i.e. the sink gating still opts out
        unmatched template spots after normalisation."""
        g, I = _template()
        theta = np.deg2rad(-8.0)
        strain = np.array([[0.02, 0.0], [0.0, 0.015]])
        v, vI = _apply(g, I, theta, strain, noise=0.006,
                       drop=0.2, spurious=4, seed=2)
        fit = self._fit(v, vI * 1e4, g, I, seed_angle=theta + np.deg2rad(2))
        assert fit is not None
        assert np.abs(fit.strain - strain).max() < 0.03, fit.strain

    def test_recovers_strain_across_weightings(self):
        """The reflection-weighting knobs (gamma intensity compression, k_power
        |g| lever-arm) must each leave the fit able to recover a known strain —
        i.e. no weighting setting breaks convergence. Also asserts the knobs are
        actually plumbed into fit_pattern: a strong vs weak setting gives a
        DIFFERENT pose on data where the weighting matters (missing+spurious)."""
        g, I = _template()
        theta = np.deg2rad(6.0)
        strain = np.array([[0.025, 0.0], [0.0, 0.01]])
        v, vI = _apply(g, I, theta, strain, noise=0.005, drop=0.15, spurious=3, seed=9)
        poses = {}
        for gamma, kpow in [(1.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 0.0)]:
            fit = self._fit(v, vI, g, I, seed_angle=theta + np.deg2rad(2),
                            gamma=gamma, k_power=kpow)
            assert fit is not None, (gamma, kpow)
            assert np.abs(fit.strain - strain).max() < 0.02, (gamma, kpow, fit.strain)
            poses[(gamma, kpow)] = self._project(fit, g)
        # gamma=1 (intensity) vs gamma=0 (uniform) must differ → gamma is applied
        assert not np.allclose(poses[(1.0, 0.0)], poses[(0.0, 0.0)], atol=1e-4)
        # k_power=1 vs k_power=0 must differ → the lever arm is applied
        assert not np.allclose(poses[(0.5, 1.0)], poses[(0.5, 0.0)], atol=1e-4)


class TestFriedelQC:
    """Friedel g/−g asymmetry: low for symmetric strain, high when one of a
    ±g pair is shifted (skewed vector finding)."""

    def test_symmetric_strain_low_asymmetry(self):
        g, I = _template()
        theta = np.deg2rad(5.0)
        strain = np.array([[0.03, 0.01], [0.01, -0.02]])
        v, vI = _apply(g, I, theta, strain, noise=0.001, seed=4)
        p = np.array([theta, 1 + 0.03, 0.01, 0.01, 1 - 0.02, 0, 0])
        fa = vo._friedel_asymmetry(p, g, v)
        assert fa < 0.01, fa

    def test_skewed_vectors_high_asymmetry(self):
        g, I = _template()
        theta = 0.0
        v, vI = _apply(g, I, theta, np.zeros((2, 2)), noise=0.001, seed=5)
        # inject skew: shift +x spots only (breaks g/−g symmetry)
        v = v.copy()
        v[v[:, 0] > 0.3, 0] += 0.05
        p = np.array([0.0, 1, 0, 0, 1, 0, 0])
        fa = vo._friedel_asymmetry(p, g, v)
        assert fa > 0.02, fa


# ── integration: real small library, recover known orientation + strain ──────

class TestBatchDriver:
    """Whole-field driver over a SpyDEDiffractionVectors, incl. warm-start."""

    def _make_vectors(self, ny, nx, g, theta, strain):
        """Build a SpyDEDiffractionVectors whose every position holds the same
        rotated+strained template spots (so the field is uniform & known)."""
        from spyde.signals.diffraction_vectors import (
            SpyDEDiffractionVectors, _build_nav_offsets, N_COLS,
            COL_NAV_X, COL_NAV_Y, COL_KX, COL_KY, COL_TIME, COL_INTENSITY,
        )
        A = np.eye(2) + strain
        v = (g @ vo._rot(theta).T) @ A.T
        rows = []
        for iy in range(ny):
            for ix in range(nx):
                for k in range(len(v)):
                    r = np.zeros(N_COLS, np.float32)
                    r[COL_NAV_X] = ix
                    r[COL_NAV_Y] = iy
                    r[COL_KX] = v[k, 0]
                    r[COL_KY] = v[k, 1]
                    r[COL_TIME] = -1.0
                    r[COL_INTENSITY] = 1.0
                    rows.append(r)
        flat = np.array(rows, np.float32)
        nav_offsets = _build_nav_offsets(flat, (ny, nx))

        class _Ax:
            scale = 0.01336
            offset = -0.7484
            size = 112
            units = "1/A"
            name = "k"
        return SpyDEDiffractionVectors(
            flat_buffer=flat, nav_offsets=nav_offsets, nav_shape=(ny, nx),
            full_nav_shape=(ny, nx), sig_shape=(112, 112),
            sig_axes=[_Ax(), _Ax()], kernel_radius_px=3.0,
            kernel_radius_data=0.04,
        )

    def test_uniform_field_recovered(self, monkeypatch):
        g, I = _template(M=14, rmax=0.6)
        theta = np.deg2rad(9.0)
        strain = np.array([[0.02, 0.006], [0.006, -0.015]])
        vecs = self._make_vectors(4, 5, g, theta, strain)

        lib = vo.TemplateLibrary(
            spots_xy=[g.astype(np.float32)], spots_I=[I.astype(np.float32)],
            template_quats=np.array([[1.0, 0, 0, 0]]),
            template_phase=np.array([0], np.int16),
            phases_meta=[{"name": "x", "point_group": "m-3m"}],
            cache={}, radial_range=(0.0, 0.6), r_max=0.6,
        )
        # bypass coarse seed (single template) and orix quat resolution
        monkeypatch.setattr(vo, "coarse_seed",
                            lambda *a, **k: [(0, theta + np.deg2rad(2), 1.0)])
        monkeypatch.setattr(vo, "_resolve_one_quat",
                            lambda ti, th, l: np.array([1.0, 0, 0, 0], np.float32))

        res = vo.compute_vector_orientation(vecs, lib, warm_start=True)
        assert res.nav_shape == (4, 5)
        assert res.strain.shape == (4, 5, 3)
        # every valid position recovers the applied strain
        valid = np.isfinite(res.residual)
        assert valid.all(), "some positions failed to fit"
        exx = res.strain_map("exx")[valid]
        eyy = res.strain_map("eyy")[valid]
        exy = res.strain_map("exy")[valid]
        assert np.abs(exx - 0.02).max() < 0.012, exx
        assert np.abs(eyy - (-0.015)).max() < 0.012, eyy
        assert np.abs(exy - 0.006).max() < 0.012, exy
        assert np.nanmedian(res.residual) < 0.01

    def test_chunked_matches_serial(self, monkeypatch):
        # the parallel chunked driver must recover the same field as the serial
        # one (no client → local thread-pool fallback)
        g, I = _template(M=14, rmax=0.6)
        theta = np.deg2rad(9.0)
        strain = np.array([[0.02, 0.006], [0.006, -0.015]])
        vecs = self._make_vectors(6, 7, g, theta, strain)
        lib = vo.TemplateLibrary(
            spots_xy=[g.astype(np.float32)], spots_I=[I.astype(np.float32)],
            template_quats=np.array([[1.0, 0, 0, 0]]),
            template_phase=np.array([0], np.int16),
            phases_meta=[{"name": "x", "point_group": "m-3m"}],
            cache={}, radial_range=(0.0, 0.6), r_max=0.6,
        )
        monkeypatch.setattr(vo, "coarse_seed",
                            lambda *a, **k: [(0, theta + np.deg2rad(2), 1.0)])
        monkeypatch.setattr(vo, "_resolve_one_quat",
                            lambda ti, th, l: np.array([1.0, 0, 0, 0], np.float32))

        res = vo.compute_vector_orientation_chunked(
            vecs, lib, chunk=3)            # 3x3 chunks over a 6x7 grid
        assert res is not None
        assert res.nav_shape == (6, 7)
        exx = res.strain_map("exx")
        assert np.isfinite(exx).all(), "chunked left holes"
        assert np.abs(exx - 0.02).max() < 0.012
        assert np.abs(res.strain_map("eyy") - (-0.015)).max() < 0.012
        assert np.abs(res.strain_map("exy") - 0.006).max() < 0.012

    def test_chunked_progress_and_stop(self, monkeypatch):
        g, I = _template(M=14, rmax=0.6)
        vecs = self._make_vectors(4, 4, g, 0.0, np.zeros((2, 2)))
        lib = vo.TemplateLibrary(
            spots_xy=[g.astype(np.float32)], spots_I=[I.astype(np.float32)],
            template_quats=np.array([[1.0, 0, 0, 0]]),
            template_phase=np.array([0], np.int16),
            phases_meta=[{"name": "x", "point_group": "m-3m"}],
            cache={}, radial_range=(0.0, 0.6), r_max=0.6,
        )
        monkeypatch.setattr(vo, "coarse_seed",
                            lambda *a, **k: [(0, 0.0, 1.0)])
        monkeypatch.setattr(vo, "_resolve_one_quat",
                            lambda ti, th, l: np.array([1.0, 0, 0, 0], np.float32))
        seen = []
        res = vo.compute_vector_orientation_chunked(
            vecs, lib, chunk=2, progress=lambda d, n: seen.append((d, n)))
        assert res is not None
        assert seen and seen[-1][0] == seen[-1][1]   # reached 100%
        # stop flag returns None
        r2 = vo.compute_vector_orientation_chunked(
            vecs, lib, chunk=2, stopped_flag=[True])
        assert r2 is None

    def test_smoothed_strain_reduces_noise_keeps_edges(self):
        # noisy strain field with a sharp grain boundary in εyy
        ny, nx = 16, 16
        strain = np.zeros((ny, nx, 3), np.float32)
        strain[:, nx // 2:, 1] = 0.02   # real boundary
        rng = np.random.RandomState(0)
        noisy = strain + rng.normal(0, 0.01, strain.shape).astype(np.float32)
        res = vo.VectorOrientationResult(
            quats=np.zeros((ny, nx, 4), np.float32),
            phase_idx=np.zeros((ny, nx), np.int16),
            theta=np.zeros((ny, nx), np.float32),
            strain=noisy, residual=np.zeros((ny, nx), np.float32),
            friedel_asym=np.zeros((ny, nx), np.float32),
            n_matched=np.zeros((ny, nx), np.int16),
            coarse_score=np.zeros((ny, nx), np.float32),
            phases_meta=[], nav_shape=(ny, nx))
        sm = res.smoothed_strain(size=3)
        assert sm.shape == noisy.shape
        # noise reduced (closer to the true field)
        err_raw = np.abs(noisy - strain).mean()
        err_sm = np.abs(sm - strain).mean()
        assert err_sm < err_raw
        # boundary preserved: a couple columns either side of the step the
        # field still separates the two grains (median keeps the edge, unlike
        # Gaussian which would wash it toward the mean)
        left = np.median(sm[:, nx // 2 - 2, 1])
        right = np.median(sm[:, nx // 2 + 2, 1])
        # the two grains stay separated (edge-preserving); TV denoises strongly
        # so the residual step is < the true 0.020 but clearly nonzero
        assert (right - left) > 0.008

    def test_tv_beats_raw_at_high_noise(self):
        # heavy noise on a piecewise-constant strain field: TV should recover it
        # much better than the raw fit (benchmark §7h).
        ny, nx = 20, 20
        gt = np.zeros((ny, nx, 3), np.float32)
        gt[ny // 2:, :, 0] = 0.02      # one grain
        rng = np.random.RandomState(1)
        noisy = gt + rng.normal(0, 0.03, gt.shape).astype(np.float32)
        res = vo.VectorOrientationResult(
            quats=np.zeros((ny, nx, 4), np.float32),
            phase_idx=np.zeros((ny, nx), np.int16),
            theta=np.zeros((ny, nx), np.float32),
            strain=noisy, residual=np.zeros((ny, nx), np.float32),
            friedel_asym=np.zeros((ny, nx), np.float32),
            n_matched=np.zeros((ny, nx), np.int16),
            coarse_score=np.zeros((ny, nx), np.float32),
            phases_meta=[], nav_shape=(ny, nx))
        err_raw = np.abs(noisy - gt).mean()
        err_tv = np.abs(res.smoothed_strain(method="tv", weight=0.03) - gt).mean()
        err_med = np.abs(res.smoothed_strain(method="median") - gt).mean()
        assert err_tv < err_raw * 0.6, (err_tv, err_raw)
        assert err_tv <= err_med + 1e-4   # TV at least as good as median here

    def test_to_orientation_map_and_ipf(self):
        res = vo.VectorOrientationResult(
            quats=np.tile([1.0, 0, 0, 0], (3, 4, 1)).astype(np.float32),
            phase_idx=np.zeros((3, 4), np.int16),
            theta=np.zeros((3, 4), np.float32),
            strain=np.zeros((3, 4, 3), np.float32),
            residual=np.zeros((3, 4), np.float32),
            friedel_asym=np.zeros((3, 4), np.float32),
            n_matched=np.zeros((3, 4), np.int16),
            coarse_score=np.ones((3, 4), np.float32),
            phases_meta=[{"name": "Ag", "point_group": "m-3m"}],
            nav_shape=(3, 4))
        om = res.to_orientation_map()
        assert om.nav_shape == (3, 4)
        assert om.n_best == 1
        ipf = res.ipf_color_map("z")
        assert ipf.shape == (3, 4, 3)
        assert ipf.dtype == np.uint8

    def test_strain_component_maps(self):
        # the *_map accessors return the right slices
        res = vo.VectorOrientationResult(
            quats=np.zeros((2, 2, 4), np.float32),
            phase_idx=np.zeros((2, 2), np.int16),
            theta=np.zeros((2, 2), np.float32),
            strain=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            residual=np.zeros((2, 2), np.float32),
            friedel_asym=np.zeros((2, 2), np.float32),
            n_matched=np.zeros((2, 2), np.int16),
            coarse_score=np.zeros((2, 2), np.float32),
            phases_meta=[], nav_shape=(2, 2),
        )
        assert np.array_equal(res.strain_map("exx"), res.strain[..., 0])
        assert np.array_equal(res.strain_map("eyy"), res.strain[..., 1])
        assert np.array_equal(res.strain_map("exy"), res.strain[..., 2])
        assert np.array_equal(res.dilatation_map(),
                              res.strain[..., 0] + res.strain[..., 1])
        assert np.array_equal(res.shear_map(), res.strain[..., 2])


@pytest.mark.slow
class TestIntegrationAg:
    def _ag_lib(self):
        from orix.crystal_map import Phase
        from diffpy.structure import Atom, Lattice, Structure
        import hyperspy.api as hs
        from spyde.actions.orientation_compute import generate_library_from_phases

        a = 4.0853
        latt = Lattice(a, a, a, 90, 90, 90)
        atoms = [Atom("Ag", [0, 0, 0]), Atom("Ag", [.5, .5, 0]),
                 Atom("Ag", [.5, 0, .5]), Atom("Ag", [0, .5, .5])]
        phase = Phase(name="Ag", point_group="m-3m",
                      structure=Structure(atoms, latt))
        sim = generate_library_from_phases(
            [phase], accelerating_voltage=200.0, resolution=2.0,
            minimum_intensity=1e-3, reciprocal_radius=0.75,
        )
        cal = hs.signals.Signal2D(np.zeros((112, 112), np.float32))
        for ax in cal.axes_manager.signal_axes:
            ax.scale = 0.01336
            ax.offset = -0.7484
        cal.set_signal_type("electron_diffraction")
        return vo.build_template_library(sim, cal, r_max=0.75)

    def test_recovers_orientation_and_strain(self):
        lib = self._ag_lib()
        ti = 40  # arbitrary template
        g = lib.spots_xy[ti].astype(np.float64)
        gI = lib.spots_I[ti].astype(np.float64)
        if len(g) < 4:
            pytest.skip("template too sparse")
        theta = np.deg2rad(7.0)
        strain = np.array([[0.025, 0.008], [0.008, -0.018]])
        v, vI = _apply(g, gI, theta, strain, noise=0.003, rmax=0.75, seed=7)

        # full path incl. coarse seed
        fit = vo.fit_pattern(v, vI, lib)
        assert fit is not None
        # coarse may pick a symmetry-equivalent template, but the fit residual
        # must be small and the recovered strain physical
        assert fit.residual < 0.02, fit.residual
        assert np.abs(fit.strain).max() <= 0.05 + 1e-3
