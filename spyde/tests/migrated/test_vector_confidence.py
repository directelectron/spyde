"""Every diffraction vector carries a positional confidence and uncertainty.

The neural method records each disk's heatmap peak (``confidence``) and the
centre's standard deviation (``sigma``); ``spyde.models.centre`` also offers
the soft-argmax of the heatmap as an opt-in centre. Vectors are
``(nav_x, nav_y, kx, ky, time, intensity, confidence, sigma)``; buffers and
files written with the first six columns load with the last two NaN.

The network runs in ONE subprocess (``RESULT_JSON <mode> {...}`` lines, then
``os._exit(0)``), as in ``test_neural_detect.py``: torch teardown can segfault
inside the pytest process on Windows.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from spyde.signals.diffraction_vectors import (
    COL_CONFIDENCE, COL_INTENSITY, COL_KX, COL_KY, COL_NAV_X, COL_NAV_Y, COL_SIGMA, COL_TIME,
    COLUMN_NAMES, LEGACY_N_COLS, N_COLS, SpyDEDiffractionVectors, _AxisLite,
)

_DRIVER = textwrap.dedent(r"""
    import json, os, sys
    import numpy as np
    from scipy.special import erfc

    def dynamical_frames(n=40, size=128, radius=6.0, seed=0):
        # Disks whose fill is uneven and points a different way in every frame
        # and reflection (the dynamical-scattering signature); truth = the
        # geometric outline centre.
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[:size, :size].astype(np.float64)
        g1, g2 = np.array([0.0, 26.0]), np.array([24.0, 4.0])
        frames, truth = [], []
        for _ in range(n):
            c0 = np.array([size / 2, size / 2]) + rng.uniform(-1, 1, 2)
            f = np.full((size, size), 3.0)
            cs = []
            for h in range(-2, 3):
                for k in range(-2, 3):
                    c = c0 + h * g1 + k * g2 + rng.normal(0, 0.3, 2)
                    if np.any(c < radius + 4) or np.any(c > size - radius - 5):
                        continue
                    r = np.hypot(yy - c[0], xx - c[1])
                    edge = 0.5 * erfc((r - radius) / (np.sqrt(2) * 0.6))
                    th = rng.uniform(0, 2 * np.pi)
                    u = ((xx - c[1]) * np.cos(th) + (yy - c[0]) * np.sin(th)) / radius
                    lobe = np.exp(-((xx - c[1] - 0.45 * radius * np.cos(th)) ** 2
                                    + (yy - c[0] - 0.45 * radius * np.sin(th)) ** 2)
                                  / (2 * (0.35 * radius) ** 2))
                    mag = rng.uniform(0.3, 0.8)
                    fill = np.clip(1 + mag * u, 0, None) + 0.8 * mag * lobe
                    amp = 400 if (h, k) == (0, 0) else rng.uniform(20, 120)
                    f += amp * edge * fill
                    cs.append(c)
            frames.append(rng.poisson(f).astype(np.float32))
            truth.append(np.array(cs))
        return np.stack(frames), truth

    def run_mode(mode):
        from spyde import models
        out = {}
        if mode == "dynamical":
            frames, truth = dynamical_frames()
            model, dev = models.get_cpu_model(None)
            for centre in ("softargmax", "offset"):
                errs = []
                for f, t in zip(frames, truth):
                    p = np.asarray(models.detect(model, f, dev, thresh=0.3, spot_diameter=12.0,
                                                 centre=centre)).reshape(-1, 3)
                    d = np.hypot(p[:, None, 0] - t[None, :, 0], p[:, None, 1] - t[None, :, 1])
                    j = d.argmin(0)
                    ok = d[j, np.arange(len(t))] < 4
                    errs.append(d[j, np.arange(len(t))][ok])
                e = np.concatenate(errs)
                out[centre] = dict(n=int(len(e)), rms=float(np.sqrt(np.mean(e ** 2))))
        elif mode == "columns":
            from spyde.actions.find_vectors_neural import _find_vectors_single_frame_neural
            frames, truth = dynamical_frames(n=2)
            _, _, peaks = _find_vectors_single_frame_neural(frames[0], 0.3, 4, spot_radius=6.0)
            out["shape"] = list(peaks.shape)
            out["confidence_min"] = float(np.min(peaks[:, 3]))
            out["confidence_max"] = float(np.max(peaks[:, 3]))
            out["sigma_finite"] = bool(np.isfinite(peaks[:, 4]).all())
            out["sigma_max"] = float(np.max(peaks[:, 4]))
            # the bright direct beam is the best-localised disk
            beam = int(np.argmax(peaks[:, 2]))
            out["beam_has_smallest_sigma"] = bool(peaks[beam, 4] <= np.min(peaks[:, 4]) + 1e-6)
        elif mode == "parity":
            # batch (detect_batch + the shared peak record) vs the single-frame
            # preview, both on the CPU model, Spot size fixed
            from spyde.actions.find_vectors_neural import (
                _counts_radius, _find_vectors_single_frame_neural, _neural_peaks)
            frames, _ = dynamical_frames(n=4)
            model, dev = models.get_cpu_model(None)
            batch = models.detect_batch(model, frames, dev, thresh=0.3, spot_diameter=12.0,
                                        with_width=True)
            worst = 0.0
            for f, p in zip(frames, batch):
                b = _neural_peaks(f, np.asarray(p, np.float32), _counts_radius(f, 6.0))
                s = _find_vectors_single_frame_neural(f, 0.3, 4, spot_radius=6.0)[2]
                assert b.shape == s.shape, (b.shape, s.shape)
                worst = max(worst, float(np.nanmax(np.abs(b - s))))
            out["max_abs_difference"] = worst
        elif mode == "gpu_parity":
            import torch
            if not torch.cuda.is_available():
                out["skipped"] = True
            else:
                import torch.nn.functional as F
                F.linear(torch.zeros(1, 1, device="cuda"), torch.zeros(1, 1, device="cuda"))
                frames, _ = dynamical_frames(n=4)
                m_cpu, d_cpu = models.get_cpu_model(None)
                m_gpu, d_gpu = models.get_model(None)
                a = models.detect_batch(m_cpu, frames, d_cpu, thresh=0.3, spot_diameter=12.0, with_width=True)
                b = models.detect_batch(m_gpu, frames, d_gpu, thresh=0.3, spot_diameter=12.0, with_width=True)
                worst, counts = 0.0, []
                for x, y in zip(a, b):
                    counts.append((len(x), len(y)))
                    if len(x) == len(y) and len(x):
                        ox, oy = np.lexsort((x[:, 1], x[:, 0])), np.lexsort((y[:, 1], y[:, 0]))
                        worst = max(worst, float(np.abs(x[ox, :2] - y[oy, :2]).max()))
                out["counts_equal"] = all(i == j for i, j in counts)
                out["max_position_difference"] = worst
                out["device"] = str(d_gpu)
        return out

    for mode in sys.argv[1:]:
        print("RESULT_JSON", mode, json.dumps(run_mode(mode)), flush=True)
    sys.stdout.flush()
    os._exit(0)
""")


@pytest.fixture(scope="module")
def neural_results():
    proc = subprocess.run([sys.executable, "-c", _DRIVER, "dynamical", "columns", "parity", "gpu_parity"],
                          capture_output=True, text=True, timeout=900)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            _, mode, payload = line.split(" ", 2)
            out[mode] = json.loads(payload)
    if not out:
        pytest.fail(f"neural driver produced no results:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return out


class TestNeuralCentres:
    def test_softargmax_beats_the_decoded_position_under_dynamical_fill(self, neural_results):
        """On disks whose fill is uneven and points a different way in every
        frame (40 frames, 839 matched disks, seed 0): the opt-in soft-argmax is
        0.33 px RMS from the true outline centre, the default decode (argmax
        pixel + offset head) 0.37 px. (On real SPED-Ag the soft-argmax carries a
        ~0.7 % radial bias, which is why it is not the default.)"""
        r = neural_results["dynamical"]
        assert r["softargmax"]["n"] == r["offset"]["n"] > 500
        assert r["softargmax"]["rms"] < r["offset"]["rms"]
        assert r["softargmax"]["rms"] < 0.36

    def test_every_neural_peak_has_confidence_and_sigma(self, neural_results):
        r = neural_results["columns"]
        assert r["shape"][1] == 5
        assert 0.3 <= r["confidence_min"] <= r["confidence_max"] <= 1.0
        assert r["sigma_finite"] and 0 < r["sigma_max"] < 5.0   # the faintest disk: ~2 px
        assert r["beam_has_smallest_sigma"]

    def test_batch_and_preview_give_the_same_record(self, neural_results):
        assert neural_results["parity"]["max_abs_difference"] < 1e-4

    def test_gpu_and_cpu_agree(self, neural_results):
        r = neural_results["gpu_parity"]
        if r.get("skipped"):
            pytest.skip("no CUDA device")
        assert r["counts_equal"]
        assert r["max_position_difference"] < 0.02


class TestPositionalSigma:
    def test_formula(self):
        """Thompson, Larson & Webb: sigma^2 = (s^2 + a^2/12)/N + 8 pi s^4 b^2/(a^2 N^2)."""
        from spyde.models.centre import positional_sigma
        s, n, b2 = 1.5, 1000.0, 4.0
        expected = np.sqrt((s * s + 1 / 12) / n + 8 * np.pi * s ** 4 * b2 / (n * n))
        assert positional_sigma(s, n, b2) == pytest.approx(expected, rel=1e-5)

    def test_brighter_is_more_precise_and_no_signal_is_nan(self):
        from spyde.models.centre import positional_sigma
        out = positional_sigma([1.0, 1.0, 1.0], [100.0, 10000.0, 0.0], [2.0, 2.0, 2.0])
        assert out[1] < out[0]
        assert np.isnan(out[2])


def _vectors(n_cols=N_COLS):
    rng = np.random.default_rng(1)
    rows = []
    for iy in range(3):
        for ix in range(4):
            for _ in range(3):
                r = np.full(n_cols, np.nan, np.float32)
                r[COL_NAV_X], r[COL_NAV_Y], r[COL_TIME] = ix, iy, -1
                r[COL_KX], r[COL_KY], r[COL_INTENSITY] = rng.normal(0, 1, 3)
                if n_cols == N_COLS:
                    r[COL_CONFIDENCE], r[COL_SIGMA] = rng.uniform(0.3, 1), rng.uniform(0.01, 0.1)
                rows.append(r)
    axes = [_AxisLite(scale=0.01, offset=-1.0, size=200, units="1/A", name="kx"),
            _AxisLite(scale=0.01, offset=-1.0, size=200, units="1/A", name="ky")]
    return SpyDEDiffractionVectors.from_arrays(
        np.array(rows), (3, 4), sig_shape=(200, 200), sig_axes=axes,
        kernel_radius_px=5.0, kernel_radius_data=0.05)


class TestVectorColumns:
    def test_layout(self):
        assert COLUMN_NAMES[COL_CONFIDENCE] == "confidence"
        assert COLUMN_NAMES[COL_SIGMA] == "sigma"
        assert _vectors().flat_buffer.shape[1] == N_COLS == 8

    def test_legacy_buffer_gains_nan_columns(self):
        v = _vectors(LEGACY_N_COLS)
        assert v.flat_buffer.shape[1] == N_COLS
        assert np.isnan(v.flat_buffer[:, COL_CONFIDENCE]).all()
        assert np.isnan(v.flat_buffer[:, COL_SIGMA]).all()
        assert v.to_dense().shape[-1] == N_COLS

    def test_npz_round_trip_keeps_both_columns(self, tmp_path):
        v = _vectors()
        v.save(str(tmp_path / "v.npz"))
        w = SpyDEDiffractionVectors.load(str(tmp_path / "v.npz"))
        np.testing.assert_array_equal(w.flat_buffer, v.flat_buffer)

    def test_zspy_round_trip_keeps_both_columns(self, tmp_path):
        import hyperspy.api as hs
        from spyde.signals.dense_diffraction_vectors import from_dense_signal, to_dense_signal
        v = _vectors()
        to_dense_signal(v).save(str(tmp_path / "v.zspy"))
        w = from_dense_signal(hs.load(str(tmp_path / "v.zspy")))
        np.testing.assert_array_equal(w.flat_buffer, v.flat_buffer)

    def test_old_six_column_zspy_loads_with_nan_columns(self, tmp_path):
        """A file saved before confidence/sigma existed: (N, 6) data, six column names."""
        import hyperspy.api as hs
        from spyde.signals.dense_diffraction_vectors import (
            META_ROOT, from_dense_signal, to_dense_signal)
        v = _vectors()
        sig = to_dense_signal(v)
        old = hs.signals.Signal1D(np.ascontiguousarray(v.flat_buffer[:, :LEGACY_N_COLS]))
        old.metadata.add_dictionary(sig.metadata.as_dictionary())
        old.metadata.set_item(f"{META_ROOT}.column_names", list(COLUMN_NAMES[:LEGACY_N_COLS]))
        old.save(str(tmp_path / "old.zspy"))
        w = from_dense_signal(hs.load(str(tmp_path / "old.zspy")))
        np.testing.assert_array_equal(w.flat_buffer[:, :LEGACY_N_COLS], v.flat_buffer[:, :LEGACY_N_COLS])
        assert np.isnan(w.flat_buffer[:, COL_CONFIDENCE:]).all()

    def test_accessors_still_work(self):
        v = _vectors()
        assert v.at(1, 2).shape == (3, N_COLS)
        assert v.kxy_at(1, 2).shape == (3, 2)
        assert v.count_map().sum() == 36
        assert v.cluster(eps=10, min_samples=1).shape == (36,)


class TestNonNeuralMethods:
    def test_nxcorr_leaves_confidence_and_sigma_nan(self):
        """Only the neural method measures them; NXCORR/DoG fill NaN, which the
        strain fit treats as unweighted."""
        from spyde.actions.find_vectors import PEAK_COLS, _find_vectors_chunk, _make_disk
        yy, xx = np.mgrid[:64, :64]
        frame = (100.0 * (np.hypot(yy - 32, xx - 32) <= 4)).astype(np.float32) + 1
        block = np.broadcast_to(frame, (2, 2, 64, 64)).copy()
        disk = _make_disk(4)
        stats = (disk.size, float(disk.mean()), float(np.sqrt(np.sum((disk - disk.mean()) ** 2) / disk.size)))
        out = _find_vectors_chunk(block, 0, 2, 0.0, 4, 0.5, 3, True, None, None, stats)
        assert out.shape[-1] == PEAK_COLS
        found = out[np.isfinite(out[..., 0])]
        assert len(found) >= 4
        assert np.isnan(found[:, 3:]).all()


class TestOverlayShowsConfidence:
    def test_unsure_vectors_are_their_own_faint_group(self):
        from spyde.actions.vector_overlay import UNSURE_BELOW, DetectorPixels, found_vector_offsets
        v = _vectors()
        rows = v.at(1, 2).copy()
        rows[:, COL_KX] = rows[:, COL_KY] = 0.0                   # inside the detector
        rows[:, COL_CONFIDENCE] = [0.9, UNSURE_BELOW - 0.1, np.nan]
        out = found_vector_offsets(rows=rows, pixels=DetectorPixels.from_axes(v.sig_axes))
        assert len(out["found"]) == 2 and len(out["unsure"]) == 1


class TestMaskedMedian:
    def test_matches_numpy_median_row_by_row(self):
        from spyde.models.centre_refine import masked_median

        rng = np.random.default_rng(0)
        values = rng.normal(size=(50, 30)).astype(np.float32)
        mask = rng.random((50, 30)) < 0.4
        mask[0] = False
        mask[1] = False
        mask[1, 3] = True
        got = masked_median(values, mask)
        want = [np.median(v[m]) if m.any() else 0.0 for v, m in zip(values, mask)]
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
