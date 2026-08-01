"""
test_drift_nonrigid.py — the non-rigid solve recovers a KNOWN warp.

The plan's acceptance criterion for A2-A5 is exactly that: apply a synthetic
distortion whose field is known exactly, fit it, and check the fit removes it.
Anything weaker (loss went down, the field is smooth, it ran without raising)
would pass with a solver that fits noise.

Ground truth is built by warping with the SAME resampler the solver uses, so the
test measures the SOLVER and not the difference between two interpolators. The
residual is then compared against the distorted-vs-reference residual, i.e. "did
it recover most of what we put in", which is scale-free and does not encode a
tolerance nobody can justify.

CPU on purpose: torch-CUDA under the pytest process segfaults on Windows
(CLAUDE.md), and these fits are tiny.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyde.drift import nonrigid as nr
from spyde.drift.model import DriftModel


H = W = 64
DEV = "cpu"


def _textured_frame(seed: int = 0) -> np.ndarray:
    """A frame with structure at several scales.

    Registration needs gradients everywhere: a field of one blob is happy to
    slide sideways, so a solver could score well while recovering the wrong
    field. Blobs plus fine noise gives an unambiguous optimum.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W), np.float32)
    for _ in range(14):
        cy, cx = rng.uniform(6, H - 6), rng.uniform(6, W - 6)
        s = rng.uniform(2.0, 4.5)
        img += rng.uniform(0.5, 1.5) * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s * s))
    img += 0.05 * rng.standard_normal((H, W)).astype(np.float32)
    return img


def _warp_np(frame: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    t = torch.as_tensor(np.asarray(frame, np.float32))[None]
    out = nr.warp_frame(torch, t,
                        torch.as_tensor(np.asarray(dy, np.float32))[None],
                        torch.as_tensor(np.asarray(dx, np.float32))[None],
                        fill_nan=False)
    return out[0].numpy().copy()


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / max(a.std(), 1e-6)
    b = (b - b.mean()) / max(b.std(), 1e-6)
    return float(np.mean((a - b) ** 2))


class TestWarp:
    def test_zero_displacement_is_identity(self):
        f = _textured_frame()
        out = _warp_np(f, np.zeros((H, W), np.float32), np.zeros((H, W), np.float32))
        assert np.allclose(out, f, atol=1e-5)

    def test_out_of_bounds_is_nan_not_zero(self):
        """The locked edge policy: uncovered pixels are NaN, never invented.

        Zero-filling here is the bug that nucleates a spurious edge 'particle'
        downstream, which is why this is asserted rather than assumed.
        """
        f = _textured_frame()
        dy = np.full((H, W), 40.0, np.float32)      # push most rows off the top
        t = torch.as_tensor(f)[None]
        out = nr.warp_frame(torch, t, torch.as_tensor(dy)[None],
                            torch.as_tensor(np.zeros((H, W), np.float32))[None],
                            fill_nan=True)[0].numpy()
        assert np.isnan(out).any(), "no NaN produced outside coverage"
        assert not np.isnan(out).all(), "everything went out of bounds"

    def test_translation_matches_a_known_roll(self):
        """An integer displacement must reproduce an exact roll."""
        f = _textured_frame()
        out = _warp_np(f, np.full((H, W), -3.0, np.float32),
                       np.full((H, W), 0.0, np.float32))
        want = np.roll(f, 3, axis=0)
        # Interior only: the roll wraps where the warp runs out of data.
        assert np.allclose(out[6:-6, 6:-6], want[6:-6, 6:-6], atol=1e-4)


class TestScanKnotRecovery:
    def _stack(self, amp: float = 3.0, n: int = 4):
        """Frames distorted by a known SLOW-AXIS-varying displacement."""
        base = _textured_frame()
        rows = np.linspace(-1.0, 1.0, H, dtype=np.float32)
        frames, truth = [], []
        for i in range(n):
            a = amp * (i + 1) / n
            dy = np.repeat((a * rows)[:, None], W, axis=1)      # varies down rows
            dx = np.zeros((H, W), np.float32)
            frames.append(_warp_np(base, dy, dx))
            truth.append(dy)
        return base, np.stack(frames), np.stack(truth)

    def test_recovers_most_of_a_known_scan_distortion(self):
        base, frames, _ = self._stack()
        model = nr.solve_nonrigid(frames, model=nr.SCAN_KNOT, reference=base,
                                  n_knots=3, steps=220, lr=0.35,
                                  smooth_weight=0.05, temporal_weight=0.05,
                                  device=DEV)
        assert model.kind == nr.SCAN_KNOT
        before = np.mean([_mse(f, base) for f in frames])
        after = np.mean([_mse(np.nan_to_num(nr.apply_nonrigid(f, model, i), nan=0.0)
                              + np.isnan(nr.apply_nonrigid(f, model, i)) * base, base)
                         for i, f in enumerate(frames)])
        assert after < 0.45 * before, (
            f"the fit removed too little of the known warp: {before:.4f} -> {after:.4f}")

    def test_the_fitted_field_varies_down_the_slow_axis(self):
        """A scan-knot fit must not collapse to a constant offset.

        A constant is the degenerate solution that a rigid solve already
        provides; if that is all this produces, the model is not earning its
        parameters.
        """
        base, frames, _ = self._stack()
        model = nr.solve_nonrigid(frames, model=nr.SCAN_KNOT, reference=base,
                                  n_knots=3, steps=220, lr=0.35,
                                  smooth_weight=0.05, temporal_weight=0.05,
                                  device=DEV)
        dy, _dx = nr.displacement_for_frame(model, len(frames) - 1)
        spread = float(dy[:, 0].max() - dy[:, 0].min())
        assert spread > 0.5, f"fitted field is nearly constant down rows ({spread:.3f} px)"

    def test_a_row_is_constant_across_the_fast_axis(self):
        """Physical contract: one row is acquired at one slow coordinate."""
        base, frames, _ = self._stack()
        model = nr.solve_nonrigid(frames, model=nr.SCAN_KNOT, reference=base,
                                  n_knots=2, steps=40, device=DEV)
        dy, dx = nr.displacement_for_frame(model, 0)
        assert np.allclose(dy, dy[:, :1], atol=1e-6)
        assert np.allclose(dx, dx[:, :1], atol=1e-6)


class TestDenseRecovery:
    def _stack(self, amp: float = 2.5, n: int = 3):
        """Frames distorted by a known field that varies in BOTH directions.

        Deliberately not expressible by any scan-knot model, so this exercises
        the case the second parameterisation exists for.
        """
        base = _textured_frame(seed=3)
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        frames = []
        for i in range(n):
            a = amp * (i + 1) / n
            dy = a * np.sin(2 * np.pi * x / W).astype(np.float32)
            dx = a * np.cos(2 * np.pi * y / H).astype(np.float32)
            frames.append(_warp_np(base, dy, dx))
        return base, np.stack(frames)

    def test_recovers_most_of_a_known_2d_deformation(self):
        base, frames = self._stack()
        model = nr.solve_nonrigid(frames, model=nr.DENSE, reference=base,
                                  grid=(6, 6), steps=260, lr=0.35,
                                  smooth_weight=0.02, temporal_weight=0.02,
                                  device=DEV)
        assert model.kind == nr.DENSE
        before = np.mean([_mse(f, base) for f in frames])
        after = []
        for i, f in enumerate(frames):
            got = nr.apply_nonrigid(f, model, i)
            m = np.isnan(got)
            got = np.where(m, base, got)          # score coverage only
            after.append(_mse(got, base))
        after = float(np.mean(after))
        assert after < 0.5 * before, (
            f"the fit removed too little of the known deformation: {before:.4f} -> {after:.4f}")

    def test_dense_field_is_not_forced_constant_across_a_row(self):
        """The dense model's whole point: it can vary along the fast axis too."""
        base, frames = self._stack()
        model = nr.solve_nonrigid(frames, model=nr.DENSE, reference=base,
                                  grid=(6, 6), steps=200, lr=0.35,
                                  smooth_weight=0.02, temporal_weight=0.02,
                                  device=DEV)
        dy, _ = nr.displacement_for_frame(model, len(frames) - 1)
        row_spread = float(np.abs(dy[H // 2] - dy[H // 2].mean()).max())
        assert row_spread > 0.1, f"dense field is constant across a row ({row_spread:.3f})"


class TestModelContract:
    def test_rigid_component_is_preserved(self):
        base = _textured_frame()
        frames = np.stack([base, base])
        rigid = DriftModel(shifts=np.array([[0.0, 0.0], [1.5, -2.0]], np.float32))
        model = nr.solve_nonrigid(frames, model=nr.SCAN_KNOT, reference=base,
                                  rigid=rigid, steps=20, device=DEV)
        assert np.allclose(model.shifts, rigid.shifts), (
            "the rigid component must survive the non-rigid fit — a caller that "
            "applies only `shifts` should still get the rigid answer")

    def test_extra_carries_everything_needed_to_rebuild_the_field(self):
        base = _textured_frame()
        frames = np.stack([base, base])
        model = nr.solve_nonrigid(frames, model=nr.DENSE, reference=base,
                                  grid=(3, 3), steps=10, device=DEV)
        for key in ("params", "field_shape", "grid"):
            assert key in model.extra, f"extra is missing {key!r}"
        dy, dx = nr.displacement_for_frame(model, 0)
        assert dy.shape == (H, W) and dx.shape == (H, W)

    def test_a_rigid_model_is_refused_not_silently_zero(self):
        rigid = DriftModel(shifts=np.zeros((2, 2), np.float32))
        with pytest.raises(ValueError, match="not a non-rigid fit"):
            nr.displacement_for_frame(rigid, 0)

    def test_unknown_model_name_is_refused(self):
        with pytest.raises(ValueError, match="model must be one of"):
            nr.solve_nonrigid(np.zeros((2, H, W), np.float32), model="wobble")

    def test_cancel_stops_the_fit(self):
        base = _textured_frame()
        frames = np.stack([base, base])
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 3

        model = nr.solve_nonrigid(frames, model=nr.SCAN_KNOT, reference=base,
                                  steps=500, cancel=cancel, device=DEV)
        assert calls["n"] <= 6, "cancel was not honoured promptly"
        assert model.kind == nr.SCAN_KNOT
