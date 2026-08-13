"""
test_motion.py — the motion driver, and the integrity of the vendored compute.

**These do not re-test upstream's numerics.** That code is vendored verbatim
from de_ground_crew and is validated over there against cryoSPARC and EMPIAR
data this repo does not have. Re-asserting its behaviour here would be a second
weaker oracle that goes stale, and the first attempt at this port proved the
cost: a hand-transcription reproduced an already-fixed bug and missed the whole
v3 aligner.

So what IS tested:

* the vendored files are pristine — drift is a failure, not a discovery;
* the boundary — no Qt, no dask, came along for the ride;
* the DRIVER's own contract — the sign reconciliation, throw, cancellation,
  fail-loud pass-through, and the result keys the UI indexes.

One end-to-end alignment runs on a known drift, because a green unit suite over
glue that never actually invokes v3 would be worthless.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from de_groundcrew import motion
from de_groundcrew.motion import driver

VENDOR = Path(driver.__file__).parent.parent / "external" / "gc_motion"


def _shift(a, dy, dx):
    from de_groundcrew.external.gc_motion._worker_extracts import _apply_shift_fourier
    return _apply_shift_fourier(np.asarray(a, dtype=np.float32), dy, dx)


def _scene(n=128, seed=0):
    """Band-limited noise plus hard points — what correlation needs."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (n, n))
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    img = np.real(np.fft.ifft2(np.fft.fft2(img)
                               * np.exp(-(np.sqrt(fy ** 2 + fx ** 2) / 0.08) ** 2)))
    img = (img - img.min()) / (float(np.ptp(img)) or 1.0)
    for cy, cx in ((30, 40), (70, 90), (100, 25)):
        img[cy - 2:cy + 3, cx - 2:cx + 3] += 2.0
    return img.astype(np.float32)


def _drifting(shifts, n=128):
    base = _scene(n)
    return np.stack([_shift(base, dy, dx) for dy, dx in shifts]).astype(np.float32)


class TestVendorIsPristine:
    """The manifest's hashes, re-checked.

    If someone edits a vendored file instead of re-syncing, this fails — which
    is the whole reason the hashes are written down.
    """

    def _manifest_rows(self):
        text = (VENDOR / "MANIFEST.md").read_text()
        # | `file.py` | `upstream/path` | lines | `hash16` |
        return re.findall(
            r"^\|\s*`([^`]+)`\s*\|\s*`[^`]+`\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]+)`\s*\|$",
            text, re.M)

    def test_the_manifest_lists_every_vendored_module(self):
        listed = {r[0] for r in self._manifest_rows()}
        on_disk = {p.name for p in VENDOR.glob("*.py")
                   if p.name not in ("__init__.py", "_worker_extracts.py")}
        assert listed == on_disk, (
            f"manifest and directory disagree: only-in-manifest={listed - on_disk}, "
            f"only-on-disk={on_disk - listed}")

    @pytest.mark.parametrize("name,lines,digest", [
        pytest.param(*r, id=r[0]) for r in
        re.findall(r"^\|\s*`([^`]+)`\s*\|\s*`[^`]+`\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]+)`\s*\|$",
                   (Path(driver.__file__).parent.parent / "external" / "gc_motion"
                    / "MANIFEST.md").read_text(), re.M)])
    def test_file_matches_its_recorded_hash(self, name, lines, digest):
        raw = (VENDOR / name).read_bytes()
        # Undo the permitted IMPORT REWRITES before hashing — see MANIFEST.md —
        # so every other byte is still checked against upstream.
        raw = (raw
               .replace(b"from de_groundcrew.external.gc_motion._motion_correction_v2 import (",
                        b"from _motion_correction_v2 import (")
               .replace(b"from de_groundcrew.external.gc_motion._worker_extracts import (",
                        b"from workers.motion_correction_worker import (")
               .replace(b"from de_groundcrew.external.gc_motion.dose_weighting import (\n            dose_weight_map, frame_doses)",
                        b"from workers.dose_weighting import dose_weight_map, frame_doses"))
        assert hashlib.sha256(raw).hexdigest()[:16] == digest, (
            f"{name} has been edited — re-sync from upstream instead")
        assert len(raw.decode().splitlines()) == int(lines)

    def test_the_only_app_references_are_the_listed_import_rewrites(self):
        # Vendored code may name the vendored package (that is the rewrite) but
        # must never reach into the APP — that would make re-syncing a merge.
        for p in VENDOR.glob("*.py"):
            if p.name == "__init__.py":
                continue
            for line in p.read_text().splitlines():
                if "de_groundcrew" in line:
                    assert "de_groundcrew.external.gc_motion" in line, (
                        f"{p.name}: {line.strip()!r} reaches outside the vendor")


class TestBoundary:
    def test_no_qt_came_with_the_vendored_code(self):
        # The Qt workers were deliberately left behind; only the compute was
        # taken. A PySide import here means one slipped through.
        out = subprocess.run(
            [sys.executable, "-c",
             "import de_groundcrew.motion, sys; "
             "print([m for m in sys.modules if m.startswith(('PySide','PyQt'))])"],
            capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]", f"Qt leaked: {out.stdout}"

    def test_no_dask_or_hyperspy_either(self):
        out = subprocess.run(
            [sys.executable, "-c",
             "import de_groundcrew.motion, sys; "
             "print([m for m in sys.modules if m.split('.')[0] in "
             "('dask','hyperspy','pyxem','distributed')])"],
            capture_output=True, text=True, timeout=180)
        assert out.stdout.strip() == "[]", f"heavy deps leaked: {out.stdout}"


class TestDriverContract:
    """The glue this app actually owns."""

    def test_recovers_a_known_drift_in_both_axes(self):
        # The one end-to-end run. Without it the rest of this file is a green
        # suite over glue that never invokes v3.
        truth = [(0.0, 0.0), (1.5, -1.0), (3.0, -2.0),
                 (4.5, -3.0), (6.0, -4.0), (7.5, -5.0)]
        r = driver.align_stack(_drifting(truth), apix=1.0)

        gy = np.array(r["shifts_y_smooth"]); gy -= gy[0]
        gx = np.array(r["shifts_x_smooth"]); gx -= gx[0]
        assert np.allclose(gy, [t[0] for t in truth], atol=0.5), gy
        assert np.allclose(gx, [t[1] for t in truth], atol=0.5), gx

    def test_the_x_sign_is_reconciled(self):
        # LOAD-BEARING. v3 returns x negated (MotionCor3 convention) and the
        # rest of the pipeline wants it internal. Get this wrong and everything
        # still runs while the drift plot is mirrored in x — so assert the
        # DIRECTION against a drift that is unambiguous in sign.
        truth = [(0.0, 0.0), (0.0, -2.0), (0.0, -4.0), (0.0, -6.0)]
        r = driver.align_stack(_drifting(truth), apix=1.0)
        gx = np.array(r["shifts_x_smooth"]); gx -= gx[0]
        assert gx[-1] < -3.0, f"x drift came back {gx} — sign is inverted"

    def test_alignment_sharpens_the_sum(self):
        r = driver.align_stack(
            _drifting([(0, 0), (2, 1), (4, 2), (6, 3), (8, 4)]), apix=1.0)
        assert float(r["aligned_sum"].std()) > float(r["unaligned_sum"].std())

    def test_result_carries_every_key_the_ui_indexes(self):
        r = driver.align_stack(_drifting([(0, 0), (1, 1), (2, 2), (3, 3)]))
        for k in ("aligned_sum", "unaligned_sum", "aligned_fft",
                  "shifts_x_raw", "shifts_y_raw",
                  "shifts_x_smooth", "shifts_y_smooth",
                  "n_frames", "bin_factor", "throw",
                  "low_confidence", "failure_reason", "confidence_signals"):
            assert k in r, k

    def test_raw_aliases_smooth_because_v3_emits_no_second_curve(self):
        # RELION does the same. Inventing a "smoothed" curve the solver never
        # produced would put a line on the drift plot that means nothing.
        r = driver.align_stack(_drifting([(0, 0), (1, 1), (2, 2), (3, 3)]))
        assert r["shifts_x_raw"] == r["shifts_x_smooth"]
        assert r["shifts_y_raw"] == r["shifts_y_smooth"]

    def test_throw_discards_leading_frames(self):
        stack = _drifting([(0, 0), (9, 9), (2, 2), (3, 3), (4, 4), (5, 5)])
        r = driver.align_stack(stack, throw=2)
        assert r["throw"] == 2 and r["n_frames"] == 4
        assert len(r["shifts_y_smooth"]) == 4

    def test_throw_always_leaves_at_least_two_frames(self):
        r = driver.align_stack(_drifting([(0, 0), (1, 1), (2, 2)]), throw=99)
        assert r["n_frames"] >= 2

    def test_a_single_frame_is_refused(self):
        with pytest.raises(ValueError, match="at least 2 frames"):
            driver.align_stack(_drifting([(0, 0)]))

    def test_cancellation_raises_rather_than_returning_a_partial_result(self):
        with pytest.raises(driver.Cancelled):
            driver.align_stack(_drifting([(0, 0), (1, 1), (2, 2), (3, 3)]),
                               should_cancel=lambda: True)

    def test_progress_is_reported(self):
        msgs: list[str] = []
        driver.align_stack(_drifting([(0, 0), (1, 1), (2, 2), (3, 3)]),
                           progress=msgs.append)
        assert msgs, "no progress was reported"

    def test_both_modes_run(self):
        for mode in driver.MODES:
            r = driver.align_stack(_drifting([(0, 0), (1, 1), (2, 2), (3, 3)]),
                                   mode=mode)
            assert r["n_frames"] == 4

    def test_an_unknown_mode_falls_back_rather_than_crashing(self):
        # Upstream's adapter does the same. A bad preset name should not lose
        # someone's alignment.
        r = driver.align_stack(_drifting([(0, 0), (1, 1), (2, 2), (3, 3)]),
                               mode="turbo")
        assert r["n_frames"] == 4


class TestFailLoud:
    def test_confidence_is_reported_as_a_result_not_an_exception(self):
        r = driver.align_stack(_drifting([(0, 0), (1, 1), (2, 2), (3, 3)]))
        assert isinstance(r["low_confidence"], bool)
        assert isinstance(r["failure_reason"], str)

    def test_a_clean_alignment_is_not_flagged(self):
        r = driver.align_stack(
            _drifting([(0, 0), (1.5, -1), (3, -2), (4.5, -3)]), apix=1.0)
        assert r["low_confidence"] is False, r["failure_reason"]


class TestGainOrientation:
    def test_the_index_order_is_upstreams(self):
        # The index is a stored setting; reordering silently changes the
        # meaning of every saved orientation.
        assert driver.ORIENTATION_LABELS[0] == "Identity"
        assert len(driver.ORIENTATION_LABELS) == 8

    def test_ranking_returns_eight_scores_and_a_separation(self):
        # The tier thresholds assume all EIGHT were scored — the separation is
        # a median over them — so a short list would silently mis-tier.
        scores, sep = driver.rank_gain_orientations(_scene(64), _scene(64, seed=2) + 1.0)
        assert len(scores) == 8
        assert sep > 0

    def test_the_tier_boundaries_are_upstreams(self):
        # Calibrated on 219 real gain pairings; not ours to re-pick.
        assert driver.classify_gain_tier(6.0) == "ok"
        assert driver.classify_gain_tier(1.0) == "fail"
        assert driver.classify_gain_tier(1.2) == "weak"


class TestPowerSpectrum:
    def test_a_non_square_frame_yields_a_square_spectrum(self):
        # Upstream's crop-to-square: Thon rings must render circular on a
        # non-square sensor. The hand-port missed this entirely.
        out = driver.log_fft(_scene(128)[:64, :])
        assert out.shape[0] == out.shape[1]

    def test_the_dc_term_does_not_swamp_the_spectrum(self):
        out = driver.log_fft(_scene() + 1000.0)
        assert float((out > 128).mean()) < 0.5
        assert out.std() > 1.0


class TestLoading:
    def test_an_unsupported_extension_is_refused_by_name(self):
        with pytest.raises(ValueError, match="Unsupported movie format"):
            driver.load_movie_stack("/tmp/nope.hspy")

    def test_round_trips_through_mrc(self, tmp_path):
        img = _scene(64)
        p = str(tmp_path / "out.mrc")
        assert driver.save_image(img, p) == p
        back, meta = driver.load_movie_stack(p)
        assert back.shape == (1, 64, 64), "a single frame must load as 3-D"
        assert meta["n_frames"] == 1 and meta["filename"] == "out.mrc"
        assert np.allclose(back[0], img, atol=1e-5)

    def test_round_trips_through_tiff(self, tmp_path):
        img = _scene(64)
        p = str(tmp_path / "out.tif")
        driver.save_image(img, p)
        back, _ = driver.load_movie_stack(p)
        assert np.allclose(back[0], img, atol=1e-5)


class TestLocalMotion:
    def test_runs_on_phase_one_output_and_returns_its_documented_shape(self):
        stack = _drifting([(0, 0), (1, 1), (2, 2), (3, 3)])
        g = driver.align_stack(stack, apix=1.0)
        r = driver.correct_local_motion(
            stack, shifts_y=g["shifts_y_smooth"], shifts_x=g["shifts_x_smooth"],
            bin_factor=1, patch_size=128)
        assert r["corrected_sum"].shape == stack.shape[1:]
        assert r["corrected_fft"].shape[0] == r["corrected_fft"].shape[1]
        assert r["n_patches"] >= 1
        assert r["ps_cc"] >= 128, "the CC patch floor is upstream's, keep it"

    def test_cancellation_propagates(self):
        stack = _drifting([(0, 0), (1, 1), (2, 2), (3, 3)])
        with pytest.raises(driver.Cancelled):
            driver.correct_local_motion(
                stack, shifts_y=[0] * 4, shifts_x=[0] * 4,
                bin_factor=1, patch_size=128, should_cancel=lambda: True)


class TestPublicSurface:
    def test_the_package_exports_what_the_session_uses(self):
        for name in ("align_stack", "correct_local_motion", "load_movie_stack",
                     "load_gain", "save_image", "log_fft", "MODES",
                     "ORIENTATION_LABELS", "rank_gain_orientations", "classify_gain_tier", "Cancelled"):
            assert hasattr(motion, name), name

    def test_callers_do_not_reach_into_the_vendored_package(self):
        # The seam has to stay small enough that re-syncing is a file copy.
        app = Path(driver.__file__).parent.parent
        offenders = []
        for p in app.rglob("*.py"):
            if "external" in p.parts or p.name == "driver.py":
                continue
            if "gc_motion" in p.read_text():
                offenders.append(p.name)
        assert sorted(offenders) == ["__init__.py", "motion_session.py"], (
            f"unexpected direct users of the vendored code: {offenders}")
