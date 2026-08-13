"""
motion_session.py — the Motion mode's backend controller.

Holds the loaded movie stack and drives the ported S.T.A.C.K. compute
(`de_groundcrew.motion`). Kept out of `session.py` because it owns a lot of
state — a stack, a gain, two figures and a worker thread — and none of it has
anything to do with the live camera.

**Everything long-running goes on a worker thread**, and results come back
through `dispatch_to_main`. Not politeness: alignment is minutes on a real
movie, and the asyncio loop that runs it is the same one serving every other
message. The old app had the same rule with `QThread`; only the transport
changed.

The viewer is PUSH here, unlike Imaging. A movie is a fixed array already in
memory, so there is no server to pull tiles from — frames are painted straight
into the figure.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from de_shell.ipc import emit, emit_error, emit_status
from de_shell.plotting.figure import FigureView

from de_groundcrew import motion

log = logging.getLogger(__name__)

#: Views the pane can show. `aligned` and `corrected` only exist after a run.
VIEWS = ("raw", "unaligned", "aligned", "corrected")


class MotionController:
    """One movie stack, its alignment, and the two figures that show it."""

    def __init__(self, session) -> None:
        self._session = session
        self._dispatch = session._dispatch_to_main

        self.stack: np.ndarray | None = None
        self.meta: dict = {}
        self.gain: np.ndarray | None = None
        self.gain_name: str = ""
        #: "ok" / "weak" / "fail" from the last gain validation, or "".
        self.gain_tier: str = ""
        self.orientation: int = 0
        self.result: dict | None = None
        self.local_result: dict | None = None

        self._view = "raw"
        self._frame_idx = 0
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()

        self.image = FigureView(session.next_window_id(), title="Motion")
        self.fft = FigureView(session.next_window_id(), title="FFT")
        session.register_window_controller(self.image.window_id, self.image)
        session.register_window_controller(self.fft.window_id, self.fft)
        self._opened = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def close(self) -> None:
        self._cancel.set()
        if self._worker is not None:
            self._worker.join(timeout=5)
        for fig in (self.image, self.fft):
            try:
                fig.close()
            except Exception as e:
                log.debug("closing a motion figure failed: %s", e)

    def _run_async(self, name: str, fn) -> bool:
        """Run *fn* on a worker thread. One at a time.

        Returns whether it started — refusing rather than queueing, because the
        controller's state (stack, gain, result) is shared and a second run
        would race the first.
        """
        if self.busy:
            emit_status("Motion: a job is already running")
            return False
        self._cancel.clear()

        def _body() -> None:
            try:
                fn()
            except motion.Cancelled:
                self._dispatch(lambda: emit_status("Cancelled"))
            except Exception as e:
                log.exception("%s failed", name)
                self._dispatch(lambda: emit_error(f"{name} failed: {e}"))
            finally:
                self._dispatch(self._emit_state)

        self._worker = threading.Thread(target=_body, daemon=True,
                                        name=f"motion-{name}")
        self._worker.start()
        self._emit_state()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def _progress(self, msg: str) -> None:
        self._dispatch(lambda: emit_status(msg))

    # ── Actions ───────────────────────────────────────────────────────────────

    def open_stack(self, path: str) -> None:
        def _load() -> None:
            stack, meta = motion.load_movie_stack(path)

            def _apply() -> None:
                self.stack, self.meta = stack, meta
                self.result = self.local_result = None
                self._frame_idx = 0
                self._view = "raw"
                self._open_figures(stack.shape[1:])
                emit_status(f"{meta['filename']} — {meta['n_frames']} frames "
                            f"{meta['width']}×{meta['height']}")
                self._paint()
                self._emit_state()
            self._dispatch(_apply)

        self._run_async("Loading stack", _load)

    def load_test_stack(self, n_frames: int = 6, size: int = 128) -> None:
        """A synthetic drifting movie, for development and the e2e.

        Mirrors SpyDE's `load_test_data_*` actions: no file dialog, no fixture
        on disk, and the drift is KNOWN — a linear ramp — so a correct
        alignment produces a visibly sharper sum and a straight trajectory.
        The scene is band-limited noise plus a few hard points, which is what
        cross-correlation needs; smooth blobs alone give an ambiguous peak.
        """
        def _make() -> None:
            rng = np.random.default_rng(0)
            img = rng.normal(0, 1, (size, size))
            fy = np.fft.fftfreq(size)[:, None]
            fx = np.fft.fftfreq(size)[None, :]
            r = np.sqrt(fy ** 2 + fx ** 2)
            img = np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-(r / 0.08) ** 2)))
            img = (img - img.min()) / (float(np.ptp(img)) or 1.0)
            for cy, cx in ((size // 4, size // 3), (size // 2, size // 2),
                           (3 * size // 4, size // 5)):
                img[cy - 2:cy + 3, cx - 2:cx + 3] += 2.0

            from de_groundcrew.external.gc_motion._worker_extracts import (
                _apply_shift_fourier)
            stack = np.stack([_apply_shift_fourier(img.astype(np.float32),
                                                   1.5 * i, -1.0 * i)
                              for i in range(n_frames)]).astype(np.float32)
            meta = {"n_frames": n_frames, "height": size, "width": size,
                    "filename": "synthetic_drift.mrc", "path": "",
                    "dtype": str(stack.dtype)}

            def _apply() -> None:
                self.stack, self.meta = stack, meta
                self.result = self.local_result = None
                self._frame_idx, self._view = 0, "raw"
                self._open_figures(stack.shape[1:])
                emit_status(f"Test movie — {n_frames} frames {size}×{size}")
                self._paint()
                self._emit_state()
            self._dispatch(_apply)

        self._run_async("Building test movie", _make)

    def open_gain(self, path: str) -> None:
        def _load() -> None:
            gain = motion.load_gain(path)
            import os
            name = os.path.basename(path)

            def _apply() -> None:
                self.gain, self.gain_name = gain, name
                emit_status(f"Gain: {name}")
                self._emit_state()
            self._dispatch(_apply)

        self._run_async("Loading gain", _load)

    def validate_gain(self) -> None:
        """Score all eight orientations and adopt the best."""
        if self.stack is None or self.gain is None:
            emit_error("Load a stack and a gain reference first")
            return

        def _check() -> None:
            scores, separation = motion.rank_gain_orientations(
                self.stack[0], self.gain)
            tier = motion.classify_gain_tier(separation)

            def _apply() -> None:
                if not scores:
                    emit_error("No gain orientation fits this frame")
                    return
                _score, label, idx = scores[0]
                self.orientation = idx
                self.gain_tier = tier
                # A "fail" tier means the eight orientations scored alike, i.e.
                # this gain does not fit this sensor at all — a wrong camera
                # type or a corrupt reference. Adopting the best of eight ties
                # would ruin every frame silently, so say so loudly.
                msg = f"Gain orientation: {label} · separation {separation:.2f} ({tier})"
                if tier == "fail":
                    emit_error(msg + " — this gain does not appear to fit this "
                                     "sensor; check it is the right camera")
                elif tier == "weak":
                    emit_status(msg + " — weak, verify before relying on it")
                else:
                    emit_status(msg)
                emit({"type": "motion_gain_scores", "tier": tier,
                      "separation": float(separation),
                      "scores": [{"label": l, "index": i, "score": float(s)}
                                 for s, l, i in scores]})
                self._emit_state()
            self._dispatch(_apply)

        self._run_async("Gain validation", _check)

    def align(self, *, mode: str, throw: int, local: bool,
              patch_size: int, bin_factor: int = 2, apix: float = 1.0) -> None:
        if self.stack is None:
            emit_error("Load a movie stack first")
            return

        def _go() -> None:
            result = motion.align_stack(
                self.stack, gain=self.gain, gain_orientation=self.orientation,
                apix=apix, mode=mode, throw=throw,
                progress=self._progress, should_cancel=self._cancel.is_set)

            local_result = None
            if local:
                local_result = motion.correct_local_motion(
                    self.stack, gain=self.gain,
                    gain_orientation=self.orientation,
                    shifts_y=result["shifts_y_smooth"],
                    shifts_x=result["shifts_x_smooth"],
                    bin_factor=bin_factor, patch_size=patch_size,
                    throw=throw, progress=self._progress,
                    should_cancel=self._cancel.is_set)

            def _apply() -> None:
                self.result = result
                self.local_result = local_result
                self._view = "corrected" if local_result else "aligned"
                self._paint()
                self._emit_shifts()
                if result.get("low_confidence"):
                    # v3 refuses an implausible alignment rather than showing
                    # one. Surfacing it as an ERROR, not a status line, is the
                    # point of fail-loud — the result is still displayed so it
                    # can be inspected, but nobody uses it by accident.
                    emit_error("Alignment is not trustworthy: "
                               + (result.get("failure_reason")
                                  or "confidence check failed"))
                else:
                    emit_status(f"Aligned {result['n_frames']} frames"
                                + (f" + local ({local_result['n_patches']} patches)"
                                   if local_result else ""))
            self._dispatch(_apply)

        self._run_async("Alignment", _go)

    def save(self, path: str) -> None:
        img = self._current_result_image()
        if img is None:
            emit_error("Nothing to save — run an alignment first")
            return

        def _write() -> None:
            written = motion.save_image(img, path)
            self._dispatch(lambda: emit_status(f"Saved {written}"))

        self._run_async("Saving", _write)

    def set_frame(self, idx: int) -> None:
        if self.stack is None:
            return
        self._frame_idx = max(0, min(int(idx), self.stack.shape[0] - 1))
        if self._view != "raw":
            self._view = "raw"
        self._paint()
        self._emit_state()

    def set_view(self, view: str) -> None:
        if view not in VIEWS:
            emit_error(f"Unknown view: {view!r}")
            return
        self._view = view
        self._paint()
        self._emit_state()

    # ── Display ───────────────────────────────────────────────────────────────

    def _open_figures(self, shape) -> None:
        if self._opened:
            return
        self.image.open(shape)
        self.fft.open(shape)
        self._opened = True

    def _current_result_image(self) -> np.ndarray | None:
        if self._view == "corrected" and self.local_result:
            return self.local_result["corrected_sum"]
        if self._view == "aligned" and self.result:
            return self.result["aligned_sum"]
        if self._view == "unaligned" and self.result:
            return self.result["unaligned_sum"]
        return None

    def _paint(self) -> None:
        """Paint the selected view and its power spectrum. Main thread only."""
        if self.stack is None:
            return
        img = self._current_result_image()
        if img is None:
            img = self.stack[self._frame_idx]
        self.image.show(np.asarray(img))

        # The FFT is the point of the mode — it is how you SEE that alignment
        # worked — so it tracks whatever is displayed. Cached for the aligned
        # views, which computed it in the worker rather than on this thread.
        spectrum = None
        if self._view == "corrected" and self.local_result:
            spectrum = self.local_result["corrected_fft"]
        elif self._view == "aligned" and self.result:
            spectrum = self.result["aligned_fft"]
        if spectrum is None:
            try:
                spectrum = motion.log_fft(np.asarray(img))
            except Exception as e:
                log.debug("power spectrum failed: %s", e)
                return
        self.fft.show(np.asarray(spectrum))

    def _emit_shifts(self) -> None:
        if not self.result:
            return
        emit({
            "type": "motion_shifts",
            "shifts_x_raw": list(self.result["shifts_x_raw"]),
            "shifts_y_raw": list(self.result["shifts_y_raw"]),
            "shifts_x_smooth": list(self.result["shifts_x_smooth"]),
            "shifts_y_smooth": list(self.result["shifts_y_smooth"]),
            "n_frames": self.result["n_frames"],
            "throw": self.result.get("throw", 0),
        })

    def _emit_state(self) -> None:
        emit({
            "type": "motion_state",
            "loaded": self.stack is not None,
            "busy": self.busy,
            "filename": self.meta.get("filename"),
            "n_frames": int(self.meta.get("n_frames", 0)),
            "width": int(self.meta.get("width", 0)),
            "height": int(self.meta.get("height", 0)),
            "frame": self._frame_idx,
            "view": self._view,
            "gain": self.gain_name or None,
            "gain_tier": self.gain_tier,
            "orientation": self.orientation,
            "orientations": list(motion.ORIENTATION_LABELS),
            "has_result": self.result is not None,
            "low_confidence": bool((self.result or {}).get("low_confidence", False)),
            "failure_reason": (self.result or {}).get("failure_reason", ""),
            "modes": list(motion.MODES),
            "has_local": self.local_result is not None,
            "image_window": self.image.window_id,
            "fft_window": self.fft.window_id,
        })
