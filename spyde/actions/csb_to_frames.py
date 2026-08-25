"""
csb_to_frames.py — "To Frames": an event stream → an ordinary in-situ movie.

A CSB file has no frames in the usable sense. It is a stream of detected
electron events at the camera's raw cadence (390 µs on the test movie), and a
single raw frame of it is mostly empty pixels. An image only exists once you
pick an exposure and integrate, which is why the file opens as a stack of
integrated time planes in the first place.

This action re-cuts that choice. Given an exposure — or the frame rate you want
in fps, which is how it is usually thought about — it rebuilds the movie at
that cadence as **its own dataset** (``_add_signal`` → a new signal tree and
its own navigator/signal window pair, not a node under the source), lazily, one
plane per dask block. A different exposure is a different movie, not a
transformation of the one on screen, and keeping the source open beside it is
the point. From there it is an ordinary lazy in-situ signal: the navigator
scrub, Play/Fast-Forward, virtual imaging, the movie editor and everything else
work on it with no knowledge that it came from an event stream.

Lazy is not an optimisation here, it is the only option: the test movie is
8192², so one plane is 268 MB and a 50-plane stack is 13 GB. Nothing is
integrated until a plane is actually looked at.
"""
from __future__ import annotations

import logging

import numpy as np

from de_shell.ipc import emit_error, emit_status

log = logging.getLogger(__name__)

DEFAULTS = dict(fps=0.0, exposure_ms=0.0, frames_per_plane=0, bin=0)


def _source_path(signal) -> str | None:
    """The .csb this signal was read from, or None if it was not."""
    try:
        om = signal.original_metadata
        if "csb" in om:
            return str(om.csb.path)
    except Exception as e:
        log.debug("reading the CSB source path failed: %s", e)
    return None


def is_csb(signal) -> bool:
    """True when a signal came from a CSB event stream — the gate for this
    action, since re-cutting the exposure only means anything for one."""
    return _source_path(signal) is not None


def _resolve_exposure(ds, params) -> float:
    """Seconds per output plane, from whichever knob the caller used.

    Three ways of saying the same thing, because all three get asked for: a
    frame RATE (fps) is what an in-situ experiment is described by, an EXPOSURE
    (ms) is what the camera is set to, and a FRAME COUNT is the atomic unit the
    stream actually quantises to.
    """
    dt = ds.frame_duration
    if dt <= 0:
        raise ValueError("this CSB file declares no frame cadence, so it has "
                         "no time axis to re-cut")
    n = int(params.get("frames_per_plane") or 0)
    if n > 0:
        return max(1, n) * dt
    ms = float(params.get("exposure_ms") or 0.0)
    if ms > 0:
        return max(ms * 1e-3, dt)
    fps = float(params.get("fps") or 0.0)
    if fps > 0:
        return max(1.0 / fps, dt)
    raise ValueError("choose a frame rate, an exposure or a number of frames "
                     "per plane")


def csb_to_frames(ctx, action_name: str = "To Frames", **params):
    """Toolbar entry point (ActionContext convention)."""
    plot, session = ctx.plot, ctx.session
    tree = getattr(plot, "signal_tree", None)
    if tree is None or session is None:
        emit_error("To Frames: no active dataset")
        return None
    src = tree.root
    path = _source_path(src)
    if path is None:
        emit_error("To Frames only applies to a CSB event stream")
        return None

    from spyde.actions.lifecycle import supersede
    handle = supersede(getattr(tree, "_to_frames_handle", None), tree)
    tree._to_frames_handle = handle

    def _work():
        if handle.stopped:
            return
        try:
            _rebuild(session, tree, src, path, params)
        except Exception as e:
            emit_error(f"To Frames failed: {e}")
            log.exception("To Frames failed")
        finally:
            handle.retire()

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="csb-to-frames")
    return None


def _rebuild(session, tree, src, path: str, params) -> None:
    """Re-cut the stream at a new exposure and add it to the tree."""
    import hyperspy.api as hs
    from spyde.external.rsciio_csb._api import (
        _dataset, _axes, _resolve_bin, lazy_stack, plane_counts,
    )

    backend = str(params.get("backend") or "auto")
    ds = _dataset(path, backend)
    bin_factor = _resolve_bin(ds, params.get("bin"))
    exposure = _resolve_exposure(ds, params)
    bounds = ds._time_bounds(exposure, 0.0, ds.duration)
    if not bounds:
        raise ValueError(
            f"a {exposure * 1e3:.4g} ms exposure selects no frames from this "
            f"{ds.duration * 1e3:.4g} ms movie")

    # Report the exposure in RAW FRAMES as well as in time. "16 planes at
    # 10 ms" does not tell you how much of the stream each one summed, and at
    # 1 frame/plane you are looking at the camera's actual output rather than
    # any integration of it — which is a different thing to be looking at.
    per_plane = bounds[0][1] - bounds[0][0]
    raw = ("1 raw frame each — no integration" if per_plane == 1
           else f"{per_plane} raw frames each")
    emit_status(f"To Frames: {len(bounds)} planes at "
                f"{exposure * 1e3:.4g} ms ({1.0 / exposure:.4g} fps), {raw}…")

    data = lazy_stack(path, bounds, ds.shape, bin_factor, np.float32, backend)
    sig = hs.signals.Signal2D(data).as_lazy()
    for axis, spec in zip(
            list(sig.axes_manager.navigation_axes)
            + list(sig.axes_manager.signal_axes),
            _axes(ds, bounds, bin_factor)):
        axis.name, axis.units = spec["name"], spec["units"]
        axis.scale, axis.offset = spec["scale"], spec["offset"]
    # `insitu` is what gates Play / Fast-Forward in toolbars.yaml; the nav axis
    # is already named "time" in seconds, which is what _is_movie_time_axis
    # looks for, but set it explicitly rather than relying on the sniff.
    try:
        sig.set_signal_type("insitu")
    except Exception as e:
        log.debug("set_signal_type(insitu) on the re-cut movie failed: %s", e)

    base = src.metadata.get_item("General.title", "CSB")
    sig.metadata.General.title = f"{base} @ {exposure * 1e3:.4g} ms"
    sig.original_metadata.add_dictionary({"csb": {
        **{k: v for k, v in ds.csb.info().items()},
        "exposure_s": exposure,
        "frames_per_plane": [int(f1 - f0) for f0, f1 in bounds],
        "plane_frame_bounds": [[int(f0), int(f1)] for f0, f1 in bounds],
        "bin": bin_factor,
        "backend": backend,
    }})

    # The free navigator again — re-cutting the exposure changes the planes, so
    # the overview has to be recomputed, but it still costs no payload reads.
    # Calibrated from the NEW signal: a 1-D selector derives its index from the
    # navigator's own axis, so an uncalibrated one pins every scrub position to
    # plane 0.
    from spyde.backend._session_files import calibrated_nav_signal
    nav = calibrated_nav_signal(plane_counts(ds, bounds), sig)

    def _add():
        session._add_signal(sig, source_path=path, navigator_override=nav)
        emit_status(f"To Frames: {len(bounds)} planes at "
                    f"{exposure * 1e3:.4g} ms ({1.0 / exposure:.4g} fps), {raw}")

    dispatch = getattr(session, "_dispatch_to_main", None)
    if dispatch is not None:
        dispatch(_add)
    else:
        _add()
