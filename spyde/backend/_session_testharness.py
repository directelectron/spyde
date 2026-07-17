"""
_session_testharness.py — TestHarnessMixin extracted from session.py.

Test-only data loaders + scripted-interaction entry points (synthetic/example
data, headless nav-drag, headless orientation). These back the Playwright e2e
suite and are gated behind the ``_TEST_ACTIONS_ENABLED`` check that lives in
``dispatch_action`` (ActionRouterMixin) — these are just the method bodies.

The mixin only USES ``self.<attr>`` (``self._plots``, ``self.signal_trees``,
``self._add_signal`` …) which are initialised / provided by the final Session.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
import hyperspy.api as hs

from spyde.backend import ipc
from spyde.backend.ipc import emit_error

log = logging.getLogger(__name__)


class TestHarnessMixin:
    def _dump_dask_state(self, only: str | None = None) -> None:
        """Log a compact snapshot of the dask cluster's state at WARNING level
        (so it lands in the Playwright harness's captured stderr even at
        SPYDE_LOG_LEVEL=WARNING): scheduler task-state histogram, per-worker
        executing/ready counts, and the call stacks of executing tasks.

        THE debugging tool for "the compute looks stuck" reports — a spec (or
        a human) fires `backendAction(page, 'dump_dask_state')` and reads the
        [dask-state] lines from ctx.backend.logBuffer. Runs on a worker thread
        (the client calls are sync round-trips; never block the main loop).

        ``only`` restricts to a single call type ("scheduler" | "info" |
        "call_stack") — used by the stall probe to isolate WHICH round-trip
        unsticks frozen task delivery."""
        def _work():
            try:
                dm = getattr(self, "dask_manager", None)
                client = getattr(dm, "client", None) if dm is not None else None
                if client is None:
                    log.warning("[dask-state] no dask client (threaded mode?)")
                    return
                log.warning("[dask-state] client=%s scheduler=%s only=%s",
                            client.status, client.scheduler.address, only)
                # Loop-identity forensics: is the dask client/scheduler running
                # on the app's MAIN asyncio loop (whose timers are dead unless
                # stdin wakes it) instead of its own LoopRunner thread?
                try:
                    c_loop = getattr(getattr(client, "loop", None),
                                     "asyncio_loop", None)
                    main_loop = getattr(self, "_main_loop", None)
                    sched = getattr(getattr(getattr(self, "dask_manager", None),
                                            "_cluster", None), "scheduler", None)
                    s_loop = getattr(getattr(sched, "loop", None),
                                     "asyncio_loop", None)
                    log.warning(
                        "[dask-state] loops: client=%r scheduler=%r main=%r "
                        "client_is_main=%s scheduler_is_main=%s",
                        c_loop, s_loop, main_loop,
                        c_loop is main_loop, s_loop is main_loop)
                except Exception as e:
                    log.warning("[dask-state] loop forensics failed: %s", e)
                # Task-state histogram straight from the scheduler.
                if only in (None, "scheduler"):
                    try:
                        counts = client.run_on_scheduler(
                            lambda dask_scheduler: {
                                s: sum(1 for t in dask_scheduler.tasks.values()
                                       if t.state == s)
                                for s in {t.state for t in
                                          dask_scheduler.tasks.values()}})
                    except Exception as e:
                        counts = f"<unavailable: {e}>"
                    log.warning("[dask-state] scheduler task states: %s", counts)
                if only in (None, "info"):
                    info = client.scheduler_info(n_workers=-1).get("workers", {})
                    for addr, w in info.items():
                        m = w.get("metrics", {})
                        log.warning("[dask-state] worker %s executing=%s ready=%s "
                                    "in_memory=%s cpu=%s%%", addr,
                                    m.get("executing"), m.get("ready"),
                                    m.get("in_memory"), m.get("cpu"))
                if only in (None, "call_stack"):
                    try:
                        stacks = client.call_stack()
                        if stacks:
                            for addr, tasks in stacks.items():
                                for key, frames in tasks.items():
                                    log.warning("[dask-state] EXECUTING %s :: %s\n%s",
                                                addr, key, "\n".join(frames[-6:]))
                        else:
                            log.warning("[dask-state] no tasks executing anywhere")
                    except Exception as e:
                        log.warning("[dask-state] call_stack failed: %s", e)
                log.warning("[dask-state] dump(only=%s) complete", only)
            except Exception as e:
                log.warning("[dask-state] dump failed: %s", e)

        threading.Thread(target=_work, daemon=True, name="dump-dask-state").start()

    def _load_test_data(self) -> None:
        """Load a synthetic 4D-STEM dataset (no file, no Dask, no download).

        Test-only entry point so Playwright can exercise the full live
        navigator→signal interaction deterministically. Each nav position has a
        distinct single bright pixel so a selector move produces a visibly
        different diffraction pattern.
        """
        import numpy as np
        # set_signal_type("electron_diffraction") imports pyxem — don't race the
        # startup prewarm's import (partially-initialized-module poisoning makes
        # the cast silently fail, and every diffraction-gated toolbar action
        # vanishes from the synthetic dataset).
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()
        nav, sig = (8, 8), (32, 32)
        data = np.zeros(nav + sig, dtype=np.float32)
        for i in range(nav[0]):
            for j in range(nav[1]):
                data[i, j, (i * 4) % 32, (j * 4) % 32] = 255.0
                data[i, j, 16, 16] = 60.0  # faint common center
        s = hs.signals.Signal2D(data)
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type(electron_diffraction) on synthetic data failed: %s", e)
        self._add_signal(s, source_path="test_data")

    def _load_test_data_lazy(self) -> None:
        """Synthetic LAZY 4D-STEM data — exercises the lazy+Dask path (Future
        compute, worker-thread display) that the eager `_load_test_data` doesn't.
        The central disk intensity varies per nav position so a virtual image of
        it is clearly structured (not uniform/black)."""
        import numpy as np
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()   # see _load_test_data — don't race the prewarm
        nav, sig = (8, 8), (32, 32)
        yy, xx = np.mgrid[0:32, 0:32]
        disk = ((xx - 16) ** 2 + (yy - 16) ** 2 <= 20).astype(np.float32)
        data = np.zeros(nav + sig, dtype=np.float32)
        for i in range(nav[0]):
            for j in range(nav[1]):
                data[i, j] = disk * (50.0 + i * 15.0 + j * 10.0)
        s = hs.signals.Signal2D(data).as_lazy()
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type(electron_diffraction) on synthetic data failed: %s", e)
        # CALIBRATE the signal axes (scale != 1, beam-centred). This is the real
        # scenario that exposed the "VI is just black" mask bug — anyplotlib ROI
        # widgets report PIXEL coords, so the detector mask must be built in pixel
        # space, not physical units. A scale=1 dataset hides that class of bug, so
        # the lazy test data is deliberately calibrated to guard against it.
        for ax in s.axes_manager.signal_axes:
            ax.scale = 0.1
            ax.offset = -(ax.size / 2.0) * 0.1
            ax.units = "1/nm"
        self._add_signal(s, source_path="test_data_lazy")

    def _load_test_data_lazy_chunked(self) -> None:
        """Test-only: LAZY 4D-STEM with MULTIPLE navigation chunks, so a crosshair
        drag crosses chunk boundaries and exercises the real distributed
        future→shm→PlotUpdateWorker→paint path (the in-chunk synthetic 8×8 is one
        chunk and never round-trips a worker). Each nav position has a single
        bright pixel at a position that varies with (iy, ix), so a frame change is
        unambiguous. nav=(24,24), signal=(32,32), nav chunks of 8 → a 3×3 chunk
        grid; signal axes span the full frame (storage-aligned)."""
        import numpy as np
        import dask.array as da
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()   # see _load_test_data — don't race the prewarm
        ny, nx, ky, kx = 24, 24, 32, 32
        data = np.zeros((ny, nx, ky, kx), dtype=np.float32)
        for i in range(ny):
            for j in range(nx):
                data[i, j, (i * 1) % ky, (j * 1) % kx] = 255.0
                data[i, j, 16, 16] = 60.0  # faint common centre
        dask_data = da.from_array(data, chunks=(8, 8, ky, kx))  # 3×3 nav chunk grid
        s = hs.signals.Signal2D(dask_data).as_lazy()
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type on chunked lazy data failed: %s", e)
        self._add_signal(s, source_path="test_data_lazy_chunked")

    def _load_test_data_si_grains(self) -> None:
        """Test-only: BUNDLED synthetic Si-grains 4-D STEM (pyxem.data.si_grains —
        6×6 nav × 128×128 signal, generated on the fly, NO download). Unlike the
        featureless-disk fixtures it has a real reciprocal lattice with crisp
        spots, so the orientation overlay's matched template lands on actual
        peaks — the offline/CI counterpart to sped_ag for OM/overlay tests."""
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()   # don't race the startup prewarm's pyxem import
        import pyxem.data as pxd
        s = pxd.si_grains()
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type on si_grains failed: %s", e)
        # Centre the beam in pixel space (the bundled data carries offset=0, i.e.
        # k=0 at pixel 0); the overlay/matcher map k→px via (k-offset)/scale, so a
        # centred offset puts the direct beam mid-detector like a real scan.
        for ax in s.axes_manager.signal_axes:
            ax.offset = -(ax.size / 2.0) * float(ax.scale)
        self._add_signal(s, source_path="test_data_si_grains")

    def _load_test_data_sped_ag(self) -> None:
        """Test-only: load the REAL sped_ag 4-D STEM scan (pyxem.data.sped_ag —
        208×64 patterns of 112×112, a strained Ag SPED dataset with genuine
        diffraction spots). Unlike the synthetic disk fixtures this has a real
        reciprocal lattice, so the orientation overlay's matched-template spots
        land on actual diffraction peaks — needed to SEE the overlay working
        (not just render). Downloads on first use (pooch-cached)."""
        import pyxem.data as pxd
        s = pxd.sped_ag(allow_download=True)
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type on sped_ag failed: %s", e)
        self._add_signal(s, source_path="test_data_sped_ag")

    def _load_test_data_line(self, payload: dict | None = None) -> None:
        """Test-only: a small calibrated 1-D signal (``Signal1D``) with NO
        navigation axes — so exactly ONE signal plot window is created
        (``is_navigator=False``) and no navigator. Backs the report line-panel
        e2e (dropping a 1-D window into a report makes a ``kind="line"`` panel).

        A 512-point synthetic spectrum: a slowly-rising baseline plus two
        Gaussian peaks at distinct positions, so the rendered curve has clear
        structure (a recoloured line / a set label + legend is pixel-visible),
        and a calibrated x-axis (0.5 eV/px) so the line panel carries real axis
        units. Mirrors ``test_report_line_panels.py``'s ``signal1d_dataset``
        fixture construction."""
        import numpy as np
        # A 1-D signal never casts to a diffraction/insitu type, but the pyxem
        # import prewarm still races _add_signal's plot machinery — keep the
        # ensure_heavy_imports guard the other loaders use (see _load_test_data).
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()
        payload = payload or {}
        n = int(payload.get("size", 512))
        x = np.arange(n, dtype=np.float32)
        baseline = 0.15 + 0.35 * (x / n)
        def _gauss(c, w, a):
            return a * np.exp(-0.5 * ((x - c) / w) ** 2)
        y = (baseline + _gauss(n * 0.30, n * 0.02, 1.0)
             + _gauss(n * 0.65, n * 0.03, 0.7)).astype(np.float32)
        s = hs.signals.Signal1D(y)
        ax = s.axes_manager.signal_axes[0]
        ax.name, ax.units, ax.scale, ax.offset = "Energy", "eV", 0.5, 0.0
        s.metadata.set_item("General.title", "1D Spectrum")
        self._add_signal(s, source_path="test_data_line")

    def _load_test_data_movie(self, payload: dict | None = None) -> None:
        """Test-only: synthetic in-situ MOVIE (1-D time nav × LARGE 2-D frames) —
        the GPU large-image consumer, with NO file and NO download so the
        WebGPU/tile specs run everywhere. Frames are 2048² by default (≥1024
        edge → anyplotlib tile mode via SpyDE's GpuTileBackend; >1 Mpx → the
        WebGPU image path above GPU_IMAGE_THRESHOLD).

        Frame content is designed for coordinate/parity assertions:
          - x+y gradient + a bright TOP-LEFT block + dimmer BOTTOM-RIGHT block:
            any flip / mirror / mis-registration of the render is visible;
          - a bright vertical band whose position encodes the FRAME INDEX, so a
            scrub visibly changes the frame (and a stale frame is detectable);
          - a fine 2-px checkerboard patch at the centre: aliases to grey in the
            downsampled overview, resolves in the crisp native detail tile.

        Chunked 1 frame/chunk lazy, like a real .mrc in-situ movie (Live-Display
        §3: each nav move is a small cold read of just that frame).
        """
        import numpy as np
        import dask.array as da
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()   # see _load_test_data — don't race the prewarm
        payload = payload or {}
        n = int(payload.get("size", 2048))
        n_frames = int(payload.get("frames", 6))
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        base = (xx / n) * 250.0 + (yy / n) * 250.0
        base[(yy < n // 6) & (xx < n // 6)] = 1000.0          # TOP-LEFT block
        base[(yy > 5 * n // 6) & (xx > 5 * n // 6)] = 800.0   # BOTTOM-RIGHT block
        # Fine checkerboard patch (2-px pitch) in the centre quarter.
        cb = slice(3 * n // 8, 5 * n // 8)
        checker = (((xx[cb, cb] // 2).astype(np.int32)
                    + (yy[cb, cb] // 2).astype(np.int32)) % 2) * 400.0 + 200.0
        frames = np.empty((n_frames, n, n), dtype=np.uint16)
        for t in range(n_frames):
            f = base.copy()
            f[cb, cb] = checker
            x0 = (t + 1) * n // (n_frames + 2)
            f[:, x0:x0 + n // 32] = 900.0                     # frame-index band
            frames[t] = f.astype(np.uint16)
        stack = da.from_array(frames, chunks=(1, n, n))       # 1 frame/chunk
        s = hs.signals.Signal2D(stack).as_lazy()
        for ax, unit in zip(s.axes_manager.signal_axes, ("nm", "nm")):
            ax.scale = 0.5
            ax.units = unit
        # CALIBRATE the TIME axis (like the DE-MRC reader gives an in-situ movie):
        # 0.05 s/frame → real-time Play runs at 20 fps. Without this the axis keeps
        # scale=1 (→ 1 fps) and the movie crawls; a calibrated axis exercises the
        # real-time pacing path a real dataset takes.
        tax = s.axes_manager.navigation_axes[0]
        tax.name, tax.units, tax.scale = "time", "s", 0.05
        s.set_signal_type("insitu")   # gates the Play/Fast Forward toolbar buttons
        self._add_signal(s, source_path="test_data_movie")

    def _test_add_second_navigator(self) -> None:
        """Test-only: register a second NAMED 1-D navigator trace on the
        in-situ MOVIE tree (root ``_signal_type == "insitu"``), so Playwright
        can exercise the stacked-navigator chip strip
        (``navigator_views._stack_navigators``) without needing a real second
        navigator source. ``load_test_data_movie``'s synthetic movie only
        ever gets one navigator ("base") — the chip strip needs
        ``len(navigator_signals) >= 2`` to appear at all.

        Targets the LAST loaded in-situ tree specifically (not just
        ``signal_trees[-1]``) — a test may load other datasets (e.g. the
        4D-STEM negative-gate fixture) after the movie, which would otherwise
        make this add the trace to the WRONG (non-1-D-nav) tree and fail the
        shape check in ``add_navigator_signal``.

        Builds a simple sine trace over the same 1-D navigation shape as the
        tree root (``add_navigator_signal`` enforces identical total shape),
        named "trace", and re-emits the navigator chip options so the
        renderer's chip strip shows up immediately.
        """
        import numpy as np
        from hyperspy.signal import BaseSignal

        try:
            tree = next((t for t in reversed(self.signal_trees)
                         if getattr(t.root, "_signal_type", None) == "insitu"), None)
            if tree is None:
                emit_error("test_add_second_navigator: no in-situ movie tree loaded")
                return
            n = int(tree.root.axes_manager.navigation_shape[0])
            trace = 0.5 * np.sin(np.linspace(0, 4 * np.pi, n)) + 1.0
            sig = BaseSignal(trace.astype(np.float32))
            sig.axes_manager[0].name = "time"
            tree.add_navigator_signal("trace", sig)
            log.info("test_add_second_navigator: added 'trace' navigator (n=%d)", n)
        except Exception as e:
            log.exception("test_add_second_navigator failed")
            emit_error(f"test_add_second_navigator failed: {e}")

    def _test_nav_drag(self, targets: list) -> None:
        """Test-only: drive the navigator crosshair through a list of (x, y) nav
        cells server-side and report, per move, whether the SIGNAL plot's painted
        data actually changed. Bypasses the iframe widget harness so a Playwright
        test can deterministically exercise the distributed future→shm→worker→
        paint path. Emits {"type":"nav_drag_result", ...}.

        For each target: set the crosshair widget position, fire the selector,
        then poll the signal plot's current_data (the painted numpy frame) until
        it changes or a timeout. Records CHANGED / NO-CHANGE + the DP's argmax so
        we can see WHICH frame painted.
        """
        import time as _time
        try:
            tree = self.signal_trees[-1] if self.signal_trees else None
            if tree is None or tree.navigator_plot_manager is None:
                ipc.emit({"type": "nav_drag_result", "error": "no navigator tree"})
                return
            mgr = tree.navigator_plot_manager
            pw = next(iter(mgr.navigation_selectors.keys()))
            sel = mgr.navigation_selectors[pw][0]
            cross = getattr(sel, "_crosshair_selector", sel)
            child = next(iter(sel.children.keys()))

            def _frame_sig():
                d = getattr(child, "current_data", None)
                if isinstance(d, np.ndarray) and d.size:
                    return (int(d.argmax()), float(d.sum()))
                return None

            def _cd_kind():
                d = getattr(child, "current_data", None)
                return type(d).__name__

            def _set_pos(sel_obj, x, y):
                """Move the navigator widget to FRAME/CELL index (x, y). A 2-D
                navigator crosshair uses cx/cy in IMAGE-PIXEL (index) coords; a 1-D
                (movie/time) navigator is an InfiniteLineSelector wrapping a
                VLineWidget whose ``.x`` is in DATA coordinates of the (possibly
                calibrated) time axis — NOT the frame index. A real iframe drag
                reports xdata in data coords, so we must convert the requested
                frame index through the axis calibration (``x_data = offset +
                index*scale``); setting ``w.x = index`` directly on a calibrated
                axis (e.g. a movie's 0.05 s/frame time axis) lands the selector at
                ``round(index/scale)``, which clips to the last frame and freezes
                the navigator (see anyplotlib 2-D widget pixel-coords memo: 1-D
                widgets use data coords, 2-D use pixels). The composite
                IntegratingSelector1D delegates to its active inner selector via
                ``.selector``."""
                inner = getattr(sel_obj, "selector", sel_obj)
                w = getattr(inner, "_widget", None) or getattr(sel_obj, "_widget", None)
                if w is None:
                    raise RuntimeError("selector has no widget")
                if hasattr(w, "cx"):          # 2-D crosshair (pixel/index coords)
                    w.cx = float(x)
                    w.cy = float(y)
                else:                          # 1-D VLineWidget — DATA coords
                    from spyde.drawing.selectors.selector1d import _signal_axis
                    scale, offset = _signal_axis(inner)
                    w.x = offset + float(x) * scale

            results = []
            prev = _frame_sig()
            for (x, y) in targets:
                try:
                    _set_pos(cross, x, y)
                except Exception as e:
                    results.append({"x": x, "y": y, "changed": False, "err": str(e)})
                    continue
                sel.delayed_update_data(force=True)
                cur = prev
                t_end = _time.monotonic() + 3.0
                while _time.monotonic() < t_end:
                    cur = _frame_sig()
                    if cur is not None and cur != prev:
                        break
                    _time.sleep(0.03)
                changed = cur is not None and cur != prev
                results.append({"x": x, "y": y, "changed": bool(changed),
                                "sig": cur, "prev": prev, "cd_kind": _cd_kind()})
                prev = cur
            n_changed = sum(1 for r in results if r.get("changed"))
            ipc.emit({"type": "nav_drag_result", "total": len(targets),
                  "changed": n_changed, "results": results})
            log.info("[REDRAW] test_nav_drag: %d/%d moves changed the DP",
                     n_changed, len(targets))
        except Exception as e:
            log.exception("test_nav_drag failed")
            ipc.emit({"type": "nav_drag_result", "error": str(e)})

    def _test_region_scrub(self, payload: dict) -> None:
        """Test-only: exercise the TIERED nav read's EXPENSIVE tier via an
        integrating REGION. Switches the navigator to integrate mode, sets the
        region widget oversized (to prove the extent cap clamps its geometry),
        then scrubs the region to a couple of positions and reports, per move,
        whether the SIGNAL plot's painted frame changed (an ndarray landed, i.e.
        the async submit_graph read painted). Emits {"type":"region_scrub_result"}.
        """
        import time as _time
        from spyde.drawing.selectors.base_selector import MAX_REGION_EXTENT_PER_DIM
        try:
            tree = self.signal_trees[-1] if self.signal_trees else None
            if tree is None or tree.navigator_plot_manager is None:
                ipc.emit({"type": "region_scrub_result", "error": "no navigator tree"})
                return
            mgr = tree.navigator_plot_manager
            pw = next(iter(mgr.navigation_selectors.keys()))
            sel = mgr.navigation_selectors[pw][0]
            child = next(iter(sel.children.keys()))

            # Switch to integrate mode → the region sub-selector becomes active.
            sel.set_integrating(True)
            region = getattr(sel, "selector", sel)
            w = getattr(region, "_widget", None)
            if w is None:
                ipc.emit({"type": "region_scrub_result", "error": "no region widget"})
                return

            def _frame_sig():
                d = getattr(child, "current_data", None)
                if isinstance(d, np.ndarray) and d.size:
                    return (int(d.argmax()), float(d.sum()))
                return None

            clamp_info = {}
            results = []
            prev = _frame_sig()
            # A 1-D (movie/time) region uses x0/x1; a 2-D region uses x/y/w/h.
            is_1d = hasattr(w, "x0")
            # 1-D range widget lives in DATA coords (like the VLine) — convert the
            # requested frame-index positions through the axis calibration so a
            # calibrated time axis (movie: 0.05 s/frame) doesn't clip everything to
            # the last frame. 2-D region is in pixel/index coords.
            _scale1d, _offset1d = (1.0, 0.0)
            if is_1d:
                from spyde.drawing.selectors.selector1d import _signal_axis
                _scale1d, _offset1d = _signal_axis(region)
            positions = payload.get("positions") or ([[10], [40], [90]] if is_1d else [[3, 3], [20, 20]])
            for pos in positions:
                try:
                    if is_1d:
                        # Set an OVERSIZED span (60 indices) → must clamp to <=16.
                        x0d = _offset1d + float(pos[0]) * _scale1d
                        w.x0 = x0d
                        w.x1 = x0d + 60.0 * _scale1d
                    else:
                        w.x = float(pos[0])
                        w.y = float(pos[1])
                        w.w = 60.0   # oversized → clamp to <=16
                        w.h = 60.0
                    region._clamp_extent()
                except Exception as e:
                    results.append({"pos": pos, "changed": False, "err": str(e)})
                    continue
                # Record the clamped extent for the first move. The 1-D widget is
                # in DATA units, so divide the span by the axis scale to report it
                # in INDEX units — matching the index-based cap the spec asserts.
                if not clamp_info:
                    if is_1d:
                        span_data = abs(float(w.x1) - float(w.x0))
                        span_idx = span_data / abs(_scale1d) if _scale1d else span_data
                        clamp_info = {"span": span_idx}
                    else:
                        clamp_info = {"w": float(w.w), "h": float(w.h)}
                sel.delayed_update_data(force=True)
                cur = prev
                t_end = _time.monotonic() + 5.0
                while _time.monotonic() < t_end:
                    cur = _frame_sig()
                    if cur is not None and cur != prev:
                        break
                    _time.sleep(0.03)
                changed = cur is not None and cur != prev
                results.append({"pos": pos, "changed": bool(changed),
                                "sig": cur, "prev": prev})
                prev = cur
            n_changed = sum(1 for r in results if r.get("changed"))
            ipc.emit({"type": "region_scrub_result", "total": len(positions),
                      "changed": n_changed, "clamp": clamp_info, "results": results,
                      "cap": MAX_REGION_EXTENT_PER_DIM})
            log.info("[REDRAW] test_region_scrub: %d/%d region moves painted; clamp=%s cap=%d",
                     n_changed, len(positions), clamp_info, MAX_REGION_EXTENT_PER_DIM)
        except Exception as e:
            log.exception("test_region_scrub failed")
            ipc.emit({"type": "region_scrub_result", "error": str(e)})

    def _load_test_vectors(self) -> None:
        """Test-only: load a small calibrated 4D-STEM stack (two disks per
        pattern) and run Find Diffraction Vectors on it, so the vectors-image
        window opens cleanly (no picker, no wizard occlusion). Lets Playwright
        exercise the downstream vector actions (Vector Virtual Imaging / Vector
        Orientation Mapping) E2E."""
        import numpy as np
        import hyperspy.api as hs
        nav, sig = (6, 6), (32, 32)
        yy, xx = np.mgrid[0:32, 0:32]
        # Four disks per pattern (≥4 vectors) so the downstream Vector
        # Orientation per-pattern fit actually runs (not skipped for too-few).
        spots = [(16, 16), (23, 9), (8, 21), (22, 24)]
        pat = np.zeros(sig, dtype=np.float32)
        for sxx, syy in spots:
            pat += ((xx - sxx) ** 2 + (yy - syy) ** 2 <= 7).astype(np.float32)
        data = np.zeros(nav + sig, dtype=np.float32)
        for i in range(nav[0]):
            for j in range(nav[1]):
                data[i, j] = pat * 100.0
        s = hs.signals.Signal2D(data)
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type(electron_diffraction) on synthetic data failed: %s", e)
        for ax in s.axes_manager.signal_axes:
            ax.scale = 0.1
            ax.offset = -(ax.size / 2.0) * 0.1
            ax.units = "1/nm"
        self._add_signal(s, source_path="test_vectors")

        src = next((p for p in self._plots
                    if not p.is_navigator and p.plot_state is not None), None)
        if src is None:
            emit_error("load_test_vectors: no active signal")
            return
        from spyde.actions.context import ActionContext
        from spyde.actions.find_vectors_action import find_diffraction_vectors
        ctx = ActionContext(plot=src, params={}, action_name="Find Diffraction Vectors")
        find_diffraction_vectors(
            ctx, sigma=1.0, kernel_radius=5, threshold=0.4,
            min_distance=3, subpixel=True,
        )

    def _run_test_orientation(self, plot, payload=None) -> None:
        """Test-only Orientation Mapping with a built-in phase (no CIF dialog), so
        the full OM workflow can be exercised E2E (incl. lazy data) without a file
        picker. Mirrors `orientation_action.orientation_mapping`.

        The phase MUST match the loaded data or the match fails (black IPF, no
        overlay). ``payload={"phase": "si"|"ag"}`` selects it; default Si, since
        the primary test dataset is the bundled ``si_grains`` scan. Use "ag" for a
        real ``sped_ag`` load."""
        payload = payload or {}
        src = plot or next(
            (p for p in self._plots if not p.is_navigator and p.plot_state is not None),
            None,
        )
        # The overlay must land on the SIGNAL diffraction pattern, not the
        # navigator. If the action arrived from the navigator window (or no plot
        # was passed and the first match was a navigator), fall back to the first
        # non-navigator signal plot so src_dp_plot is always the DP.
        if src is not None and getattr(src, "is_navigator", False):
            src = next(
                (p for p in self._plots
                 if not p.is_navigator and p.plot_state is not None),
                None,
            )
        if src is None:
            emit_error("run_test_orientation: no active signal")
            return
        tree = getattr(src, "signal_tree", None)
        if tree is None:
            emit_error("run_test_orientation: no signal tree")
            return

        def _work():
            try:
                from orix.crystal_map import Phase
                from diffpy.structure import Atom, Lattice, Structure
                from spyde.actions.orientation_action import run_orientation
                which = str(payload.get("phase", "si")).lower()
                if which == "ag":
                    # FCC silver (a=4.0853 Å) — matches the real sped_ag scan.
                    structure = Structure(
                        atoms=[Atom("Ag", [0, 0, 0]), Atom("Ag", [.5, .5, 0]),
                               Atom("Ag", [.5, 0, .5]), Atom("Ag", [0, .5, .5])],
                        lattice=Lattice(4.0853, 4.0853, 4.0853, 90, 90, 90),
                    )
                    phase = Phase(name="Ag", space_group=225, structure=structure)
                else:
                    # Diamond-cubic silicon (a=5.4307 Å) — matches the bundled
                    # si_grains test scan so the template lands on its real spots.
                    structure = Structure(
                        atoms=[Atom("Si", [0, 0, 0]), Atom("Si", [.5, .5, 0]),
                               Atom("Si", [.5, 0, .5]), Atom("Si", [0, .5, .5]),
                               Atom("Si", [.25, .25, .25]), Atom("Si", [.75, .75, .25]),
                               Atom("Si", [.75, .25, .75]), Atom("Si", [.25, .75, .75])],
                        lattice=Lattice(5.4307, 5.4307, 5.4307, 90, 90, 90),
                    )
                    phase = Phase(name="Si", space_group=227, structure=structure)
                run_orientation(
                    self, tree.root, tree, [phase],
                    dict(accelerating_voltage=200.0, resolution=8.0),
                    dict(n_best=3, gamma=0.5), src_dp_plot=src,
                )
            except Exception as e:
                emit_error(f"run_test_orientation failed: {e}")
                log.exception("run_test_orientation failed")

        threading.Thread(target=_work, daemon=True, name="test-orientation").start()
