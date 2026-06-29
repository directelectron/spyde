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

from spyde.backend.ipc import emit, emit_error

log = logging.getLogger(__name__)


class TestHarnessMixin:
    def _load_test_data(self) -> None:
        """Load a synthetic 4D-STEM dataset (no file, no Dask, no download).

        Test-only entry point so Playwright can exercise the full live
        navigator→signal interaction deterministically. Each nav position has a
        distinct single bright pixel so a selector move produces a visibly
        different diffraction pattern.
        """
        import numpy as np
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
                emit({"type": "nav_drag_result", "error": "no navigator tree"})
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

            results = []
            prev = _frame_sig()
            for (x, y) in targets:
                try:
                    cross._widget.cx = float(x)
                    cross._widget.cy = float(y)
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
            emit({"type": "nav_drag_result", "total": len(targets),
                  "changed": n_changed, "results": results})
            log.info("[REDRAW] test_nav_drag: %d/%d moves changed the DP",
                     n_changed, len(targets))
        except Exception as e:
            log.exception("test_nav_drag failed")
            emit({"type": "nav_drag_result", "error": str(e)})

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
