"""
Staged Find-Diffraction-Vectors wizard backend (Qt parity):

  fv_preview → attaches a LIVE found-peaks preview overlay to the source DP so
               red circles update as you tune the sliders / move the navigator.
  fv_tune    → live-updates the preview sliders (σ / kernel radius / threshold /
               min distance / subpixel) and redraws at the current crosshair —
               NO full-dataset compute.
  fv_run     → full-dataset batch with the tuned params → a new vectors-image
               window with `tree.diffraction_vectors` attached.
  fv_stop    → removes the live preview overlay (caret closed).

Memory safety: the preview only slices/computes a small nav window (radius
ceil(3σ)) around the crosshair — never the full dataset (see
test_find_vectors_memory for the batch-compute contract).
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=25.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _calibrated_diffraction_4d(nav=(4, 5), sig=(24, 24), scale=0.1):
    """A 4D-STEM stack: every pattern has a bright disk at the centre (so the
    NXCORR peak finder reliably finds at least one peak)."""
    data = np.zeros(nav + sig, dtype=np.float32)
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - 12) ** 2 + (yy - 12) ** 2 <= 16).astype(np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * 100.0
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = 0.0
        ax.units = "1/nm"
    return s


class TestFindVectorsWizard:
    def test_preview_tune_run(self):
        from spyde.backend.session import Session
        from spyde.actions.find_vectors_action import (
            fv_preview, fv_tune, fv_run, fv_stop,
        )
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1))
            time.sleep(0.4)
            src = _signal_plot(session)
            tree = src.signal_tree

            # ── Tune: start the live preview ─────────────────────────────────
            fv_preview(session, src, {
                "sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                "min_distance": 3, "subpixel": True,
            })
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is not None), \
                "live preview overlay never attached"
            prev = tree._fv_preview
            assert prev._mg is not None              # circle marker group exists

            # The centred disk produces at least one peak at the current crosshair.
            # _offsets_for is now the engine's compute → returns {offsets, response}.
            assert _wait(lambda: len(prev._offsets_for(0, 0)["offsets"]) >= 1), \
                "preview found no peaks on a disk pattern"

            # ── Tune: change params live (no new window) ─────────────────────
            before_trees = len(session.signal_trees)
            fv_tune(session, src, {
                "sigma": 1.0, "kernel_radius": 7, "threshold": 0.3,
                "min_distance": 4, "subpixel": False,
            })
            assert _wait(lambda: prev.kernel_radius == 7 and prev.subpixel is False)
            assert abs(prev.threshold - 0.3) < 1e-9
            assert len(session.signal_trees) == before_trees   # tune never computes

            # ── Compute: full-dataset batch → a new vectors window ───────────
            fv_run(session, src, {
                "sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                "min_distance": 3, "subpixel": True,
            })
            assert _wait(lambda: len(session.signal_trees) == before_trees + 1,
                         timeout=40), "vectors window never opened"
            vtree = session.signal_trees[-1]
            assert _wait(lambda: getattr(vtree, "diffraction_vectors", None) is not None,
                         timeout=40), "diffraction_vectors never attached"
            assert int(vtree.diffraction_vectors.count_map().sum()) > 0

            # Running drops the live preview (the final overlay replaces it).
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is None)
            # …and attaches the persistent found-vector overlay on the source DP.
            assert _wait(lambda: getattr(tree, "_vector_overlay", None) is not None)
        finally:
            session.shutdown()

    def test_stop_removes_preview(self):
        from spyde.backend.session import Session
        from spyde.actions.find_vectors_action import fv_preview, fv_stop
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1))
            time.sleep(0.4)
            src = _signal_plot(session)
            tree = src.signal_tree

            fv_preview(session, src, {"sigma": 1.0, "kernel_radius": 5,
                                      "threshold": 0.4, "min_distance": 3,
                                      "subpixel": True})
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is not None)
            prev = tree._fv_preview

            fv_stop(session, src, {})
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is None)
            assert prev._mg is None                  # marker group removed
        finally:
            session.shutdown()


class TestPreviewBeamstopToggle:
    def test_set_params_beamstop_auto_toggles_mask(self):
        """Toggling 'Mask beam stop' applies/clears the mask on the live preview.
        Detection is ASYNC (a ~400-frame scan runs on a bg thread so it never
        blocks the live re-render), so we wait for the scan to land."""
        import time
        from spyde.actions.vector_overlay import FindVectorsPreviewOverlay

        # a synthetic signal whose scan-mean has a dark beam-stop bar
        ny, nx, ky, kx = 6, 6, 32, 32
        f = np.full((ky, kx), 1000.0, np.float32)
        f[:24, 14:18] = 1.0                       # dark bar in every pattern
        data = np.broadcast_to(f, (ny, nx, ky, kx)).copy()
        sig = hs.signals.Signal2D(data)

        ov = FindVectorsPreviewOverlay.__new__(FindVectorsPreviewOverlay)
        ov.signal = sig
        ov.sigma = 0.0; ov.kernel_radius = 5; ov.threshold = 0.4
        ov.min_distance = 3; ov.subpixel = True; ov.method = "dog"
        ov.dog_sigma1 = 0.8; ov.dog_sigma2 = 2.0
        ov.beamstop_mask = None
        ov._last_iyix = (3, 3)
        ov.show_transform = False
        ov._hidden = False
        ov._beamstop_wanted = False
        ov._beamstop_scanning = False
        ov._engine = None        # set_params skips the async re-render request

        # toggle ON → async scan kicked off; wait for it to apply the mask
        ov.set_params(beamstop_auto=True)
        t0 = time.time()
        while time.time() - t0 < 5.0 and ov.beamstop_mask is None:
            time.sleep(0.02)
        assert ov.beamstop_mask is not None, "beam-stop scan never applied a mask"
        assert ov.beamstop_mask[10, 15]           # on the bar

        # toggle OFF → mask cleared (synchronous)
        ov.set_params(beamstop_auto=False)
        assert ov.beamstop_mask is None

        # toggle ON again → cached, applied without a re-scan
        ov.set_params(beamstop_auto=True)
        assert ov.beamstop_mask is not None

    def test_mask_overlay_pushed_to_plot(self):
        """The detected beam-stop mask is drawn as a translucent overlay on the
        DP plot (set_overlay_mask), re-pushed cheaply on dilation change, and
        CLEARED when the stop is toggled off — all without re-detecting (the
        static stop is cached as `_beamstop_raw`)."""
        from spyde.actions.vector_overlay import FindVectorsPreviewOverlay

        calls = []                       # captured (mask px or None) per push
        class _DP:
            def set_overlay_mask(self, mask, color="#ff4444", alpha=0.4):
                calls.append(None if mask is None else int(np.count_nonzero(mask)))

        ov = FindVectorsPreviewOverlay.__new__(FindVectorsPreviewOverlay)
        ov.dp_plot = _DP()
        ov.beamstop_mask = None
        ov.beamstop_dilate = 5
        ov._beamstop_wanted = True
        raw = np.zeros((64, 64), bool)
        yy, xx = np.ogrid[:64, :64]
        raw[(yy - 32) ** 2 + (xx - 32) ** 2 < 36] = True   # disk, undilated
        ov._beamstop_raw = raw

        ov._apply_dilation()
        assert calls[-1] is not None and calls[-1] >= int(raw.sum())
        px5 = calls[-1]

        # bigger dilation → bigger overlay, NO re-scan (raw is reused)
        ov.beamstop_dilate = 12
        ov._apply_dilation()
        assert calls[-1] > px5

        # toggle off → overlay cleared
        ov._beamstop_wanted = False
        ov.beamstop_mask = None
        ov._push_mask_overlay()
        assert calls[-1] is None

    def test_show_transform_compute_and_render(self):
        """With show_transform on, the engine compute returns BOTH peaks and the
        response, and _render_payload pushes the response image to the DP plot
        (via the Plot's set_data so it persists) plus the markers."""
        from spyde.actions.vector_overlay import FindVectorsPreviewOverlay

        ny, nx, ky, kx = 4, 4, 32, 32
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:ky, 0:kx]
        base = rng.normal(50, 3, (ky, kx)).astype(np.float32)
        for c in ((10, 10), (22, 22)):
            base += 300 * np.exp(-((yy - c[0]) ** 2 + (xx - c[1]) ** 2) / (2 * 1.3 ** 2))
        data = np.broadcast_to(base, (ny, nx, ky, kx)).copy()
        sig = hs.signals.Signal2D(data)

        pushed = []
        marked = []
        hist = []
        class _DP:                       # Plot-level transform paint (new contract)
            needs_auto_level = False
            def set_transform_image(self, arr, levels=None): pushed.append((arr, levels))
            def _emit_histogram(self, arr, lo, hi, threshold=None): hist.append((lo, hi, threshold))
        class _MG:
            def set(self, **kw): marked.append(kw.get("offsets"))

        ov = FindVectorsPreviewOverlay.__new__(FindVectorsPreviewOverlay)
        ov.signal = sig; ov.dp_plot = _DP(); ov._mg = _MG()
        ov.sigma = 0.0; ov.kernel_radius = 5; ov.threshold = 8.0
        ov.min_distance = 3; ov.subpixel = True; ov.method = "dog"
        ov.dog_sigma1 = 0.8; ov.dog_sigma2 = 2.0; ov.beamstop_mask = None
        ov.show_transform = True; ov._hidden = False
        ov._last_iyix = (2, 2)

        # 1) compute returns a payload with peaks AND a response (no side-effect)
        payload = ov._offsets_for(2, 2)
        assert isinstance(payload, dict)
        assert payload["response"] is not None
        assert payload["response"].shape == (ky, kx)
        assert len(payload["offsets"]) >= 2
        assert not pushed, "compute must not paint (render does)"

        # 2) render pushes the response image to the DP and the markers, with the
        #    display floor (clim-min) snapped to the detector threshold and a
        #    histogram threshold marker emitted.
        ov._render_payload(payload)
        assert pushed and pushed[-1][0].shape == (ky, kx)
        assert pushed[-1][1][0] == 8.0          # clim-min == threshold
        assert hist and hist[-1][2] == 8.0       # histogram threshold marker
        assert marked and len(marked[-1]) >= 2

    def test_nxcorr_transform_clim_ceiling_is_fixed(self):
        """Moving the threshold in correlation view must NOT move the clim ceiling
        (the earlier flash: hi was tied to threshold, so nudging it washed the
        image to white). NXCORR ceiling is a fixed 1.0; only the floor moves."""
        from spyde.actions.vector_overlay import FindVectorsPreviewOverlay
        pushed = []
        class _DP:
            needs_auto_level = False
            def set_transform_image(self, arr, levels=None): pushed.append(levels)
            def _emit_histogram(self, *a, **k): pass
        class _MG:
            def set(self, **kw): pass

        ov = FindVectorsPreviewOverlay.__new__(FindVectorsPreviewOverlay)
        ov.dp_plot = _DP(); ov._mg = _MG(); ov.method = "nxcorr"
        ov.show_transform = True; ov._hidden = False

        resp = np.linspace(-1, 0.64, 32 * 32).reshape(32, 32).astype(np.float32)
        for thr in (0.5, 0.6, 0.7, 0.9):
            ov.threshold = thr
            ov._render_payload({"offsets": np.zeros((0, 2), np.float32),
                                "response": resp})
            lo, hi = pushed[-1]
            assert hi == 1.0, f"NXCORR clim ceiling moved to {hi} (flash)"
            assert abs(lo - thr) < 1e-6, f"clim floor != threshold ({lo} vs {thr})"
