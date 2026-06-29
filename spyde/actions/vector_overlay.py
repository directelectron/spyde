"""
vector_overlay.py — live marker overlays on a diffraction-pattern plot.

The Qt app drew the found diffraction vectors (and the matched orientation
template) as scatter markers on top of the live diffraction pattern, updating as
you moved the navigator.  This is the Electron/anyplotlib equivalent: it adds a
``circles`` MarkerGroup to the DP plot and re-pushes its offsets whenever the
navigator selection changes (via :attr:`BaseSelector.index_hooks`).

Coordinate system (the subtle part):

* diffraction vectors are stored in *calibrated* units (kx, ky in 1/nm); the
  rendered-disk frames map them to pixels with ``px = (k - offset) / scale``
  using ``sig_axes[0]`` for the column (kx, width) and ``sig_axes[1]`` for the
  row (ky, height) — see ``_render_disks_block``.
* anyplotlib markers with ``transform="data"`` are addressed in *image-pixel*
  coordinates (no axis scale/offset; extent only relabels ticks).  So we convert
  calibrated → pixel here, the same way the renderer does, and the markers land
  exactly on the rendered disks.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)


def _stats(arr) -> str:
    """Compact mean/std/min/max/shape summary for debug logging of image data."""
    try:
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return "empty"
        return (f"shape={tuple(a.shape)} mean={a.mean():.3g} std={a.std():.3g} "
                f"min={a.min():.3g} max={a.max():.3g}")
    except Exception:
        return "??"


def _indices_to_iyix(indices):
    """A navigator crosshair reports ``[[ix, iy]]`` (cx=column=nav-x,
    cy=row=nav-y). Return ``(iy, ix)`` for the vectors' ``at(iy, ix)`` — the
    SPATIAL pair (last two nav coords). For a higher-D navigator (e.g. a 5-D
    stack: ``[stack, ix, iy]``) the extra leading coords are dropped here; the
    caller reads them via :func:`_indices_lead_nav`."""
    idx = np.asarray(indices)
    if idx.ndim >= 2:
        idx = idx[0]
    # The last two coords are the spatial (x, y) pair from the innermost
    # crosshair; anything before them is a higher-level navigator (stack) index.
    ix, iy = int(idx[-2]), int(idx[-1])
    return iy, ix


def _indices_lead_nav(indices):
    """Return the LEADING navigation coords (everything before the spatial x,y
    pair) as a tuple, in data-axis order — e.g. ``(stack,)`` for a 5-D stack, or
    ``()`` for a plain 4-D scan. These index the navigation axes ABOVE the 2-D
    scan and are sliced as fixed positions before the spatial window."""
    idx = np.asarray(indices)
    if idx.ndim >= 2:
        idx = idx[0]
    if idx.shape[-1] <= 2:
        return ()
    return tuple(int(v) for v in idx[:-2])


def _clip_to_bounds(px, W, H, slack=8.0):
    """Drop marker offsets that fall outside the detector (with a small ``slack``
    so edge disks still show). Find Vectors can emit a few spurious peaks far
    outside the frame (sub-pixel refinement / ghost-cell artifacts) whose
    calibrated coords map to pixel positions like 24000 — drawing a circle there
    litters the plot with off-screen / giant arcs."""
    if px is None or len(px) == 0:
        return px
    m = ((px[:, 0] >= -slack) & (px[:, 0] <= (W - 1) + slack) &
         (px[:, 1] >= -slack) & (px[:, 1] <= (H - 1) + slack))
    return px[m]


def _navigator_selectors_for(tree, dp_plot):
    """Navigator selectors that drive ``dp_plot`` (so the overlay tracks the same
    navigation that updates the DP image)."""
    npm = getattr(tree, "navigator_plot_manager", None)
    if npm is None:
        return []
    out = []
    for sel in npm.all_navigation_selectors:
        if dp_plot in getattr(sel, "active_children", []):
            out.append(sel)
    out = out or list(npm.all_navigation_selectors)
    # Dedup: a composite navigator selector (IntegratingSSelector2D) exposes its
    # crosshair AND rectangle sub-selectors; registering the overlay hook on both
    # fires it twice per nav move (double compute → transform image flashes).
    # Keep one selector per parent (or per identity).
    seen, uniq = set(), []
    for sel in out:
        key = id(getattr(sel, "parent", sel) or sel)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(sel)
    return uniq


class _DPOverlay:
    """Base for a single circle-marker overlay that tracks the navigator.

    Subclasses set ``dp_plot``, ``name``, ``_color``, ``_radius_px`` (and call
    ``_calibrate(sig_axes)`` if they convert calibrated kx,ky → pixels) and
    implement ``_offsets_for(iy, ix) -> (N, 2)`` image-pixel offsets. All the
    chrome — attach + navigator wiring + push + show/hide + remove — lives here.

    While hidden the overlay still TRACKS the navigator (``_last_iyix`` updates),
    it just doesn't draw, so re-showing redraws the current frame.
    """

    _hidden = False
    _last_iyix = (0, 0)
    _lead_nav: tuple = ()   # leading nav coords above the 2-D scan (e.g. stack idx)
    name = "overlay"
    _color = "#ff3030"
    _radius_px = 4.0
    # How the per-position compute runs (see live_overlay.LiveOverlayEngine):
    # "sync" for cheap overlays (immediate), "thread"/"future" for heavy ones so
    # the compute never holds the navigator's serialised update lock.
    _overlay_mode = "sync"
    _engine = None

    def _calibrate(self, sig_axes) -> None:
        self._x_scale = float(sig_axes[0].scale) or 1.0
        self._x_off = float(sig_axes[0].offset)
        self._y_scale = float(sig_axes[1].scale) or 1.0
        self._y_off = float(sig_axes[1].offset)

    def _to_px(self, xy) -> np.ndarray:
        """Calibrated (kx, ky) → image-pixel offsets — the convention anyplotlib
        ``transform="data"`` markers use (no axis scale/offset)."""
        xy = np.asarray(xy, dtype=np.float64)
        if xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        mx = (xy[:, 0] - self._x_off) / self._x_scale
        my = (xy[:, 1] - self._y_off) / self._y_scale
        return np.column_stack([mx, my]).astype(np.float32)

    def _offsets_for(self, iy, ix) -> np.ndarray:   # pragma: no cover
        raise NotImplementedError

    def _marker_kwargs(self, offsets) -> dict:
        return {"offsets": offsets}

    def attach(self, tree):
        plot2d = getattr(self.dp_plot, "_plot2d", None)
        if plot2d is None:
            return self
        self._mg = plot2d.add_circles(
            np.zeros((0, 2), dtype=np.float32), name=self.name,
            radius=float(self._radius_px), edgecolors=self._color, facecolors=None,
            linewidths=1.5, alpha=1.0, transform="data",
        )
        self._engine = self._make_engine(tree)
        self._selectors = _navigator_selectors_for(tree, self.dp_plot)
        seeded = False
        for sel in self._selectors:
            # Guard against a double-register (re-attach / two selectors sharing a
            # plot): the same hook firing twice means two computes per nav move
            # and the transform image flashing between them.
            if self._on_indices not in sel.index_hooks:
                sel.index_hooks.append(self._on_indices)
            # Seed from the last-known position so markers appear immediately.
            if sel.current_indices is not None:
                self._on_indices(sel.current_indices)
                seeded = True
        # ALWAYS seed the engine, even when no selector had a committed position
        # yet (navigator hasn't painted) or none were found. Without this the
        # overlay sat blank until a *future* nav move — the silent failure that
        # made the matched-template markers never appear in a fresh view. The
        # hooks above are still registered, so a later nav move refreshes it.
        if not seeded:
            self._engine.request(*self._last_iyix)
        if not self._selectors:
            log.warning("[overlay:%s] attached with NO navigator selectors — "
                        "markers will not track the navigator until one exists",
                        self.name)
        else:
            log.debug("[overlay:%s] attached on %d selector(s), seeded=%s",
                      self.name, len(self._selectors), seeded)
        return self

    def _make_engine(self, tree):
        """Build the reactive-overlay engine: re-run ``_offsets_for`` off the
        navigator thread on each move and push the result. Subclasses with a
        multi-group payload override ``_render_payload``."""
        from spyde.drawing.live_overlay import LiveOverlayEngine
        client = None
        if self._overlay_mode == "future":
            client = getattr(tree, "client", None)
        return LiveOverlayEngine(
            self._offsets_for, self._render_payload,
            mode=self._overlay_mode, client=client, name=self.name,
        )

    def _render_payload(self, payload) -> None:
        """Render a computed payload (default: a single offsets array)."""
        if not self._hidden:
            self._push(payload)

    def _on_indices(self, indices):
        self._last_iyix = _indices_to_iyix(indices)
        # Leading navigation coords above the 2-D scan (e.g. the stack index of a
        # 5-D stack); sliced as fixed positions in _blurred_frame.
        self._lead_nav = _indices_lead_nav(indices)
        log.debug("[overlay:%s] nav move -> lead=%s (%s,%s) engine=%s hidden=%s",
                  self.name, self._lead_nav, self._last_iyix[0], self._last_iyix[1],
                  self._engine is not None, self._hidden)
        if self._engine is not None and not self._hidden:
            self._engine.request(*self._last_iyix)

    def _push(self, offsets):
        if self._mg is None:
            return
        try:
            self._mg.set(**self._marker_kwargs(offsets))
        except Exception:
            try:
                self._mg.set(offsets=offsets)
            except Exception as e:
                log.debug("pushing overlay marker offsets failed: %s", e)

    def set_visible(self, visible: bool) -> None:
        self._hidden = not bool(visible)
        if self._hidden:
            self._push(np.zeros((0, 2), dtype=np.float32))
        elif self._engine is not None:
            self._engine.request(*self._last_iyix)      # recompute current frame
        else:
            self._push(self._offsets_for(*self._last_iyix))

    def remove(self):
        if self._engine is not None:
            self._engine.stop()
            self._engine = None
        for sel in self._selectors:
            if self._on_indices in sel.index_hooks:
                sel.index_hooks.remove(self._on_indices)
        self._selectors = []
        if self._mg is not None:
            try:
                self._mg.remove()
            except Exception as e:
                log.debug("removing overlay marker group failed: %s", e)
            self._mg = None


class VectorOverlay(_DPOverlay):
    """A live found-vectors circle overlay bound to a DP plot and its navigator."""

    def __init__(self, dp_plot, vecs, *, color="#ff3030", name="found_vectors",
                 radius_px=None):
        self.dp_plot = dp_plot
        self.vecs = vecs
        self.name = name
        self._color = color
        self._mg = None
        self._selectors = []
        self._calibrate(vecs.sig_axes)
        self._W = int(vecs.sig_axes[0].size)
        self._H = int(vecs.sig_axes[1].size)
        if radius_px is None:
            radius_px = getattr(vecs, "kernel_radius_px", 4.0)
        # Cap to a small fraction of the detector so the circles can never swamp
        # the pattern (e.g. on a tiny/degenerate frame a fixed px radius looks
        # enormous); the disk it outlines is `kernel_radius_px` so this is the
        # detection radius for normal data.
        cap = max(4.0, 0.08 * min(self._W, self._H))
        self._radius_px = float(np.clip(float(radius_px), 2.0, cap))

    def _offsets_for(self, iy, ix) -> np.ndarray:
        try:
            # Show the CURRENT stack/time slice only (lead = the outer nav coords),
            # not every slice overlaid — matches the DP being viewed.
            px = self._to_px(self.vecs.kxy_at_nav(iy, ix, lead=self._lead_nav))
        except Exception:
            return np.zeros((0, 2), dtype=np.float32)
        return _clip_to_bounds(px, self._W, self._H)


def attach_vector_overlay(dp_plot, vecs, tree, *, color="#ff3030",
                          name="found_vectors", radius_px=None) -> VectorOverlay:
    """Add a live found-vectors marker overlay to ``dp_plot`` and wire it to the
    navigator selectors of ``tree``. Returns the :class:`VectorOverlay`."""
    return VectorOverlay(dp_plot, vecs, color=color, name=name,
                         radius_px=radius_px).attach(tree)


class OrientationOverlay(_DPOverlay):
    """Overlay the best-matching template's simulated spots on the DP, live.

    Re-runs single-pattern template matching (via the prebuilt matching cache,
    ~5 ms) at the navigator position and draws the resulting spots — same idea as
    the Qt orientation live-refine scatter, minus the IPF refine UI.
    """

    def __init__(self, dp_plot, signal, sim, matching_cache, *,
                 gamma=0.5, max_radius=None, normalize_templates=True,
                 scale_override=None, min_intensity=0.0,
                 color="#30ff60", name="orientation_template", radius_px=4.0):
        self.dp_plot = dp_plot
        self.signal = signal
        self.sim = sim
        self.cache = matching_cache
        self.gamma = float(gamma)
        self.max_radius = max_radius
        self.normalize_templates = bool(normalize_templates)
        self.scale_override = scale_override
        self.min_intensity = float(min_intensity)
        self.name = name
        self._color = color
        self._radius_px = max(2.0, float(radius_px))
        self._mg = None
        self._selectors: list = []
        self._last_iyix = (0, 0)
        # Serialise matching: nav-move (selector thread) and refine-slider
        # (dispatch thread) both call pyxem's numba matcher, whose default
        # workqueue layer is NOT thread-safe under concurrent calls.
        self._match_lock = threading.Lock()
        self._calibrate(signal.axes_manager.signal_axes)

    def set_refine_params(self, **params) -> None:
        """Live-update the Refine sliders (gamma / scale / min-intensity /
        normalize) and re-draw the matched template at the CURRENT crosshair
        position — the Qt "3 Refine" tab behaviour."""
        if "gamma" in params and params["gamma"] is not None:
            self.gamma = float(params["gamma"])
        if "normalize_templates" in params:
            self.normalize_templates = bool(params["normalize_templates"])
        if "scale_override" in params:
            so = params["scale_override"]
            self.scale_override = float(so) if so not in (None, "", 0) else None
        if "min_intensity" in params and params["min_intensity"] is not None:
            self.min_intensity = float(params["min_intensity"])
        iy, ix = self._last_iyix
        self._push(self._offsets_for(iy, ix))

    def _frame(self, iy, ix):
        frame = self.signal.data[iy, ix]
        if hasattr(frame, "compute"):      # lazy/dask: one small pattern only
            frame = frame.compute()
        return np.asarray(frame, dtype=float)

    def _offsets_for(self, iy, ix) -> np.ndarray:
        from spyde.actions.orientation_compute import best_match_spots
        try:
            frame = self._frame(iy, ix)
            with self._match_lock:
                coords = best_match_spots(
                    frame, self.sim, self.cache, gamma=self.gamma,
                    max_radius=self.max_radius,
                    normalize_templates=self.normalize_templates,
                    scale_override=self.scale_override,
                    original_scale=self._x_scale,
                    min_intensity=self.min_intensity,
                )
        except Exception as e:
            # Don't swallow blind: a failed match must be distinguishable from a
            # genuine no-spots result (it hid a silent overlay failure once).
            log.debug("[overlay:orient] best_match_spots FAILED nav=(%s,%s): %r",
                      iy, ix, e)
            return np.zeros((0, 2), dtype=np.float32)
        n = 0 if coords is None else len(coords)
        log.debug("[overlay:orient] nav=(%s,%s) -> %d spots", iy, ix, n)
        if coords is None or len(coords) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return self._to_px(coords)


def attach_orientation_overlay(dp_plot, signal, sim, matching_cache, tree, *,
                               gamma=0.5, max_radius=None,
                               normalize_templates=True,
                               color="#30ff60") -> OrientationOverlay:
    """Add a live matched-template spot overlay to ``dp_plot``, wired to the
    navigator selectors of ``tree``. Returns the :class:`OrientationOverlay`."""
    return OrientationOverlay(
        dp_plot, signal, sim, matching_cache, gamma=gamma, max_radius=max_radius,
        normalize_templates=normalize_templates, color=color,
    ).attach(tree)


class FindVectorsPreviewOverlay(_DPOverlay):
    """Live found-peaks preview on the DP, BEFORE the batch compute.

    Qt parity: the Find-Vectors caret drew a red scatter that re-ran
    single-frame peak finding as you tuned the sliders (σ / kernel radius /
    threshold / min distance / subpixel) and moved the navigator, so you could
    dial the parameters in before committing to the full-dataset compute.

    Runs :func:`find_vectors._find_vectors_single_frame` on the nav-blurred frame
    at the crosshair. Memory-safe: only a small nav window (radius ``ceil(3σ)``)
    is sliced and ``.compute()``-ed for lazy data — never the full dataset.
    """

    # Everything (peaks AND the transform image) is computed off the navigator
    # thread via the shared LiveOverlayEngine and rendered when ready, with the
    # in-flight compute cancelled when a newer position/param arrives — the SAME
    # greedy future-cancel / latest-wins pipeline the navigator image uses. No
    # bespoke threads, no synchronous recompute in set_params.
    _overlay_mode = "thread"

    def __init__(self, dp_plot, signal, *, sigma=1.0, kernel_radius=5,
                 threshold=0.5, min_distance=5, subpixel=True,
                 method="nxcorr", dog_sigma1=0.8, dog_sigma2=2.0,
                 beamstop_mask=None, show_transform=False,
                 color="#ff3030", name="fv_preview"):
        self.dp_plot = dp_plot
        self.signal = signal
        self.sigma = float(sigma)
        self.kernel_radius = int(kernel_radius)
        self.threshold = float(threshold)
        self.min_distance = int(min_distance)
        self.subpixel = bool(subpixel)
        self.method = str(method).lower()
        self.dog_sigma1 = float(dog_sigma1)
        self.dog_sigma2 = float(dog_sigma2)
        self.beamstop_mask = beamstop_mask
        # When True the dp plot shows the detector's TRANSFORMED image (DoG
        # band-pass / NXCORR correlation) instead of the raw pattern, so the
        # user sees what the peak finder sees. _raw_levels saves the plot's
        # contrast so it can be restored when toggled off.
        self.show_transform = bool(show_transform)
        self._raw_levels = None
        self.name = name
        self._color = color
        self._radius_px = max(1, int(kernel_radius))
        self._mg = None
        self._selectors = []
        self._lock = threading.Lock()
        # beam-stop auto-detect runs async (see _request_beamstop)
        self._beamstop_wanted = beamstop_mask is not None
        self._beamstop_scanning = False
        self.beamstop_dilate = 5

    def _marker_radius(self) -> float:
        """Circle radius for the markers, in pixels. For NXCORR it's the disk
        kernel radius. For DoG the natural spot scale is the σ₁ band-pass: a
        Gaussian of width σ has a blob radius ≈ √2·σ (where the LoG/DoG response
        peaks), so draw circles of ~√2·σ₁ to outline the detected spot size."""
        if self.method == "dog":
            return max(1.0, float(np.sqrt(2.0) * self.dog_sigma1))
        return float(self.kernel_radius)

    # The peak radius tracks the (live-tunable) detector scale on every push.
    def _marker_kwargs(self, offsets) -> dict:
        return {"offsets": offsets, "radius": self._marker_radius()}

    def set_params(self, **p) -> None:
        """Live-update the tuning sliders and redraw at the current crosshair."""
        if p.get("sigma") is not None:
            self.sigma = float(p["sigma"])
        if p.get("kernel_radius") is not None:
            self.kernel_radius = max(1, int(p["kernel_radius"]))
        if p.get("threshold") is not None:
            self.threshold = float(p["threshold"])
        if p.get("min_distance") is not None:
            self.min_distance = max(1, int(p["min_distance"]))
        if "subpixel" in p and p["subpixel"] is not None:
            self.subpixel = bool(p["subpixel"])
        if p.get("method") is not None:
            self.method = str(p["method"]).lower()
        if p.get("dog_sigma1") is not None:
            self.dog_sigma1 = float(p["dog_sigma1"])
        if p.get("dog_sigma2") is not None:
            self.dog_sigma2 = float(p["dog_sigma2"])
        if p.get("beamstop_dilate") is not None:
            new_dil = max(0, int(p["beamstop_dilate"]))
            if new_dil != getattr(self, "beamstop_dilate", 5):
                self.beamstop_dilate = new_dil
                # dilation is a cheap maximum_filter on the CACHED raw mask — the
                # static stop is NOT re-detected. Only re-dilate if it's on.
                if getattr(self, "_beamstop_wanted", False):
                    self._apply_dilation()
        if "beamstop_auto" in p and p["beamstop_auto"] is not None:
            want = bool(p["beamstop_auto"])
            self._beamstop_wanted = want
            if want:
                # non-blocking: kicks off the ~400-frame scan on a bg thread the
                # first time and applies + re-renders when it lands. Must NOT
                # block set_params (it runs on the live-tune path).
                self._request_beamstop()
            else:
                self.beamstop_mask = None
                self._push_mask_overlay()           # clears the overlay
        if "show_transform" in p and p["show_transform"] is not None:
            new_show = bool(p["show_transform"])
            if new_show != self.show_transform:
                # enter/leave transform view: the lock suppresses the navigator's
                # raw-frame paint so the overlay owns the image while on.
                log.debug("[fv-preview] transform view %s -> %s (lock %s)",
                          self.show_transform, new_show,
                          "ON" if new_show else "OFF")
                if hasattr(self.dp_plot, "set_transform_active"):
                    self.dp_plot.set_transform_active(new_show)
                if not new_show:
                    self._restore_raw_frame()  # toggled OFF → put the raw DP back
            self.show_transform = new_show
        log.debug("[fv-preview] set_params method=%s thr=%.3g md=%d sig=%.2g "
                  "dog=(%.2g,%.2g) beamstop=%s show_transform=%s keys=%s",
                  self.method, self.threshold, self.min_distance, self.sigma,
                  self.dog_sigma1, self.dog_sigma2,
                  "yes" if self.beamstop_mask is not None else "no",
                  self.show_transform, sorted(p.keys()))
        # A param change is just another reason to recompute the current frame —
        # route it through the SAME engine (latest-wins, cancellable) as a
        # navigator move. No synchronous recompute, no extra thread.
        if self._engine is not None:
            self._engine.request(*self._last_iyix)

    def _apply_dilation(self) -> None:
        """Dilate the cached UNDILATED beam-stop mask to the current radius — a
        cheap maximum_filter, NO re-scan. The detection (reading frames) ran once;
        the static stop never needs re-detecting just to change the dilation."""
        raw = getattr(self, "_beamstop_raw", None)
        if raw is None:
            self.beamstop_mask = None
            self._push_mask_overlay()
            return
        from spyde.actions.find_vectors import _dilate_mask
        r = int(getattr(self, "beamstop_dilate", 5))
        self.beamstop_mask = _dilate_mask(raw, r) if r > 0 else raw
        self._push_mask_overlay()

    def _push_mask_overlay(self) -> None:
        """Show the current beam-stop mask as a translucent overlay on the DP
        plot (client-side composite in the anyplotlib iframe — no recompute).
        Cleared when the mask is None or the beam stop is toggled off."""
        dp = getattr(self, "dp_plot", None)
        if dp is None or not hasattr(dp, "set_overlay_mask"):
            return
        mask = self.beamstop_mask if getattr(self, "_beamstop_wanted", False) else None
        try:
            dp.set_overlay_mask(mask, color="#ff8a3d", alpha=0.35)
        except Exception as e:
            log.debug("[fv-preview] push mask overlay failed: %s", e)

    def _request_beamstop(self) -> None:
        """Detect the beam stop ONCE (cached), then dilate cheaply on demand.

        The slow part is reading frames off the (multi-GB lazy) dataset — done a
        SINGLE time on a bg thread and cached as the UNDILATED mask. Changing the
        dilation radius just re-runs a maximum_filter on the cache (instant), it
        does NOT re-scan. Must not block set_params (runs on the live-tune path)."""
        if hasattr(self, "_beamstop_raw"):
            self._apply_dilation()                  # cached → just (re)dilate
            return
        if getattr(self, "_beamstop_scanning", False):
            return                                  # a scan is already in flight
        self._beamstop_scanning = True

        def _scan():
            try:
                from spyde.actions.find_vectors import _auto_beamstop_from_signal
                nav_dim = self.signal.axes_manager.navigation_dimension
                # dilate=0 → the raw (undilated) stop mask; we dilate locally.
                raw = _auto_beamstop_from_signal(self.signal, nav_dim, dilate=0)
            except Exception as e:
                log.debug("preview beam-stop detection failed: %s", e)
                raw = None
            self._beamstop_raw = raw                # may be None (no stop found)
            self._beamstop_scanning = False
            log.debug("[fv-preview] beam-stop scan done: raw=%s",
                      "none" if raw is None else f"{int(raw.sum())}px")
            if getattr(self, "_beamstop_wanted", False):
                self._apply_dilation()
                if self._engine is not None:
                    self._engine.request(*self._last_iyix)

        threading.Thread(target=_scan, daemon=True, name="fv-beamstop").start()

    # ── geometry ──────────────────────────────────────────────────────────────
    def _blurred_frame(self, iy, ix) -> np.ndarray:
        """Nav-space Gaussian blur matching the batch (``sigma`` over the 2-D scan
        dims, 0 over signal dims). Slice a small nav window of radius ``ceil(3σ)``
        around (iy,ix), blur it, and return the centre frame.

        For a higher-D navigator (5-D stack), the leading nav axes (``_lead_nav``,
        e.g. the stack index) are sliced at their FIXED current position first, so
        the spatial window + blur operate on the selected stack's 2-D scan exactly
        as for a plain 4-D dataset."""
        data = self.signal.data
        # Drop the leading (non-spatial) nav axes at their current index, so the
        # remaining array is (y, x, *signal) — the 4-D-style layout the rest of
        # this method assumes. ``_lead_nav`` is in data-axis order.
        lead = tuple(int(v) for v in (self._lead_nav or ()))
        nav_dim = 2 + len(lead)
        if lead:
            sig_ndim = data.ndim - nav_dim
            # Clamp each leading index to its axis (a stale higher-grid position
            # mustn't IndexError; matches the navigator's own clamp).
            lead = tuple(min(max(0, v), int(data.shape[i]) - 1)
                         for i, v in enumerate(lead))
            data = data[lead]   # fancy/scalar index of the leading axes
        ny, nx = int(data.shape[0]), int(data.shape[1])
        r = int(np.ceil(3 * self.sigma)) if self.sigma > 0 else 0
        y0, y1 = max(0, iy - r), min(ny, iy + r + 1)
        x0, x1 = max(0, ix - r), min(nx, ix + r + 1)
        block = data[y0:y1, x0:x1]
        if hasattr(block, "compute"):
            # Compute this tiny window on the LOCAL threaded scheduler, NOT the
            # distributed cluster. The navigator drives the same lazy signal's
            # CachedDaskArray and aggressively cancels surrounding-block futures
            # (cancel_surrounding); a distributed preview slice shares those
            # block futures, so a navigator move cancels the preview's chunk
            # mid-read → "get_inds … cancelled for reason: lost dependencies"
            # and the preview stops updating. The local scheduler reads the slice
            # independently and is unaffected by the navigator's cancels.
            try:
                block = block.compute(scheduler="threads")
            except Exception:
                # Fall back to whatever default the array carries (e.g. already
                # in-RAM numpy via a Future-backed slice) rather than failing.
                block = np.asarray(block.compute())
        block = np.asarray(block, dtype=np.float32)
        if r > 0 and self.sigma > 0:
            from scipy.ndimage import gaussian_filter
            sig_tuple = (self.sigma, self.sigma) + (0,) * (block.ndim - 2)
            block = gaussian_filter(block, sigma=sig_tuple)
        return block[iy - y0, ix - x0]

    def remove(self):
        # leaving the preview must release the transform lock so the navigator
        # can paint raw frames again.
        if hasattr(self.dp_plot, "set_transform_active"):
            try:
                self.dp_plot.set_transform_active(False)
            except Exception as e:
                log.debug("releasing transform lock on remove failed: %s", e)
        # clear the beam-stop overlay so it doesn't linger after the wizard closes
        if hasattr(self.dp_plot, "set_overlay_mask"):
            try:
                self.dp_plot.set_overlay_mask(None)
            except Exception as e:
                log.debug("clearing mask overlay on remove failed: %s", e)
        super().remove()

    def _restore_raw_frame(self) -> None:
        """Re-display the raw diffraction pattern after the transform toggle is
        turned off — re-slice the current crosshair frame and push it back via
        the PLOT (sets current_data so it persists)."""
        if not hasattr(self.dp_plot, "set_data"):
            return
        try:
            iy, ix = self._last_iyix
            frame = self._blurred_frame(iy, ix)
            finite = frame[np.isfinite(frame)]
            levels = ((float(finite.min()), float(finite.max()))
                      if finite.size else None)
            self.dp_plot.needs_auto_level = True
            self.dp_plot.set_data(np.asarray(frame, dtype=np.float32), levels=levels)
            log.debug("[fv-preview] restored RAW frame nav=(%s,%s) %s",
                      iy, ix, _stats(frame))
        except Exception as e:
            log.debug("restoring raw frame after transform toggle failed: %s", e)

    # ── one compute → one payload → one render, all via the engine ─────────────
    def _offsets_for(self, iy, ix):
        """The engine's compute. Returns a payload dict {offsets, response} so a
        SINGLE cancellable compute produces both the peak markers AND (when
        requested) the transformed image. No side-effects on the plot here —
        rendering happens in _render_payload (which may run on a worker thread)."""
        from spyde.actions.find_vectors import _find_peaks_single_frame
        try:
            frame = self._blurred_frame(iy, ix)
            params = dict(
                method=self.method, kernel_radius=self.kernel_radius,
                threshold=self.threshold, min_distance=self.min_distance,
                subpixel=self.subpixel, dog_sigma1=self.dog_sigma1,
                dog_sigma2=self.dog_sigma2,
            )
            if self.show_transform:
                peaks, response = _find_peaks_single_frame(
                    frame, params, beamstop_mask=self.beamstop_mask,
                    with_response=True)
            else:
                peaks = _find_peaks_single_frame(
                    frame, params, beamstop_mask=self.beamstop_mask)
                response = None
        except Exception as e:
            log.debug("[fv-preview] compute FAILED nav=(%s,%s): %r", iy, ix, e)
            return {"offsets": np.zeros((0, 2), np.float32), "response": None}

        if peaks is None or len(peaks) == 0:
            offsets = np.zeros((0, 2), np.float32)
        else:
            # peaks=[ky_row, kx_col, val] px → markers (x=col=kx, y=row=ky).
            offsets = np.column_stack([peaks[:, 1], peaks[:, 0]]).astype(np.float32)
        # Diagnostic: count returned peaks that fall on a masked pixel. If this is
        # ever > 0 the BACKEND mask exclusion is wrong; if it's 0 but circles still
        # appear over the brown overlay, the mismatch is in the RENDERER (marker vs
        # mask coordinate/orientation), not the detector.
        under = -1
        m = self.beamstop_mask
        if m is not None and len(offsets):
            yi = np.clip(np.round(offsets[:, 1]).astype(int), 0, m.shape[0] - 1)
            xi = np.clip(np.round(offsets[:, 0]).astype(int), 0, m.shape[1] - 1)
            under = int(m[yi, xi].sum())
        log.info("[fv-preview] COMPUTE nav=(%s,%s) method=%s thr=%.3g md=%d kr=%d "
                 "npeaks=%d under_mask=%d beamstop=%s show_transform=%s",
                 iy, ix, self.method, self.threshold, self.min_distance,
                 self.kernel_radius, len(offsets), under,
                 "yes" if m is not None else "no", self.show_transform)
        return {"offsets": offsets, "response": response}

    def _render_payload(self, payload) -> None:
        """Render the compute payload: the transformed image (if any) THEN the
        markers on top. Runs latest-wins via the engine; safe on a worker thread
        (set_data / marker push are GIL-protected)."""
        if self._hidden or not isinstance(payload, dict):
            log.debug("[fv-preview] render SKIPPED (hidden=%s dict=%s)",
                      self._hidden, isinstance(payload, dict))
            return
        response = payload.get("response")
        offsets = payload.get("offsets", np.zeros((0, 2), np.float32))
        if (self.show_transform and response is not None
                and hasattr(self.dp_plot, "set_transform_image")):
            try:
                r = np.asarray(response, dtype=np.float32)
                finite = r[np.isfinite(r)]
                # Contrast window: floor = the detector threshold, ceiling = the
                # response's robust max — the two are INDEPENDENT. The earlier bug
                # tied the ceiling to the threshold (hi = threshold + 1 when the
                # 99.5%ile fell below threshold), so nudging the threshold blew the
                # window open and washed the image to white (the reported flash).
                # For NXCORR the response is in [-1, 1] so a FIXED ceiling of 1.0
                # is the stablest choice (never moves frame-to-frame); for DoG SNR
                # use the 99th-percentile max (no fixed scale). The floor is the
                # threshold, clamped just under the ceiling.
                if self.method == "dog":
                    hi = float(np.percentile(finite, 99.0)) if finite.size else 1.0
                else:
                    hi = 1.0                       # NXCORR score ceiling, fixed
                lo = float(self.threshold)
                if lo >= hi:
                    lo = hi - 1e-3                 # keep a non-degenerate window
                self.dp_plot.needs_auto_level = False
                self.dp_plot.set_transform_image(r, levels=(lo, hi))
                if hasattr(self.dp_plot, "_emit_histogram"):
                    self.dp_plot._emit_histogram(r, lo, hi, threshold=float(self.threshold))
                log.debug("[fv-preview] render TRANSFORM %s clim=(%.3g,%.3g) "
                          "threshold=%.3g + %d markers", _stats(r), lo, hi,
                          float(self.threshold), len(offsets))
            except Exception as e:
                log.debug("[fv-preview] render transform failed: %s", e)
        else:
            log.debug("[fv-preview] render MARKERS-only n=%d (show_transform=%s "
                      "response=%s)", len(offsets), self.show_transform,
                      response is not None)
        # Re-assert the beam-stop overlay AFTER the image paint: the image dims
        # are now set, and set_data does not clear the mask so this is just a
        # cheap state re-push (idempotent if the mask is unchanged).
        self._push_mask_overlay()
        self._push(offsets)


class VectorOrientationOverlay(_DPOverlay):
    """Live Vector-Orientation refine preview: the FITTED template (green) over
    the MEASURED vectors (red) on the vectors diffraction pattern, tracking the
    navigator (Qt parity — the vector-OM Refine scatter).

    At each navigator position it fits the pose (`fit_pattern`) of the vectors
    against the template library and draws ``A·Rot(θ)·g_template + t`` as green
    circles; the measured vectors are drawn red. The fit is ~tens of ms and runs
    on the selector thread, serialised by a lock.
    """

    def __init__(self, dp_plot, vecs, lib, params=None, *,
                 color_meas="#ff3030", color_tmpl="#30ff60", radius_px=None,
                 on_fit=None):
        self.dp_plot = dp_plot
        self.vecs = vecs
        self.lib = lib
        self.params = dict(params or {})
        self.on_fit = on_fit          # callback(fit_or_None) for the Refine readout
        self.name_meas = "vom_measured"
        self.name_tmpl = "vom_template"
        self._mg_meas = None
        self._mg_tmpl = None
        self._selectors: list = []
        self._last_iyix = (0, 0)
        self._hidden = False
        self._lock = threading.Lock()
        self._calibrate(vecs.sig_axes)
        if radius_px is None:
            radius_px = getattr(vecs, "kernel_radius_px", 4.0)
        self._radius_px = max(2.0, float(radius_px))

    def set_params(self, **params) -> None:
        self.params.update({k: v for k, v in params.items() if v is not None})
        if not self._hidden:
            self._push(*self._offsets_for(*self._last_iyix))

    # Two marker groups (measured + template) → override the single-group chrome.
    def set_visible(self, visible: bool) -> None:
        self._hidden = not bool(visible)
        if self._hidden:
            empty = np.zeros((0, 2), dtype=np.float32)
            self._push(empty, empty)
        elif self._engine is not None:
            self._engine.request(*self._last_iyix)
        else:
            self._push(*self._offsets_for(*self._last_iyix))

    def _emit_fit(self, fit) -> None:
        if self.on_fit is not None:
            try:
                self.on_fit(fit)
            except Exception as e:
                log.debug("on_fit overlay callback failed: %s", e)

    def _offsets_for(self, iy, ix):
        from spyde.actions.vector_orientation import (
            fit_pattern, project_spots, DEFAULTS, COL_KX, COL_KY, COL_INTENSITY,
        )
        try:
            rows = np.asarray(self.vecs.at(iy, ix))
        except Exception:
            self._emit_fit(None)
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        if rows.size == 0:
            self._emit_fit(None)
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        meas_xy = rows[:, [COL_KX, COL_KY]].astype(np.float64)
        meas_px = self._to_px(meas_xy)
        if len(rows) < 4:
            self._emit_fit(None)
            return meas_px, np.zeros((0, 2), np.float32)

        mI = rows[:, COL_INTENSITY].astype(np.float64)
        P = {**DEFAULTS, **self.params}
        try:
            with self._lock:
                fit = fit_pattern(meas_xy, mI, self.lib, P)
        except Exception:
            self._emit_fit(None)
            return meas_px, np.zeros((0, 2), np.float32)
        if fit is None:
            self._emit_fit(None)
            return meas_px, np.zeros((0, 2), np.float32)
        self._emit_fit(fit)

        p7 = np.zeros(7, np.float64)
        p7[0] = float(fit.theta)
        p7[1:5] = np.asarray(fit.affine, float).reshape(-1)
        p7[5:7] = np.asarray(fit.translation, float)
        g = np.asarray(self.lib.spots_xy[int(fit.template_idx)], np.float64)
        model_xy = project_spots(p7, g)
        return meas_px, self._to_px(model_xy)

    def attach(self, tree):
        plot2d = getattr(self.dp_plot, "_plot2d", None)
        if plot2d is None:
            return self
        self._mg_meas = plot2d.add_circles(
            np.zeros((0, 2), np.float32), name=self.name_meas,
            radius=self._radius_px, edgecolors="#ff3030", facecolors=None,
            linewidths=1.5, alpha=1.0, transform="data",
        )
        self._mg_tmpl = plot2d.add_circles(
            np.zeros((0, 2), np.float32), name=self.name_tmpl,
            radius=self._radius_px, edgecolors="#30ff60", facecolors=None,
            linewidths=1.5, alpha=1.0, transform="data",
        )
        # Synchronous (inline) on purpose: the pose fit (fit_pattern) is only
        # ~tens of ms (it doesn't block on a .compute() like the peak preview),
        # and it touches a process-global resource that is NOT safe to run from
        # a worker thread concurrently with a direct call. The engine in "sync"
        # mode keeps the unified workflow without that hazard.
        from spyde.drawing.live_overlay import LiveOverlayEngine
        self._engine = LiveOverlayEngine(
            self._offsets_for, lambda payload: self._push(*payload),
            mode="sync", name=self.name_meas,
        )
        self._selectors = _navigator_selectors_for(tree, self.dp_plot)
        seeded = False
        for sel in self._selectors:
            sel.index_hooks.append(self._on_indices)
            if sel.current_indices is not None:
                self._on_indices(sel.current_indices)
                seeded = True
        if not seeded:
            self._engine.request(*self._last_iyix)
        return self

    def _on_indices(self, indices):
        iy, ix = _indices_to_iyix(indices)
        self._last_iyix = (iy, ix)
        if self._engine is not None and not self._hidden:
            self._engine.request(iy, ix)

    def _push(self, meas, tmpl):
        for mg, off in ((self._mg_meas, meas), (self._mg_tmpl, tmpl)):
            if mg is None:
                continue
            try:
                mg.set(offsets=off)
            except Exception as e:
                log.debug("pushing meas/tmpl marker offsets failed: %s", e)

    def remove(self):
        if self._engine is not None:
            self._engine.stop()
            self._engine = None
        for sel in self._selectors:
            if self._on_indices in sel.index_hooks:
                sel.index_hooks.remove(self._on_indices)
        self._selectors = []
        for attr in ("_mg_meas", "_mg_tmpl"):
            mg = getattr(self, attr, None)
            if mg is not None:
                try:
                    mg.remove()
                except Exception as e:
                    log.debug("removing meas/tmpl marker group failed: %s", e)
                setattr(self, attr, None)


def attach_vector_orientation_overlay(dp_plot, vecs, lib, tree, *, params=None,
                                      on_fit=None) -> VectorOrientationOverlay:
    """Add a live Vector-Orientation refine overlay (red measured vectors + green
    fitted template) to ``dp_plot``, wired to ``tree``'s navigator selectors.
    ``on_fit(fit_or_None)`` fires after each fit for the Refine strain readout."""
    return VectorOrientationOverlay(dp_plot, vecs, lib, params=params,
                                    on_fit=on_fit).attach(tree)


def attach_find_vectors_preview(dp_plot, signal, tree, *, sigma=1.0,
                                kernel_radius=5, threshold=0.5, min_distance=5,
                                subpixel=True, method="nxcorr", dog_sigma1=0.8,
                                dog_sigma2=2.0, beamstop_mask=None,
                                beamstop_auto=False, show_transform=False,
                                color="#ff3030") -> FindVectorsPreviewOverlay:
    """Add a live found-peaks preview overlay to ``dp_plot``, wired to the
    navigator selectors of ``tree``. Returns the :class:`FindVectorsPreviewOverlay`."""
    ov = FindVectorsPreviewOverlay(
        dp_plot, signal, sigma=sigma, kernel_radius=kernel_radius,
        threshold=threshold, min_distance=min_distance, subpixel=subpixel,
        method=method, dog_sigma1=dog_sigma1, dog_sigma2=dog_sigma2,
        beamstop_mask=beamstop_mask, show_transform=show_transform, color=color,
    ).attach(tree)
    if beamstop_auto:
        ov._beamstop_wanted = True
        ov._request_beamstop()        # async; applies + re-renders when ready
    return ov
