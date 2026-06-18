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

import threading

import numpy as np


def _indices_to_iyix(indices):
    """A navigator crosshair reports ``[[ix, iy]]`` (cx=column=nav-x,
    cy=row=nav-y). Return ``(iy, ix)`` for the vectors' ``at(iy, ix)``."""
    idx = np.asarray(indices)
    if idx.ndim >= 2:
        idx = idx[0]
    ix, iy = int(idx[0]), int(idx[1])
    return iy, ix


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
    # Fall back to all navigator selectors if none claim this plot directly.
    return out or list(npm.all_navigation_selectors)


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
    name = "overlay"
    _color = "#ff3030"
    _radius_px = 4.0

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
        self._selectors = _navigator_selectors_for(tree, self.dp_plot)
        seeded = False
        for sel in self._selectors:
            sel.index_hooks.append(self._on_indices)
            # Seed from the last-known position so markers appear immediately.
            if sel.current_indices is not None:
                self._on_indices(sel.current_indices)
                seeded = True
        if not seeded:
            self._push(self._offsets_for(*self._last_iyix))
        return self

    def _on_indices(self, indices):
        self._last_iyix = _indices_to_iyix(indices)
        if not self._hidden:
            self._push(self._offsets_for(*self._last_iyix))

    def _push(self, offsets):
        if self._mg is None:
            return
        try:
            self._mg.set(**self._marker_kwargs(offsets))
        except Exception:
            try:
                self._mg.set(offsets=offsets)
            except Exception:
                pass

    def set_visible(self, visible: bool) -> None:
        self._hidden = not bool(visible)
        empty = np.zeros((0, 2), dtype=np.float32)
        self._push(empty if self._hidden else self._offsets_for(*self._last_iyix))

    def remove(self):
        for sel in self._selectors:
            try:
                sel.index_hooks.remove(self._on_indices)
            except ValueError:
                pass
        self._selectors = []
        if self._mg is not None:
            try:
                self._mg.remove()
            except Exception:
                pass
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
        if radius_px is None:
            radius_px = getattr(vecs, "kernel_radius_px", 4.0)
        self._radius_px = max(2.0, float(radius_px))

    def _offsets_for(self, iy, ix) -> np.ndarray:
        try:
            return self._to_px(self.vecs.kxy_at(iy, ix))
        except Exception:
            return np.zeros((0, 2), dtype=np.float32)


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
        except Exception:
            return np.zeros((0, 2), dtype=np.float32)
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

    def __init__(self, dp_plot, signal, *, sigma=1.0, kernel_radius=5,
                 threshold=0.5, min_distance=5, subpixel=True,
                 color="#ff3030", name="fv_preview"):
        self.dp_plot = dp_plot
        self.signal = signal
        self.sigma = float(sigma)
        self.kernel_radius = int(kernel_radius)
        self.threshold = float(threshold)
        self.min_distance = int(min_distance)
        self.subpixel = bool(subpixel)
        self.name = name
        self._color = color
        self._radius_px = max(1, int(kernel_radius))
        self._mg = None
        self._selectors = []
        self._lock = threading.Lock()

    # The peak radius tracks the (live-tunable) kernel radius on every push.
    def _marker_kwargs(self, offsets) -> dict:
        return {"offsets": offsets, "radius": float(self.kernel_radius)}

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
        iy, ix = self._last_iyix
        self._push(self._offsets_for(iy, ix))

    # ── geometry ──────────────────────────────────────────────────────────────
    def _blurred_frame(self, iy, ix) -> np.ndarray:
        """Nav-space Gaussian blur matching the batch (``sigma`` over nav dims,
        0 over signal dims). Slice a small nav window of radius ``ceil(3σ)``
        around (iy,ix), blur it, and return the centre frame."""
        data = self.signal.data
        ny, nx = int(data.shape[0]), int(data.shape[1])
        r = int(np.ceil(3 * self.sigma)) if self.sigma > 0 else 0
        y0, y1 = max(0, iy - r), min(ny, iy + r + 1)
        x0, x1 = max(0, ix - r), min(nx, ix + r + 1)
        block = data[y0:y1, x0:x1]
        if hasattr(block, "compute"):          # lazy: only this small window
            block = block.compute()
        block = np.asarray(block, dtype=np.float32)
        if r > 0 and self.sigma > 0:
            from scipy.ndimage import gaussian_filter
            sig_tuple = (self.sigma, self.sigma) + (0,) * (block.ndim - 2)
            block = gaussian_filter(block, sigma=sig_tuple)
        return block[iy - y0, ix - x0]

    def _offsets_for(self, iy, ix) -> np.ndarray:
        from spyde.actions.find_vectors import _find_vectors_single_frame
        try:
            frame = self._blurred_frame(iy, ix)
            with self._lock:
                _, _, peaks = _find_vectors_single_frame(
                    frame, self.kernel_radius, self.threshold, self.min_distance,
                    subpixel=self.subpixel,
                )
        except Exception:
            return np.zeros((0, 2), dtype=np.float32)
        if peaks is None or len(peaks) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        # peaks = [ky_row, kx_col, value] in PIXELS → markers (x=col=kx, y=row=ky).
        return np.column_stack([peaks[:, 1], peaks[:, 0]]).astype(np.float32)

    # ── attach / update ─────────────────────────────────────────────────────
    def attach(self, tree):
        plot2d = getattr(self.dp_plot, "_plot2d", None)
        if plot2d is None:
            return self
        self._mg = plot2d.add_circles(
            np.zeros((0, 2), dtype=np.float32), name=self.name,
            radius=float(self.kernel_radius), edgecolors=self._color,
            facecolors=None, linewidths=1.5, alpha=1.0, transform="data",
        )
        self._selectors = _navigator_selectors_for(tree, self.dp_plot)
        seeded = False
        for sel in self._selectors:
            sel.index_hooks.append(self._on_indices)
            if sel.current_indices is not None:
                self._on_indices(sel.current_indices)
                seeded = True
        # Always render the preview immediately, even if no selector has fired
        # yet (current_indices is None until the first navigator move) — the Qt
        # caret showed peaks the moment you opened it.
        if not seeded:
            iy, ix = self._last_iyix
            self._push(self._offsets_for(iy, ix))
        return self

    def _on_indices(self, indices):
        iy, ix = _indices_to_iyix(indices)
        self._last_iyix = (iy, ix)
        self._push(self._offsets_for(iy, ix))

    def _push(self, offsets):
        if self._mg is None:
            return
        try:
            self._mg.set(offsets=offsets, radius=float(self.kernel_radius))
        except Exception:
            try:
                self._mg.set(offsets=offsets)
            except Exception:
                pass

    def remove(self):
        for sel in self._selectors:
            try:
                sel.index_hooks.remove(self._on_indices)
            except ValueError:
                pass
        self._selectors = []
        if self._mg is not None:
            try:
                self._mg.remove()
            except Exception:
                pass
            self._mg = None


class VectorOrientationOverlay:
    """Live Vector-Orientation refine preview: the FITTED template (green) over
    the MEASURED vectors (red) on the vectors diffraction pattern, tracking the
    navigator (Qt parity — the vector-OM Refine scatter).

    At each navigator position it fits the pose (`fit_pattern`) of the vectors
    against the template library and draws ``A·Rot(θ)·g_template + t`` as green
    circles; the measured vectors are drawn red. The fit is ~tens of ms and runs
    on the selector thread, serialised by a lock.
    """

    def __init__(self, dp_plot, vecs, lib, params=None, *,
                 color_meas="#ff3030", color_tmpl="#30ff60", radius_px=None):
        self.dp_plot = dp_plot
        self.vecs = vecs
        self.lib = lib
        self.params = dict(params or {})
        self.name_meas = "vom_measured"
        self.name_tmpl = "vom_template"
        self._mg_meas = None
        self._mg_tmpl = None
        self._selectors: list = []
        self._last_iyix = (0, 0)
        self._hidden = False
        self._lock = threading.Lock()

        sig_axes = vecs.sig_axes
        self._x_scale = float(sig_axes[0].scale) or 1.0
        self._x_off = float(sig_axes[0].offset)
        self._y_scale = float(sig_axes[1].scale) or 1.0
        self._y_off = float(sig_axes[1].offset)
        if radius_px is None:
            radius_px = getattr(vecs, "kernel_radius_px", 4.0)
        self._radius_px = max(2.0, float(radius_px))

    def set_params(self, **params) -> None:
        self.params.update({k: v for k, v in params.items() if v is not None})
        iy, ix = self._last_iyix
        if not self._hidden:
            self._push(*self._offsets_for(iy, ix))

    def set_visible(self, visible: bool) -> None:
        self._hidden = not bool(visible)
        if self._hidden:
            empty = np.zeros((0, 2), dtype=np.float32)
            self._push(empty, empty)
        else:
            iy, ix = self._last_iyix
            self._push(*self._offsets_for(iy, ix))

    def _to_px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        if xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        mx = (xy[:, 0] - self._x_off) / self._x_scale
        my = (xy[:, 1] - self._y_off) / self._y_scale
        return np.column_stack([mx, my]).astype(np.float32)

    def _offsets_for(self, iy, ix):
        from spyde.actions.vector_orientation import (
            fit_pattern, project_spots, DEFAULTS, COL_KX, COL_KY, COL_INTENSITY,
        )
        try:
            rows = np.asarray(self.vecs.at(iy, ix))
        except Exception:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        if rows.size == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        meas_xy = rows[:, [COL_KX, COL_KY]].astype(np.float64)
        meas_px = self._to_px(meas_xy)
        if len(rows) < 4:
            return meas_px, np.zeros((0, 2), np.float32)

        mI = rows[:, COL_INTENSITY].astype(np.float64)
        P = {**DEFAULTS, **self.params}
        try:
            with self._lock:
                fit = fit_pattern(meas_xy, mI, self.lib, P)
        except Exception:
            return meas_px, np.zeros((0, 2), np.float32)
        if fit is None:
            return meas_px, np.zeros((0, 2), np.float32)

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
        self._selectors = _navigator_selectors_for(tree, self.dp_plot)
        seeded = False
        for sel in self._selectors:
            sel.index_hooks.append(self._on_indices)
            if sel.current_indices is not None:
                self._on_indices(sel.current_indices)
                seeded = True
        if not seeded:
            iy, ix = self._last_iyix
            self._push(*self._offsets_for(iy, ix))
        return self

    def _on_indices(self, indices):
        iy, ix = _indices_to_iyix(indices)
        self._last_iyix = (iy, ix)
        if self._hidden:
            return
        self._push(*self._offsets_for(iy, ix))

    def _push(self, meas, tmpl):
        for mg, off in ((self._mg_meas, meas), (self._mg_tmpl, tmpl)):
            if mg is None:
                continue
            try:
                mg.set(offsets=off)
            except Exception:
                pass

    def remove(self):
        for sel in self._selectors:
            try:
                sel.index_hooks.remove(self._on_indices)
            except ValueError:
                pass
        self._selectors = []
        for attr in ("_mg_meas", "_mg_tmpl"):
            mg = getattr(self, attr, None)
            if mg is not None:
                try:
                    mg.remove()
                except Exception:
                    pass
                setattr(self, attr, None)


def attach_vector_orientation_overlay(dp_plot, vecs, lib, tree, *, params=None
                                      ) -> VectorOrientationOverlay:
    """Add a live Vector-Orientation refine overlay (red measured vectors + green
    fitted template) to ``dp_plot``, wired to ``tree``'s navigator selectors."""
    return VectorOrientationOverlay(dp_plot, vecs, lib, params=params).attach(tree)


def attach_find_vectors_preview(dp_plot, signal, tree, *, sigma=1.0,
                                kernel_radius=5, threshold=0.5, min_distance=5,
                                subpixel=True, color="#ff3030") -> FindVectorsPreviewOverlay:
    """Add a live found-peaks preview overlay to ``dp_plot``, wired to the
    navigator selectors of ``tree``. Returns the :class:`FindVectorsPreviewOverlay`."""
    return FindVectorsPreviewOverlay(
        dp_plot, signal, sigma=sigma, kernel_radius=kernel_radius,
        threshold=threshold, min_distance=min_distance, subpixel=subpixel,
        color=color,
    ).attach(tree)
