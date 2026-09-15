"""Live overlays on a diffraction-pattern plot: the found vectors, the
Find-Vectors tuning preview, the matched orientation template, the strain
reference selection and the vector-orientation refine.

Each one is a child of the node its window displays, added with
``BaseSignalTree.add_overlay`` and evaluated at the navigator's position
through the readers the base pattern is read with. This module holds the
per-position functions and the calls that add them; the navigator wiring, the
drawing and the teardown belong to the tree and the plot.

Coordinate system: diffraction vectors and simulated spots are in calibrated
units (kx, ky), while anyplotlib markers drawn with ``transform="data"`` are
addressed in image pixels. :class:`DetectorPixels` converts between them the
same way the rendered disk frames do, so a marker lands on its disk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from spyde.drawing.overlay_node import NavigationPosition

log = logging.getLogger(__name__)

# How far outside the detector a marker may still sit, so an edge disk shows.
_MARKER_SLACK_PX = 8.0


@dataclass(frozen=True)
class DetectorPixels:
    """Calibrated (kx, ky) converted to image-pixel marker offsets.

    ``pixel = (value - offset) / scale``, the conversion the rendered disk
    frames make, so markers and disks share one coordinate system."""

    x_scale: float
    x_offset: float
    y_scale: float
    y_offset: float
    width: int
    height: int
    #: Multiply an axis-unit value by this to get Å⁻¹ (10 for axes in nm⁻¹).
    #: Simulated spots arrive in Å⁻¹ whatever the detector is labelled, so the
    #: overlay needs the factor to put them back on the displayed axes.
    inverse_angstrom_factor: float = 1.0

    @classmethod
    def from_axes(cls, signal_axes, *, inverse_angstrom_factor=None
                  ) -> "DetectorPixels":
        from spyde.reciprocal_units import inverse_angstrom_factor as unit_factor

        x_axis, y_axis = signal_axes[0], signal_axes[1]
        if inverse_angstrom_factor is None:
            # Axis units alone answer this for nm⁻¹ and Å⁻¹; mrad would need
            # the wavelength, which an axis record does not carry, so an
            # unconvertible label falls back to 1.0 and draws where the data is.
            inverse_angstrom_factor = unit_factor(getattr(x_axis, "units", "")) or 1.0
        return cls(float(x_axis.scale) or 1.0, float(x_axis.offset),
                   float(y_axis.scale) or 1.0, float(y_axis.offset),
                   int(x_axis.size), int(y_axis.size),
                   float(inverse_angstrom_factor))

    @classmethod
    def from_signal(cls, signal) -> "DetectorPixels":
        """:meth:`from_axes` for a live signal, which can also resolve a
        detector calibrated in mrad (that conversion needs the beam energy)."""
        from spyde.reciprocal_units import axis_unit_factor

        return cls.from_axes(signal.axes_manager.signal_axes,
                             inverse_angstrom_factor=axis_unit_factor(signal) or 1.0)

    def to_inverse_angstrom(self, xy) -> np.ndarray:
        """Axis-unit ``(N, 2)`` values → Å⁻¹, the unit every simulated library
        and every fit tolerance is expressed in."""
        return np.asarray(xy, dtype=np.float64) * self.inverse_angstrom_factor

    def inverse_angstrom_to_pixels(self, xy) -> np.ndarray:
        """Å⁻¹ ``(N, 2)`` values → image-pixel offsets, going through whatever
        unit the detector axes currently display."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return self.to_pixels(xy / self.inverse_angstrom_factor)

    @property
    def max_scale(self) -> float:
        """The coarser of the two axis scales, for converting a pixel radius
        into calibrated units."""
        return max(abs(self.x_scale), abs(self.y_scale))

    def to_pixels(self, xy) -> np.ndarray:
        """``(N, 2)`` float32 pixel offsets for calibrated ``(N, 2)`` values."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.column_stack([(xy[:, 0] - self.x_offset) / self.x_scale,
                                (xy[:, 1] - self.y_offset) / self.y_scale]
                               ).astype(np.float32)

    def clipped(self, xy) -> np.ndarray:
        """:meth:`to_pixels`, with the offsets off the detector dropped.

        Peak finding can emit a few spurious peaks far outside the frame whose
        calibrated coordinates map to pixel positions like 24000, and a circle
        there litters the plot with giant arcs."""
        pixels = self.to_pixels(xy)
        if len(pixels) == 0:
            return pixels
        inside = (
            (pixels[:, 0] >= -_MARKER_SLACK_PX)
            & (pixels[:, 0] <= self.width - 1 + _MARKER_SLACK_PX)
            & (pixels[:, 1] >= -_MARKER_SLACK_PX)
            & (pixels[:, 1] <= self.height - 1 + _MARKER_SLACK_PX))
        return pixels[inside]

    def marker_radius(self, radius_px) -> float:
        """A circle radius capped to a small fraction of the detector, so the
        circles can never swamp the pattern on a tiny frame."""
        cap = max(4.0, 0.08 * min(self.width, self.height))
        return float(np.clip(float(radius_px), 2.0, cap))


class VectorRows:
    """One navigation position's vector rows, as a recipe's per-position
    argument.

    ``at(*index)`` takes the WHOLE navigation index, so a 5-D scan's leading
    time coordinate selects that slice's rows rather than every slice's."""

    def __init__(self, vectors):
        self.vectors = vectors

    def at(self, *index):
        vectors = self.vectors
        if len(index) == len(vectors.full_nav_shape):
            return vectors.slice_at(*index)
        # A stale higher-grid position, or a navigator with fewer coords than
        # the store has axes: read the spatial pair across every outer step.
        if len(index) < 2:
            return vectors.at(0, 0)
        return vectors.at(int(index[-2]), int(index[-1]))


def remove_overlay_node(tree, node) -> None:
    """Take an overlay node off ``tree``, tolerating a missing tree or node."""
    if tree is None or node is None:
        return
    try:
        tree.remove_overlay(node)
    except Exception as e:
        log.debug("removing overlay node %r failed: %s",
                  getattr(node, "name", node), e)


def clear_tree_overlay(tree, attribute: str) -> None:
    """Remove the overlay node held on ``tree.<attribute>`` and forget it, so
    re-running an action never stacks two sets of markers."""
    remove_overlay_node(tree, getattr(tree, attribute, None))
    setattr(tree, attribute, None)


def overlay_static(node) -> dict:
    """The static arguments an overlay node's function is called with."""
    return dict(node.signal._map_recipe.static)


def _add_overlay(tree, parent_signal, function, **kwargs):
    """Add an overlay node and draw it at the navigator's current position, so
    it appears without waiting for a move."""
    from spyde.drawing.overlays import refresh_overlays_for

    node = tree.add_overlay(parent_signal, function, **kwargs)
    refresh_overlays_for(tree)
    return node


# ── found vectors on the source or result diffraction pattern ────────────────

def found_vector_offsets(*, rows, pixels: DetectorPixels) -> dict:
    """The found vectors at one navigation position, as marker offsets."""
    rows = np.asarray(rows)
    if rows.size == 0:
        return {"found": np.zeros((0, 2), dtype=np.float32)}
    from spyde.signals.diffraction_vectors import COL_KX, COL_KY
    return {"found": pixels.clipped(rows[:, [COL_KX, COL_KY]])}


def attach_vector_overlay(vecs, tree, *, color="#ff3030", name="found_vectors",
                          radius_px=None, signal=None):
    """Draw the found vectors as circles on the windows showing ``signal`` (the
    node they were found on), tracking the navigator. Returns the node."""
    pixels = DetectorPixels.from_axes(vecs.sig_axes)
    if radius_px is None:
        radius_px = getattr(vecs, "kernel_radius_px", 4.0)
    style = {"radius": pixels.marker_radius(radius_px), "edgecolors": color,
             "facecolors": None, "linewidths": 1.5, "alpha": 1.0}
    return _add_overlay(
        tree, signal if signal is not None else tree.root, found_vector_offsets,
        name=name, source=False, groups={"found": ("circles", style)},
        iterating={"rows": VectorRows(vecs)}, static={"pixels": pixels},
    )


# ── the Find Vectors tuning preview ──────────────────────────────────────────

def _blurred_centre_frame(window, centre, sigma: float) -> np.ndarray:
    """The centre frame of a navigation window, Gaussian-blurred over the two
    innermost navigation axes.

    That is the batch's blur: ``(sigma, sigma, 0, 0)`` over a scan, with any
    leading axis (a 5-D stack's time index) held at the requested position
    rather than blurred across. ``centre`` is None when there is no blur to
    apply and the recipe handed over one frame instead of a window."""
    window = np.asarray(window, dtype=np.float32)
    if centre is None:
        return window
    centre = tuple(int(v) for v in centre)
    while len(centre) > 2:
        window = window[centre[0]]
        centre = centre[1:]
    if sigma > 0 and window.ndim > 2:
        from scipy.ndimage import gaussian_filter
        spread = (sigma,) * len(centre) + (0,) * (window.ndim - len(centre))
        window = gaussian_filter(window, sigma=spread)
    for position in centre:
        window = window[position]
    return window


def _transform_levels(response, method: str, threshold: float):
    """The contrast window for the detector's transformed image.

    Floor and ceiling are independent, so moving the threshold slider never
    moves the ceiling and washes the image out. A correlation score lives in
    [-1, 1], where a fixed ceiling of 1.0 holds still frame to frame; a
    band-pass signal-to-noise ratio has no fixed scale and takes its ceiling
    from the response's own 99th percentile."""
    finite = response[np.isfinite(response)]
    if method == "dog":
        high = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    else:
        high = 1.0
    low = float(threshold)
    if low >= high:
        low = high - 1e-3
    return low, high


def find_vectors_preview(window, centre=None, *, params: dict, sigma: float,
                         beamstop_mask, show_transform: bool) -> dict:
    """The peaks the detector finds in the frame under the crosshair, the image
    it found them in when the transform view is on, and the beam stop it
    excluded.

    With a navigation blur the recipe hands over the window and the requested
    position inside it; with no blur it hands over that one frame."""
    from spyde.actions.find_vectors import _find_peaks_single_frame

    frame = _blurred_centre_frame(window, centre, float(sigma))
    if show_transform:
        peaks, response = _find_peaks_single_frame(
            frame, params, beamstop_mask=beamstop_mask, with_response=True)
    else:
        peaks = _find_peaks_single_frame(frame, params,
                                         beamstop_mask=beamstop_mask)
        response = None

    if peaks is None or len(peaks) == 0:
        offsets = np.zeros((0, 2), np.float32)
    else:
        # peaks are [ky_row, kx_col, value] in pixels; a marker is (x, y).
        offsets = np.column_stack([peaks[:, 1], peaks[:, 0]]).astype(np.float32)
    threshold = float(params.get("threshold", 0.0))
    value = {"peaks": {"data": offsets, "radius": _preview_marker_radius(params)},
             "transform": None, "mask": beamstop_mask, "threshold": threshold}
    if response is not None:
        response = np.asarray(response, dtype=np.float32)
        levels = _transform_levels(
            response, str(params.get("method", "")).lower(), threshold)
        value["transform"] = {"data": response, "levels": levels}
    return value


def _preview_marker_radius(params: dict) -> float:
    """The circle radius for the preview's peaks, in pixels. For NXCORR it is
    the disk kernel radius; for DoG the natural spot scale is the band-pass,
    where a Gaussian of width sigma peaks at a blob radius of sqrt(2)*sigma."""
    if str(params.get("method", "")).lower() == "dog":
        return max(1.0, float(np.sqrt(2.0) * float(params.get("dog_sigma1", 0.8))))
    return max(1.0, float(params.get("kernel_radius", 5)))


def _detector_params(params: dict) -> dict:
    """The keys the single-frame detector reads, taken from the wizard's
    parameter set."""
    return {key: params[key] for key in
            ("method", "kernel_radius", "threshold", "min_distance", "subpixel",
             "model_id", "bg_sigma", "spot_radius", "dog_sigma1", "dog_sigma2")
            if key in params}


def _beamstop_for(tree, signal, params: dict):
    """The beam-stop mask the preview excludes, dilated to the requested
    radius. Detection reads a sample of frames once and is cached on the tree,
    so changing the dilation is a cheap filter on that cache."""
    if not params.get("beamstop_auto"):
        return None
    from spyde.actions.find_vectors import _auto_beamstop_from_signal, _dilate_mask

    if not hasattr(tree, "_fv_beamstop_raw"):
        tree._fv_beamstop_raw = _auto_beamstop_from_signal(
            signal, signal.axes_manager.navigation_dimension, dilate=0)
    raw = tree._fv_beamstop_raw
    if raw is None:
        return None
    dilate = int(params.get("beamstop_dilate", 5) or 0)
    return _dilate_mask(raw, dilate) if dilate > 0 else raw


def _preview_depth(signal, sigma: float):
    """The navigation neighbourhood the blur needs, flat on every axis above
    the two the scan is blurred over, so a stack's time axis is never crossed
    and the window stays ``(2d+1)`` frames per blurred axis."""
    radius = int(np.ceil(3 * sigma)) if sigma > 0 else 0
    navigation_dimension = int(signal.axes_manager.navigation_dimension)
    if radius == 0 or navigation_dimension <= 2:
        return radius
    return (0,) * (navigation_dimension - 2) + (radius, radius)


def attach_find_vectors_preview(dp_plot, signal, tree, params: dict,
                                *, color="#ff3030"):
    """Draw the peaks the detector finds under the crosshair, live, while the
    Find Vectors caret is tuning. Returns the node.

    ``params`` is the wizard's coerced parameter set. The navigation blur is a
    neighbourhood of radius ``ceil(3 sigma)`` on the recipe, so the preview
    sees the same frame the batch does. The beam stop, if one is wanted, lands
    a moment later: detecting it reads frames, which does not belong on the
    thread the caret is dispatched on."""
    sigma = float(params.get("sigma", 0.0))
    style = {"radius": _preview_marker_radius(params), "edgecolors": color,
             "facecolors": None, "linewidths": 1.5, "alpha": 1.0}
    node = _add_overlay(
        tree, signal, find_vectors_preview, name="fv_preview",
        depth=_preview_depth(signal, sigma),
        expensive=str(params.get("method", "")).lower() == "neural",
        groups={"peaks": ("circles", style),
                "transform": ("transform", {}),
                "mask": ("mask", {"color": "#ff8a3d", "alpha": 0.35})},
        static={"params": _detector_params(params), "sigma": sigma,
                "beamstop_mask": None,
                "show_transform": bool(params.get("show_transform"))},
        on_value=lambda value: _emit_preview_histogram(dp_plot, value),
    )
    request_beamstop(tree, node, params)
    return node


def request_beamstop(tree, node, params: dict) -> None:
    """Put the beam-stop mask the preview excludes on the overlay node, off the
    caller's thread.

    Detection reads a sample of frames, so it runs on the compute backend's
    overlay lane and the mask reaches the recipe from the done callback. The
    preview returns the mask under its own group every move, so the drawing
    follows the static argument and needs no push of its own. Turning the stop
    off needs no scan and applies straight away."""
    if not params.get("beamstop_auto"):
        if overlay_static(node).get("beamstop_mask") is not None:
            tree.replace_overlay_static(node, beamstop_mask=None)
        return
    backend = getattr(getattr(tree, "session", None), "compute_backend", None)
    signal = node.parent.signal
    if backend is None:
        log.debug("[fv-preview] no compute backend for the beam-stop estimate")
        return

    def _apply(finished) -> None:
        try:
            mask = finished.result()
        except Exception as e:
            log.debug("[fv-preview] beam-stop detection failed: %s", e)
            return
        if not node.attached:
            return              # the caret closed while the stop was found
        tree.replace_overlay_static(node, beamstop_mask=mask)

    try:
        backend.submit_overlay(
            lambda: _beamstop_for(tree, signal, params)).add_done_callback(_apply)
    except Exception as e:
        log.debug("[fv-preview] the beam-stop estimate was not submitted: %s", e)


def _emit_preview_histogram(plot, value) -> None:
    """Mark the detector threshold on the window's histogram, so the slider has
    a scale to move against. Runs on the painter thread, after the image."""
    transform = value.get("transform") if isinstance(value, dict) else None
    if transform is None or not hasattr(plot, "_emit_histogram"):
        return
    image, (low, high) = transform["data"], transform["levels"]
    try:
        plot._emit_histogram(image, low, high,
                             threshold=float(value.get("threshold", low)))
    except Exception as e:
        log.debug("[fv-preview] emitting the transform histogram failed: %s", e)


def remove_find_vectors_preview(tree) -> None:
    """Drop the live preview. Its groups go with the node, the beam-stop mask
    among them."""
    clear_tree_overlay(tree, "_fv_preview")


def tune_find_vectors_preview(tree, node, params: dict) -> None:
    """Apply a new parameter set to the live preview and redraw at the current
    crosshair position. The circle radius rides the next value, so nothing is
    rebuilt for it; the beam stop, if it changed, lands from the overlay lane."""
    sigma = float(params.get("sigma", 0.0))
    node.expensive = str(params.get("method", "")).lower() == "neural"
    tree.replace_overlay_static(
        node, params=_detector_params(params), sigma=sigma,
        show_transform=bool(params.get("show_transform")))
    request_beamstop(tree, node, params)


# ── the matched orientation template ─────────────────────────────────────────

def orientation_template_spots(frame, *, pixels: DetectorPixels, sim, cache,
                               gamma, max_radius, normalize_templates,
                               scale_override, min_intensity) -> dict:
    """The best-matching template's simulated spots for one pattern."""
    from spyde.actions.orientation_compute import best_match_spots

    coords = best_match_spots(
        np.asarray(frame, dtype=float), sim, cache, gamma=float(gamma),
        max_radius=max_radius, normalize_templates=bool(normalize_templates),
        # Both scales are in the DISPLAYED unit, so their ratio — all
        # best_match_spots uses — is unit-free and needs no conversion.
        scale_override=scale_override, original_scale=pixels.x_scale,
        min_intensity=float(min_intensity))
    if coords is None or len(coords) == 0:
        return {"template": np.zeros((0, 2), dtype=np.float32)}
    # `coords` and `max_radius` are Å⁻¹ (diffsims' unit), the axes may be in
    # anything — so the placement goes through the conversion, not straight to
    # pixels.
    return {"template": pixels.inverse_angstrom_to_pixels(coords)}


def attach_orientation_overlay(signal, sim, matching_cache, tree, *,
                               gamma=0.5, max_radius=None,
                               normalize_templates=True, scale_override=None,
                               min_intensity=0.0, color="#30ff60",
                               name="orientation_template", radius_px=4.0):
    """Draw the best-matching template's spots on the windows showing
    ``signal``, re-matched at every navigator position. Returns the node."""
    # From the SIGNAL, not its axes: the simulated spots are in Å⁻¹ and putting
    # them back on a detector calibrated in mrad needs the beam energy, which
    # only the signal carries.
    pixels = DetectorPixels.from_signal(signal)
    style = {"radius": max(2.0, float(radius_px)), "edgecolors": color,
             "facecolors": None, "linewidths": 1.5, "alpha": 1.0}
    return _add_overlay(
        tree, signal, orientation_template_spots, name=name,
        groups={"template": ("circles", style)},
        static={"pixels": pixels, "sim": sim, "cache": matching_cache,
                "gamma": float(gamma), "max_radius": max_radius,
                "normalize_templates": bool(normalize_templates),
                "scale_override": scale_override,
                "min_intensity": float(min_intensity)},
    )


def set_orientation_refine_params(tree, node, **params) -> None:
    """Live-update the Refine sliders and redraw the matched template."""
    static = {}
    if params.get("gamma") is not None:
        static["gamma"] = float(params["gamma"])
    if params.get("normalize_templates") is not None:
        static["normalize_templates"] = bool(params["normalize_templates"])
    if "scale_override" in params:
        override = params["scale_override"]
        static["scale_override"] = (float(override)
                                    if override not in (None, "", 0) else None)
    if params.get("min_intensity") is not None:
        static["min_intensity"] = float(params["min_intensity"])
    if static:
        tree.replace_overlay_static(node, **static)


# ── the strain reference-spot selection ──────────────────────────────────────

_STRAIN_SELECTED_COLOR = "#30ff60"      # green = drives the fit
_STRAIN_EXCLUDED_COLOR = "#6c7086"      # grey  = ignored
_STRAIN_ARROW_COLOR = "#fab387"         # orange displacement arrows


def strain_selection(*, rows, position, pixels: DetectorPixels, ref_yx,
                     ref_spots, selected, match_radius_px) -> dict:
    """The reference reflections on the reference pixel, and the per-spot
    displacement arrows everywhere else.

    On the reference pixel the reference spots are drawn as circles, green for
    the ones driving the fit and grey for the excluded ones, and a double-click
    toggles them. Off it, each selected spot is joined to the nearest measured
    peak within ``match_radius_px``, which is the local displacement.

    ``on_reference`` is not a group: it tells the overlay's owner which of the
    two the value is, which is what makes the double-click a pick on the
    reference pixel and nothing anywhere else."""
    from spyde.signals.diffraction_vectors import COL_KX, COL_KY

    empty = np.zeros((0, 2), np.float32)
    no_arrows = (empty, np.zeros(0, np.float32), np.zeros(0, np.float32))
    ref_spots = np.asarray(ref_spots, float).reshape(-1, 2)
    selected = np.asarray(selected, bool).reshape(-1)
    position = tuple(int(v) for v in np.atleast_1d(position))
    on_reference = tuple(position[-2:]) == tuple(int(v) for v in ref_yx)

    if on_reference:
        chosen = ref_spots[selected] if len(ref_spots) else empty
        excluded = ref_spots[~selected] if len(ref_spots) else empty
        return {"selected": pixels.clipped(chosen),
                "excluded": pixels.clipped(excluded), "arrows": no_arrows,
                "on_reference": True}

    reference = ref_spots[selected] if len(ref_spots) else empty
    rows = np.asarray(rows)
    if len(reference) == 0 or rows.size == 0:
        return {"selected": empty, "excluded": empty, "arrows": no_arrows,
                "on_reference": False}
    reference_px = pixels.to_pixels(reference)
    measured_px = pixels.to_pixels(rows[:, [COL_KX, COL_KY]])
    tails, across, down = [], [], []
    radius_squared = float(match_radius_px) ** 2
    for x, y in reference_px:
        distance = (measured_px[:, 0] - x) ** 2 + (measured_px[:, 1] - y) ** 2
        nearest = int(np.argmin(distance))
        if distance[nearest] <= radius_squared:
            tails.append([x, y])
            across.append(float(measured_px[nearest, 0] - x))
            down.append(float(measured_px[nearest, 1] - y))
    if not tails:
        return {"selected": empty, "excluded": empty, "arrows": no_arrows,
                "on_reference": False}
    return {"selected": empty, "excluded": empty, "on_reference": False,
            "arrows": (np.asarray(tails, np.float32),
                       np.asarray(across, np.float32),
                       np.asarray(down, np.float32))}


class StrainSelectionOverlay:
    """The strain reference selection: an overlay node plus the double-click
    that toggles a reference spot in or out of the fit.

    The selection mask is the node's ``selected`` static argument and is the
    single source of truth for which reference vectors drive the strain fit; a
    toggle replaces it and the navigator redraw does the drawing."""

    def __init__(self, plot, vecs, tree, *, ref_yx, ref_spots, selected=None,
                 match_radius_px=6.0, radius_px=None, on_toggle=None):
        self.plot = plot
        self.tree = tree
        self.on_toggle = on_toggle
        self._removed = False
        self._click = None
        self.pixels = DetectorPixels.from_axes(vecs.sig_axes)
        if radius_px is None:
            radius_px = getattr(vecs, "kernel_radius_px", 4.0)
        self.radius_px = self.pixels.marker_radius(radius_px)
        spots = np.asarray(ref_spots if ref_spots is not None
                           else np.zeros((0, 2)), float).reshape(-1, 2)
        # Start with nothing marked: the user picks which reflections drive the
        # fit rather than every reflection at a fresh reference pixel.
        mask = (np.zeros(len(spots), dtype=bool) if selected is None
                else np.asarray(selected, dtype=bool).reshape(-1))
        circles = {"facecolors": None, "alpha": 1.0, "radius": self.radius_px}
        displayed = getattr(plot.plot_state, "current_signal", None)
        self.node = _add_overlay(
            tree, tree.root if displayed is None else displayed, strain_selection,
            name="strain_select", source=False,
            groups={
                "excluded": ("circles", dict(circles, linewidths=1.3,
                                             alpha=0.9,
                                             edgecolors=_STRAIN_EXCLUDED_COLOR)),
                "selected": ("circles", dict(circles, linewidths=2.0,
                                             edgecolors=_STRAIN_SELECTED_COLOR)),
                "arrows": ("arrows", {"edgecolors": _STRAIN_ARROW_COLOR,
                                      "linewidths": 1.6}),
            },
            iterating={"rows": VectorRows(vecs),
                       "position": NavigationPosition()},
            static={"pixels": self.pixels,
                    "ref_yx": (int(ref_yx[0]), int(ref_yx[1])),
                    "ref_spots": spots, "selected": mask,
                    "match_radius_px": float(match_radius_px)},
        )
        self._wire_double_click(plot)

    def _wire_double_click(self, plot) -> None:
        # A single click is ambiguous with panning on a 2-D anyplotlib panel,
        # so a discrete pick uses "double_click" (the Particle Picker example
        # makes the same choice).
        from spyde.drawing.selectors.base_selector import event_handler_fn

        plot2d = getattr(plot, "_plot2d", None)
        if plot2d is None:
            return
        self._click = event_handler_fn(self._on_click)
        try:
            plot2d.add_event_handler(self._click, "double_click")
        except Exception as e:
            log.debug("[strain-select] wiring the double-click failed: %s", e)

    # ── the state the controller drives ──────────────────────────────────────
    @property
    def ref_spots(self) -> np.ndarray:
        return overlay_static(self.node)["ref_spots"]

    @property
    def selected(self) -> np.ndarray:
        return overlay_static(self.node)["selected"]

    @property
    def ref_yx(self) -> tuple:
        return overlay_static(self.node)["ref_yx"]

    @property
    def match_radius_px(self) -> float:
        return overlay_static(self.node)["match_radius_px"]

    def selected_reference(self) -> np.ndarray:
        """The selected reference spots, calibrated: what the fit uses."""
        spots, mask = self.ref_spots, self.selected
        return spots[mask] if len(spots) else spots

    def set_reference(self, ref_yx, ref_spots, selected=None) -> None:
        """New reference pixel. Without an explicit ``selected``, CARRY FORWARD
        the current selection: every marked spot stays marked at the new pixel
        if a peak is still within ``match_radius_px`` of its old calibrated
        position, the same nearest-peak rule the arrows use."""
        spots = np.asarray(ref_spots, float).reshape(-1, 2)
        mask = (np.asarray(selected, bool).reshape(-1) if selected is not None
                else self._carry_forward(spots))
        self.tree.replace_overlay_static(
            self.node, ref_yx=(int(ref_yx[0]), int(ref_yx[1])),
            ref_spots=spots, selected=mask)

    def _carry_forward(self, new_spots: np.ndarray) -> np.ndarray:
        spots, mask = self.ref_spots, self.selected
        marked = spots[mask] if len(spots) else np.zeros((0, 2))
        carried = np.zeros(len(new_spots), dtype=bool)
        if len(new_spots) == 0 or len(marked) == 0:
            return carried
        # The pixel match radius in calibrated units, as the arrows use it.
        radius_squared = (self.match_radius_px * self.pixels.max_scale) ** 2
        for x, y in marked:
            distance = ((new_spots[:, 0] - x) ** 2 + (new_spots[:, 1] - y) ** 2)
            nearest = int(np.argmin(distance))
            if distance[nearest] <= radius_squared:
                carried[nearest] = True
        return carried

    def set_match_radius(self, radius_px: float) -> None:
        self.tree.replace_overlay_static(
            self.node, match_radius_px=max(1.0, float(radius_px)))

    def set_visible(self, visible: bool) -> None:
        self.tree.set_overlay_visible(self.node, visible)

    def remove(self) -> None:
        # anyplotlib has no remove_event_handler, so the handler is made inert
        # instead: with the node gone there is nothing left to toggle.
        self._removed = True
        self._click = None
        remove_overlay_node(self.tree, self.node)

    # ── the double-click ─────────────────────────────────────────────────────
    def _on_click(self, event=None) -> None:
        """Toggle the reference spot under the cursor in or out of the fit.

        A spot is picked only while this window's navigator sits on the
        reference pixel, which is what the last value drawn here says.

        anyplotlib's double-click event carries xdata/ydata, which for a
        calibrated diffraction pattern are physical kx, ky, the space the
        reference spots are already stored in, so the hit test needs no
        conversion."""
        spots = self.ref_spots
        if self._removed or event is None or len(spots) == 0:
            return
        drawn = self.plot.last_overlay_value(self.node) or {}
        if not drawn.get("on_reference"):
            return
        try:
            x, y = float(event.xdata), float(event.ydata)
        except Exception as e:
            log.debug("[strain-select] the click carried no position: %s", e)
            return
        distance = (spots[:, 0] - x) ** 2 + (spots[:, 1] - y) ** 2
        nearest = int(np.argmin(distance))
        hit_radius = (self.radius_px + 4.0) * self.pixels.max_scale
        if distance[nearest] > hit_radius ** 2:
            return
        mask = self.selected.copy()
        mask[nearest] = not mask[nearest]
        self.tree.replace_overlay_static(self.node, selected=mask)
        if self.on_toggle is not None:
            try:
                self.on_toggle()
            except Exception as e:
                log.debug("[strain-select] the toggle callback failed: %s", e)


def attach_strain_selection_overlay(dp_plot, vecs, tree, *, ref_yx, ref_spots,
                                    selected=None, match_radius_px=6.0,
                                    on_toggle=None) -> StrainSelectionOverlay:
    """Add the interactive strain reference-spot selection to ``dp_plot``."""
    return StrainSelectionOverlay(
        dp_plot, vecs, tree, ref_yx=ref_yx, ref_spots=ref_spots,
        selected=selected, match_radius_px=match_radius_px,
        on_toggle=on_toggle)


# ── the vector-orientation refine preview ────────────────────────────────────

def vector_orientation_fit(*, rows, pixels: DetectorPixels, lib,
                           params: dict) -> dict:
    """The measured vectors and the fitted template at one position.

    The pose (theta, A, t) is fitted against the template library and the
    template is drawn as ``A.Rot(theta).g + t``."""
    from spyde.actions.vector_orientation import (
        fit_pattern, project_spots, DEFAULTS, COL_KX, COL_KY, COL_INTENSITY,
    )

    rows = np.asarray(rows)
    if rows.size == 0:
        return {"measured": None, "template": None, "fit": None}
    # The vectors are in the detector's own units and everything the fit
    # touches — the template library, the soft-assign bandwidths, the no-match
    # sink — is in Å⁻¹, so they are converted rather than compared across units.
    measured = pixels.to_inverse_angstrom(rows[:, [COL_KX, COL_KY]])
    measured_px = pixels.to_pixels(rows[:, [COL_KX, COL_KY]].astype(np.float64))
    if len(rows) < 4:
        return {"measured": measured_px, "template": None, "fit": None}

    try:
        fit = fit_pattern(measured, rows[:, COL_INTENSITY].astype(np.float64),
                          lib, {**DEFAULTS, **params})
    except Exception as e:
        # A pose that will not converge still has measured vectors to draw,
        # and the caret's readout has to be told there is no fit.
        log.debug("the vector-orientation fit failed at this position: %s", e)
        fit = None
    if fit is None:
        return {"measured": measured_px, "template": None, "fit": None}
    pose = np.zeros(7, np.float64)
    pose[0] = float(fit.theta)
    pose[1:5] = np.asarray(fit.affine, float).reshape(-1)
    pose[5:7] = np.asarray(fit.translation, float)
    spots = np.asarray(lib.spots_xy[int(fit.template_idx)], np.float64)
    return {"measured": measured_px,
            "template": pixels.inverse_angstrom_to_pixels(project_spots(pose, spots)),
            "fit": fit}


def attach_vector_orientation_overlay(vecs, lib, tree, *, params=None,
                                      radius_px=None, on_fit=None):
    """Draw the measured vectors (red) and the fitted template (green) on the
    vectors diffraction pattern. Returns the node."""
    # Take the unit factor from the library rather than re-deriving it from the
    # axis records: the library was built against the live signal and so can
    # resolve a detector in mrad, which an axis record alone cannot. Both sides
    # of this overlay then convert by the same number as the whole-field fit.
    pixels = DetectorPixels.from_axes(
        vecs.sig_axes,
        inverse_angstrom_factor=float(getattr(lib, "inverse_angstrom_factor", 1.0)))
    if radius_px is None:
        radius_px = getattr(vecs, "kernel_radius_px", 4.0)
    style = {"radius": max(2.0, float(radius_px)), "facecolors": None,
             "linewidths": 1.5, "alpha": 1.0}
    return _add_overlay(
        tree, tree.root, vector_orientation_fit, name="vom_refine",
        source=False,
        groups={"measured": ("circles", dict(style, edgecolors="#ff3030")),
                "template": ("circles", dict(style, edgecolors="#30ff60"))},
        iterating={"rows": VectorRows(vecs)},
        static={"pixels": pixels, "lib": lib, "params": dict(params or {})},
        on_value=(None if on_fit is None
                  else lambda value: on_fit(value.get("fit")
                                            if isinstance(value, dict) else None)),
    )


def set_vector_orientation_params(tree, node, **params) -> None:
    """Live-update the vector-orientation refine weights and redraw."""
    merged = dict(overlay_static(node).get("params", {}))
    merged.update({k: v for k, v in params.items() if v is not None})
    tree.replace_overlay_static(node, params=merged)
