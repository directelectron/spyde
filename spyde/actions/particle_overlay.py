"""
particle_overlay.py — the live particle overlay + the stacked navigator lanes.

Plan B9 (overlay and editing), C2 (events on the navigator) and C3 (trails,
integer lanes are step plots). Modelled on :mod:`spyde.actions.vector_overlay`:
marker groups on the signal plot, re-pushed from an ``index_hook`` on the tree's
navigation selectors, torn down through ``lifecycle.replace_tree_attr``.

Two coordinate systems, and the whole overlay hangs on telling them apart
--------------------------------------------------------------------------
anyplotlib 2-D markers drawn with ``transform="data"`` are addressed in
**image-pixel** coordinates — no axis scale or offset (``masks._signal_k_grids``
documents the bug class; building geometry in physical units gives an empty or
misplaced overlay on any calibrated axis). But the two things this module draws
live in *different* spaces already:

* **contours** are stored by ``measure_frame`` as int16 **pixel** ``(y, x)`` — so
  they only need the axis swap to ``(x, y)``, never a scale;
* **centroids** (``COL["y"]``/``COL["x"]``) are **calibrated**, ``pixel * scale``
  — so they must be divided by ``particles.scale`` to reach widget space.

Dividing a contour or forgetting to divide a centroid both produce an overlay
that is plausible at ``scale == 1`` and silently wrong everywhere else, which is
exactly how the ``_signal_k_grids`` bug survived. The synthetic fixture is
``scale=0.5`` on purpose, and ``test_particle_overlay.py`` pins the conversion
there.

A click arrives in the THIRD space: anyplotlib's ``Event`` carries only
``xdata``/``ydata`` (the JS emits ``img_x``/``img_y`` too, but anyplotlib's Event
dataclass has no such field and drops them), and those are the panel's *physical*
data coordinates. So the hit test converts the click back to pixels through the
**plot's own** signal axes rather than assuming they agree with
``particles.scale``.

One marker group per colour — not one group with a colour array
---------------------------------------------------------------
anyplotlib's 1-D marker path accepts ``fill_color``/``color`` as arrays parallel
to the items; its **2-D** path does not (``drawMarkers2d`` in ``figure_esm.js``
binds one ``fillStyle``/``strokeStyle`` for the whole set, and an array there is
an invalid canvas colour that silently leaves the previous style in place). The
overlay lives on a 2-D panel, so "colour per track" means one group per colour:
six accent buckets plus one grey for untracked rows. That is also why the trail
needs a group per (colour, age) pair — a polyline cannot fade along its own
length in this wire format.

The consequence is ~30 marker groups, and ``MarkerGroup.set`` re-serialises the
WHOLE registry on every call, so the groups are updated through
:func:`_push_groups`, which mutates them and pushes **once** per frame.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from spyde.signals.particles import COL, MEASURED_COLUMNS, N_COLUMNS

log = logging.getLogger(__name__)


# ── palette ──────────────────────────────────────────────────────────────────

#: SpyDE's six accents, in order. Track ids cycle through them by ``id % 6`` so
#: the mapping is a pure function of the id: the same track keeps its colour as
#: the user scrubs, across a re-open, and between the overlay and any other
#: surface that colours by track (the kymograph, the report embed).
TRACK_COLORS: tuple[str, ...] = (
    "#89b4fa",   # blue    — track 0
    "#f38ba8",   # red
    "#a6e3a1",   # green
    "#f9e2af",   # yellow
    "#cba6f7",   # mauve
    "#94e2d5",   # teal
)

#: Untracked rows (``track_id < 0``: the linker has not run, or this row was
#: created by an edit and has not been re-linked). Grey, not a seventh accent —
#: "no identity yet" must not read as "identity number six".
UNTRACKED_COLOR = "#6c7086"

#: The selected particle's outline. Deliberately outside :data:`TRACK_COLORS` so
#: selection is never confusable with a track colour.
SELECTED_COLOR = "#ffffff"

#: Fill opacity. 25% keeps the underlying image readable at the plan's target
#: density (hundreds of particles per frame) — the outline carries the shape.
FILL_ALPHA = 0.25
OUTLINE_WIDTH = 1.0
SELECTED_WIDTH = 2.0

#: Trailing window, in frames. ~8 is enough to read a direction without the
#: trails of a dense field crossing into an unreadable mesh.
DEFAULT_TRAIL_FRAMES = 8

#: How many opacity steps the trail fades through. A polyline carries ONE alpha
#: in this wire format (see the module docstring), so each step is a separate
#: marker group per colour: 6 x 4 = 24 groups. Four steps is where the ramp stops
#: looking banded on an 8-frame window; eight would double the group count for a
#: difference that is invisible at 1 px line width.
TRAIL_FADE_STEPS = 4

#: Head-dot radius in image pixels.
HEAD_RADIUS_PX = 2.5

#: Event lane colours, plan C2. Order is ``track.EVENT_KINDS``.
EVENT_COLORS: dict[str, str] = {
    "birth": "#a6e3a1",    # green
    "death": "#f38ba8",    # red
    "merge": "#cba6f7",    # mauve
    "split": "#f9e2af",    # yellow
}

#: Properties shown in the selected particle's readout. Three, not twelve: the
#: readout sits ON the frame beside the particle, so it competes with the data.
READOUT_COLUMNS: tuple[str, ...] = ("area", "equiv_diameter", "circularity")

#: The caret's parameter schema — the single host-agnostic source of truth
#: (README §4.2), resolved by ``registry.wizard_parameters("part")``. Same dict
#: spec as a ``toolbars.yaml`` ``parameters:`` block.
PARAMETERS: dict[str, dict] = {
    "show_trails": {
        "name": "Trails", "type": "bool", "default": False,
        "description": "Fade the last N frames of each track behind a head dot.",
    },
    "trail_frames": {
        "name": "Trail length", "type": "int", "default": DEFAULT_TRAIL_FRAMES,
        "min": 2, "max": 60, "step": 1,
        "description": "Trailing window, in frames.",
    },
    "region_select": {
        "name": "Region select", "type": "bool", "default": False,
        "description": "Rubber-band box for selecting many particles at once.",
    },
}


def track_color(track_id) -> str:
    """Colour for a track id.

    Parameters
    ----------
    track_id
        The ``track_id`` column value. Negative (or NaN) means "not linked" and
        maps to :data:`UNTRACKED_COLOR`.

    Returns
    -------
    str
        A ``#rrggbb`` hex string from :data:`TRACK_COLORS`, cycling every six.
    """
    try:
        tid = int(track_id)
    except (TypeError, ValueError):
        return UNTRACKED_COLOR
    if tid < 0:
        return UNTRACKED_COLOR
    return TRACK_COLORS[tid % len(TRACK_COLORS)]


def fade(color: str, alpha: float) -> str:
    """``#rrggbb`` + an 8-bit alpha byte → ``#rrggbbaa``.

    Canvas accepts 8-digit hex, which is how a trail segment carries its own
    opacity even though the group's ``linewidth``/``color`` are scalars.
    """
    a = int(round(float(np.clip(alpha, 0.0, 1.0)) * 255))
    return f"{color}{a:02x}"


def trail_alphas(steps: int = TRAIL_FADE_STEPS) -> list[float]:
    """Opacity per age step, newest first. Linear from 1.0 down to 0.2.

    Not down to 0: the oldest step of a trail still has to be visible, or the
    window reads as shorter than it is.
    """
    n = max(1, int(steps))
    if n == 1:
        return [1.0]
    return [1.0 - 0.8 * i / (n - 1) for i in range(n)]


# ── geometry ─────────────────────────────────────────────────────────────────

def frame_from_indices(indices, nav_to_frame=None) -> int:
    """Particle-frame index from a navigation selector's committed indices.

    A particle tree's navigation space is 1-D (time), so the first raveled
    coordinate IS the frame — the same read ``navigator_views._StackedNavCursor``
    makes. *nav_to_frame*, when given, maps a source navigation index onto a
    particle frame (the inverse of ``tree.nav_map``), for an overlay drawn on the
    SOURCE movie rather than on the particle tree's own label movie.
    """
    try:
        idx = int(np.asarray(indices).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return 0
    if nav_to_frame is not None:
        return int(nav_to_frame.get(idx, idx))
    return idx


def centroids_px(rows: np.ndarray, scale: float) -> np.ndarray:
    """``(n, 2)`` float32 ``(x, y)`` marker offsets from property rows.

    Centroids are stored calibrated (``pixel * scale``), markers want image
    pixels — see the module docstring. Column order flips too: the row carries
    ``(y, x)``, a marker offset is ``(x, y)``.
    """
    rows = np.asarray(rows)
    if rows.size == 0:
        return np.zeros((0, 2), np.float32)
    s = float(scale) or 1.0
    return np.column_stack([rows[:, COL["x"]] / s,
                            rows[:, COL["y"]] / s]).astype(np.float32)


def contour_xy(particles, index: int) -> np.ndarray:
    """One particle's outline as ``(k, 2)`` float32 ``(x, y)`` image pixels.

    Contours are already stored in pixels, so this is an axis swap and a dtype
    change — deliberately NOT a division by ``scale``.
    """
    c = particles.contour_at(int(index))
    if len(c) < 3:
        return np.zeros((0, 2), np.float32)
    return np.column_stack([c[:, 1], c[:, 0]]).astype(np.float32)


def _axis_scale_offset(plot) -> tuple[float, float, float, float]:
    """``(x_scale, x_offset, y_scale, y_offset)`` of the plot's displayed signal.

    Used to convert a click's ``xdata``/``ydata`` (physical) back to the image
    pixels every marker lives in. Falls back to an identity mapping when the plot
    has no calibrated axes, which is also what anyplotlib does — with no
    ``x_axis`` array on the panel it reports ``xdata == img_x``.
    """
    try:
        state = getattr(plot, "plot_state", None)
        sig = getattr(state, "current_signal", None)
        ax = sig.axes_manager.signal_axes
        return (float(ax[0].scale) or 1.0, float(ax[0].offset),
                float(ax[1].scale) or 1.0, float(ax[1].offset))
    except Exception:
        return 1.0, 0.0, 1.0, 0.0


def _navigator_selectors_for(tree, plot) -> list:
    """Navigation selectors that drive *plot*.

    Same resolution (and the same composite-selector dedup) as
    ``vector_overlay._navigator_selectors_for``: a composite navigator selector
    exposes both its crosshair and its region sub-selector, and registering the
    hook on both fires two redraws per navigator move.
    """
    npm = getattr(tree, "navigator_plot_manager", None)
    if npm is None:
        return []
    out = [sel for sel in npm.all_navigation_selectors
           if plot in getattr(sel, "active_children", [])]
    out = out or list(npm.all_navigation_selectors)
    seen, uniq = set(), []
    for sel in out:
        key = id(getattr(sel, "parent", sel) or sel)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(sel)
    return uniq


# ── marker pushes ────────────────────────────────────────────────────────────

def _push_groups(plot2d, updates: dict) -> None:
    """Apply many marker-group updates with ONE panel push.

    ``MarkerGroup.set`` re-serialises the whole marker registry and pushes the
    panel state on every call. A single navigator move touches the seven fill
    groups, the seven head-dot groups and up to twenty-four trail groups, so
    doing it through ``set`` would be ~30 full serialisations (and, in the
    Electron host, ~30 PLOTAPP lines) per frame. Mutating the group dicts and
    pushing once is the identical result for a thirtieth of the transport.

    Falls back to per-group ``set`` if ``_push_markers`` ever goes away.
    """
    if not updates:
        return
    pusher = getattr(plot2d, "_push_markers", None)
    if pusher is None:
        for group, kwargs in updates.items():
            try:
                group.set(**kwargs)
            except Exception as exc:
                log.debug("[particles] marker set failed: %s", exc)
        return
    for group, kwargs in updates.items():
        try:
            group._data.update(kwargs)
        except Exception as exc:
            log.debug("[particles] marker update failed: %s", exc)
    try:
        pusher()
    except Exception as exc:
        log.debug("[particles] marker push failed: %s", exc)


# ── the overlay ──────────────────────────────────────────────────────────────

class ParticleOverlay:
    """Filled, track-coloured particle outlines on a signal plot, live.

    Draws frame *t*'s particles as 25%-filled polygons with a 1 px outline,
    coloured by ``track_id % 6``; the selected particle gets a white outline, its
    id and a short property readout. Optional trails fade the last *N* frames of
    each track behind a head dot marking "here, now".

    Parameters
    ----------
    plot
        The :class:`~spyde.drawing.plots.plot.Plot` to draw on (the label movie's
        signal plot, or the source movie's).
    particles
        The :class:`~spyde.signals.particles.SpyDEParticles` store. Held by
        reference and MUTATED IN PLACE by the edit methods, because the lazy
        label movie closes over this exact object (see
        ``particle_tree.open_particle_tree``).
    events
        Optional :class:`~spyde.particles.track.ParticleEvent` list. Carried so a
        consumer reading the overlay has them to hand; the navigator lanes read
        ``tree.particle_events`` directly. Nothing here DRAWS them yet — plan C2's
        third surface (a birth/death badge flashed on the frame during playback)
        is not implemented.
    nav_map
        The tree's ``nav_map`` (particle frame → source navigation index). Given
        only when the overlay is drawn on the SOURCE movie, where the navigator
        indexes the source grid; it is inverted internally.
    frame_provider
        ``fn(t) -> (h, w) ndarray`` returning the intensity image for frame *t*.
        Used to re-measure after a merge or split. ``None`` leaves the intensity
        columns NaN on edited rows rather than inventing them.
    trail_frames, show_trails
        Trailing window length and whether trails start on.
    on_select, on_edit
        Callbacks fired after a selection change / an edit, so a caret or table
        dock can follow. Both are called with no arguments.
    """

    def __init__(self, plot, particles, *, events: Sequence = (),
                 nav_map=None, frame_provider: Callable[[int], np.ndarray] | None = None,
                 trail_frames: int = DEFAULT_TRAIL_FRAMES, show_trails: bool = False,
                 on_select: Callable[[], None] | None = None,
                 on_edit: Callable[[], None] | None = None,
                 name: str = "particles"):
        self.plot = plot
        self.particles = particles
        self.events = list(events or ())
        self.name = str(name)
        self.frame_provider = frame_provider
        self.trail_frames = max(1, int(trail_frames))
        self.show_trails = bool(show_trails)
        self.on_select = on_select
        self.on_edit = on_edit

        self.tree = None
        self.selected: list[int] = []
        self.hovered: int | None = None
        self._frame = 0
        self._hidden = False
        self._groups: dict[str, Any] = {}
        self._selectors: list = []
        self._handlers: list = []
        self._region_widget = None
        # Latest-wins: a navigator move computes off the main thread and marshals
        # the push; teardown bumps FIRST so a superseded payload never lands.
        self._gen = 0
        # nav index → particle frame, only when the two grids differ.
        self._nav_to_frame = None
        if nav_map is not None:
            nav = np.asarray(nav_map, np.int64).ravel()
            if not np.array_equal(nav, np.arange(nav.size)):
                self._nav_to_frame = {int(v): i for i, v in enumerate(nav)}
        self._prev_settled_ms = None
        # The store's identity at the last redraw. An edit rebuilds the buffers,
        # so any cached global index the caret is holding is stale; the counter
        # is what a consumer compares against.
        self.revision = 0

    # ── attach / detach ──────────────────────────────────────────────────────

    def attach(self, tree) -> "ParticleOverlay":
        """Create the marker groups, wire the navigator and the click handlers."""
        self.tree = tree
        plot2d = getattr(self.plot, "_plot2d", None)
        if plot2d is None:
            # Cosmetic, so not fatal — but say so, rather than leaving the user
            # with a silently missing overlay (vector_overlay's rule).
            log.warning("[particles] overlay skipped: plot has no live 2-D plot "
                        "(figure iframe not loaded?)")
            return self
        self._build_groups(plot2d)
        self._wire_events(plot2d)
        self._wire_navigator(tree)
        self._redraw()
        return self

    def _build_groups(self, plot2d) -> None:
        """One group per colour (see the module docstring), created empty.

        CREATION ORDER IS DRAW ORDER, and it is by marker TYPE, not by group:
        ``MarkerRegistry.to_wire_list`` walks its type dicts in the order they
        were first touched and flattens each one's groups. So the first
        ``add_lines`` puts every trail underneath every polygon, and the first
        ``add_circles`` puts every head dot on top of both — which is the z-order
        this wants (a head dot hidden under a fill is not a head dot). Within the
        polygons type the selected outline is added after the fills, so it draws
        over them.
        """
        empty_poly: list = []
        empty_off = np.zeros((0, 2), np.float32)
        empty_seg = np.zeros((0, 2, 2), np.float32)
        colors = list(TRACK_COLORS) + [UNTRACKED_COLOR]

        for i, color in enumerate(colors):                       # bottom: trails
            for step, alpha in enumerate(trail_alphas(TRAIL_FADE_STEPS)):
                self._groups[f"trail{i}_{step}"] = plot2d.add_lines(
                    empty_seg, name=f"{self.name}_trail_{i}_{step}",
                    edgecolors=fade(color, alpha), linewidths=1.5,
                    transform="data")
        for i, color in enumerate(colors):                       # then the fills
            self._groups[f"fill{i}"] = plot2d.add_polygons(
                empty_poly, name=f"{self.name}_fill_{i}", facecolors=color,
                edgecolors=color, linewidths=OUTLINE_WIDTH, alpha=FILL_ALPHA,
                transform="data")
        self._groups["selected"] = plot2d.add_polygons(
            empty_poly, name=f"{self.name}_selected", facecolors=None,
            edgecolors=SELECTED_COLOR, linewidths=SELECTED_WIDTH,
            transform="data")
        for i, color in enumerate(colors):                       # then head dots
            self._groups[f"head{i}"] = plot2d.add_circles(
                empty_off, name=f"{self.name}_head_{i}", radius=HEAD_RADIUS_PX,
                facecolors=color, edgecolors=color, linewidths=1.0, alpha=1.0,
                transform="data")
        self._groups["labels"] = plot2d.add_texts(                # top: labels
            empty_off, [], name=f"{self.name}_labels", color=SELECTED_COLOR,
            fontsize=11, transform="data")

    def _wire_events(self, plot2d) -> None:
        """Click-to-select and hover-to-label.

        ``double_click`` for the discrete pick, matching the strain reference
        picker: a single click on a 2-D panel is ambiguous with panning, and
        anyplotlib's own pan/click disambiguation has moved between versions.

        Hover runs off ``pointer_settled``, not ``pointer_move``: a label that
        needed an IPC round trip per mouse move would put hundreds of messages a
        second on the same stdout line protocol the nav painter uses. Settling is
        also the correct semantics — the label answers "what am I pointing at",
        which is only a question once the pointer has stopped.
        """
        from spyde.drawing.selectors.base_selector import event_handler_fn
        for event_type, method in (("double_click", self._on_click),
                                   ("pointer_settled", self._on_settled)):
            handler = event_handler_fn(method)
            self._handlers.append(handler)
            try:
                plot2d.add_event_handler(handler, event_type)
            except Exception as exc:
                log.debug("[particles] wiring %s failed: %s", event_type, exc)
        try:
            self._prev_settled_ms = plot2d._state.get("pointer_settled_ms")
            plot2d.configure_pointer_settled(180)
        except Exception as exc:
            log.debug("[particles] enabling pointer_settled failed: %s", exc)

    def _wire_navigator(self, tree) -> None:
        self._selectors = _navigator_selectors_for(tree, self.plot)
        for sel in self._selectors:
            if self._on_indices not in sel.index_hooks:
                sel.index_hooks.append(self._on_indices)
            if getattr(sel, "current_indices", None) is not None:
                self._frame = frame_from_indices(sel.current_indices,
                                                 self._nav_to_frame)
        if not self._selectors:
            log.warning("[particles] attached with NO navigator selectors — the "
                        "overlay will not follow the frame")

    def remove(self) -> None:
        """Detach every hook, drop every marker group. Idempotent."""
        self._gen += 1                      # teardown bumps FIRST (README §6)
        for sel in self._selectors:
            if self._on_indices in sel.index_hooks:
                sel.index_hooks.remove(self._on_indices)
        self._selectors = []
        self.set_region_select(False)
        plot2d = getattr(self.plot, "_plot2d", None)
        if plot2d is not None and self._prev_settled_ms is not None:
            try:
                plot2d.configure_pointer_settled(int(self._prev_settled_ms))
            except Exception as exc:
                log.debug("[particles] restoring pointer_settled failed: %s", exc)
        for group in self._groups.values():
            try:
                group.remove()
            except Exception as exc:
                log.debug("[particles] removing marker group failed: %s", exc)
        self._groups = {}
        # anyplotlib has no remove_event_handler; dropping our references lets
        # the wrappers (weakly registered) be collected, and an empty _groups
        # makes every handler a no-op in the meantime.
        self._handlers = []

    # ── navigator ────────────────────────────────────────────────────────────

    def _on_indices(self, indices) -> None:
        """Navigation moved. Runs on the ``_NavDispatcher`` thread."""
        frame = frame_from_indices(indices, self._nav_to_frame)
        if frame == self._frame:
            return
        self._frame = frame
        self._request_redraw()

    def _request_redraw(self) -> None:
        """Compute the payload here, push it on the asyncio main thread.

        The payload is pure numpy over one frame (plus the trail window), so it
        is cheap enough to build on whichever thread asked. The PUSH is a figure
        update and must be marshalled — CLAUDE.md's threading contract.
        """
        if self._hidden or not self._groups:
            return
        gen = self._gen = self._gen + 1
        payload = self._payload(self._frame)
        session = getattr(self.plot, "session", None)
        dispatch = getattr(session, "_dispatch_to_main", None)
        if dispatch is None:
            self._apply(payload, gen)
            return
        dispatch(lambda: self._apply(payload, gen))

    def _apply(self, payload: dict, gen: int) -> None:
        if gen != self._gen or not self._groups:
            return                          # superseded, or torn down
        plot2d = getattr(self.plot, "_plot2d", None)
        if plot2d is None:
            return
        _push_groups(plot2d, {self._groups[key]: kwargs
                              for key, kwargs in payload.items()
                              if key in self._groups})

    def _redraw(self) -> None:
        """Rebuild and push immediately (an edit / a selection, not a nav move)."""
        if self._hidden or not self._groups:
            return
        gen = self._gen = self._gen + 1
        self._apply(self._payload(self._frame), gen)

    # ── the payload ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_payload() -> dict:
        """Every group key mapped to the contents that draw nothing."""
        out: dict[str, dict] = {}
        for i in range(len(TRACK_COLORS) + 1):
            out[f"fill{i}"] = {"vertices_list": []}
            out[f"head{i}"] = {"offsets": np.zeros((0, 2), np.float32)}
            for step in range(TRAIL_FADE_STEPS):
                out[f"trail{i}_{step}"] = {"segments": np.zeros((0, 2, 2), np.float32)}
        out["selected"] = {"vertices_list": []}
        out["labels"] = {"offsets": np.zeros((0, 2), np.float32), "texts": []}
        return out

    def _payload(self, t: int) -> dict:
        """Every marker group's contents for frame *t*. Pure — no plot access.

        Returned as ``{group_key: set_kwargs}`` so the whole overlay is testable
        without a live figure, which is the only way to assert the head-dot rule
        and the pixel conversion in a headless suite.
        """
        n_colors = len(TRACK_COLORS) + 1
        n_steps = TRAIL_FADE_STEPS
        out: dict[str, dict] = {}
        fills: list[list] = [[] for _ in range(n_colors)]
        heads: list[list] = [[] for _ in range(n_colors)]
        trails: list[list[list]] = [[[] for _ in range(n_steps)]
                                    for _ in range(n_colors)]

        if 0 <= int(t) < self.particles.n_frames and self.particles.has_masks:
            for gi in self.particles.indices_at(int(t)):
                poly = contour_xy(self.particles, int(gi))
                if len(poly) < 3:
                    continue
                fills[self._bucket(gi)].append(poly)

        if self.show_trails:
            self._fill_trails(int(t), trails, heads)

        for i in range(n_colors):
            out[f"fill{i}"] = {"vertices_list": fills[i]}
            out[f"head{i}"] = {"offsets": (np.asarray(heads[i], np.float32)
                                           if heads[i]
                                           else np.zeros((0, 2), np.float32))}
            for step in range(n_steps):
                segs = trails[i][step]
                out[f"trail{i}_{step}"] = {
                    "segments": (np.asarray(segs, np.float32) if segs
                                 else np.zeros((0, 2, 2), np.float32))}

        out["selected"] = {"vertices_list": self._selected_polys(int(t))}
        offsets, texts = self._labels(int(t))
        out["labels"] = {"offsets": offsets, "texts": texts}
        return out

    def _bucket(self, gi: int) -> int:
        """Colour-group index for a global particle row (last = untracked)."""
        tid = int(self.particles.flat_buffer[int(gi), COL["track_id"]])
        if tid < 0:
            return len(TRACK_COLORS)
        return tid % len(TRACK_COLORS)

    def _fill_trails(self, t: int, trails, heads) -> None:
        """Trail segments + head dots for the window ending at frame *t*.

        **A dead track draws no head dot.** The dot means "the particle is HERE
        NOW", so it is drawn only for a track with a detection at exactly *t* —
        which covers both a track that died before *t* and one inside its
        ``memory`` gap, without either needing to be special-cased or the
        ``LinkResult`` needing to be around. Found by looking at a render: a
        track that died at frame 16 was still painting a head dot at 18 because
        its trajectory still intersected the trailing window, and it read as a
        real particle the segmenter had stopped filling (plan C3).

        Walks the window's FRAMES rather than indexing tracks, for the reason
        ``track._extract_events`` gives: a ``{track: row}`` map over a whole
        movie is 1.5M dict entries at the plan's target scale, to answer a
        question that only ever spans the last few frames.
        """
        scale = float(self.particles.scale) or 1.0
        t0 = max(0, t - self.trail_frames + 1)
        per_track: dict[int, list[tuple[int, float, float]]] = {}
        for f in range(t0, min(t, self.particles.n_frames - 1) + 1):
            rows = self.particles.at(f)
            if len(rows) == 0:
                continue
            xs = rows[:, COL["x"]] / scale
            ys = rows[:, COL["y"]] / scale
            tids = rows[:, COL["track_id"]].astype(np.int64)
            for k in range(len(rows)):
                tid = int(tids[k])
                if tid < 0:
                    continue            # an untracked row has no trajectory
                per_track.setdefault(tid, []).append((f, float(xs[k]), float(ys[k])))

        n_steps = len(trail_alphas(TRAIL_FADE_STEPS))
        for tid, pts in per_track.items():
            bucket = tid % len(TRACK_COLORS)
            for (_f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
                # Age is measured at the segment's NEWER end, so the piece next
                # to the head is the brightest one.
                step = min(n_steps - 1,
                           (t - f1) * n_steps // max(1, self.trail_frames))
                trails[bucket][step].append([[x0, y0], [x1, y1]])
            if pts and pts[-1][0] == t:
                heads[bucket].append([pts[-1][1], pts[-1][2]])

    def _selected_polys(self, t: int) -> list:
        """Outlines of the selected particles that are visible in frame *t*."""
        if not self.selected or not self.particles.has_masks:
            return []
        out = []
        for gi in self.selected:
            if not self._in_frame(gi, t):
                continue
            poly = contour_xy(self.particles, gi)
            if len(poly) >= 3:
                out.append(poly)
        return out

    def _labels(self, t: int) -> tuple[np.ndarray, list[str]]:
        """Text labels — SELECTION and HOVER only.

        Always-on ids were rejected in the plan for a measured reason: legible on
        a nine-particle mock-up, a wall of numbers at 500 particles per frame.
        """
        shown = list(self.selected)
        if self.hovered is not None and self.hovered not in shown:
            shown.append(self.hovered)
        shown = [gi for gi in shown if self._in_frame(gi, t)]
        if not shown:
            return np.zeros((0, 2), np.float32), []
        rows = self.particles.flat_buffer[np.asarray(shown, np.int64)]
        scale = float(self.particles.scale) or 1.0
        offsets = centroids_px(rows, scale)
        # Anchor OUTSIDE the body, up and to the right: anyplotlib draws text
        # left-aligned / top-baselined from the offset, so anchoring on the
        # centroid lays the readout across the particle it describes (seen in the
        # app). One body radius clears it.
        radii = rows[:, COL["equiv_diameter"]] / (2.0 * scale)
        radii = np.where(np.isfinite(radii), radii, 0.0) + 2.0
        offsets[:, 0] += radii
        offsets[:, 1] -= radii
        texts = [self.describe(gi) for gi in shown]
        return offsets, texts

    def _in_frame(self, gi: int, t: int) -> bool:
        if not 0 <= int(gi) < self.particles.n_particles:
            return False
        lo, hi = self.particles.t_offsets[int(t)], self.particles.t_offsets[int(t) + 1]
        return bool(lo <= int(gi) < hi)

    def describe(self, gi: int) -> str:
        """One-line id + property readout for particle *gi*."""
        row = self.particles.flat_buffer[int(gi)]
        tid = int(row[COL["track_id"]])
        head = f"#{int(gi)}" if tid < 0 else f"track {tid}"
        units = self.particles.units
        parts = [head]
        for name in READOUT_COLUMNS:
            value = float(row[COL[name]])
            if not np.isfinite(value):
                continue
            suffix = {"area": f" {units}²", "equiv_diameter": f" {units}"}.get(name, "")
            parts.append(f"{name.replace('_', ' ')} {value:.3g}{suffix}")
        return "  ".join(parts)

    # ── display state ────────────────────────────────────────────────────────

    def set_visible(self, visible: bool) -> None:
        """Show or hide every group.

        A hidden overlay still TRACKS the navigator (``_frame`` keeps moving), so
        re-showing draws the frame the user is on rather than the one they were
        on when it went away — the same contract ``vector_overlay`` keeps.
        """
        self._hidden = not bool(visible)
        if not self._hidden:
            self._redraw()
            return
        gen = self._gen = self._gen + 1
        self._apply(self._empty_payload(), gen)

    def set_trails(self, enabled: bool, frames: int | None = None) -> None:
        """Toggle trails and set the trailing window length."""
        self.show_trails = bool(enabled)
        if frames is not None:
            self.trail_frames = max(1, int(frames))
        self._redraw()

    def set_frame(self, t: int) -> None:
        """Programmatically move to particle frame *t* (playback, a table jump)."""
        self._frame = int(t)
        self._redraw()

    # ── selection ────────────────────────────────────────────────────────────

    def select(self, indices: Iterable[int] | int | None) -> list[int]:
        """Select particles by GLOBAL index — the external hook a table row uses.

        Returns the resulting selection. Indices outside the store are dropped
        rather than raising: the caret's row list can lag an edit by a frame, and
        a stale row must not take the backend down with it.
        """
        if indices is None:
            wanted: list[int] = []
        elif isinstance(indices, (int, np.integer)):
            wanted = [int(indices)]
        else:
            wanted = [int(i) for i in indices]
        n = self.particles.n_particles
        self.selected = [i for i in wanted if 0 <= i < n]
        self._after_select()
        return list(self.selected)

    def select_track(self, track_id: int, *, frame: int | None = None) -> list[int]:
        """Select a track's detection in one frame (default: the current one)."""
        t = self._frame if frame is None else int(frame)
        if not 0 <= t < self.particles.n_frames:
            return self.select([])
        rows = self.particles.at(t)
        hit = np.nonzero(rows[:, COL["track_id"]].astype(np.int64) == int(track_id))[0]
        base = int(self.particles.t_offsets[t])
        return self.select([base + int(k) for k in hit])

    def select_region(self, x0, y0, x1, y1, *, frame: int | None = None) -> list[int]:
        """Rubber-band selection: every particle in *frame* whose centroid falls
        inside the image-pixel box ``(x0, y0)-(x1, y1)``."""
        t = self._frame if frame is None else int(frame)
        if not 0 <= t < self.particles.n_frames:
            return self.select([])
        rows = self.particles.at(t)
        if len(rows) == 0:
            return self.select([])
        pts = centroids_px(rows, self.particles.scale)
        lo_x, hi_x = sorted((float(x0), float(x1)))
        lo_y, hi_y = sorted((float(y0), float(y1)))
        inside = ((pts[:, 0] >= lo_x) & (pts[:, 0] <= hi_x)
                  & (pts[:, 1] >= lo_y) & (pts[:, 1] <= hi_y))
        base = int(self.particles.t_offsets[t])
        return self.select([base + int(k) for k in np.nonzero(inside)[0]])

    def pick(self, px: float, py: float, *, frame: int | None = None) -> int | None:
        """Nearest-centroid hit test at image-pixel ``(px, py)``.

        The hit radius is the particle's OWN ``equiv_diameter`` (floored at
        4 px), not a constant: a fixed radius that feels right for a 50 px body
        makes a 5 px one unclickable, and one sized for the small body picks a
        neighbour when the field is dense. Same reasoning as the linker's
        adaptive merge radius.
        """
        t = self._frame if frame is None else int(frame)
        if not 0 <= t < self.particles.n_frames:
            return None
        rows = self.particles.at(t)
        if len(rows) == 0:
            return None
        scale = float(self.particles.scale) or 1.0
        pts = centroids_px(rows, scale)
        d2 = (pts[:, 0] - float(px)) ** 2 + (pts[:, 1] - float(py)) ** 2
        k = int(np.argmin(d2))
        radius = rows[k, COL["equiv_diameter"]] / scale
        radius = max(4.0, float(radius) if np.isfinite(radius) else 0.0)
        if d2[k] > radius ** 2:
            return None
        return int(self.particles.t_offsets[t]) + k

    def clear_selection(self) -> list[int]:
        return self.select([])

    def _after_select(self) -> None:
        self._redraw()
        if self.on_select is not None:
            try:
                self.on_select()
            except Exception as exc:
                log.debug("[particles] on_select failed: %s", exc)

    def _on_click(self, event=None) -> None:
        if event is None or not self._groups:
            return
        try:
            px, py = self._event_px(event)
        except Exception as exc:
            log.debug("[particles] click had no usable coordinates: %s", exc)
            return
        hit = self.pick(px, py)
        self.select([] if hit is None else [hit])

    def _on_settled(self, event=None) -> None:
        if event is None or not self._groups:
            return
        try:
            px, py = self._event_px(event)
        except Exception:
            return
        hit = self.pick(px, py)
        if hit == self.hovered:
            return
        self.hovered = hit
        self._redraw()

    def _event_px(self, event) -> tuple[float, float]:
        """A pointer event's position in IMAGE PIXELS.

        anyplotlib's Python ``Event`` carries only ``xdata``/``ydata`` (the JS
        payload's ``img_x``/``img_y`` have no field on the dataclass and are
        dropped), and those are the panel's PHYSICAL data coordinates. Markers
        are in image pixels, so the click is converted back through the plot's
        own axes — not through ``particles.scale``, which is the store's
        calibration and need not be the displayed signal's.
        """
        xs, xo, ys, yo = _axis_scale_offset(self.plot)
        return (float(event.xdata) - xo) / xs, (float(event.ydata) - yo) / ys

    # ── rubber band ──────────────────────────────────────────────────────────

    def set_region_select(self, enabled: bool) -> None:
        """Show/hide the rubber-band rectangle used for bulk selection.

        A widget rather than a drag on the canvas: the panel's own drag is pan,
        and anyplotlib's rectangle widget already owns the handles, the clamping
        and the pointer events.
        """
        plot2d = getattr(self.plot, "_plot2d", None)
        if not enabled:
            if self._region_widget is not None and plot2d is not None:
                try:
                    plot2d.remove_widget(self._region_widget)
                except Exception as exc:
                    log.debug("[particles] removing region widget failed: %s", exc)
            self._region_widget = None
            return
        if self._region_widget is not None or plot2d is None:
            return
        h, w = self.particles.frame_shape
        try:
            widget = plot2d.add_rectangle_widget(
                x=w * 0.25, y=h * 0.25, w=w * 0.5, h=h * 0.5, color=SELECTED_COLOR)
        except Exception as exc:
            log.debug("[particles] adding region widget failed: %s", exc)
            return
        from spyde.drawing.selectors.base_selector import event_handler_fn
        handler = event_handler_fn(self._on_region)
        self._handlers.append(handler)
        try:
            widget.add_event_handler(handler, "pointer_up")
        except Exception as exc:
            log.debug("[particles] wiring region widget failed: %s", exc)
        self._region_widget = widget
        self._on_region()

    def _on_region(self, _event=None) -> None:
        widget = self._region_widget
        if widget is None:
            return
        try:
            x, y = float(widget.x), float(widget.y)
            w, h = float(widget.w), float(widget.h)
        except Exception as exc:
            log.debug("[particles] reading region widget failed: %s", exc)
            return
        self.select_region(x, y, x + w, y + h)

    # ── editing ──────────────────────────────────────────────────────────────

    def delete(self, indices: Iterable[int] | None = None) -> int:
        """Delete particles (default: the selection). Returns how many went."""
        idx = sorted({int(i) for i in (self.selected if indices is None else indices)})
        if not idx:
            return 0
        removed = delete_particles(self.particles, idx)
        self._record("delete", indices=idx, frame=self._frame)
        self.selected = []
        self._after_edit()
        return removed

    def merge(self, indices: Iterable[int] | None = None) -> int:
        """Merge particles into one, re-measure, return its new global index."""
        idx = sorted({int(i) for i in (self.selected if indices is None else indices)})
        if len(idx) < 2:
            raise ValueError("merge needs at least two particles")
        new_index = merge_particles(self.particles, idx,
                                    frame_image=self._frame_image_for(idx[0]))
        self._record("merge", indices=idx, frame=self._frame, result=[new_index])
        self.selected = [new_index]
        self._after_edit()
        return new_index

    def split(self, index: int | None = None, line=None) -> tuple[int, int]:
        """Split one particle along *line* and re-measure both halves.

        *line* is ``((x0, y0), (x1, y1))`` in IMAGE PIXELS — the space the drawn
        line widget reports in.
        """
        if index is None:
            if len(self.selected) != 1:
                raise ValueError("split needs exactly one selected particle")
            index = self.selected[0]
        if line is None:
            raise ValueError("split needs a cut line")
        pair = split_particle(self.particles, int(index), line,
                              frame_image=self._frame_image_for(int(index)))
        self._record("split", indices=[int(index)], frame=self._frame,
                     line=[[float(v) for v in pt] for pt in line],
                     result=list(pair))
        self.selected = list(pair)
        self._after_edit()
        return pair

    def _frame_image_for(self, gi: int):
        if self.frame_provider is None:
            return None
        t = int(self.particles.flat_buffer[int(gi), COL["t"]])
        try:
            return np.asarray(self.frame_provider(t))
        except Exception as exc:
            log.debug("[particles] frame_provider(%d) failed: %s", t, exc)
            return None

    def _record(self, kind: str, **fields) -> dict:
        return record_edit(self.tree, self.particles, kind, **fields)

    def _after_edit(self) -> None:
        self.revision += 1
        self.hovered = None
        self._redraw()
        if self.on_edit is not None:
            try:
                self.on_edit()
            except Exception as exc:
                log.debug("[particles] on_edit failed: %s", exc)


def attach_particle_overlay(plot, particles, tree, **kwargs) -> ParticleOverlay:
    """Attach a :class:`ParticleOverlay` to *plot*, wired to *tree*'s navigator.

    Stored on the tree as ``tree._particle_overlay`` via
    ``lifecycle.replace_tree_attr``, so re-running never stacks two overlays and
    ``BaseSignalTree.close()`` reaps it.
    """
    from spyde.actions.lifecycle import replace_tree_attr
    return replace_tree_attr(
        tree, "_particle_overlay",
        lambda: ParticleOverlay(plot, particles, **kwargs).attach(tree))


# ── edits on the store ───────────────────────────────────────────────────────
#
# Separate from the overlay on purpose: an edit is a transformation of the CSR
# table and nothing about it needs a figure, so it is testable (and scriptable)
# on its own. Every one of them mutates the store IN PLACE — the lazy label movie
# closes over that exact object (``particle_tree.open_particle_tree``), so
# swapping in a new SpyDEParticles would leave the open window rendering the old
# contours forever. That is the same trap ``particles_action._adopt`` exists for.

def _row_frames(particles) -> np.ndarray:
    """``(n,)`` int64 frame index per row, from the CSR pointers."""
    return np.repeat(np.arange(particles.n_frames, dtype=np.int64),
                     np.diff(particles.t_offsets))


def _splice(particles, *, drop: Sequence[int] = (),
            add: Sequence[tuple[int, np.ndarray, np.ndarray | None]] = ()) -> np.ndarray:
    """Rebuild the store in place with rows removed and/or added.

    *add* is a sequence of ``(frame, row, contour)``. Added rows land at the END
    of their frame's block (a stable sort by frame over the kept rows followed by
    the new ones), so existing global indices shift by at most the deletions
    ahead of them and never by an insertion into the middle of a frame.

    Returns the ``(len(add),)`` global indices the added rows ended up at.

    Vectorised rather than looped: the plan's target is 1.5M rows, and an edit
    that walked them in Python would take seconds for a single click. The only
    loop is over the handful of rows being added.
    """
    n = particles.n_particles
    keep = np.ones(n, bool)
    if len(drop):
        keep[np.asarray(list(drop), np.int64)] = False

    frames = _row_frames(particles)
    kept_frames = frames[keep]
    kept_rows = particles.flat_buffer[keep]

    add = list(add)
    add_frames = np.asarray([int(f) for f, _r, _c in add], np.int64)
    add_rows = (np.asarray([r for _f, r, _c in add], np.float32).reshape(-1, N_COLUMNS)
                if add else np.zeros((0, N_COLUMNS), np.float32))

    all_frames = np.concatenate([kept_frames, add_frames])
    all_rows = np.concatenate([kept_rows, add_rows], axis=0)
    order = np.argsort(all_frames, kind="stable")

    particles.flat_buffer = np.ascontiguousarray(all_rows[order], np.float32)
    counts = np.bincount(all_frames, minlength=particles.n_frames)
    particles.t_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    if particles.contours is not None:
        lens = np.diff(particles.contour_offsets)
        starts = particles.contour_offsets[:-1]
        add_polys = [np.zeros((0, 2), np.int16) if c is None
                     else np.asarray(c, np.int16).reshape(-1, 2) for _f, _r, c in add]
        add_lens = np.asarray([len(p) for p in add_polys], np.int64)
        all_lens = np.concatenate([lens[keep], add_lens])
        # Point sources: kept polygons index the OLD contour array; added ones
        # index a small appended block. One concatenation, then one gather.
        pool = np.concatenate([particles.contours] + add_polys, axis=0) \
            if add_polys else particles.contours
        add_starts = len(particles.contours) + np.concatenate(
            [[0], np.cumsum(add_lens)])[:-1].astype(np.int64)
        all_starts = np.concatenate([starts[keep], add_starts])

        new_lens = all_lens[order]
        new_off = np.concatenate([[0], np.cumsum(new_lens)]).astype(np.int64)
        within = np.arange(int(new_off[-1])) - np.repeat(new_off[:-1], new_lens)
        src = np.repeat(all_starts[order], new_lens) + within
        particles.contours = np.ascontiguousarray(pool[src], np.int16)
        particles.contour_offsets = new_off

    # Where each added row landed: its position in `order`.
    rank = np.empty(order.size, np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank[len(kept_rows):]


def delete_particles(particles, indices: Sequence[int]) -> int:
    """Drop rows from the store, in place. Returns the number removed."""
    idx = sorted({int(i) for i in indices if 0 <= int(i) < particles.n_particles})
    if not idx:
        return 0
    _splice(particles, drop=idx)
    return len(idx)


def _full_mask(particles, index: int) -> np.ndarray:
    """One particle's boolean mask, at full frame size."""
    mask, (y0, x0, y1, x1) = particles.mask_at(int(index))
    h, w = particles.frame_shape
    out = np.zeros((h, w), bool)
    out[y0:y1, x0:x1] |= mask
    return out


def _remeasure(labels: np.ndarray, frame_image, t: int, scale: float):
    """``measure_frame`` on a purpose-built label image. Returns (rows, contours)."""
    from spyde.particles import measure_frame
    return measure_frame(labels, frame_image, t=int(t), scale=float(scale))


def merge_particles(particles, indices: Sequence[int], *, frame_image=None) -> int:
    """Union the masks of two or more particles, re-measure, return the new index.

    Every input must be in the SAME frame — merging across frames is not an
    editing operation, it is a linking one, and doing it here would silently
    produce a row whose ``t`` contradicts its CSR block.

    The merged row inherits the ``track_id`` of the LARGEST input, matching the
    linker's merge semantics (a large body absorbs a smaller one, ``track.py``);
    the absorbed track simply stops, which is what a re-link would also conclude.
    """
    idx = sorted({int(i) for i in indices})
    if len(idx) < 2:
        raise ValueError("merge needs at least two particles")
    if not particles.has_masks:
        raise ValueError("cannot merge without outlines "
                         "(segmentation ran with store_masks=False)")
    frames = {int(particles.flat_buffer[i, COL["t"]]) for i in idx}
    if len(frames) != 1:
        raise ValueError(f"merge needs one frame; got {sorted(frames)}")
    t = frames.pop()

    from skimage.measure import label as connected_components

    union = np.zeros(particles.frame_shape, bool)
    for i in idx:
        union |= _full_mask(particles, i)
    # Connected-component labelling, NOT ``union.astype(int32)``: casting the
    # boolean union gives every pixel label 1, and ``regionprops`` measures a
    # label as ONE region whether or not it is connected — so two discs on
    # opposite sides of the frame would merge "successfully" into a row whose
    # centroid sits in the empty space between them. 8-connectivity, because two
    # bodies meeting at a corner are touching.
    components = connected_components(union, connectivity=2)
    n_components = int(components.max())
    if n_components == 0:
        raise ValueError("the merged region measured as empty")
    if n_components > 1:
        raise ValueError(
            f"the selected particles do not touch — merging them would make "
            f"{n_components} disconnected regions, not one")
    rows, contours = _remeasure(components.astype(np.int32), frame_image, t,
                                particles.scale)
    if len(rows) != 1:
        raise ValueError(f"the merged region measured as {len(rows)} rows, not 1")

    areas = particles.flat_buffer[np.asarray(idx), COL["area"]]
    survivor = idx[int(np.argmax(areas))]
    rows[0, COL["track_id"]] = particles.flat_buffer[survivor, COL["track_id"]]
    rows[0, COL["label"]] = particles.flat_buffer[np.asarray(idx), COL["label"]].min()

    added = _splice(particles, drop=idx, add=[(t, rows[0], contours[0])])
    return int(added[0])


def split_particle(particles, index: int, line, *, frame_image=None) -> tuple[int, int]:
    """Cut one particle along *line* and re-measure both halves.

    *line* is ``((x0, y0), (x1, y1))`` in IMAGE PIXELS. The cut is the infinite
    line through those two points; pixels of the particle's mask fall on one side
    or the other by the sign of the 2-D cross product. An infinite line rather
    than a segment because a user drawing a cut across a blob naturally starts
    and ends outside it, and a segment-only rule would leave the ends uncut.

    The larger half keeps the parent's ``track_id``; the smaller is left
    untracked (-1), because which fragment continues the track is exactly the
    question a re-link answers and guessing it here would fabricate an identity.
    """
    if not particles.has_masks:
        raise ValueError("cannot split without outlines "
                         "(segmentation ran with store_masks=False)")
    gi = int(index)
    (x0, y0), (x1, y1) = ((float(line[0][0]), float(line[0][1])),
                          (float(line[1][0]), float(line[1][1])))
    if (x1 - x0) == 0.0 and (y1 - y0) == 0.0:
        raise ValueError("the cut line has zero length")

    mask = _full_mask(particles, gi)
    h, w = particles.frame_shape
    yy, xx = np.mgrid[0:h, 0:w]
    side = (xx - x0) * (y1 - y0) - (yy - y0) * (x1 - x0)
    labels = np.zeros((h, w), np.int32)
    labels[mask & (side < 0)] = 1
    labels[mask & (side >= 0)] = 2
    if not labels.any() or labels.max() < 2 or not (labels == 1).any():
        raise ValueError("the cut line does not divide this particle")

    t = int(particles.flat_buffer[gi, COL["t"]])
    rows, contours = _remeasure(labels, frame_image, t, particles.scale)
    if len(rows) != 2:
        raise ValueError(
            f"the cut produced {len(rows)} regions, not 2 — the line probably "
            "clipped a corner or the halves are disconnected")

    parent_track = particles.flat_buffer[gi, COL["track_id"]]
    keeper = int(np.argmax(rows[:, COL["area"]]))
    rows[keeper, COL["track_id"]] = parent_track
    rows[1 - keeper, COL["track_id"]] = -1.0

    added = _splice(particles, drop=[gi],
                    add=[(t, rows[k], contours[k]) for k in range(2)])
    return int(added[0]), int(added[1])


def record_edit(tree, particles, kind: str, **fields) -> dict:
    """Record a manual correction on the tree AND in the store's provenance.

    Two places, deliberately. ``tree.particle_edits`` is the live log a re-run
    reads so it does not silently discard the user's corrections; the copy in
    ``particles.provenance["edits"]`` travels with ``SpyDEParticles.save`` and is
    stamped into the tree's commit provenance, so a corrected result is still
    reproducible from the file alone.
    """
    record = {"kind": str(kind), "at": time.time(), **fields}
    if tree is not None:
        record["revision"] = len(getattr(tree, "particle_edits", None) or ()) + 1
        edits = list(getattr(tree, "particle_edits", None) or ())
        edits.append(record)
        tree.particle_edits = edits
        commit = getattr(tree, "_commit_provenance", None)
        if isinstance(commit, dict):
            commit["edits"] = list(edits)
    if particles is not None:
        provenance = dict(particles.provenance or {})
        provenance["edits"] = list(provenance.get("edits") or ()) + [record]
        particles.provenance = provenance
    return record


def pending_edits(tree) -> list[dict]:
    """The manual corrections recorded on *tree*, oldest first.

    **This is the seam a re-segmentation must consult.** A re-run rebuilds the
    store from the raw frames, which discards every edit unless it reads this
    first — and discarding them silently is the failure the plan calls out (B9).
    Nothing in ``particles_action`` reads it yet; that is the segmentation
    workstream's half of the contract, and it is a list of plain JSON-safe dicts
    precisely so it can cross that boundary (and the IPC) unchanged.
    """
    return list(getattr(tree, "particle_edits", None) or ())


# ── the navigator lanes (plan C2) ────────────────────────────────────────────

#: Lane names, in stacking order.
LANE_COUNT = "count"
LANE_SIZE = "mean size"
LANE_EVENTS = "events"

#: Row each event kind occupies inside the event lane, top to bottom.
EVENT_ROWS: dict[str, int] = {"birth": 3, "death": 2, "merge": 1, "split": 0}


def step_trace(values, x=None) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate samples so a polyline draws a ``steps-post`` staircase.

    **The count lane is integer data and must be drawn as a step.** A straight
    interpolation between frames puts the visual transition half a frame early,
    so a nucleation at frame 8 reads as 7 (plan C3, found by looking at a
    render). anyplotlib's ``Axes.plot`` has no ``drawstyle``, so the staircase is
    built into the DATA: each sample is held until the next x, then jumps.
    Continuous quantities (mean size) stay plain lines.

    Returns
    -------
    (x, y)
        Both ``(2 * n,)``. The final sample is held one step wide so the last
        frame is as visible as every other one.
    """
    y = np.asarray(values, np.float32).ravel()
    n = y.size
    if n == 0:
        return np.zeros(0, np.float64), np.zeros(0, np.float32)
    xs = np.arange(n, dtype=np.float64) if x is None else np.asarray(x, np.float64).ravel()
    width = float(xs[1] - xs[0]) if n > 1 else 1.0
    edges = np.concatenate([xs, [xs[-1] + width]])
    out_x = np.repeat(edges, 2)[1:-1]
    out_y = np.repeat(y, 2)
    return out_x, out_y


def _event_points(events, kind: str, scale: float, offset: float) -> np.ndarray:
    """``(n, 2)`` ``(x, row)`` marker offsets for one event kind."""
    frames = [int(e.frame) for e in events if getattr(e, "kind", None) == kind]
    if not frames:
        return np.zeros((0, 2), np.float32)
    row = float(EVENT_ROWS.get(kind, 0))
    return np.column_stack([np.asarray(frames, np.float64) * scale + offset,
                            np.full(len(frames), row)]).astype(np.float32)


def publish_navigator_lanes(session, tree, *, plot=None) -> bool:
    """Publish ``tree.nav_traces`` as three stacked 1-D navigator lanes.

    ``count(t)``, ``mean size(t)`` and a dedicated event lane with a colour per
    kind, stacked as rows on one shared time axis with a single logical cursor
    wired to the tree's REAL 1-D navigation selector — so dragging any lane moves
    the movie and playback moves every lane's line.

    Reuses ``navigator_views``' stacked-cursor machinery wholesale
    (:class:`~spyde.actions.navigator_views._StackedNavCursor`, its registry and
    its teardown), because the hard part of a stacked navigator is the two-way
    cursor sync and its re-entrancy guard, and that is already written and
    tested. What is NOT reusable is ``_stack_navigators``' figure builder: it
    draws every lane as a plain line, and the count lane must be a step and the
    event lane is markers rather than a trace at all.

    Each lane also lands in ``tree.navigator_signals`` so the chip strip lists it
    and the existing ``select_navigator`` path can re-stack any subset.

    Returns
    -------
    bool
        True if the figure was emitted.
    """
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    import hyperspy.api as hs

    from spyde.actions.figure_registry import keep_alive
    from spyde.actions.navigator_views import (
        STACKED_LABEL, _real_nav_selector, _selector_axis, _stacked_cursors,
        _StackedNavCursor, _teardown_stacked,
    )
    from spyde.backend.ipc import emit
    from spyde.drawing.plots.plot import finalize_figure_html

    traces = dict(getattr(tree, "nav_traces", None) or {})
    if LANE_COUNT not in traces:
        log.debug("[particles] no nav_traces on the tree; nothing to publish")
        return False
    if plot is None:
        plot = _first_nav_plot(tree)
    window_id = getattr(plot, "window_id", None)
    if window_id is None:
        log.debug("[particles] no navigator window to publish lanes onto")
        return False

    count = np.asarray(traces[LANE_COUNT], np.float32)
    size = np.asarray(traces.get("size", np.full(count.shape, np.nan)), np.float32)
    events = list(getattr(tree, "particle_events", None) or ())

    # Register the traces as named navigators too, so the chip strip offers them.
    for name, data in ((LANE_COUNT, count), (LANE_SIZE, size)):
        if name in getattr(tree, "navigator_signals", {}):
            continue
        try:
            tree.add_navigator_signal(name, hs.signals.Signal1D(np.nan_to_num(data)))
        except Exception as exc:
            log.debug("[particles] registering navigator %r failed: %s", name, exc)

    selector = _real_nav_selector(session, int(window_id))
    scale, offset = _selector_axis(selector) if selector is not None else (1.0, 0.0)
    current = 0
    if selector is not None and getattr(selector, "current_indices", None) is not None:
        try:
            current = int(np.asarray(selector.current_indices).ravel()[0])
        except Exception:
            current = 0
    cursor_x = current * scale + offset

    _teardown_stacked(session, plot)
    try:
        fig, axes = apl.subplots(3, 1, sharex=True)
        panels = np.array(axes, dtype=object).ravel()
        widgets = []

        step_x, step_y = step_trace(count, np.arange(count.size) * scale + offset)
        lanes = [
            (panels[0], LANE_COUNT, step_y, step_x),
            (panels[1], LANE_SIZE, np.nan_to_num(size),
             np.arange(size.size) * scale + offset),
        ]
        # NB every setter below goes on the PLOT the panel returns, never on the
        # Axes: anyplotlib's `Axes` has no `set_title`/`set_ylim` at all, so the
        # guarded calls silently did nothing and the event lane came out
        # autoscaled to the invisible baseline (verified in the app — the event
        # rows sat off-scale and nothing drew).
        for panel, title, ydata, xdata in lanes:
            line = panel.plot(ydata, axes=[xdata], label=title)
            _set_title(line, title)
            widgets.append(_add_cursor(line, cursor_x))

        # A transparent baseline establishes the x axis; the events themselves
        # are markers, one group (and one row) per kind so the colours mean
        # something.
        base = panels[2].plot(np.zeros(count.size, np.float32),
                              axes=[np.arange(count.size) * scale + offset],
                              label=LANE_EVENTS, alpha=0.0)
        _set_title(base, LANE_EVENTS)
        for kind, color in EVENT_COLORS.items():
            base.add_points(_event_points(events, kind, scale, offset),
                            name=f"event_{kind}", sizes=5.0, color=color,
                            facecolors=color, alpha=1.0, label=kind)
        try:
            # Fixed rows, NOT autoscaled: the lane must read the same whether the
            # movie contains one kind of event or all four, so a birth is always
            # on the top row.
            base.set_ylim(-0.5, max(EVENT_ROWS.values()) + 0.5)
        except Exception as exc:
            log.debug("[particles] setting event-lane ylim failed: %s", exc)
        widgets.append(_add_cursor(base, cursor_x))

        widgets = [w for w in widgets if w is not None]
        if len(widgets) >= 2 and selector is not None:
            _stacked_cursors(session)[int(window_id)] = _StackedNavCursor(
                session, int(window_id), widgets, selector)

        fig_id = _electron.register(fig)
        html = finalize_figure_html(fig, fig_id)
        keep_alive(int(window_id), fig)
        emit({"type": "figure", "fig_id": fig_id, "window_id": window_id,
              "html": html, "title": " / ".join((LANE_COUNT, LANE_SIZE, LANE_EVENTS)),
              "is_navigator": True,
              "view_label": STACKED_LABEL, "view_kind": "stacked"})
        return True
    except Exception as exc:
        log.exception("[particles] publishing navigator lanes failed: %s", exc)
        return False


def maybe_stack_particle_lanes(session, plot, tree, names) -> bool:
    """Lane hook for ``navigator_views.select_navigator``.

    A particle tree's lanes do not render as plain lines: ``count`` is integer
    data and must be a STEP, and the event lane is coloured markers rather than a
    trace at all. ``_stack_navigators`` draws every named navigator the same way,
    so when the user ⇧-clicks the chips on a particle tree the generic builder
    would silently produce the wrong picture — a nucleation at frame 8 reading as
    7 is exactly the failure plan C3 calls out.

    Returns True when it built the lanes (the caller must then return).
    """
    if not getattr(tree, "nav_traces", None) or LANE_COUNT not in tree.nav_traces:
        return False
    wanted = {str(n) for n in (names or ())}
    if not wanted & {LANE_COUNT, LANE_SIZE}:
        return False        # the user picked other navigators; not our business
    return publish_navigator_lanes(session, tree, plot=plot)


def _add_cursor(panel, x: float):
    try:
        return panel.add_vline_widget(x=float(x), color="#ff9100")
    except Exception as exc:
        log.debug("[particles] adding lane cursor failed: %s", exc)
        return None


def _set_title(plot1d, title: str) -> None:
    """Title a lane. Takes the PLOT, not the Axes — see the note in the builder."""
    try:
        plot1d.set_title(title)
    except Exception as exc:
        log.debug("[particles] set_title on lane failed: %s", exc)


def _first_nav_plot(tree):
    manager = getattr(tree, "navigator_plot_manager", None)
    if manager is None:
        return None
    for window in list(manager.plot_windows.keys()):
        plots = manager.plots.get(window) or []
        if plots:
            return plots[0]
    return None


def _first_signal_plot(tree):
    for plot in list(getattr(tree, "signal_plots", None) or ()):
        if getattr(plot, "plot_state", None) is not None:
            return plot
    return None


# ── staged handlers (registry.STAGED_HANDLERS, key "part") ───────────────────

def _resolve(session, plot):
    """``(tree, overlay)`` for a handler. Prefers the clicked plot's own tree,
    falling back to any tree that carries particles — the caret's window may
    resolve to the count-map navigator rather than the label movie, the same way
    ``lifecycle.resolve_vectors`` handles the vectors case."""
    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    if tree is not None and getattr(tree, "particles", None) is not None:
        return tree, getattr(tree, "_particle_overlay", None)
    for candidate in getattr(session, "signal_trees", None) or ():
        if getattr(candidate, "particles", None) is not None:
            return candidate, getattr(candidate, "_particle_overlay", None)
    return tree, getattr(tree, "_particle_overlay", None) if tree is not None else None


def _source_frame_provider(tree):
    """``fn(t) -> frame`` reading ONE frame of the tree's source movie.

    Used to re-measure intensity after an edit. Reads a single frame and computes
    only that slice — never the movie (CLAUDE.md memory-safety rule). Returns
    None when there is no source, in which case edited rows keep NaN intensities
    rather than fabricated ones.
    """
    source = getattr(tree, "source_node", None)
    data = getattr(source, "data", None)
    if data is None:
        return None

    def read(t: int):
        frame = data[int(t)]
        if hasattr(frame, "compute"):
            frame = frame.compute()
        return np.asarray(frame)

    return read


def part_open(session, plot, payload=None) -> None:
    """Caret mounted → attach the overlay to the tree's signal plot."""
    from spyde.actions.lifecycle import wait_for_particles
    from spyde.backend.ipc import emit_error, emit_status

    payload = payload or {}
    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    if tree is None:
        emit_error("Particle Overlay: no active dataset")
        return
    if getattr(tree, "particles", None) is None:
        # The segmentation attach gap (plan Wave 0): seg_run opens its window
        # early and attaches tree.particles only at finalize.
        if wait_for_particles(session, plot,
                              lambda: part_open(session, plot, payload),
                              what="Particle Overlay"):
            return
        emit_error("Particle Overlay needs a segmentation result (no particles).")
        return

    target = _first_signal_plot(tree) or plot
    overlay = attach_particle_overlay(
        target, tree.particles, tree,
        events=list(getattr(tree, "particle_events", None) or ()),
        nav_map=getattr(tree, "nav_map", None),
        frame_provider=_source_frame_provider(tree),
        trail_frames=int(payload.get("trail_frames", DEFAULT_TRAIL_FRAMES)),
        show_trails=bool(payload.get("show_trails", False)),
        on_select=lambda: _emit_selection(tree),
    )
    if overlay is None:
        emit_error("Particle Overlay could not attach to this window")
        return
    emit_status(f"Particle overlay on — {tree.particles.n_particles} particles")
    _emit_selection(tree)


def part_close(session, plot, payload=None) -> None:
    """Caret unmounted → tear the overlay down."""
    from spyde.actions.lifecycle import replace_tree_attr
    tree, _overlay = _resolve(session, plot)
    if tree is not None:
        replace_tree_attr(tree, "_particle_overlay", None)


def part_tune(session, plot, payload=None) -> None:
    """Live parameter change: trails on/off and the trailing window length."""
    payload = payload or {}
    _tree, overlay = _resolve(session, plot)
    if overlay is None:
        return
    overlay.set_trails(bool(payload.get("show_trails", overlay.show_trails)),
                       payload.get("trail_frames"))


def part_select(session, plot, payload=None) -> None:
    """External selection hook — a table row, a track id, or a region.

    Payload accepts ``{"indices": [...]}`` (global particle indices),
    ``{"track_id": n}``, or ``{"region": [x0, y0, x1, y1]}`` in image pixels.
    """
    payload = payload or {}
    tree, overlay = _resolve(session, plot)
    if overlay is None:
        return
    if "region" in payload:
        overlay.select_region(*[float(v) for v in payload["region"]])
    elif "track_id" in payload:
        overlay.select_track(int(payload["track_id"]))
    else:
        overlay.select(payload.get("indices") or [])
    _emit_selection(tree)


def part_region_mode(session, plot, payload=None) -> None:
    """Show/hide the rubber-band selection rectangle."""
    payload = payload or {}
    _tree, overlay = _resolve(session, plot)
    if overlay is not None:
        overlay.set_region_select(bool(payload.get("enabled", True)))


def part_delete(session, plot, payload=None) -> None:
    """Delete the selected particles (or ``payload["indices"]``)."""
    from spyde.backend.ipc import emit_error, emit_status
    payload = payload or {}
    tree, overlay = _resolve(session, plot)
    if overlay is None:
        emit_error("Particle Overlay: nothing to edit")
        return
    try:
        removed = overlay.delete(payload.get("indices"))
    except Exception as exc:
        emit_error(f"Delete particles failed: {exc}")
        return
    emit_status(f"Deleted {removed} particle{'' if removed == 1 else 's'}")
    _emit_selection(tree)


def part_merge(session, plot, payload=None) -> None:
    """Merge the selected particles into one and re-measure."""
    from spyde.backend.ipc import emit_error, emit_status
    payload = payload or {}
    tree, overlay = _resolve(session, plot)
    if overlay is None:
        emit_error("Particle Overlay: nothing to edit")
        return
    try:
        index = overlay.merge(payload.get("indices"))
    except Exception as exc:
        emit_error(f"Merge particles failed: {exc}")
        return
    emit_status(f"Merged into particle {index}")
    _emit_selection(tree)


def part_split(session, plot, payload=None) -> None:
    """Split the selected particle along ``payload["line"]`` (image pixels)."""
    from spyde.backend.ipc import emit_error, emit_status
    payload = payload or {}
    tree, overlay = _resolve(session, plot)
    if overlay is None:
        emit_error("Particle Overlay: nothing to edit")
        return
    try:
        a, b = overlay.split(payload.get("index"), payload.get("line"))
    except Exception as exc:
        emit_error(f"Split particle failed: {exc}")
        return
    emit_status(f"Split into particles {a} and {b}")
    _emit_selection(tree)


def part_lanes(session, plot, payload=None) -> None:
    """Build (or rebuild) the three stacked navigator lanes."""
    from spyde.backend.ipc import emit_error
    tree, _overlay = _resolve(session, plot)
    if tree is None or not getattr(tree, "nav_traces", None):
        emit_error("Particle lanes: this dataset has no navigator traces")
        return
    if not publish_navigator_lanes(session, tree):
        emit_error("Particle lanes: could not build the stacked navigator")


def _json_float(value) -> float | None:
    """A property value as JSON, with NaN mapped to ``null``.

    ``ipc.emit`` calls ``json.dumps`` with the default ``allow_nan=True``, which
    writes the bare token ``NaN`` — not valid JSON, so ``JSON.parse`` on the
    Electron side throws and the whole message is dropped. And NaN is the NORMAL
    value here: ``measure_frame`` leaves every intensity column NaN when it runs
    without an intensity image, which is exactly what an edit made with no
    ``frame_provider`` produces.
    """
    v = float(value)
    return v if np.isfinite(v) else None


def _emit_selection(tree) -> None:
    """Tell the renderer which particles are selected, with their properties.

    One message rather than a per-row query: the table dock highlights the
    selection and shows the readout, and both come from the same rows.
    """
    from spyde.backend.ipc import emit
    overlay = getattr(tree, "_particle_overlay", None) if tree is not None else None
    if overlay is None:
        return
    rows = []
    for gi in overlay.selected:
        row = overlay.particles.flat_buffer[int(gi)]
        record = {"index": int(gi), "frame": int(row[COL["t"]]),
                  "track_id": int(row[COL["track_id"]]),
                  "color": track_color(row[COL["track_id"]])}
        record.update({name: _json_float(row[COL[name]]) for name in MEASURED_COLUMNS})
        rows.append(record)
    emit({"type": "particle_selection",
          "window_id": getattr(overlay.plot, "window_id", None),
          "frame": int(overlay._frame), "revision": int(overlay.revision),
          "indices": [int(i) for i in overlay.selected], "particles": rows})


# ── toolbar entry ────────────────────────────────────────────────────────────

class _OverlayHandle:
    """What the toolbar tracks so DESELECTING the action tears the overlay down.

    ``Session._track_action_artifacts`` records any object a toolbar action
    returns that exposes ``active_children``, lights the button, and calls
    ``.close()`` on it when the user clicks the lit button again.

    That is the ONLY un-toggle path the renderer offers: ``FloatingToolbar``'s
    click handler sends ``set_action_active(active=false)`` for an action it
    believes is live — never a second ``toolbar_action``. So an action that
    toggles by inspecting its own state can be turned ON but never OFF. Verified
    in the app: the second click re-ran ``part_open`` and the overlay stayed up.
    """

    active_children: tuple = ()

    def __init__(self, session, plot):
        self.session = session
        self.plot = plot

    def close(self) -> None:
        part_close(self.session, self.plot, {})


def particle_overlay(ctx, action_name: str = "Particle Overlay", **params):
    """Toolbar toggle: attach the overlay and hand the toolbar its teardown handle.

    Self-contained rather than a no-op wizard parent, so the overlay works from
    the toolbar alone; the ``part_*`` staged handlers are the same operations for
    a caret to drive once one exists.
    """
    from spyde.backend import ipc

    plot = getattr(ctx, "plot", None)
    session = getattr(plot, "session", None)
    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    if tree is None:
        ipc.emit_error("Particle Overlay: no active dataset")
        return None
    if getattr(tree, "_particle_overlay", None) is not None:
        # Already up (a stale artifact, or a caret opened it): close so the click
        # is still a toggle even when the artifact path is not in play.
        part_close(session, plot, {})
        window_id = getattr(plot, "window_id", None)
        if window_id is not None:
            ipc.emit({"type": "action_active", "window_id": window_id,
                      "name": action_name, "active": False})
        return None
    part_open(session, plot, params)
    if getattr(tree, "_particle_overlay", None) is None:
        return None                     # open failed; it emitted its own error
    return _OverlayHandle(session, plot)


def particle_lanes(ctx, action_name: str = "Particle Lanes", **params):
    """Toolbar entry: (re)build the three stacked navigator lanes."""
    plot = getattr(ctx, "plot", None)
    session = getattr(plot, "session", None)
    part_lanes(session, plot, params)
    return None
