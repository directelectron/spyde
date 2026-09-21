"""
multiangle_navigator.py — the POLAR ANGLE NAVIGATOR for a multi-angle acquisition.

A multi-angle acquisition is N datasets ("members") taken at different (tilt,
azimuth) angles about one point (:mod:`spyde.multiangle`), opened as one tree::

    Aligned Stack   (angle, y, x | ky, kx)   5-D root
    └── Summed      (y, x | ky, kx)          displayed on open

The angle is the 5th dimension, and the tree navigates it with an ordinary 1-D
line. That line is honest about the DATA — one member per index, carrying each
member's intensity — but it says nothing about the acquisition: the members are
not ordered in any single coordinate across shells, so the axis is a plain index
axis (0…N−1) and the degrees live only in the model. This window is where the
degrees are drawn.

It draws the acquisition as it physically happened: one ring per shell, at the
shell's tilt, with one point per member at ``(tilt·sin θ, tilt·cos θ)``. So a
MISSING angle reads as a gap in the ring rather than as a number nobody checked,
and two members that happen to share an azimuth across shells sit at different
radii — which is also why a pick resolves to a MEMBER INDEX, never an azimuth.

**The picture of the angles is the toggle.** On the Summed node every angle is
lit, because every one of them is in the frame on screen; clicking one switches
the display to the 5-D stack at that angle. On the stack node the live angle
carries a draggable handle, and clicking the ring or dragging that handle round
it scrubs the 5th dimension. One picture, both states.

**It drives the tree's REAL selector and keeps no copy of the position.** A pick
moves the tree's own 1-D navigation selector; an ``index_hook`` on that selector
moves the ring's highlight. So a move from anywhere else — the 1-D line,
playback, a programmatic step, a node switch's forced re-fire — reaches the ring
by the same path a pick does, and the two can never disagree. This is the
binding :mod:`spyde.actions.navigator_views` uses for its stacked 1-D cursors.

A bare-figure window (no registered ``Plot``), so it registers a controller for
dispatch and teardown and keeps its figure alive per window — see §6 of
``spyde/actions/README.md``.
"""
from __future__ import annotations

import logging

import numpy as np

from de_shell.actions.figure_registry import keep_alive
from spyde.drawing.selectors.base_selector import event_handler_fn

logger = logging.getLogger(__name__)

#: Ring outline, and the tilt label beside it.
RING_COLOR = "#555c66"
LABEL_COLOR = "#9aa4b2"
#: A member that is contributing to the frame on screen, and one that is not.
LIT_COLOR = "#ff9100"
DIM_COLOR = "#4a5560"
#: The live angle, on the stack node — the one the frame on screen comes from.
HIGHLIGHT_COLOR = "#ffffff"
#: Point outline — dark, so neighbouring members on a crowded ring stay separate.
POINT_EDGE_COLOR = "#11151a"

MEMBER_POINT_SIZE = 9

#: Free space around the outermost ring, as a fraction of its radius. Enough for
#: the tilt labels, which sit just outside each ring.
_VIEW_MARGIN = 0.28
#: Gap between a ring and its tilt label, as a fraction of the outermost radius.
_LABEL_GAP = 0.05

#: Attribute the open window is cached under, so a second request raises the
#: existing window instead of piling up duplicates.
_WINDOW_ATTRIBUTE = "_multiangle_navigator"


# ─────────────────────────────────────────────────────────────────────────────
# Reading the acquisition off the tree
# ─────────────────────────────────────────────────────────────────────────────

def multiangle_model(tree):
    """The :class:`~spyde.multiangle.model.MultiAngleModel` behind *tree*, or None.

    Every composed node carries the same model, so the first one found answers
    for the tree.
    """
    from spyde.multiangle.model import model_from_metadata
    from spyde.multiangle.recipe import recipe_for

    nodes = list(getattr(tree, "walk", lambda: [])())
    # The stack's recipe first: a SUM's recipe carries zeroed offsets on a
    # reopened tree (its members are the already-aligned planes), which is
    # right for reading it and wrong as a description of the acquisition.
    for node in nodes:
        recipe = recipe_for(node.signal)
        if recipe is not None and recipe.has_angle_axis:
            return recipe.model
    # A signal READ BACK FROM A FILE has no recipe — recipes are runtime
    # objects and are not written out. The metadata carries the same model,
    # which is the reason it is recorded.
    for node in nodes:
        model = model_from_metadata(node.signal)
        if model is not None:
            return model
    for node in nodes:
        recipe = recipe_for(node.signal)
        if recipe is not None:
            return recipe.model
    return None


def is_angle_stack(signal) -> bool:
    """True when *signal* keeps the angle axis, composed or read from a file."""
    from spyde.multiangle.recipe import recipe_for
    from spyde.signals.multiangle import is_multiangle_stack

    if signal is None:
        return False
    recipe = recipe_for(signal)
    if recipe is not None:
        return bool(recipe.has_angle_axis)
    return is_multiangle_stack(signal)


def stack_signal(tree, displayed=None):
    """The 5-D per-angle node — the one a pick on the ring switches to.

    *displayed* wins when it is itself a stack: a Rebin or Crop of the stack
    keeps the angle axis and the metadata, and a pick that switched to the
    first stack in the tree put a user who had binned a 97 GB acquisition
    back on the full-resolution root without a word.
    """
    from spyde.multiangle.recipe import recipe_for
    from spyde.signals.multiangle import is_multiangle_stack

    if is_angle_stack(displayed):
        return displayed
    nodes = list(getattr(tree, "walk", lambda: [])())
    for node in nodes:
        recipe = recipe_for(node.signal)
        if recipe is not None and recipe.has_angle_axis:
            return node.signal
    # Same reason: from a file there is no recipe to ask, and the angle axis
    # is what `has_angle_axis` was reporting anyway.
    for node in nodes:
        if is_multiangle_stack(node.signal):
            return node.signal
    return None


def member_positions(model) -> np.ndarray:
    """``(N, 2)`` — where each member sits on the ring picture.

    ``x = tilt·sin(azimuth)``, ``y = tilt·cos(azimuth)``: azimuth 0° points up
    and grows clockwise, which is how a tilt/azimuth pair is normally read off
    an instrument.
    """
    azimuths = np.radians(np.asarray(model.azimuths, dtype=float))
    tilts = np.asarray(model.tilts, dtype=float)
    return np.column_stack([tilts * np.sin(azimuths), tilts * np.cos(azimuths)])


def nearest_member(positions, x: float, y: float) -> int:
    """Index of the member nearest ``(x, y)`` on the ring picture.

    Distance in the PLANE, which is what makes a pick unambiguous: two members
    can share an azimuth across shells, and they are then separated only by
    their radii. An azimuth alone can never name a member.
    """
    offsets = np.asarray(positions, dtype=float) - np.array([float(x), float(y)])
    return int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))


def shell_radii(model) -> dict[int, float]:
    """``shell id → ring radius`` (the shell's mean tilt), in ascending id order.

    The mean rather than any one member's tilt: a shell groups tilts within a
    tolerance, so its members' values can differ in the last digit and a ring
    drawn at one of them would miss the others by a hair.
    """
    tilts = np.asarray(model.tilts, dtype=float)
    return {shell: float(np.mean(tilts[list(members)]))
            for shell, members in model.shells.items()}


def _widest_gap(azimuths) -> float:
    """The azimuth (radians) with the most room, for a ring's tilt label.

    A label at a fixed azimuth lands on top of whichever member happens to sit
    there — at azimuth 0 it always does, since an acquisition that starts at 0°
    is the normal case. Putting it in the ring's widest gap instead means it
    only ever collides when the ring is genuinely full, and it draws the eye to
    a missing angle rather than hiding one.
    """
    values = np.sort(np.mod(np.asarray(azimuths, dtype=float), 360.0))
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(np.radians(values[0] + 180.0))
    # Wrapping the first value round the circle makes the last gap ordinary.
    wrapped = np.append(values, values[0] + 360.0)
    gaps = np.diff(wrapped)
    widest = int(np.argmax(gaps))
    return float(np.radians(wrapped[widest] + gaps[widest] / 2.0))


def _format_degrees(value: float) -> str:
    """``1.0`` → ``"1°"``, ``0.5`` → ``"0.5°"`` — the shortest exact form."""
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return f"{text or '0'}°"


# ─────────────────────────────────────────────────────────────────────────────
# The tree's real angle selector
# ─────────────────────────────────────────────────────────────────────────────

def angle_selector(tree, n_members: int):
    """The tree's REAL 1-D navigation selector for the angle axis, or None.

    Identified by what it indexes — a one-dimensional navigator whose axis is as
    long as there are members — rather than by its class, so it keeps answering
    if the chain gains another 1-D navigator.
    """
    manager = getattr(tree, "navigator_plot_manager", None)
    if manager is None:
        return None
    for selector in manager.all_navigation_selectors:
        if _selector_axis_size(selector) == int(n_members):
            return selector
    return None


def _selector_axis_size(selector) -> int | None:
    """Length of the 1-D navigation axis *selector* indexes, or None if it is
    not a 1-D selector at all."""
    try:
        plot = selector.current_plot
        axes = plot.plot_state.current_signal.axes_manager.signal_axes
    except Exception:
        return None
    return int(axes[0].size) if len(axes) == 1 else None


def _index_hook_targets(selector) -> list:
    """Every sub-selector of *selector* whose updates should move the ring.

    A 1-D navigation selector is a composite that swaps between a crosshair and
    an integrating span, and each half keeps its own hook list. Hooking only the
    active one means the ring silently stops following the moment the user turns
    Integrate on.
    """
    candidates = [getattr(selector, name, None)
                  for name in ("_inf_line_selector", "_linear_region_selector")]
    targets = [target for target in candidates
               if getattr(target, "index_hooks", None) is not None]
    if targets:
        return targets
    return [selector] if getattr(selector, "index_hooks", None) is not None else []


# ─────────────────────────────────────────────────────────────────────────────
# The window
# ─────────────────────────────────────────────────────────────────────────────

class MultiAngleNavigatorController:
    """Owns the polar angle window: its figure, its binding to the tree's real
    angle selector, and the pick that moves it.

    Registered with ``session.register_window_controller`` so ✕-closing the
    window tears the binding down (the WindowController protocol in
    :mod:`spyde.actions.registry`), and so a staged action arriving from this
    bare-figure window — which has no ``Plot`` to resolve — still finds it.
    """

    def __init__(self, session, window_id: int, tree, model, title: str):
        self.session = session
        self.window_id = int(window_id)
        self.tree = tree
        self.model = model
        self.title = title
        self.positions = member_positions(model)
        self.radii = shell_radii(model)
        self.current_members: tuple[int, ...] = ()
        #: What the ring says the display is — the angle, or the sum over them.
        self.badge = ""
        self.selector = angle_selector(tree, model.n_members)
        self._points = None
        self._handle = None
        self._axes = None
        #: True while the pointer still holds the handle — JS owns its position
        #: then, and writing to it would fight the drag.
        self._dragging = False
        # Bound once and kept: a fresh bound method is a new object on every
        # attribute access, so a handler looked up again could not be found by
        # identity to detach it. The pointer handler is additionally wrapped,
        # because anyplotlib tags its handlers with an attribute and a bound
        # method cannot take one.
        self._index_hook = self._on_selector_index
        self._pointer_handler = event_handler_fn(self._on_pointer)
        self._handle_handler = event_handler_fn(self._on_handle)
        self._hooked: list = []
        self._closed = False

    # ── build + emit ────────────────────────────────────────────────────────
    def build(self) -> bool:
        """Draw the rings and emit the window. False if the figure failed."""
        import anyplotlib as apl
        import anyplotlib._electron as _electron

        from spyde.drawing.plots.plot import finalize_figure_html
        from de_shell.ipc import emit

        try:
            figure, axes = apl.subplots(1, 1)
            # `or 1.0` covers an acquisition whose only shell is at zero tilt,
            # where every point is the origin and a zero-width view has none.
            outer = max(self.radii.values(), default=0.0) or 1.0
            limit = outer * (1.0 + _VIEW_MARGIN)
            panel = np.array(axes, dtype=object).ravel()[0].axes2d(
                xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
            self._draw_rings(panel, outer)
            self._points = panel.scatter(
                self.positions[:, 0], self.positions[:, 1],
                s=MEMBER_POINT_SIZE, c=[LIT_COLOR] * self.model.n_members,
                edgecolors=POINT_EDGE_COLOR)
            # The live angle is a DRAGGABLE handle rather than a drawn marker:
            # dragging it round the ring is how the 5th dimension is scrubbed,
            # and a plain click on the panel is the other way to pick an angle.
            self._handle = panel.add_point_widget(
                float(self.positions[0, 0]), float(self.positions[0, 1]),
                color=HIGHLIGHT_COLOR, show_crosshair=False)
            self._handle.add_event_handler(self._handle_handler,
                                           "pointer_move", "pointer_up")
            self._axes = panel
            panel.add_event_handler(self._pointer_handler, "pointer_down")
            # Before the figure is turned into HTML, so the window opens in the
            # state it belongs in rather than correcting itself a moment later.
            self.refresh()

            figure_id = _electron.register(figure)
            keep_alive(self.window_id, figure)
            emit({"type": "figure", "fig_id": figure_id,
                  "window_id": self.window_id,
                  "html": finalize_figure_html(figure, figure_id),
                  "title": self.title, "is_navigator": False})
            # A figure message does not rename a window, so say the name too.
            emit({"type": "window_title", "window_ids": [self.window_id],
                  "title": self.title})
        except Exception as e:
            logger.exception("building the multi-angle ring failed: %s", e)
            return False
        return True

    def _draw_rings(self, panel, outer: float) -> None:
        """One outline per shell, each labelled with its tilt."""
        turn = np.linspace(0.0, 2.0 * np.pi, 181)
        tilts = np.asarray(self.model.tilts, dtype=float)
        for shell, radius in self.radii.items():
            panel.plot(radius * np.cos(turn), radius * np.sin(turn),
                       color=RING_COLOR, linewidth=1.0)
            label = _format_degrees(tilts[self.model.shells[shell][0]])
            azimuth = _widest_gap(self.model.azimuths[list(self.model.shells[shell])])
            offset = radius + outer * _LABEL_GAP
            panel.text(offset * np.sin(azimuth), offset * np.cos(azimuth),
                       label, color=LABEL_COLOR, fontsize=11)

    # ── what the ring currently shows ───────────────────────────────────────
    def showing_one_angle(self) -> bool:
        """True when the tree displays the per-angle stack rather than a sum.

        Read from the tree every time instead of remembered, so a node switch
        made anywhere — the Workflow panel, a script — reaches the ring.
        """
        # Not the recipe alone: a reopened tree has none, and asking only
        # it left the ring stuck on "all angles" — every member lit, no
        # handle, and a full node switch on every pick.
        return is_angle_stack(self.displayed_signal())

    def displayed_signal(self):
        """The node the tree's signal window is showing, or None."""
        for plot in list(getattr(self.tree, "signal_plots", []) or []):
            state = getattr(plot, "plot_state", None)
            signal = getattr(state, "current_signal", None) if state else None
            if signal is not None:
                return signal
        return None

    def live_members(self) -> tuple[int, ...]:
        """The members whose data is in the frame on screen.

        On a sum that is every member — which is what "every angle lit" means,
        and why the summed node needs no highlight: no single angle is the live
        one. On the stack it is the angle axis's selection, which is normally
        one member and is the whole span when the angle axis is integrating.
        """
        if not self.showing_one_angle():
            return tuple(range(self.model.n_members))
        return self.current_members

    def refresh(self, members=None) -> None:
        """Repaint the ring for the displayed node and the live angle(s).

        Main thread only — it writes to the figure.
        """
        if self._closed or self._points is None:
            return
        if members is not None:
            self.current_members = tuple(int(member) for member in members)
        elif not self.current_members:
            self.current_members = self._selector_members()
        live = self.live_members()
        self.badge = self._badge(live)
        try:
            self._points.set(facecolors=self._member_colors(live))
            self._place_handle(live)
            self._axes.set_title(self.badge)
        except Exception as e:
            logger.debug("repainting the multi-angle ring failed: %s", e)

    def _place_handle(self, live) -> None:
        """Put the draggable handle on the live angle.

        Hidden on a sum: with every angle lit there is no single one for it to
        mark, and a handle sitting on one of them would say otherwise. An
        integrating span keeps it on the span's first angle — the lit points
        are what show the whole span.
        """
        member = int(live[0]) if live else -1
        if not self.showing_one_angle() or not 0 <= member < self.model.n_members:
            self._handle.hide()
            return
        if not self._dragging:
            x, y = self.positions[member]
            self._handle.set(_notify=False, x=float(x), y=float(y))
        self._handle.show()

    @property
    def current_member(self) -> int | None:
        """The single live angle, or None when several are."""
        return (self.current_members[0]
                if len(self.current_members) == 1 else None)

    def _member_colors(self, live) -> list[str]:
        members = set(live)
        return [LIT_COLOR if index in members else DIM_COLOR
                for index in range(self.model.n_members)]

    def _badge(self, live) -> str:
        if not live:
            return ""
        if len(live) == 1:
            member = live[0]
            return (f"{_format_degrees(self.model.tilts[member])} · "
                    f"{_format_degrees(self.model.azimuths[member])}")
        if len(live) == self.model.n_members:
            return "Σ all angles"
        return f"Σ {len(live)} angles"

    def drawn_highlight(self) -> np.ndarray:
        """``(K, 2)`` — where the ring marks the live angle: the handle's own
        position when it is showing, nothing when the display is a sum.

        Read back off the handle rather than from :attr:`current_members`, so a
        caller checking the ring is checking what is on screen.
        """
        if self._handle is None or not self._handle.visible:
            return np.empty((0, 2), dtype=float)
        return np.array([[float(self._handle.get("x")),
                          float(self._handle.get("y"))]])

    def drawn_member_colors(self) -> list[str]:
        """The per-member fill colours currently on screen."""
        if self._points is None:
            return []
        return list(self._points._data.get("facecolors") or [])

    # ── picking ─────────────────────────────────────────────────────────────
    def _on_pointer(self, event) -> None:
        """A click anywhere on the ring picks the angle nearest to it.

        Nearest rather than a hit on the point itself: the points are small and
        the answer is never ambiguous, so requiring a direct hit would only make
        the window fussy to use.
        """
        if self._closed or event.xdata is None or event.ydata is None:
            return
        self.select_member(
            nearest_member(self.positions, event.xdata, event.ydata))

    def _on_handle(self, event=None) -> None:
        """The handle was dragged — take the angle it is nearest."""
        if self._closed or self._handle is None:
            return
        self._dragging = getattr(event, "event_type", "") != "pointer_up"
        self.select_member(nearest_member(self.positions,
                                          self._handle.get("x"),
                                          self._handle.get("y")))

    def select_member(self, member: int) -> None:
        """Show *member*'s angle: move the REAL selector, switching to the
        per-angle stack first if a sum is on screen.

        The selector's widget is positioned before the node switch so the
        forced re-slice the switch fires already reads the new angle — one
        update rather than one at the old angle and another at the new.
        """
        if self._closed or not 0 <= int(member) < self.model.n_members:
            return
        self._position_selector(int(member))
        if self.showing_one_angle():
            self._refire_selector()
        else:
            self._show_stack()
        self.refresh([int(member)])

    def _position_selector(self, member: int) -> None:
        """Move the real selector's line onto *member* without updating yet.

        ``_notify=False`` because a plain write is indistinguishable from a
        drag: the selector's own handler would hear it and queue an update at
        the node we are about to leave. The update is fired deliberately below,
        once — by the node switch, or by :meth:`_refire_selector`.
        """
        widget = _selector_widget(self.selector)
        if widget is None:
            return
        scale, offset = _selector_calibration(self.selector)
        try:
            widget.set(_notify=False, x=member * scale + offset)
        except Exception as e:
            logger.debug("positioning the angle selector failed: %s", e)

    def _refire_selector(self) -> None:
        try:
            self.selector.delayed_update_data(force=True)
        except Exception as e:
            logger.debug("re-firing the angle selector failed: %s", e)

    def _show_stack(self) -> None:
        """Switch the display to the 5-D stack, through the ordinary node-switch
        path (which re-slices every navigator, this window's angle included)."""
        signal = stack_signal(self.tree, self.displayed_signal())
        plot = next(iter(getattr(self.tree, "signal_plots", []) or []), None)
        if signal is None or plot is None:
            self._refire_selector()
            return
        try:
            self.session._select_signal_node(plot, id(signal))
        except Exception as e:
            logger.debug("switching to the aligned stack failed: %s", e)
            self._refire_selector()

    # ── the selector → ring half of the binding ─────────────────────────────
    def bind_selector(self) -> None:
        """Follow the real selector, whoever moved it.

        ``index_hooks`` fire in the selector's own update with the committed
        indices, so playback, a keyboard step and a node switch's forced
        re-slice all arrive here exactly as a drag does.
        """
        for target in _index_hook_targets(self.selector):
            try:
                target.index_hooks.append(self._index_hook)
                self._hooked.append(target)
            except Exception as e:
                logger.debug("binding the ring to the angle selector "
                             "failed: %s", e)

    def _on_selector_index(self, indices) -> None:
        """Runs on the navigator dispatcher thread — marshal the repaint."""
        if self._closed:
            return
        members = _members_from_indices(indices)
        if members:
            self.session._dispatch_to_main(lambda: self.refresh(members))

    def _selector_members(self) -> tuple[int, ...]:
        """Where the real selector sits now, for the opening paint."""
        return _members_from_indices(
            getattr(self.selector, "current_indices", None))

    # ── WindowController protocol ───────────────────────────────────────────
    @property
    def source_plot(self):
        """This window's stand-in wherever a real ``Plot``/tree is required."""
        return next(iter(getattr(self.tree, "signal_plots", []) or []), None)

    def close(self) -> None:
        """Detach from the real selector and forget the figure handles.

        Detaching first: a torn-down window must not keep repainting markers
        that no longer have a window to be drawn in.
        """
        self._closed = True
        for target in self._hooked:
            try:
                target.index_hooks.remove(self._index_hook)
            except ValueError:
                pass
            except Exception as e:
                logger.debug("detaching the ring's index hook failed: %s", e)
        self._hooked = []
        self._points = self._handle = self._axes = None
        if getattr(self.tree, _WINDOW_ATTRIBUTE, None) is self:
            try:
                setattr(self.tree, _WINDOW_ATTRIBUTE, None)
            except Exception as e:
                logger.debug("clearing the cached ring window failed: %s", e)


def _members_from_indices(indices) -> tuple[int, ...]:
    """A 1-D selector's committed indices as member numbers.

    One row per selected position: one for a crosshair, several when the angle
    axis is integrating a span.
    """
    if indices is None:
        return ()
    try:
        return tuple(int(value) for value in np.asarray(indices).ravel())
    except Exception:
        return ()


def _selector_widget(selector):
    """The draggable line widget backing a 1-D navigation selector."""
    inner = getattr(selector, "_inf_line_selector", None)
    widget = getattr(inner, "_widget", None) if inner is not None else None
    return widget if widget is not None else getattr(selector, "_widget", None)


def _selector_calibration(selector) -> tuple[float, float]:
    """``(scale, offset)`` of the axis the selector indexes, so a member index
    becomes the widget's data-space position."""
    try:
        axis = (selector.current_plot.plot_state.current_signal
                .axes_manager.signal_axes[0])
        return float(axis.scale), float(axis.offset)
    except Exception:
        return 1.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Opening, and the staged actions
# ─────────────────────────────────────────────────────────────────────────────

def open_multiangle_navigator(session, tree):
    """Open *tree*'s polar angle navigator, or refresh the one already open.

    Returns the controller — or None for a tree that is not a multi-angle
    acquisition, or one whose angle axis has no navigation selector to drive.
    """
    from de_shell.ipc import emit_error

    existing = getattr(tree, _WINDOW_ATTRIBUTE, None)
    if existing is not None and not existing._closed:
        existing.refresh()
        return existing

    model = multiangle_model(tree)
    if model is None:
        emit_error("The angle navigator needs a multi-angle acquisition.")
        return None
    if angle_selector(tree, model.n_members) is None:
        emit_error("The angle navigator needs the acquisition's angle "
                   "navigator to be open.")
        return None

    controller = MultiAngleNavigatorController(
        session, session.next_window_id(), tree, model, _window_title(tree))
    if not controller.build():
        return None
    controller.bind_selector()
    session.register_window_controller(controller.window_id, controller)
    setattr(tree, _WINDOW_ATTRIBUTE, controller)
    return controller


def _window_title(tree) -> str:
    """``"<dataset> — Angles"``. The tilts are not in the name: they are the
    ring labels, and an acquisition with several shells would spell them out
    twice and still lose the end of the title to the window's width."""
    try:
        base = str(tree.root.metadata.get_item("General.title", "") or "")
    except Exception as e:
        logger.debug("resolving the angle-window title failed: %s", e)
        base = ""
    return f"{base} — Angles" if base else "Angles"


def _tree_for(session, plot, payload):
    """The multi-angle tree an action is aimed at.

    From the clicked plot when it has one, else from the window the action came
    from (a bare-figure window resolves through its controller), else the one
    multi-angle tree the session holds.
    """
    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    if tree is None:
        controller = session.controller_by_window_id(
            (payload or {}).get("window_id"))
        tree = getattr(controller, "tree", None)
    if tree is not None and multiangle_model(tree) is not None:
        return tree
    candidates = [candidate for candidate in getattr(session, "signal_trees", [])
                  if multiangle_model(candidate) is not None]
    return candidates[0] if len(candidates) == 1 else None


def multiangle_show_angles(session, plot, payload) -> None:
    """Staged action — open the polar angle navigator for the acting tree."""
    from de_shell.ipc import emit_error

    tree = _tree_for(session, plot, payload)
    if tree is None:
        emit_error("No multi-angle acquisition is open.")
        return
    open_multiangle_navigator(session, tree)


def multiangle_pick_angle(session, plot, payload) -> None:
    """Staged action — show one member's angle, as a pick on the ring does.

    ``payload["member"]`` is the member index. Goes through the controller so a
    programmatic pick and a click are the same code, ring highlight included.
    """
    tree = _tree_for(session, plot, payload)
    if tree is None:
        return
    controller = open_multiangle_navigator(session, tree)
    if controller is not None:
        controller.select_member(int((payload or {}).get("member", 0)))
