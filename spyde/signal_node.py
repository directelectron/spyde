from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
from hyperspy.signal import BaseSignal


@dataclass
class SignalNode:
    signal: BaseSignal
    name: str
    parent: Optional["SignalNode"]
    children: dict[str, "SignalNode"] = field(default_factory=dict)
    transformation: Optional[str] = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    # ArrayCache locality tag: True = frame N of this node's output needs only a
    # small, bounded slice of frame N of the parent's data (safe to lazily slice
    # and compute one frame at a time). None (the default, set by every call site
    # that doesn't explicitly opt in) resolves to opaque/must-materialize — a
    # fail-safe default, since arbitrary code (a console session, a future action
    # nobody remembered to tag) can't be inspected for locality automatically.
    # See spyde/array_cache/locality.py for the ancestry-walk resolver.
    local: Optional[bool] = None
    # Overlay node: its signal has no data of its own, it is evaluated at the
    # navigator position and drawn on the plot showing its parent. It gets no
    # PlotState and is not part of the workflow tree the renderer shows.
    overlay: bool = False
    # The per-position function is slow enough to freeze the navigator, so it
    # runs as one cancellable future instead of inline on the dispatcher. None
    # means the source frame decides: the overlay takes whichever tier the base
    # read of that frame would take.
    expensive: Optional[bool] = False
    # An integrating region reaches this node's source read as the whole point
    # array, so it integrates what the base frame integrates. Off by default:
    # a marker overlay follows the region's centre position.
    follows_region: bool = False
    # Drawn while True; a hidden overlay keeps its groups but pushes nothing.
    visible: bool = True
    # Group name -> (kind, style): one anyplotlib primitive per entry, created
    # when the node is added. The function returns a value per group name.
    groups: dict = field(default_factory=dict)
    # Called on the painter thread with the whole value dict each time the
    # overlay is drawn, for a node whose result also feeds a panel or a caret.
    on_value: Optional[Callable] = None
    _resolved_local: Optional[bool] = field(default=None, repr=False, compare=False)

    @property
    def attached(self) -> bool:
        """True while this node is still a child of its parent. A removed
        overlay can have a value in flight, and must not draw."""
        parent = self.parent
        return parent is not None and parent.children.get(self.name) is self
