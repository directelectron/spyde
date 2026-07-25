from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
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
    _resolved_local: Optional[bool] = field(default=None, repr=False, compare=False)
