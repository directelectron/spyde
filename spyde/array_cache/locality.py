"""Ancestry-walk resolver for the SignalNode.local tag.

Walks a node's parent chain to the nearest concrete/opaque boundary and
answers one question: is EVERY step from here back to a real backing tagged
"local" (safe to lazily slice and compute one frame at a time)? Memoized on
the node itself (SignalNode._resolved_local) so it's resolved ONCE per
view-select, never re-walked per frame — see the frame-pipeline design notes.

Scope: this only matters for nodes reached via BaseSignalTree.add_transformation
/ add_node (same-tree children). commit.py's open_result_tree/commit_result_tree
and the Python console's Session._add_signal always build a NEW tree root
(parent=None), so they resolve trivially via the root boundary below and never
need this walk at all — by construction they're already concrete or going
through the existing progressive-fill materialize path.
"""
from __future__ import annotations

from spyde.signal_node import SignalNode


def resolve_locality(node: SignalNode) -> bool:
    """True if the chain from ``node`` back to the nearest tree root is
    entirely local-tagged — i.e. ``node``'s data can be computed one frame at
    a time via a small, bounded slice of its ancestry, with no node along the
    way requiring the whole dataset to be materialized first.

    Default is opaque (False) for any untagged node — the fail-safe answer,
    since arbitrary code (a console session, a future action nobody tagged)
    can't be inspected for locality automatically.
    """
    if node._resolved_local is not None:
        return node._resolved_local

    if node.parent is None:
        # Root boundary: either the original file-backed signal, or the root
        # of a tree built by open_result_tree/commit_result_tree/the console —
        # all of which are already concrete or on the existing materialize path.
        result = True
    elif node.local is not True:
        # Untagged (None) or explicitly opaque (False) — fail-safe default.
        result = False
    else:
        result = resolve_locality(node.parent)

    node._resolved_local = result
    return result
