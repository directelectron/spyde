"""The 5-D multi-angle stack, as a signal type.

A composed acquisition saved to disk is just an array: ``(angle, y, x | ky,
kx)`` and nothing to say what the angle axis means. Reopening it gave a 5-D
dataset with no Summed node, no shells and no angle ring — the alignment
survived, everything built on top of it did not, and a dataset published that
way hands the next person an array and no way to know what it is.

Everything needed to put the tree back is already in the file: the stack IS
the aligned members, so the sums are reductions over its first axis, and
``Acquisition.multiangle`` carries the tilts, azimuths, shells and reference
that name them.

It EXTENDS ``ElectronDiffraction2D`` rather than standing beside it. The
toolchain is gated on the signal type, and a composed acquisition losing that
type has already cost this project once — every diffraction action silently
disappeared. By inheriting, a multi-angle stack is still a diffraction signal
to everything that asks, and is its own thing to the loader.

Recognition falls back to the METADATA and the angle axis, because a file
written before this type existed carries the metadata and not the type, and
because a 4-D SUM carries the metadata without being a stack.
"""
from __future__ import annotations

from pyxem.signals import ElectronDiffraction2D, LazyElectronDiffraction2D

#: Where a composed acquisition records what it is.
MULTIANGLE_METADATA = "Acquisition.multiangle"
MULTIANGLE_SIGNAL_TYPE = "spyde_multiangle"


def is_multiangle_stack(signal) -> bool:
    """True when *signal* is a saved multi-angle stack to be re-expanded.

    A 4-D SUM carries the same metadata without being a stack, so the angle
    axis is what distinguishes them.
    """
    metadata = getattr(signal, "metadata", None)
    if metadata is None or not metadata.has_item(MULTIANGLE_METADATA):
        return False
    if metadata.get_item("Signal.signal_type", "") == MULTIANGLE_SIGNAL_TYPE:
        return True
    recorded = metadata.get_item(MULTIANGLE_METADATA)
    try:
        members = int(recorded["n_members"])
    except (KeyError, TypeError, ValueError):
        return False
    data = getattr(signal, "data", None)
    shape = getattr(data, "shape", ())
    # A stack leads with one plane per member; a sum has no such axis.
    return len(shape) == 5 and int(shape[0]) == members


class MultiAngle(ElectronDiffraction2D):
    """A multi-angle acquisition with its angle axis intact.

    A diffraction signal first: everything gated on that keeps working on the
    stack, which is the whole reason for extending rather than replacing.
    """

    _signal_type = MULTIANGLE_SIGNAL_TYPE
    _signal_dimension = 2


class LazyMultiAngle(LazyElectronDiffraction2D, MultiAngle):
    """The lazy one, which is what a real acquisition always is."""

    _signal_type = MULTIANGLE_SIGNAL_TYPE
    _signal_dimension = 2
    _lazy = True
