"""
signals.py — the composed arrays as HyperSpy signals, calibrated and labelled.

A multi-angle acquisition opens as ONE tree with two nodes:

    Aligned Stack   (angle, y, x | ky, kx)   the root — every angle kept
    └── Summed      (y, x | ky, kx)          angle summed away — shown first

Summing is the transformation, so the stack is the parent: the tree then records
what was done, and the Workflow panel offers the stack as the step before the
sum rather than as a separate dataset.

Two calibration details here are easy to get wrong and silent when wrong.

**The angle axis is an INDEX axis, for a multi-shell acquisition.** HyperSpy
does have non-uniform axes (``DataAxis`` carries an explicit array of values),
so the question is not whether an axis can hold irregular angles — it is whether
any single number describes a member. It does not: a member is a (tilt,
azimuth) PAIR, and across shells the azimuths repeat, so the sequence is not
ordered. ``DataAxis`` refuses that outright — *"The non-uniform axis needs to be
ordered"* — and rightly, since ``value2index`` could not answer for a repeated
value anyway.

A SINGLE-shell acquisition is the case where one number does describe a member,
and its sorted azimuths would make a perfectly good non-uniform axis. That is
blocked today by SpyDE rather than by HyperSpy:
``spyde/drawing/selectors/selector1d.py`` resolves a position to an index with
``(x - offset) / scale``, and a ``DataAxis`` has neither attribute — the lookup
raises, is swallowed, and falls back to ``(1.0, 0.0)``, so the angle VALUE is
used as the index. Until that reads ``axis.value2index`` instead, an index axis
is the only one that navigates correctly. The geometry lives in the model and in
the angle navigator meanwhile.

**Cropping moves every origin.** The composed scan starts
``overlap_origin`` positions into the reference member and the composed detector
``detector_origin`` pixels into it, so both offsets shift. Skip that and the
reciprocal-space origin sits off the direct beam by the size of the crop —
small enough to look plausible and wrong in every g-vector derived from it.
"""
from __future__ import annotations

import numpy as np

from spyde.multiangle.compose import (
    composed_axis_offsets, stack_aligned, sum_aligned,
)
from spyde.multiangle.recipe import MultiAngleRecipe, attach_recipe

#: Name given to the leading axis of the stack node.
ANGLE_AXIS_NAME = "angle"


def _reference_axes(members, model):
    """The reference member's axes, in array order ``(y, x, ky, kx)``."""
    reference = members[model.reference]
    axes_manager = getattr(reference, "axes_manager", None)
    if axes_manager is None:
        return None
    return list(axes_manager._axes)


#: What each composed axis IS, in array order after the angle axis. A member's
#: own names come from whichever reader opened it and are frequently a guess —
#: an MRC with no sidecar reports its scan axes as ``''`` and ``'z'`` and its
#: detector axes as ``'y'``/``'x'``. Here the role is known by construction, so
#: it is stated rather than inherited. Scale, offset and units still come from
#: the member: those carry physical information this cannot invent.
_AXIS_ROLE_NAMES = ("y", "x", "ky", "kx")


def _copy_axis(source, destination, name, offset=None):
    destination.scale = float(source.scale)
    destination.units = source.units
    destination.name = name
    destination.offset = (float(source.offset) if offset is None
                          else float(offset))


def _calibrate(signal, members, model, *, leading_angle_axis: bool):
    """Carry the reference member's calibration onto a composed signal.

    The composed axes line up 1:1 with the reference's, shifted by one when the
    stack's angle axis leads. Anything the reference cannot supply is left at
    HyperSpy's default rather than invented.
    """
    reference_axes = _reference_axes(members, model)
    composed_axes = list(signal.axes_manager._axes)
    if leading_angle_axis:
        angle_axis = composed_axes[0]
        angle_axis.name = ANGLE_AXIS_NAME
        angle_axis.scale, angle_axis.offset = 1.0, 0.0
        angle_axis.units = "index"
        composed_axes = composed_axes[1:]
    if reference_axes is None or len(reference_axes) != len(composed_axes):
        return signal

    scan_origin, detector_origin = composed_axis_offsets(
        model,
        scan_offsets=(reference_axes[0].offset, reference_axes[1].offset),
        scan_scales=(reference_axes[0].scale, reference_axes[1].scale),
        detector_offsets=(reference_axes[2].offset, reference_axes[3].offset),
        detector_scales=(reference_axes[2].scale, reference_axes[3].scale),
    )
    for source, destination, name, offset in zip(
            reference_axes, composed_axes, _AXIS_ROLE_NAMES,
            (scan_origin[0], scan_origin[1],
             detector_origin[0], detector_origin[1])):
        _copy_axis(source, destination, name, offset)
    return signal


def _carry_metadata(signal, members, model):
    """Take the reference member's metadata, and record the acquisition on it."""
    reference = members[model.reference]
    metadata = getattr(reference, "metadata", None)
    if metadata is not None:
        try:
            signal.metadata = metadata.deepcopy()
            signal_type = metadata.get_item("Signal.signal_type", "")
            if signal_type:
                signal.set_signal_type(signal_type)
        except Exception:
            # Metadata is a convenience here; a member with an odd tree must not
            # stop the acquisition opening.
            pass
    try:
        signal.metadata.set_item("Acquisition.multiangle", {
            "n_members": int(model.n_members),
            "n_shells": int(model.n_shells),
            "tilts": [float(value) for value in model.tilts],
            "azimuths": [float(value) for value in model.azimuths],
            "shell_ids": [int(value) for value in model.shell_ids],
            "reference": int(model.reference),
        })
    except Exception:
        pass
    return signal


def _as_signal(data):
    """Wrap a composed array in a signal whose last two axes are the detector.

    ``Signal2D(dask_array)`` builds an EAGER signal holding a dask array, which
    looks right and then materialises the whole acquisition the first time
    anything touches ``.data``. The lazy class is the one that keeps the array
    lazy and its chunking intact.
    """
    if hasattr(data, "rechunk"):
        from hyperspy._signals.signal2d import LazySignal2D

        return LazySignal2D(data)

    import hyperspy.api as hs

    return hs.signals.Signal2D(data)


def build_stack_signal(aligned, members, model, *, title=None):
    """The 5-D root: every member on the common grid, angle axis intact."""
    signal = _as_signal(stack_aligned(aligned))
    _calibrate(signal, members, model, leading_angle_axis=True)
    _carry_metadata(signal, members, model)
    signal.metadata.set_item(
        "General.title", title or f"Multi-Angle ({model.n_members} angles)")
    return attach_recipe(signal, MultiAngleRecipe(
        members=tuple(members), model=model, has_angle_axis=True,
        member_indices=tuple(range(len(members)))))


def build_summed_signal(aligned, members, model, *, dtype=None, title=None,
                        member_indices=None):
    """The 4-D node: the members summed.

    ``member_indices`` restricts the sum to some of them — a per-shell sum,
    which is a different experiment (a different precession angle), not a
    cosmetic subset.
    """
    if member_indices is None:
        member_indices = range(len(aligned))
    member_indices = tuple(int(index) for index in member_indices)
    chosen = [aligned[index] for index in member_indices]
    chosen_members = tuple(members[index] for index in member_indices)

    data = sum_aligned(chosen, dtype=dtype)
    signal = _as_signal(data)
    _calibrate(signal, members, model, leading_angle_axis=False)
    _carry_metadata(signal, members, model)
    signal.metadata.set_item("General.title", title or "Summed")
    return attach_recipe(signal, MultiAngleRecipe(
        members=chosen_members, model=model, has_angle_axis=False,
        dtype=data.dtype, member_indices=member_indices))


def shell_titles(model):
    """``{shell_id: label}`` — the tilt each shell was taken at, for a node name."""
    titles = {}
    for shell_id, member_indices in model.shells.items():
        tilt = float(model.tilts[member_indices[0]])
        titles[shell_id] = f"Summed {tilt:g}°"
    return titles


def member_navigator_planes(navigators, model):
    """``(N, height, width)`` — each member's OWN overview on the common grid.

    The 5-D root's navigator needs the angle axis, and what goes along it
    matters: give every plane the same composed image and the angle line
    reduces to a constant, which is a navigator that cannot be navigated. One
    plane per member makes the line each member's own total intensity, so a
    tilt that collected less is visible and dragging along it means something.

    The cost is that the real-space navigator then shows ONE member's image
    while the summed node is displayed. That is the honest reading of the axis —
    it is the angle axis, and its planes are the angles.
    """
    if len(navigators) != model.n_members:
        raise ValueError(
            f"the model describes {model.n_members} members but "
            f"{len(navigators)} navigators were given")
    first = np.asarray(navigators[0])
    planes = []
    for member_index, navigator in enumerate(navigators):
        image = np.asarray(navigator)
        if image.shape != first.shape:
            raise ValueError(
                "navigators must all be the members' own scan shape; got "
                f"{image.shape} and {first.shape}")
        rows, columns = model.nav_slices(member_index, image.shape)
        planes.append(image[rows, columns].astype(np.float32))
    return np.stack(planes, axis=0)


def composite_navigator(navigators, model):
    """The composed scan overview, for free, from the members' own navigators.

    Reducing the composed 5-D array to build this would read every member in
    full. The members' navigators already exist — each one is computed when its
    member is opened — and the composed navigator is those images on the common
    grid, added. Exactly the reasoning behind ``_reader_navigator``: a navigator
    somebody already has beats one computed from the whole dataset.
    """
    if len(navigators) != model.n_members:
        raise ValueError(
            f"the model describes {model.n_members} members but "
            f"{len(navigators)} navigators were given")
    first = np.asarray(navigators[0])
    total = None
    for member_index, navigator in enumerate(navigators):
        image = np.asarray(navigator)
        if image.shape != first.shape:
            raise ValueError(
                "navigators must all be the members' own scan shape; got "
                f"{image.shape} and {first.shape}")
        rows, columns = model.nav_slices(member_index, image.shape)
        region = image[rows, columns].astype(np.float64)
        total = region if total is None else total + region
    return total
