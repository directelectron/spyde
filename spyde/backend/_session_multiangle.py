"""
_session_multiangle.py — a multi-angle acquisition opened as ONE signal tree.

:mod:`spyde.multiangle` aligns N separate 4-D datasets ("members") recorded at
different tilt magnitudes and azimuths about one point, and composes them. This
module is the step that turns that into something a user is looking at::

    Aligned Stack   (angle, y, x | ky, kx)   the tree ROOT
    └── Summed      (y, x | ky, kx)          what the window opens on
        ├── Summed 1°     (that shell's members only)
        └── Summed 0.5°

Summing is the transformation, so the stack is the parent: the tree then records
what was done, and the Workflow section of the dock offers the stack as the step
before the sum rather than as a separate dataset. The node the experiment is
*for* is the summed one, so that is the node displayed once the windows open.

Three things here are not free to change.

**Nothing reads a member's 4-D array.** The alignment is solved on two already
reduced images per member, the composition is slicing, and the display reads
through :class:`~spyde.array_cache.readers.multiangle.MultiAngleReader`. A
``.compute()`` on a composed array would pull a navigation chunk out of every
member at once — the CLAUDE.md memory-safety rule, multiplied by N.

**The navigator comes from the members, not from the composed array.** Letting
the tree build its own would reduce the whole 5-D acquisition to draw a
thumbnail. :func:`~spyde.multiangle.signals.composite_navigator` is that same
image for nothing, from images the alignment solve already needed.

**The children are added with ``add_node``, not ``add_transformation``.** A
composed signal is built by :mod:`spyde.multiangle.signals`, not by calling a
method on its parent, so there is no transformation to apply — only a node to
record.
"""
from __future__ import annotations

import logging
import math
import os
import threading

import numpy as np

from de_shell import ipc
from de_shell.ipc import emit_status, emit_error

from spyde.signals.multiangle import MULTIANGLE_METADATA

from spyde.backend._session_files import (
    SUPPORTED_EXTS, _is_supported_dataset_path, _path_ext,
)

log = logging.getLogger(__name__)

#: Scan positions per navigation chunk of the members' common grid, when the
#: composed frame's size is not known. Matches the default
#: :mod:`spyde.multiangle.load` chooses, so members loaded through either door
#: land on the same grid.
NAV_CHUNK = 32

#: What ONE composed navigation chunk should weigh. A chunk is what every
#: consumer holds whole, so bytes are the thing to fix, not a count of scan
#: positions: 32 x 32 is 134 MB of 256 x 256 uint16 patterns and 1.04 GB of
#: 507 x 501 uint32 ones, and at a gigabyte nothing downstream will take the
#: chunking as it stands — find-vectors sizes its ghost blocks to 100 MB, so it
#: rechunks, which is the one move the aligned read exists to avoid.
NAV_CHUNK_BYTES = 100 * 1024 ** 2

#: Rounding this far from a whole pixel means the integer alignment was the
#: better of two near-equal choices, which the user should hear about rather
#: than have to go looking for (see :mod:`spyde.multiangle.model` on residuals).
_COIN_TOSS_RESIDUAL = 0.4


def _check_members(members, paths) -> None:
    """Every member must be one 4-D dataset on the same grid as the others.

    Refused here, with the offending file named, rather than several steps
    later: a member of a different shape aligns to a different composed shape,
    and the failure then surfaces as a shape mismatch between arrays that no
    longer mention which file either came from.
    """
    for member, path in zip(members, paths):
        if member.data.ndim != 4:
            raise ValueError(
                f"{os.path.basename(path)} is {member.data.ndim}-D; a "
                "multi-angle member must be 4-D (scan_y, scan_x, ky, kx)")
    shapes = {tuple(int(size) for size in member.data.shape)
              for member in members}
    if len(shapes) != 1:
        raise ValueError(
            "the members must share one scan and detector shape; got "
            f"{sorted(shapes)}")


def _member_overviews(members):
    """``(images, patterns)`` — each member's real-space image and mean pattern.

    These are the two images the alignment is solved on, and the only reduction
    of a member this loader performs. Both of a member's are asked for in ONE
    call so the pass over the file serves them together instead of once each.
    Reductions stream chunk by chunk, so nothing here holds a member in memory.
    """
    import dask.array as da

    reductions = []
    for member in members:
        data = member.data
        reductions.append(data.sum(axis=(-2, -1), dtype=np.float64))
        reductions.append(data.mean(axis=(0, 1), dtype=np.float64))
    computed = da.compute(*reductions)
    return ([np.asarray(value) for value in computed[0::2]],
            [np.asarray(value) for value in computed[1::2]])


def _solve_model(paths, tilts, azimuths, images, patterns, reference: int):
    """Both alignments solved, packaged as a :class:`MultiAngleModel`."""
    from spyde.multiangle import (
        MultiAngleModel, assign_shells, solve_real_space, solve_reciprocal,
    )

    tilts = np.asarray(tilts, dtype=np.float64)
    nav_offsets, nav_residuals = solve_real_space(
        np.stack(images), reference=reference)
    dp_offsets, dp_residuals = solve_reciprocal(
        np.stack(patterns), reference=reference)
    return MultiAngleModel(
        paths=list(paths),
        tilts=tilts,
        azimuths=np.asarray(azimuths, dtype=np.float64),
        shell_ids=assign_shells(tilts),
        nav_offsets=nav_offsets,
        dp_offsets=dp_offsets,
        reference=reference,
        nav_residuals=nav_residuals,
        dp_residuals=dp_residuals,
        provenance={"solved_by": "Session.open_multiangle"},
    )


def composed_nav_chunk(members, target_bytes: int = NAV_CHUNK_BYTES) -> int:
    """Scan positions per axis in one composed navigation chunk.

    Derived from what a composed FRAME weighs rather than fixed, because that
    is what changes between acquisitions: the same 32 that is 134 MB of
    256 x 256 uint16 patterns is 1.04 GB of 507 x 501 uint32 ones, once the
    detector is bigger and summing four uint16 members has promoted the sum to
    uint32. A chunk is held whole by everyone who touches it, so its SIZE is
    the quantity to hold steady.

    ONE number for the acquisition, not one per member: the members share the
    common grid, and :func:`~spyde.multiangle.compose.member_nav_chunks` needs
    their crops to keep landing on it.

    Nothing here is an alignment constraint — a frame-chunked store addresses
    every frame separately, so any value splits no stored chunk. Falls back to
    :data:`NAV_CHUNK` when a member does not say what its frames are.
    """
    from spyde.multiangle.compose import sum_dtype

    # A signal or the bare array: the composition is handed both, and which one
    # it is says nothing about how big a chunk should be.
    arrays = [getattr(member, "data", member) for member in members]
    arrays = [array for array in arrays
              if getattr(array, "shape", None) is not None
              and getattr(array, "dtype", None) is not None
              and len(array.shape) >= 2]
    if not arrays:
        return NAV_CHUNK
    shapes = [tuple(int(s) for s in array.shape[-2:]) for array in arrays]
    dtypes = [array.dtype for array in arrays]
    frame_bytes = (shapes[0][0] * shapes[0][1]
                   * np.dtype(sum_dtype(dtypes[0], len(dtypes))).itemsize)
    if frame_bytes <= 0:
        return NAV_CHUNK
    # Per AXIS, so the chunk is nav_chunk**2 frames.
    return max(1, int(math.isqrt(max(1, int(target_bytes // frame_bytes)))))


def _aligned_arrays(members, model):
    """Every member on the common grid, aligned as it is READ where it can be.

    Three routes, tried in order, all giving identical values:

    * a MEMMAP-backed member (`.mrc`, `.de5`, raw) is re-expressed as an aligned
      read of its own file — each block reads exactly the frames belonging at
      its output positions, so the margin outside the common region is never
      read at all;
    * a store compressed FRAME BY FRAME is rebuilt on a grid whose blocks line
      up with the other members' after cropping;
    * anything else is cropped after loading, which is equally correct but
      leaves the composed navigation grid the union of the members'.

    The first two put every member on ONE grid, so the composed array carries no
    rechunk layer — nothing is moved to make the members agree.
    """
    from spyde.multiangle.compose import aligned_member
    from spyde.multiangle.load import (
        aligned_from_signal, load_aligned_store_member,
    )

    nav_chunk = composed_nav_chunk(members)
    aligned = []
    for index, member in enumerate(members):
        # Checked against None, never for truthiness: these are arrays, and
        # `a or b` asks a multi-element array whether it is true, which raises.
        array = aligned_from_signal(member, model, index, nav_chunk=nav_chunk)
        if array is None:
            array = load_aligned_store_member(
                member, model, index, nav_chunk=nav_chunk)
        if array is None:
            array = aligned_member(member, model, index)
        aligned.append(array)
    return aligned


def _stack_navigator(images, model):
    """The 5-D root's navigator: ONE PLANE PER MEMBER, on the common grid.

    The tree builds its navigator CHAIN from the root's navigation dimension and
    refuses a navigator of any other shape, so the root's navigator carries the
    angle axis. What goes along that axis decides whether the angle navigator
    works at all: repeating one composed overview makes every plane identical,
    and the 1-D angle line it reduces to is then a flat constant — a navigator
    with nothing to navigate. Per-member planes make the line each member's own
    total intensity.
    """
    import hyperspy.api as hs

    from spyde.multiangle.signals import member_navigator_planes

    return hs.signals.BaseSignal(member_navigator_planes(images, model))


def _open_angle_ring(tree) -> None:
    """Show the acquisition's angles as a ring beside the data.

    The 1-D angle navigator can be dragged along but says nothing about WHERE
    each angle sits; a multi-shell acquisition is a set of rings, and a missing
    angle is a gap in one. Failing to draw it must not fail the load — the
    dataset is open and usable without it.
    """
    try:
        from spyde.actions.multiangle_navigator import open_multiangle_navigator

        open_multiangle_navigator(tree.session, tree)
    except Exception as e:
        log.debug("opening the multi-angle ring failed: %s", e)


def _attach_node(tree, parent_signal, new_signal, name: str) -> None:
    """Record a composed node on *tree* and let the open windows display it.

    ``add_node`` rather than ``add_transformation``: the composed signal is
    already built, and there is no method on the parent that produces it. Only
    ``add_transformation`` registers the node's PlotState, so that half is done
    here — without it the Workflow panel offers a node no window can show.

    ``local=True`` because a composed frame is one frame from each member added
    together, which is exactly what the locality gate asks, and what lets the
    multi-angle reader serve the node instead of the composed dask graph.
    """
    tree.add_node(parent_signal, new_signal, name, local=True)
    tree.update_plot_states(new_signal)


def _display(tree, signal) -> None:
    """Show *signal* in every signal window of *tree*."""
    from spyde.actions.lifecycle import show_tree_node

    for plot in list(tree.signal_plots):
        show_tree_node(plot, tree, signal)


class MultiAngleLoaderMixin:
    """``open_multiangle`` — the Session half of the multi-angle loader.

    Uses ``self.load_aligned`` / ``self._add_signal`` / ``self._await_dask`` /
    ``self._dispatch_to_main`` from the final Session.
    """

    def open_multiangle(self, paths, tilts, azimuths, *,
                        reference: int = 0, reader_options=None) -> None:
        """Open N members of a multi-angle acquisition as one signal tree.

        Parameters
        ----------
        paths
            One dataset per member, in member order.
        tilts, azimuths
            The tilt magnitude and azimuth each member was taken at, degrees.
            Both must be as long as *paths*: they are what the shells are
            grouped by, and an acquisition with them missing is not one.
        reference
            Index of the member the others are aligned to.
        reader_options
            Extra arguments for opening each member, e.g.
            ``{"navigation_shape": (nx, ny)}`` for a binary format that does
            not carry its scan shape and has no sidecar describing it.

        Returns immediately; the load, the alignment solve and the composition
        run on a daemon thread.
        """
        paths = [path for path in (paths or []) if path]
        tilts = list(tilts or [])
        azimuths = list(azimuths or [])
        if len(paths) < 2:
            emit_error("A multi-angle acquisition needs at least two members.")
            return
        if len(tilts) != len(paths) or len(azimuths) != len(paths):
            emit_error(
                f"Multi-angle needs a tilt and an azimuth for each of the "
                f"{len(paths)} members; got {len(tilts)} tilts and "
                f"{len(azimuths)} azimuths.")
            return
        unreadable = [path for path in paths
                      if _path_ext(path) not in SUPPORTED_EXTS
                      or not _is_supported_dataset_path(path)]
        if unreadable:
            emit_error(
                "Cannot open — missing or unsupported: "
                f"{', '.join(os.path.basename(p) for p in unreadable)}")
            return
        if not 0 <= int(reference) < len(paths):
            emit_error(f"Reference member {reference} is outside "
                       f"0..{len(paths) - 1}.")
            return

        ipc.emit({"type": "loading", "busy": True,
                  "text": f"Opening {len(paths)} multi-angle members…"})
        emit_status(f"Opening {len(paths)} multi-angle members…")
        threading.Thread(
            target=self._load_multiangle_thread,
            args=(paths, tilts, azimuths, int(reference),
                  dict(reader_options or {})),
            daemon=True,
            name=f"load-multiangle-{len(paths)}",
        ).start()

    def _load_multiangle_thread(self, paths, tilts, azimuths,
                                reference: int, reader_options=None) -> None:
        try:
            self._await_dask()
            # Format registrations and reader fast paths must land before the
            # first read, for the reason _load_file_thread gives.
            from spyde.backend.heavy_imports import ensure_heavy_imports
            ensure_heavy_imports()

            members = [self._load_member(path, reader_options)
                       for path in paths]
            _check_members(members, paths)

            emit_status(f"Aligning {len(members)} multi-angle members…")
            images, patterns = _member_overviews(members)
            model = _solve_model(
                paths, tilts, azimuths, images, patterns, reference)

            compose_multiangle_tree(self, members, model, images)
        except Exception as e:
            ipc.emit({"type": "loading", "busy": False, "text": ""})
            log.exception("opening a multi-angle acquisition failed")
            emit_error(f"Failed to open the multi-angle acquisition: {e}")

    def _load_member(self, path: str, reader_options=None):
        """One member, lazy and chunked to span whole diffraction patterns."""
        signal = self.load_aligned(path, **(reader_options or {}))
        if isinstance(signal, list):
            if len(signal) != 1:
                raise ValueError(
                    f"{os.path.basename(path)} holds {len(signal)} signals; a "
                    "multi-angle member must be a single 4-D dataset")
            signal = signal[0]
        return signal


def compose_multiangle_tree(session, members, model, images):
    """Everything after the model exists: compose, open, attach, display, ring.

    The whole-acquisition loader (:meth:`MultiAngleLoaderMixin.open_multiangle`)
    and the staged loader dialog (``maped_commit`` in
    :mod:`spyde.backend._session_multiangle_loader`) differ only in how the
    model was arrived at — one solve in one pass, or three stages the user
    inspected in between. From here on they must not differ at all, so this is
    one function rather than two that look alike on the day they are written.

    Call from a worker thread. Returns ``(tree, summed_signal)``.

    Parameters
    ----------
    members
        The loaded lazy member signals, in model order.
    model
        The solved :class:`~spyde.multiangle.model.MultiAngleModel`.
    images
        Each member's real-space image — the same ones the real-space solve
        registered. They become the 5-D root's per-member navigator planes, so
        drawing it costs nothing further.
    """
    from spyde.multiangle.signals import (
        build_stack_signal, build_summed_signal, shell_titles,
    )

    aligned = _aligned_arrays(members, model)
    stack = build_stack_signal(aligned, members, model)
    summed = build_summed_signal(aligned, members, model)
    titles = shell_titles(model)
    # A single-shell acquisition's one shell IS the summed node; a child
    # duplicating its parent would be a node the user cannot tell apart.
    shells = [] if model.n_shells < 2 else [
        (titles[shell_id],
         build_summed_signal(aligned, members, model,
                             member_indices=member_indices,
                             title=titles[shell_id]))
        for shell_id, member_indices in model.shells.items()
    ]
    navigator = _stack_navigator(images, model)

    ipc.emit({"type": "loading", "busy": False, "text": ""})
    # The navigator depends on every member rather than on one path, so the
    # per-file sidecar cache has nothing valid to key it on.
    tree = session._add_signal(
        stack, source_path=model.paths[model.reference],
        navigator_override=navigator, enable_nav_sidecar=False)
    # The tree names its own root "root"; here the root IS the aligned stack,
    # and the Workflow panel is where the user picks between it and the sums.
    tree.root_node.name = "Aligned Stack"

    _attach_node(tree, stack, summed, "Summed")
    for title, shell_signal in shells:
        _attach_node(tree, summed, shell_signal, title)

    # Both on the main thread, and in this order: the ring reads the tree's
    # navigation selectors, which the display switch is what guarantees are in
    # place.
    session._dispatch_to_main(lambda: _display(tree, summed))
    session._dispatch_to_main(lambda: _open_angle_ring(tree))
    _report_multiangle(model, summed)
    return tree, summed


def _sum_over_angles(stack, indices, title):
    """The stack's chosen angles added, as a 4-D node.

    The saved stack IS the aligned members, so a sum over its leading axis is
    the same array the composition produced — no members, no offsets and no
    re-reading of anything are needed to get the Summed node back.

    The node carries a RECIPE, which is what makes looking at it bearable. A
    composed node displayed through its dask graph computes the whole
    enclosing block to show one frame; through the recipe it reads one frame
    per angle and adds them (CLAUDE.md, Live-Display 3 — measured at 2196 ms
    against 0.55 ms on a comparable node). The "members" here are the stack's
    own per-angle slices and the offsets are ZERO, because a saved stack is
    already aligned and shifting it again would serve the wrong position.
    """
    import numpy as np

    from spyde.multiangle.compose import sum_dtype
    from spyde.multiangle.model import MultiAngleModel, model_from_metadata
    from spyde.multiangle.recipe import MultiAngleRecipe, attach_recipe

    indices = [int(index) for index in indices]
    data = stack.data
    wanted = sum_dtype(data.dtype, len(indices))
    # `dtype=` on the sum as well as the cast: a reduction over an unsigned
    # integer array promotes to uint64 on its own, and the cast alone left a
    # freshly composed uint32 node reopening as uint64 — twice the bytes per
    # frame, and a different dtype from the one that was saved.
    summed = data[indices].astype(wanted).sum(axis=0, dtype=wanted)

    signal = stack._deepcopy_with_new_data(summed)
    # Drop the angle axis: it described planes that have just been added up,
    # and leaving it makes a 4-D array claim a 5-D calibration.
    angle = signal.axes_manager.navigation_axes[-1]
    signal.axes_manager.remove(angle)
    signal.metadata.set_item("General.title", title)
    # The sum must not keep the STACK's type: it would be recognised as a
    # stack on reopening and the rebuild would raise on a 4-D array. With no
    # member type recorded (members opened from a raw format carry none) it
    # is a diffraction signal, which is what the stack's own class extends.
    member_type = stack.metadata.get_item(
        f"{MULTIANGLE_METADATA}.member_signal_type", "")
    signal.set_signal_type(member_type or "electron_diffraction")

    recorded = model_from_metadata(stack)
    if recorded is None:
        return signal
    # Zeroed, not the recorded offsets: those describe where each member sat
    # in its OWN file, and were applied when the stack was written. Re-using
    # them would shift an already-aligned stack a second time, which serves a
    # real frame from the wrong scan position — wrong and invisible.
    members = len(recorded.paths)
    aligned = MultiAngleModel(
        paths=list(recorded.paths), tilts=recorded.tilts,
        azimuths=recorded.azimuths, shell_ids=recorded.shell_ids,
        nav_offsets=np.zeros((members, 2), dtype=np.int64),
        dp_offsets=np.zeros((members, 2), dtype=np.int64),
        reference=int(recorded.reference))
    # `inav` takes the navigation axes in display order, fastest first, so the
    # angle is LAST: (x, y, angle).
    planes = tuple(stack.inav[:, :, index] for index in indices)
    return attach_recipe(signal, MultiAngleRecipe(
        members=planes, model=aligned, has_angle_axis=False,
        dtype=summed.dtype, member_indices=tuple(indices), stack=stack))


def rebuild_multiangle_tree(session, signal, source_path=None):
    """Put the multi-angle tree back around a stack read from a file.

    A saved acquisition is otherwise just a 5-D array: the alignment survives
    and everything built on it does not, so reopening one gave no Summed node,
    no shells and no angle ring. Nothing extra is needed to rebuild them — the
    sums are reductions over the stack's own leading axis, and the tilts,
    azimuths, shells and reference come from `Acquisition.multiangle`.

    Returns the tree, or None when *signal* is not a saved stack.
    """
    from spyde.signals.multiangle import is_multiangle_stack

    if not is_multiangle_stack(signal):
        return None
    # `get_item`, not `get`: the metadata is a DictionaryTreeBrowser and has
    # no mapping interface, so `.get` raises and the caller's except turns a
    # whole acquisition back into a bare array.
    recorded = signal.metadata
    members = int(recorded.get_item(f"{MULTIANGLE_METADATA}.n_members"))
    shell_ids = [int(value) for value in recorded.get_item(
        f"{MULTIANGLE_METADATA}.shell_ids", [0] * members)]
    tilts = [float(value) for value in recorded.get_item(
        f"{MULTIANGLE_METADATA}.tilts", [0.0] * members)]

    summed = _sum_over_angles(signal, range(members), "Summed")
    # One shell IS the summed node; a child duplicating its parent is a node
    # the user cannot tell apart. Same rule the composition follows.
    by_shell: dict[int, list[int]] = {}
    for index, shell in enumerate(shell_ids):
        by_shell.setdefault(shell, []).append(index)
    shells = [] if len(by_shell) < 2 else [
        (f"{tilts[indices[0]]:g}°", _sum_over_angles(
            signal, indices, f"{tilts[indices[0]]:g}°"))
        for shell, indices in sorted(by_shell.items())
    ]

    tree = session._add_signal(signal, source_path=source_path)
    tree.root_node.name = "Aligned Stack"
    _attach_node(tree, signal, summed, "Summed")
    for title, shell_signal in shells:
        _attach_node(tree, summed, shell_signal, title)
    session._dispatch_to_main(lambda: _display(tree, summed))
    session._dispatch_to_main(lambda: _open_angle_ring(tree))
    emit_status(f"Multi-angle acquisition: {members} angles, "
                f"{len(by_shell)} shell{'s' if len(by_shell) != 1 else ''}")
    return tree


def _report_multiangle(model, summed) -> None:
    """Say what was opened, and whether the rounding was a close call."""
    emit_status(
        f"Multi-angle: {model.n_members} members in {model.n_shells} "
        f"shell{'s' if model.n_shells != 1 else ''} → "
        f"{tuple(summed.axes_manager.navigation_shape)} nav × "
        f"{tuple(summed.axes_manager.signal_shape)} signal")
    worst = max((float(np.max(np.abs(values)))
                 for values in (model.nav_residuals, model.dp_residuals)
                 if values is not None), default=0.0)
    if worst > _COIN_TOSS_RESIDUAL:
        emit_status(
            f"Multi-angle: alignment rounded by up to {worst:.2f} px — "
            "the members do not land on whole pixels.")
