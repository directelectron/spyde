"""
_session_multiangle_loader.py — the STAGED multi-angle loader.

:meth:`~spyde.backend._session_multiangle.MultiAngleLoaderMixin.open_multiangle`
opens an acquisition in one shot: load every member, solve both alignments,
compose, show. That is right when the angles are already known and the solve
can be trusted. It is the wrong shape when either is in doubt, because the only
feedback is the finished dataset — a member at the wrong tilt, or a reciprocal
solve that locked onto the wrong feature, is discovered after all the work and
fixed by starting again.

So this module is the same job in stages, with the user between them:

    Load datasets  →  Align real space  →  Align reciprocal space  →  Commit

Nothing is composed until Commit, and each stage can be re-run with different
parameters without repeating the one before it.

**One message, always the whole snapshot.** Every action here answers with a
``maped_state`` — the complete loader state, not a patch. The renderer
therefore never maintains its own copy to keep in step, which is the class of
bug a staged dialog otherwise grows: a stage that forgets to announce one field
leaves the screen describing a state the backend has left behind. The cost is a
few hundred bytes per click, which is nothing.

An action that is not instantaneous — adding files, either solve, the commit —
answers TWICE: immediately with ``busy: true``, and again when it lands. The
LAST snapshot is always the authoritative one. The renderer's controls move
only when this module echoes, so a slow stage that said nothing would read as a
frozen dialog.

**A commit reports its outcome even if the dialog has gone.** The renderer
closes the dialog on commit, so the usual "drop a superseded result" rule would
make a FAILED commit silent. See :func:`_report_commit`.

**Opening a member reads no frames.** Probing is a lazy open: the shape, dtype,
size and any angle metadata come from the header. This matters more here than
in the one-shot loader, because a user assembling an acquisition adds and
removes files repeatedly, and a probe that read data would make that unusable
on real members.

**A member is always opened UNAIDED first.** A raw binary format records the
frames and the detector but no scan grid, so the dialog has a scan-shape field
(``maped_set_scan_shape``) for those files. It is a FALLBACK: a Direct Electron
``.mrc`` ships an ``_info.txt`` sidecar naming the scan grid and the reader
finds it on its own, and a number typed into a dialog must never override one
the acquisition recorded. See :func:`_probe_member`.

**The per-member reductions are computed once and kept.** The two solves need
one real-space image and one mean diffraction pattern per member, and computing
those IS the expensive part of this loader on real data — everything else is
header reads and small-array arithmetic. They are cached on the loader state,
so re-running a solve with different parameters costs the solve alone. Neither
reduction materialises a member (CLAUDE.md memory safety): the image is a
streaming sum over the detector axes, and the mean pattern is taken over a
sample of a few hundred scan positions.

**The dialog is a set of tableaux, so the state carries pictures.** A row of
file names cannot answer the question the user actually has — "is this member
the same region of the same sample as that one?" — and a renderer cannot read
an array. So every image this loader decides something from is also published
as a small PNG data URI: each member's real-space image (``members[].preview``)
and, for the reciprocal stage, the four corner diffraction sums
(``reciprocal.corners``). They are thumbnails, normalised for looking at; the
numbers are still solved from the arrays.

**A tableau is filled before anything is solved.** It is what the user looks at
to DECIDE whether to run a stage, so it cannot be the reward for having run
one. The probe therefore hands straight over to
:func:`_start_preview_fill`, which reads the members and fills both grids one
member at a time. On a real acquisition that pass is the dominant cost of this
whole dialog, and there is no version of "show me the images" that avoids it —
so it is made visible (a snapshot per member, ``busy`` throughout) and
interruptible (a generation guard checked between members) rather than hidden.
The reductions it makes are the ones the solves want, in the same caches, so
the work is moved earlier rather than added.

**Real space is aligned from a VIRTUAL IMAGE.** A Direct Electron ``.mrc``
ships its detector's virtual images beside the data and the reader hands them
over in ``metadata._HyperSpy.navigators``, so the honest default is to align on
the picture the microscope already formed rather than on a sum over the whole
detector. ``virtual_image`` names which one; ``None`` means compute one, which
is that whole-detector sum. Only names EVERY member carries are offered — a
virtual image one member has cannot align a set.

**Reciprocal space is aligned from the SCAN CORNERS.** Summing the patterns
over four small corner blocks and fitting a plane through the four beam
positions costs a fraction of a percent of the acquisition, where measuring
every pattern's beam costs all of it. It is the same trick the DPC wizard uses
to seed its field, out of the same module (:mod:`spyde.corners`). Each corner
has its OWN extent, because a scan whose top-left sits on vacuum and whose
bottom-right sits on the sample needs two different boxes.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from de_shell import ipc
from de_shell.ipc import emit_error, emit_status

from spyde.actions.lifecycle import bump_generation, is_current, run_on_worker
from spyde.backend._session_files import (
    SUPPORTED_EXTS, _dataset_size_bytes, _is_supported_dataset_path, _path_ext,
)
from spyde.backend._session_multiangle import (
    _check_members, compose_multiangle_tree,
)
from spyde.corners import (
    CORNER_NAMES, DEFAULT_CORNER_FRACTION, corner_slice, plane_through,
)

log = logging.getLogger(__name__)

#: Scan positions the mean diffraction pattern is averaged over. A direct beam
#: is the brightest thing on the detector at every position, so a few hundred
#: patterns locate it as well as tens of thousands do — and the two reductions
#: share one pass over the member, so the sample costs arithmetic, not reads.
PATTERN_SAMPLE_POSITIONS = 256

#: Metadata fields a member may carry its own angles in, most specific first.
#: Deliberately short: a stage tilt is not a precession tilt, and guessing from
#: a field that happens to have "tilt" in its name would put a confident wrong
#: number in front of the user, which is worse than the zero they can see is a
#: placeholder.
_TILT_FIELDS = ("Spyde.multiangle.tilt",
                "Acquisition_instrument.TEM.rocking_angle",
                "Acquisition_instrument.TEM.precession_angle")
_AZIMUTH_FIELDS = ("Spyde.multiangle.azimuth",
                   "Acquisition_instrument.TEM.precession_azimuth")

#: Arguments :func:`spyde.drift.solve_translation` takes that a caller has any
#: business setting from a dialog. Anything else in a ``params`` payload is
#: ignored rather than forwarded — an unknown keyword would raise inside the
#: worker, where the user sees a failed stage instead of a rejected control.
#:
#: Only ``max_shift`` has a field on the dialog. It is the guard against a
#: periodic lattice locking the correlation onto the wrong translation, which
#: is the failure a user has to be able to reach in; the rest are here so a
#: script can reach them without a second entry point.
_SOLVER_PARAMETERS = ("upsample", "max_shift", "min_shift", "roi", "apodize",
                      "normalize", "reject_outliers")

#: How the reciprocal stage can be solved. ``corners`` sums the patterns over
#: the four corners of each scan and fits a plane through the beam positions it
#: finds there; ``beam`` fits the beam in the member's MEAN pattern;
#: ``correlate`` registers the mean patterns against one another, which is the
#: answer when there is no sharp direct beam to fit.
RECIPROCAL_METHODS = ("corners", "beam", "correlate")

#: Number of scan corners the ``corners`` method measures — four, and named in
#: :data:`spyde.corners.CORNER_NAMES`. Here so the per-corner lists on a member
#: are built from the same count the slicing uses.
N_CORNERS = len(CORNER_NAMES)

#: Longest edge of a tableau thumbnail, in pixels. Small on purpose: these are
#: panels in a grid, not data — a full-size real-space image per member would
#: put megabytes through a line-oriented stdout protocol on every snapshot, and
#: every action here answers with the WHOLE snapshot.
PREVIEW_MAX_EDGE = 128

#: Name of the window the real-space stage opens to show its own result.
ALIGNED_WINDOW_TITLE = "Real-space alignment"

#: Percentiles a thumbnail is stretched between, the same robust range the
#: progressive navigator fill uses: one saturated pixel on an electron detector
#: is ordinary, and a min/max stretch would let it black out the whole panel.
PREVIEW_PERCENTILES = (2.0, 98.0)

#: Where a reader publishes the virtual images it found beside a dataset. A
#: Direct Electron ``.mrc`` has them as sidecar files and rsciio loads them
#: here under ``_sig_<name>``; HyperSpy drops that prefix on the way in, so the
#: names a user sees are the detector's own.
VIRTUAL_IMAGE_ITEM = "_HyperSpy.navigators"

#: The defaults every host starts from. ``max_shift`` is
#: :func:`spyde.drift.solve_translation`'s own, repeated here because the
#: dialog shows it and a field whose default disagrees with the backend's wins
#: silently. ``method`` is ``corners`` because it is the fast one: it reads a
#: fraction of a percent of each member where the others read a sample of every
#: one of them.
DEFAULTS = dict(method="corners", beam_method="center_of_mass",
                half_square_width=0, max_shift=32.0,
                corner_extent=DEFAULT_CORNER_FRACTION)


# ── the state ────────────────────────────────────────────────────────────────

@dataclass
class LoaderMember:
    """One candidate member: what the probe learned, plus what the user set."""

    path: str
    name: str
    size_bytes: int = 0
    scan_shape: tuple[int, ...] | None = None
    detector_shape: tuple[int, ...] | None = None
    dtype: str | None = None
    tilt: float = 0.0
    azimuth: float = 0.0
    #: True while *azimuth* is this loader's evenly-spread guess rather than a
    #: value from the file or from the user. Only auto azimuths are re-spread
    #: when the member list changes; a number someone chose is never moved.
    azimuth_is_auto: bool = True
    #: Why this file cannot be used, from the probe. Separate from ``error``
    #: because that also carries "does not match the others", which is a fact
    #: about the SET and has to be recomputed whenever the set changes.
    load_error: str | None = None
    error: str | None = None
    #: True when the file records frames and a detector but no scan grid, so
    #: it can only be read as a member once one is supplied. Marks the rows
    #: ``maped_set_scan_shape`` re-probes — and the ones it must UNDO when the
    #: scan shape is cleared, which is why it stays set after a successful
    #: retry rather than being cleared by it.
    needs_scan_shape: bool = False
    signal: Any = None
    image: np.ndarray | None = None
    pattern: np.ndarray | None = None
    #: Virtual images this member's file carries, by name, filtered to the ones
    #: whose shape is actually this member's scan grid — a picture of anything
    #: else cannot be registered against the other members.
    virtual_images: list[str] = field(default_factory=list)
    #: The real-space thumbnail, and which virtual image it was made from —
    #: kept so a changed selection rebuilds the picture and an unchanged one
    #: does not. The source is ``None`` in two distinguishable cases: no
    #: preview at all (``preview`` is ``None`` too), or a preview made from the
    #: COMPUTED image, which is what a ``None`` selection means.
    preview: str | None = None
    preview_source: str | None = None
    #: How much of the scan each corner block spans, as a fraction of its own
    #: axis. FOUR of them, in :data:`spyde.corners.CORNER_NAMES` order, because
    #: the corners of a scan are not interchangeable — one may sit on vacuum
    #: and another on the sample.
    corner_extents: list[float] = field(
        default_factory=lambda: [DEFAULT_CORNER_FRACTION] * N_CORNERS)
    #: The diffraction pattern summed over each corner block, and its
    #: thumbnail. ``None`` where it has not been computed for the CURRENT
    #: extent — changing one corner's extent clears that corner alone.
    corner_patterns: list[np.ndarray | None] = field(
        default_factory=lambda: [None] * N_CORNERS)
    corner_previews: list[str | None] = field(
        default_factory=lambda: [None] * N_CORNERS)

    def forget_corners(self, corner: int | None = None) -> None:
        """Drop one corner's measurement, or all four."""
        for index in (range(N_CORNERS) if corner is None else [int(corner)]):
            self.corner_patterns[index] = None
            self.corner_previews[index] = None

    @property
    def pending(self) -> bool:
        """Queued, not yet probed. Its row shows a name and nothing else."""
        return self.signal is None and self.load_error is None

    @property
    def usable(self) -> bool:
        return self.error is None and self.signal is not None


@dataclass
class LoaderStage:
    """One solved alignment, or the absence of one."""

    offsets: np.ndarray | None = None
    residuals: np.ndarray | None = None

    @property
    def solved(self) -> bool:
        return self.offsets is not None

    @property
    def per_member_residuals(self) -> list[float] | None:
        """ONE number per member: how far from a whole pixel that member's
        answer was, on its worse axis.

        The solve measures a residual per AXIS, and the state keeps both — the
        model records them. What a reader needs beside a member's name is a
        single "was this a coin toss?", and the worse axis is that: an answer
        0.47 px out on one axis and 0.01 on the other was a coin toss, and a
        mean or a norm would both round that reading off.
        """
        if self.residuals is None:
            return None
        return [float(np.max(np.abs(row))) for row in self.residuals]

    @property
    def max_residual(self) -> float | None:
        if self.residuals is None:
            return None
        return float(np.max(np.abs(self.residuals))) if self.residuals.size else 0.0

    def as_message(self) -> dict:
        return {
            "solved": self.solved,
            "offsets": (None if self.offsets is None
                        else [[int(dy), int(dx)] for dy, dx in self.offsets]),
            "residuals": self.per_member_residuals,
            "max_residual": self.max_residual,
        }


@dataclass
class MultiAngleLoaderState:
    """Everything the loader dialog is showing, and nothing else.

    Lives on the Session for as long as the dialog is open
    (``session._multiangle_loader``). ``maped_close_loader`` drops it, which is
    also what stops any solve still running: a worker checks its generation
    against the state object it captured, and close bumps that first.
    """

    members: list[LoaderMember] = field(default_factory=list)
    #: Index of the member the others align to — ``None`` until there is one
    #: that can be. Every offset is a correction TOWARDS it, so it is part of
    #: what an offset array means, not a display preference.
    reference: int | None = None
    real: LoaderStage = field(default_factory=LoaderStage)
    reciprocal: LoaderStage = field(default_factory=LoaderStage)
    busy: bool = False
    message: str = ""
    #: Extra arguments for opening every member, whatever they are.
    reader_options: dict = field(default_factory=dict)
    #: ``(x, y)`` scan grid for members whose file does not record one, or
    #: ``None``. ONE for the acquisition, not one per member: the members of an
    #: acquisition share a scan grid, and a per-row field would be N copies of
    #: the same number for the user to keep in step.
    scan_shape: tuple[int, int] | None = None
    #: Which virtual image the real-space stage aligns on, by name, or ``None``
    #: to compute one. ONE for the acquisition, for the same reason the scan
    #: shape is: registering member 0's bright field against member 1's dark
    #: field would be comparing two different pictures and calling the
    #: difference drift.
    virtual_image: str | None = None
    #: Where the reciprocal stage looks for the zero beam, as
    #: ``{"cy": …, "cx": …, "half": …}`` in DETECTOR pixels, or ``None`` for
    #: the whole pattern. ONE for the acquisition, like the scan shape: the
    #: members are the same detector at the same camera length, and a region
    #: per member could be placed on a different reflection in each.
    #:
    #: It exists because the beam finder is a centre of mass, which over a
    #: whole pattern is pulled bodily towards whichever reflections a tilt
    #: happens to excite. Placed on the zero beam it measures the zero beam.
    beam_roi: dict | None = None
    #: The real-space solve's evidence as the DIALOG shows it:
    #: ``{"unaligned": <png>, "aligned": <png>, "gain": float}``, or ``None``.
    #:
    #: In the dialog because the dialog covers the screen — it is a full-screen
    #: modal, so a window opened behind it cannot be looked at or reached,
    #: and the sum aligned against the sum unaligned is the whole reason to
    #: believe a solve. The separate window remains for after the dialog closes.
    real_evidence: dict | None = None
    #: The open aligned-sum window, or ``None``. One per dialog: re-running the
    #: alignment repaints it rather than opening another, and a ✕ on it clears
    #: this through the controller's ``close``.
    aligned_window: Any = None
    #: True while *virtual_image* is this loader's own pick rather than one the
    #: user made. An acquisition that ships virtual images should align on one
    #: without being asked — that is the picture the microscope formed, and
    #: reducing the whole detector to reinvent it is the expensive way to get a
    #: worse one. Once someone chooses, including choosing the computed image,
    #: the pick is theirs and the loader stops moving it.
    virtual_image_is_auto: bool = True

    _real_generation: int = 0
    _reciprocal_generation: int = 0
    _commit_generation: int = 0
    _probe_generation: int = 0
    _preview_generation: int = 0
    #: Held across computing the per-member reductions. Two stages started in
    #: quick succession would otherwise both find the cache empty and both
    #: compute it; serialising them is the whole point, and this lock is never
    #: taken on the navigator's dispatcher, so it cannot wedge the display.
    _reduction_lock: threading.Lock = field(default_factory=threading.Lock)
    #: Held across probing, so two file drops in quick succession fill their
    #: rows one batch after the other. The ROWS are appended synchronously, in
    #: call order, so the list order never depends on which worker wins.
    _probe_lock: threading.Lock = field(default_factory=threading.Lock)
    #: Held across summing the corner blocks, for the reason
    #: :attr:`_reduction_lock` is held across the reductions: the reciprocal
    #: solve and an extent change can both want the same corner, and computing
    #: it twice is the one thing the cache exists to stop.
    _corner_lock: threading.Lock = field(default_factory=threading.Lock)
    _corner_generation: int = 0

    # ── derived ──────────────────────────────────────────────────────────────

    @property
    def shell_ids(self) -> dict[int, int]:
        """``{member index: shell id}``, for the members that can be used.

        A member that failed to open, or is still being probed, is in NO shell
        — it has no tilt anyone has vouched for, and putting it in one would
        have it counted in a shell's membership and its averages.
        """
        from spyde.multiangle import assign_shells

        usable = [index for index, member in enumerate(self.members)
                  if member.usable]
        shells = assign_shells([self.members[index].tilt for index in usable])
        return {index: int(shell) for index, shell in zip(usable, shells)}

    @property
    def available_virtual_images(self) -> list[str]:
        """The virtual images EVERY usable member carries, by name.

        An intersection rather than a union: the real-space solve registers the
        members' images against one another, so a name only some of them have
        would offer an alignment that cannot be made. Empty while there is no
        usable member — there is nothing yet for a name to be common to.
        """
        usable = [member for member in self.members if member.usable]
        if not usable:
            return []
        common = set(usable[0].virtual_images)
        for member in usable[1:]:
            common &= set(member.virtual_images)
        return sorted(common)

    @property
    def can_commit(self) -> bool:
        return (len(self.members) >= 2 and self.reference is not None
                and self.real.solved and self.reciprocal.solved)

    def cancel_alignments(self) -> None:
        """Make a solve still running drop its result on arrival."""
        bump_generation(self, "_real_generation")
        bump_generation(self, "_reciprocal_generation")

    def cancel_in_flight(self) -> None:
        """Make every worker still running drop its result on arrival."""
        self.cancel_alignments()
        bump_generation(self, "_commit_generation")
        bump_generation(self, "_probe_generation")
        bump_generation(self, "_corner_generation")
        bump_generation(self, "_preview_generation")

    def invalidate_real(self) -> None:
        """Forget the real-space solve alone — what it measured has changed.

        Choosing a different virtual image is exactly this: the members are the
        same members in the same order, so the reciprocal offsets still mean
        what they meant, but the real-space offsets came from registering a
        different set of pictures and are no longer that answer.
        """
        self.real_evidence = None
        bump_generation(self, "_real_generation")
        self.real = LoaderStage()

    def invalidate_reciprocal(self) -> None:
        """Forget the reciprocal solve alone — its measurement has changed.

        A corner extent is an input to it, in the same sense the virtual image
        is an input to the real-space solve. Dropped whichever method solved
        it: re-solving from the corners is the cheap stage by construction, so
        over-invalidating costs a click and under-invalidating shows a number
        that was measured somewhere else.
        """
        bump_generation(self, "_reciprocal_generation")
        self.reciprocal = LoaderStage()

    def invalidate_alignments(self) -> None:
        """Forget both solves — the member set they describe has changed.

        Both offset arrays are one row per member, in member order, relative to
        one reference. A member added, removed or promoted to reference makes
        them not merely stale but indexed against a list that no longer exists,
        so they would be drawn against the wrong member's name. Dropping them
        is the only honest answer.
        """
        self.cancel_alignments()
        self.real = LoaderStage()
        self.reciprocal = LoaderStage()

    def build_model(self, *, nav_offsets=None, dp_offsets=None):
        """The :class:`~spyde.multiangle.model.MultiAngleModel` to compose from.

        Either offset array can be substituted for the solve's, because both
        are asked for before their stage has finished: the aligned-sum window
        asks real-space questions before the reciprocal stage has run, and the
        real-space stage scores CANDIDATE offsets while choosing between them,
        when the state holds no answer yet. A model cannot be built without
        both arrays.
        """
        from spyde.multiangle import MultiAngleModel, assign_shells

        tilts = np.asarray([member.tilt for member in self.members],
                           dtype=np.float64)
        return MultiAngleModel(
            paths=[member.path for member in self.members],
            tilts=tilts,
            azimuths=np.asarray([member.azimuth for member in self.members],
                                dtype=np.float64),
            shell_ids=assign_shells(tilts),
            nav_offsets=(self.real.offsets if nav_offsets is None
                         else nav_offsets),
            dp_offsets=(self.reciprocal.offsets if dp_offsets is None
                        else dp_offsets),
            reference=self.reference,
            nav_residuals=self.real.residuals,
            dp_residuals=self.reciprocal.residuals,
            provenance={"solved_by": "the multi-angle loader dialog"},
        )


class MultiAngleLoaderStateMixin:
    """The Session's half of the staged loader: it holds the state.

    A class attribute rather than an ``__init__`` assignment, because the
    Session composes its mixins without calling theirs — and "no dialog open"
    is exactly what ``None`` says.
    """

    #: The open loader dialog's state, or None when no dialog is open.
    _multiangle_loader: MultiAngleLoaderState | None = None

    def discard_multiangle_loader(self) -> None:
        """Drop the loader state, so nothing still solving can land.

        The aligned-sum window goes with the dialog: it is the dialog's own
        evidence for a stage nothing is standing in any more. Closed through
        ``_forget_window`` rather than by dropping the reference, so the
        renderer removes it and its figure is released.
        """
        state = self._multiangle_loader
        if state is not None:
            state.cancel_in_flight()
            window = state.aligned_window
            if window is not None:
                self._forget_window(window.window_id)
        self._multiangle_loader = None


# ── the one message back ─────────────────────────────────────────────────────

def state_message(state: MultiAngleLoaderState | None) -> dict:
    """The complete ``maped_state`` snapshot. ``None`` means no dialog is open.

    Everything is a plain Python number or list: a numpy scalar serialises as
    a string through the JSON encoder's ``default=str`` fallback, which reaches
    the renderer as ``"3"`` rather than ``3`` and compares wrong everywhere
    downstream.

    Three things a reader has to be able to rely on:

    * ``offsets`` (a ``[dy, dx]`` pair per member) and ``residuals`` (one
      number per member — see :attr:`LoaderStage.per_member_residuals`) are
      both in ``members`` ORDER, so entry *i* belongs to ``members[i]``. Any
      change to the list drops them rather than leaving them to be drawn
      against the wrong names;
    * ``shells[].members`` and ``reference`` are member ``index`` values;
    * ``reference`` and a member's ``shell`` are ``null`` when there is none —
      no member can be the reference yet, or that member is not in a shell
      because it failed to open or has not been probed.

    The tableaux add pictures to that, and every one of them can be ``null``:
    a ``preview`` before the member has an image to make one from, a
    ``reciprocal.corners`` entry before that corner has been summed. ``null``
    means "not measured yet", never "measured and empty", so a renderer draws a
    placeholder rather than a black panel.
    """
    if state is None:
        return {"type": "maped_state", "members": [], "reference": None,
                "shells": [], "scan_shape": None,
                "virtual_image": None, "available_virtual_images": [],
                "beam_roi": None,
                "real": {**LoaderStage().as_message(), "evidence": None},
                "reciprocal": {**LoaderStage().as_message(), "corners": {}},
                "busy": False, "message": "", "can_commit": False}

    shell_ids = state.shell_ids
    members = [
        {
            "index": index,
            "path": member.path,
            "name": member.name,
            "scan_shape": (None if member.scan_shape is None
                           else [int(v) for v in member.scan_shape]),
            "detector_shape": (None if member.detector_shape is None
                               else [int(v) for v in member.detector_shape]),
            "dtype": member.dtype,
            "size_bytes": int(member.size_bytes),
            "tilt": float(member.tilt),
            "azimuth": float(member.azimuth),
            "shell": shell_ids.get(index),
            "error": member.error,
            "preview": member.preview,
            "virtual_images": list(member.virtual_images),
        }
        for index, member in enumerate(state.members)
    ]

    shells: dict[int, list[int]] = {}
    for index, shell in shell_ids.items():
        shells.setdefault(shell, []).append(index)

    return {
        "type": "maped_state",
        "members": members,
        "reference": (None if state.reference is None
                      else int(state.reference)),
        "shells": [
            {"shell": shell,
             "tilt": min(float(state.members[i].tilt) for i in indices),
             "members": sorted(indices)}
            for shell, indices in sorted(shells.items())
        ],
        "scan_shape": (None if state.scan_shape is None
                       else [int(v) for v in state.scan_shape]),
        "virtual_image": state.virtual_image,
        "available_virtual_images": state.available_virtual_images,
        "beam_roi": (None if not state.beam_roi else
                     {key: float(value)
                      for key, value in state.beam_roi.items()}),
        "real": {**state.real.as_message(),
                 "evidence": state.real_evidence},
        "reciprocal": {**state.reciprocal.as_message(),
                       "corners": _corners_message(state)},
        "busy": bool(state.busy),
        "message": str(state.message),
        "can_commit": bool(state.can_commit),
    }


def _corners_message(state: MultiAngleLoaderState) -> dict:
    """``{"<member index>": {previews, extents}}`` — the corner tableau.

    Keyed by the member's index as a STRING, because this is a JSON object and
    a JSON object has no integer keys. Every member has an entry, including one
    that failed to open: the tableau is a grid with a row per member, and a
    missing key would shift every row below it onto the wrong name.

    ``previews`` and ``extents`` are both four long and both in ROW-MAJOR
    corner order — top left, top right, bottom left, bottom right — which is
    :data:`spyde.corners.CORNER_NAMES` and the index ``maped_set_corner_extent``
    takes. It is a stated contract, not an accident of how the slices happen to
    be built: a renderer laying the four out as a 2 × 2 grid and labelling them
    has no way to notice a reordering, and the picture would simply be wrong.
    """
    return {
        str(index): {"previews": list(member.corner_previews),
                     "extents": [float(extent)
                                 for extent in member.corner_extents]}
        for index, member in enumerate(state.members)
    }


def _emit_state(state: MultiAngleLoaderState | None) -> None:
    ipc.emit(state_message(state))


def _on_main(session, work) -> None:
    """Run *work* on the asyncio main thread (inline when there is no loop)."""
    dispatch = getattr(session, "_dispatch_to_main", None)
    if dispatch is None:
        work()
    else:
        dispatch(work)


# ── the tableau thumbnails ───────────────────────────────────────────────────

def _stretched(array: np.ndarray) -> np.ndarray:
    """*array* as 8-bit grey, stretched over its own robust range.

    Each panel is stretched on ITS OWN percentiles rather than a range shared
    across the tableau. A tableau is there to answer "is this the same region
    of the same sample?", and members at different tilts collect wildly
    different total intensity — a shared range would render the dimmest members
    black and lose the very comparison the grid is for.
    """
    values = np.asarray(array, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size:
        low, high = np.percentile(finite, PREVIEW_PERCENTILES)
        if not high > low:
            low, high = float(finite.min()), float(finite.max())
    else:
        low = high = 0.0
    if not high > low:
        # One value everywhere (or nothing finite at all): mid grey says "this
        # is flat", where a 0-or-255 panel would read as data.
        return np.full(values.shape, 128, dtype=np.uint8)
    scaled = (np.nan_to_num(values, nan=low) - low) / (high - low)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def _thumbnail(array) -> str | None:
    """*array* as a ``data:image/png;base64,…`` thumbnail, or ``None``.

    ``None`` for anything that is not a 2-D picture — a member that has not
    been reduced yet, a corner that has not been summed — so the renderer can
    tell "not measured" from "measured and blank" without a second field.

    Never UPSCALED. A 24 × 28 scan is a 24 × 28 PNG and the renderer stretches
    it; enlarging here would ship interpolated pixels that say nothing the
    small ones did not.
    """
    if array is None:
        return None
    values = np.asarray(array)
    if values.ndim != 2 or values.size == 0:
        return None

    import base64
    import io

    from PIL import Image

    image = Image.fromarray(_stretched(values), mode="L")
    longest = max(image.size)
    if longest > PREVIEW_MAX_EDGE:
        scale = PREVIEW_MAX_EDGE / float(longest)
        image = image.resize(
            (max(1, int(round(image.width * scale))),
             max(1, int(round(image.height * scale)))),
            Image.BOX)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return ("data:image/png;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii"))


# ── the virtual images a member already carries ──────────────────────────────

def _navigators(signal):
    """The reader's virtual images as ``{name: signal}``, or an empty dict."""
    metadata = getattr(signal, "metadata", None)
    try:
        if metadata is None or not metadata.has_item(VIRTUAL_IMAGE_ITEM):
            return {}
        navigators = metadata.get_item(VIRTUAL_IMAGE_ITEM)
        return {str(name): navigators.get_item(name)
                for name in navigators.keys()}
    except Exception as e:
        # A reader is free to put anything under its own metadata node, and a
        # shape we did not expect must cost the user a missing dropdown entry
        # rather than a member that will not open.
        log.debug("reading the virtual images failed: %s", e)
        return {}


def _virtual_image(signal, name: str) -> np.ndarray | None:
    """One named virtual image as an array, or ``None`` if it has none.

    Reads no frames of the dataset: these are small sidecar pictures the reader
    already loaded, one value per scan position.
    """
    navigator = _navigators(signal).get(str(name))
    data = getattr(navigator, "data", None)
    return None if data is None else np.asarray(data)


def _usable_virtual_images(signal, scan_shape) -> list[str]:
    """The names whose picture is this member's SCAN, sorted.

    A file may carry images of other things entirely — an external camera
    frame, a differently binned view. Only one on the scan grid can be
    registered against another member's, so the rest are not offered at all
    rather than offered and then failing inside a solve.
    """
    if scan_shape is None:
        return []
    expected = tuple(int(size) for size in scan_shape)
    names = []
    for name, navigator in _navigators(signal).items():
        data = getattr(navigator, "data", None)
        if data is not None and tuple(int(v) for v in data.shape) == expected:
            names.append(name)
    return sorted(names)


def _member_image(state: MultiAngleLoaderState,
                  member: LoaderMember) -> np.ndarray | None:
    """The real-space image this member is aligned and previewed by.

    The chosen virtual image when there is one, otherwise the computed
    reduction — which may not exist yet, hence ``None``.
    """
    if state.virtual_image is not None:
        return _virtual_image(member.signal, state.virtual_image)
    return member.image


def _needs_tiles(state: MultiAngleLoaderState, member: LoaderMember) -> bool:
    """Whether *member* still has a tableau tile it could be given."""
    if not member.usable:
        return False
    if member.preview is None or member.preview_source != state.virtual_image:
        return True
    return any(pattern is None for pattern in member.corner_patterns)


def _refresh_previews(state: MultiAngleLoaderState) -> None:
    """Rebuild the member thumbnails the current selection has invalidated.

    Cheap to call and safe to call often: a preview already made from the
    selected image is left alone, which is what keeps a snapshot from
    re-encoding every member's PNG on every click.

    A preview the selection has invalidated is DROPPED even when there is
    nothing to replace it with yet. The old picture is a different image of the
    same member, so leaving it up would label a bright field "dark field" for
    as long as the reduction takes — and a blank tile beside a progress message
    is a state the user can read, where a confidently wrong one is not.
    """
    for member in state.members:
        if member.preview is not None \
                and member.preview_source == state.virtual_image:
            continue
        member.preview = _thumbnail(_member_image(state, member))
        member.preview_source = (state.virtual_image
                                 if member.preview is not None else None)


# ── probing ──────────────────────────────────────────────────────────────────

def _metadata_angle(signal, fields) -> float | None:
    """The first of *fields* the member actually carries, as a float."""
    metadata = getattr(signal, "metadata", None)
    for dotted in fields:
        try:
            if metadata is not None and metadata.has_item(dotted):
                return float(metadata.get_item(dotted))
        except (TypeError, ValueError) as e:
            log.debug("%s is not a number: %s", dotted, e)
    return None


def _open_lazily(session, path: str, reader_options: dict):
    """``(signal, error)`` — a lazy open that reports rather than raises."""
    try:
        return session._load_member(path, reader_options), None
    except Exception as e:
        log.debug("probing %s failed: %s", path, e)
        return None, str(e)


def _describe(member: LoaderMember, signal) -> tuple[int, ...]:
    """Record what the opened *signal* is, and return its shape."""
    shape = tuple(int(size) for size in signal.data.shape)
    signal_dimension = int(signal.axes_manager.signal_dimension)
    member.signal = signal
    member.dtype = str(signal.data.dtype)
    member.scan_shape = shape[:len(shape) - signal_dimension]
    member.detector_shape = shape[len(shape) - signal_dimension:]
    return shape


def _probe_member(session, member: LoaderMember, reader_options: dict,
                  scan_shape=None, *, read_angles: bool = True) -> None:
    """Fill in *member* by opening its file LAZILY: header only, no frames.

    A file that will not open leaves ``load_error`` set rather than raising, so
    one bad path in a multi-select does not sink the others — the user needs to
    see WHICH one, and the rest of the list is still good.

    **The file is always opened UNAIDED first**, and *scan_shape* is a retry
    for the one failure it fixes. A Direct Electron ``.mrc`` ships an
    ``_info.txt`` sidecar naming the scan grid and the reader picks it up on
    its own, so trying the scan shape first would override a file that already
    knew better with a number typed into a dialog. Unaided-first means the
    sidecar always wins and the dialog's field is a fallback.

    ``read_angles=False`` on a RE-probe: the file's metadata has not changed,
    and re-reading it would overwrite a tilt or azimuth the user has typed in
    the meantime.
    """
    path = member.path
    member.signal = None
    member.scan_shape = member.detector_shape = member.dtype = None
    member.load_error = member.error = None
    member.needs_scan_shape = False
    member.image = member.pattern = None
    member.virtual_images = []
    member.preview = member.preview_source = None
    # The corner EXTENTS survive — they are fractions of the scan, so they
    # still say what the user meant. What was measured inside them does not.
    member.forget_corners()

    if _path_ext(path) not in SUPPORTED_EXTS or not _is_supported_dataset_path(path):
        member.load_error = "missing, or not a supported dataset"
        return

    signal, error = _open_lazily(session, path, reader_options)
    if signal is None:
        member.load_error = error
        return
    shape = _describe(member, signal)

    if len(shape) == 3:
        # The common failure, and not the file's fault: a raw binary format
        # records the frames and the detector but has no concept of a scan
        # grid. Say what is MISSING, not just that the rank is wrong.
        member.needs_scan_shape = True
        if scan_shape is None:
            member.load_error = (
                f"{shape[0]} frames of {shape[1]}×{shape[2]}; this format "
                "records no scan grid. Set the scan shape.")
            return
        columns, rows = (int(size) for size in scan_shape)
        if columns * rows != shape[0]:
            # Checked HERE because the reader does not check it: asked for a
            # grid that does not fit, it silently returns that many positions
            # anyway — reading the wrong frames, or past the end of the file.
            member.load_error = (
                f"a {columns} × {rows} scan is {columns * rows} positions, "
                f"but this file holds {shape[0]} frames")
            return
        signal, error = _open_lazily(
            session, path, {**reader_options,
                            "navigation_shape": (columns, rows)})
        if signal is None:
            member.load_error = error
            return
        shape = _describe(member, signal)

    if len(shape) != 4:
        member.load_error = (
            f"{len(shape)}-D; a multi-angle member must be 4-D "
            "(scan_y, scan_x, ky, kx)")
        return

    member.virtual_images = _usable_virtual_images(signal, member.scan_shape)

    if not read_angles:
        return
    tilt = _metadata_angle(signal, _TILT_FIELDS)
    if tilt is not None:
        member.tilt = tilt
    azimuth = _metadata_angle(signal, _AZIMUTH_FIELDS)
    if azimuth is not None:
        member.azimuth = azimuth
        member.azimuth_is_auto = False


def _spread_azimuths(state: MultiAngleLoaderState) -> None:
    """Spread the members that named no azimuth evenly around the ring.

    A tilt cannot be guessed from a file's position in a list — it is the
    quantity the shells are grouped by, and inventing one would invent the
    acquisition's structure. An azimuth can: the members of a shell are
    normally stepped evenly around it, so an even spread is the right first
    draft and any of it is one field away from being corrected.

    Only members that can actually be used take part, so a file that failed to
    open — or one a second file drop has not probed yet — does not claim a
    place on the ring and shift everything after it.
    """
    auto = [member for member in state.members
            if member.azimuth_is_auto and member.usable]
    for position, member in enumerate(auto):
        member.azimuth = 360.0 * position / len(auto)


def _refresh_errors(state: MultiAngleLoaderState) -> None:
    """Recompute every member's ``error`` — including "unlike the others".

    Shape agreement is a fact about the SET, so it is decided here (after any
    change to the list) rather than at probe time: removing the odd member out
    must clear the complaint from the ones that were only ever guilty of
    disagreeing with it.
    """
    probed = [member for member in state.members
              if member.load_error is None and not member.pending]
    majority = ((probed[0].scan_shape, probed[0].detector_shape)
                if probed else None)
    for member in state.members:
        if member.load_error is not None:
            member.error = member.load_error
        elif member.pending:
            member.error = None
        elif majority is not None and (member.scan_shape,
                                       member.detector_shape) != majority:
            member.error = (
                f"{tuple(member.scan_shape)} scan × "
                f"{tuple(member.detector_shape)} detector does not match "
                f"{tuple(majority[0])} × {tuple(majority[1])}")
        else:
            member.error = None


def _settle_reference(state: MultiAngleLoaderState) -> None:
    """Point the reference at a member that can actually be one.

    The FIRST usable member, because an acquisition has no natural reference
    and asking for one before anything can be aligned is a click that only ever
    has one sensible answer. ``None`` while there is no such member.
    """
    usable = [index for index, member in enumerate(state.members)
              if member.usable]
    if state.reference not in usable:
        state.reference = usable[0] if usable else None


def _settle_virtual_image(state: MultiAngleLoaderState) -> None:
    """Point the virtual-image choice at one the member set can honour.

    Two cases. While nobody has chosen, the loader picks the first image the
    members all carry — the acquisition shipped it, so it is a better first
    draft than a sum over the whole detector that costs a pass over every
    member to produce. And a name every member had until this one arrived is no
    longer a name every member has, so a chosen one that has gone falls back to
    the computed image: aligning the rest on it while the newcomer used
    something else would be a silently different measurement per row.
    """
    available = state.available_virtual_images
    if state.virtual_image_is_auto:
        state.virtual_image = available[0] if available else None
    elif state.virtual_image is not None \
            and state.virtual_image not in available:
        state.virtual_image = None


def _resettle(state: MultiAngleLoaderState) -> None:
    """Everything that has to hold after the member list changes.

    Errors first: which members can be used decides which take a place on the
    azimuth ring, which can be the reference, and which virtual images are
    common to them all.
    """
    _refresh_errors(state)
    _spread_azimuths(state)
    _settle_reference(state)
    _settle_virtual_image(state)
    _refresh_previews(state)
    state.invalidate_alignments()


def _member_summary(state: MultiAngleLoaderState) -> str:
    """The member list in one line — how many, and which cannot be used.

    Shared by the probe and the tableau fill, because the fill takes the
    ``busy`` phase over from the probe and has to give the same answer back
    when it lets go of it.
    """
    failed = [member.name for member in state.members if member.error]
    return (f"{len(state.members)} member"
            f"{'s' if len(state.members) != 1 else ''}"
            + (f" · {len(failed)} unusable: {', '.join(failed)}"
               if failed else ""))


# ── the per-member reductions ────────────────────────────────────────────────

def _sample_positions(scan_shape, target: int):
    """Evenly spaced scan rows and columns, about *target* positions in all.

    Evenly spaced rather than the first *target* in a row, for the reason
    ``drift_action._preview_indices`` gives: a contiguous corner of the scan is
    one region of the sample, and a mean pattern taken there is that region's,
    not the acquisition's.  ``None`` when the whole scan is already small
    enough to average outright.
    """
    height, width = (int(size) for size in scan_shape)
    if height * width <= int(target):
        return None
    per_axis = max(1, int(round(math.sqrt(float(target)))))
    rows = np.unique(np.linspace(
        0, height - 1, min(height, per_axis)).round().astype(int))
    columns = np.unique(np.linspace(
        0, width - 1, min(width, per_axis)).round().astype(int))
    return rows, columns


def _member_reductions(signal):
    """``(real_space_image, mean_pattern)`` for one member, in ONE pass.

    Both are asked for in a single ``compute`` so the reads they share are done
    once — the image needs every position, so the sampled pattern rides along
    on chunks that were going to be read anyway.

    Neither reduction materialises the member: the image is a streaming sum
    over the detector axes, and the pattern is a mean over a sample of scan
    positions (CLAUDE.md memory safety — a ``.compute()`` on the 4-D array
    itself would be hundreds of GB).
    """
    import dask.array as da

    data = signal.data
    image = data.sum(axis=(-2, -1), dtype=np.float64)
    sample = _sample_positions(data.shape[:2], PATTERN_SAMPLE_POSITIONS)
    sampled = data if sample is None else data[sample[0]][:, sample[1]]
    pattern = sampled.mean(axis=(0, 1), dtype=np.float64)
    computed_image, computed_pattern = da.compute(image, pattern)
    return np.asarray(computed_image), np.asarray(computed_pattern)


def _ensure_reductions(state: MultiAngleLoaderState, members) -> None:
    """Compute what *members* is missing and keep it. Runs on a worker thread.

    Takes the member list the stage was started with rather than reading it
    back off the state: a file added while the solve is running changes what
    ``state.members`` is, and this pass must stay the one the stage's offsets
    will be indexed against.
    """
    with state._reduction_lock:
        for member in members:
            if member.image is not None and member.pattern is not None:
                continue
            emit_status(f"Reducing {member.name}…")
            member.image, member.pattern = _member_reductions(member.signal)


# ── the scan corners ─────────────────────────────────────────────────────────

def _corner_sums(signal, extents, corners) -> list[np.ndarray]:
    """The diffraction pattern summed over each of the named scan corners.

    Asked for in ONE ``compute`` so the corners a member needs are read in a
    single pass over its graph. Nothing here materialises the member (CLAUDE.md
    memory safety): each sum streams its own corner block and returns one
    detector-sized pattern, and at the default extent the four blocks together
    are about one percent of the scan.
    """
    import dask.array as da

    data = signal.data
    nav_shape = data.shape[:2]
    sums = []
    for corner in corners:
        rows, columns = corner_slice(nav_shape, corner, extents[corner])
        sums.append(data[rows, columns].sum(axis=(0, 1), dtype=np.float64))
    return [np.asarray(pattern) for pattern in da.compute(*sums)]


def _ensure_corner_sums(state: MultiAngleLoaderState, members) -> None:
    """Sum whatever corner blocks *members* is missing. Runs on a worker.

    Only the corners with no pattern for their CURRENT extent are computed,
    which is what makes adjusting one corner's widget cost one corner of one
    member rather than the whole tableau.
    """
    with state._corner_lock:
        for member in members:
            missing = [corner for corner in range(N_CORNERS)
                       if member.corner_patterns[corner] is None]
            if not missing:
                continue
            emit_status(f"Summing the scan corners of {member.name}…")
            patterns = _corner_sums(member.signal, member.corner_extents,
                                    missing)
            for corner, pattern in zip(missing, patterns):
                member.corner_patterns[corner] = pattern
                member.corner_previews[corner] = _thumbnail(pattern)


# ── filling the tableaux ─────────────────────────────────────────────────────

def _start_preview_fill(session, state: MultiAngleLoaderState) -> None:
    """Give every member its tableau tiles, without waiting to be asked.

    A tableau is what the user looks at to DECIDE whether to run a stage, so it
    cannot be the reward for having run one. This starts as soon as a probe
    finishes and fills the grid from what the members can be made to show: each
    member's real-space image, and the four corner sums for the reciprocal tab.

    **The work is moved, not added.** The reductions and corner sums land in
    exactly the caches the solves read (:func:`_ensure_reductions`,
    :func:`_ensure_corner_sums`), under the same locks, so a solve afterwards
    finds them done rather than doing them again. On a real acquisition this
    *is* the loader's dominant cost — a pass over every member — and there is no
    version of "show me the images" that avoids it. What it must not be is a
    silent freeze, hence:

    * **one member at a time, announced.** Every member that lands emits a
      snapshot, so a ten-member acquisition shows its first tile in the time
      one member takes rather than nothing until the last;
    * **interruptible.** The generation guard is checked before each member and
      between its two halves, so closing the dialog or re-probing stops the
      pass at the next member rather than running a multi-gigabyte read that
      nothing is waiting for.

    It hands over the ``busy`` phase from the stage that called it rather than
    starting a new one: dropping to ``busy: false`` in between would flicker
    every control in the dialog, and would tell a caller waiting for "the
    loader is idle" that it is, a moment before it is not. It BORROWS that
    stage's message while it works and gives it back at the end, so the line
    the user is left looking at is the one the stage meant to leave them with.
    """
    settled_message = state.message
    members = [member for member in state.members if _needs_tiles(state, member)]
    if not members:
        # Nothing to read. Announce only if a ``busy`` phase was handed over,
        # so the callers that have one are released and the ones that were
        # already idle do not send a second identical snapshot per click.
        if state.busy:
            state.busy = False
            _emit_state(state)
        return

    generation = bump_generation(state, "_preview_generation")
    state.busy = True
    state.message = (f"Reading {len(members)} member"
                     f"{'s' if len(members) != 1 else ''} for the tableau…")
    _emit_state(state)

    def _wanted(member: LoaderMember) -> bool:
        """Still this pass, and still a member of the acquisition.

        Identity rather than equality: a member holds numpy arrays, so ``in``
        would compare them elementwise and raise on the truth value.
        """
        return (is_current(state, "_preview_generation", generation)
                and any(other is member for other in state.members))

    def _landed(position: int, member: LoaderMember) -> None:
        def _apply():
            if not _wanted(member):
                return
            _refresh_previews(state)
            state.message = f"Read {position}/{len(members)} · {member.name}"
            _emit_state(state)

        _on_main(session, _apply)

    def _work():
        from spyde.backend.heavy_imports import ensure_heavy_imports

        ensure_heavy_imports()
        for position, member in enumerate(members, start=1):
            if not _wanted(member):
                return
            # Only when there is no image to be had for free: a member whose
            # file shipped the chosen virtual image is already previewable, and
            # reducing it would be computing a worse copy of a picture we have.
            if _member_image(state, member) is None:
                _ensure_reductions(state, [member])
            if not _wanted(member):
                return
            _ensure_corner_sums(state, [member])
            _landed(position, member)

    def _done(_result):
        if not is_current(state, "_preview_generation", generation):
            return
        _refresh_previews(state)
        state.busy = False
        state.message = settled_message
        _emit_state(state)

    def _failed(error):
        emit_error(f"Reading the members for the tableau failed: {error}")

        def _apply():
            if not is_current(state, "_preview_generation", generation):
                return
            state.busy = False
            state.message = f"Reading the members failed: {error}"
            _emit_state(state)

        _on_main(session, _apply)

    run_on_worker(session, _work, name="maped-previews",
                  on_done=_done, on_error=_failed)


# ── the aligned-sum window ───────────────────────────────────────────────────

#: Pixels ignored around the edge of a summed image when measuring how much
#: structure survived. The members cover different regions, so every candidate
#: alignment crops differently and leaves its own border — and a border is a
#: step, which any sharpness measure scores highly.
_STRUCTURE_MARGIN = 12

#: Scale of the background removed first. The specimen's own shape and the
#: detector's illumination are smooth and large; they carry most of the
#: variance and none of the information about whether the members landed on
#: each other.
_STRUCTURE_BACKGROUND = 8.0


def _structure(image) -> float:
    """How much fine detail an image holds, per unit brightness.

    The number behind "sharpness x N". Gradient energy was the obvious choice
    and does not work here: it is won by the border the crop creates, so a
    worse alignment that crops harder scores higher — measured on a real
    four-member acquisition, an alignment that visibly BLURRED the layers
    scored 1.5x, while the one that actually stacked them scored 0.85x. It
    still fails after excluding the border, because a smooth background ramp
    dominates what is left.

    So: drop the background, ignore the edges, and measure contrast relative
    to the mean. On the same acquisition this ranks the alignments the way the
    specimen does — the layers' own periodicity going 43 -> 294 in
    signal-to-noise as the score goes 0.88 -> 1.14.
    """
    from scipy.ndimage import gaussian_filter

    values = np.asarray(image, dtype=np.float64)
    margin = _STRUCTURE_MARGIN
    if min(values.shape) > 3 * margin:
        values = values[margin:-margin, margin:-margin]
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    filled = np.where(finite, values, np.nanmean(values[finite]))
    detail = filled - gaussian_filter(filled, _STRUCTURE_BACKGROUND)
    mean = float(np.mean(filled))
    if not np.isfinite(mean) or abs(mean) < 1e-12:
        return 0.0
    return float(np.std(detail[finite]) / abs(mean))


def _alignment_evidence(images, model) -> dict | None:
    """The members summed with the solved offsets applied, and without.

    Both panels are the SAME region of the SAME members and differ only in
    whether each member's offset was applied — the aligned one reads each
    member through its own slices, the unaligned one reads every member through
    the REFERENCE's. Same shape, same pixel count, so the sharpness numbers
    beneath them compare.

    The crop is :meth:`~spyde.multiangle.model.MultiAngleModel.nav_slices`,
    which is how the composition itself reads a member, so this shows what
    Commit will build rather than a second opinion about it. Deliberately NOT
    ``drift_action._stack_sum``: that shifts frames by interpolation, which is
    the right model for a drifting movie and the wrong one here — a multi-angle
    offset is a whole number of pixels and alignment is a read at different
    indices, never a resample.

    ``None`` when the members' offsets spread so far that the region they all
    cover is too small to look at. Reads no member: these are the real-space
    images the solve already registered.
    """
    scan_shape = np.asarray(images[0]).shape[:2]
    region = model.overlap_shape(scan_shape)
    if min(region) < 2:
        return None

    reference_slices = model.nav_slices(model.reference, scan_shape)
    aligned = np.mean(
        [np.asarray(image, dtype=np.float64)[model.nav_slices(index, scan_shape)]
         for index, image in enumerate(images)], axis=0)
    unaligned = np.mean(
        [np.asarray(image, dtype=np.float64)[reference_slices]
         for image in images], axis=0)
    before, after = _structure(unaligned), _structure(aligned)
    return {"aligned": aligned, "unaligned": unaligned,
            "gain": (after / before) if before > 0 else float("nan")}


class AlignedSumWindow:
    """The real-space solve's evidence: the members summed, aligned and not.

    Aligned, the members' features stack and the sum is sharp; unaligned, it
    blurs. A residual in pixels cannot show that, and neither can the tableau
    of members one at a time — the question is what they look like ON TOP OF
    one another, which is a picture of the thing itself.

    A bare-figure window (``actions/README.md`` §6): it registers as a window
    controller so the ✕ and :meth:`Session._forget_window` tear it down, and
    its figure is held by ``figure_registry`` for as long as the window lives.
    Display only — nothing here feeds back into a solve.
    """

    def __init__(self, session, window_id: int, state):
        self.session = session
        self.window_id = int(window_id)
        self.state = state
        self.closed = False
        self._panels: dict = {}

    @property
    def source_plot(self):
        """No plot behind this window: it is built from the loader's own
        arrays, and the dialog has no source plot to begin with."""
        return None

    def build(self, evidence: dict) -> bool:
        """Draw the pair and emit the window. False if the figure failed."""
        import anyplotlib as apl
        import anyplotlib._electron as _electron

        from de_shell.actions.figure_registry import keep_alive
        from spyde.actions.drift_action import _figure_geometry
        from spyde.drawing.plots.plot import finalize_figure_html

        try:
            figsize, aspect = _figure_geometry()
            figure, axes = apl.subplots(1, 2, figsize=figsize)
            panel = np.array(axes, dtype=object).ravel()
            self._panels = {
                "unaligned": panel[0].imshow(
                    evidence["unaligned"].astype(np.float32), cmap="gray"),
                "aligned": panel[1].imshow(
                    evidence["aligned"].astype(np.float32), cmap="gray"),
            }
            self._set_titles(evidence)

            figure_id = _electron.register(figure)
            keep_alive(self.window_id, figure)
            ipc.emit({"type": "figure", "fig_id": figure_id,
                      "window_id": self.window_id,
                      "html": finalize_figure_html(figure, figure_id),
                      "title": ALIGNED_WINDOW_TITLE, "is_navigator": False,
                      "aspect": float(aspect)})
            # A figure message does not rename a window, so say the name too.
            ipc.emit({"type": "window_title", "window_ids": [self.window_id],
                      "title": ALIGNED_WINDOW_TITLE})
        except Exception as e:
            log.exception("building the aligned-sum window failed: %s", e)
            return False
        return True

    def update(self, evidence: dict) -> None:
        """Repaint both panels — a re-run moves the window, not the window
        count."""
        if self.closed or not self._panels:
            return
        try:
            self._panels["unaligned"].set_data(
                evidence["unaligned"].astype(np.float32))
            self._panels["aligned"].set_data(
                evidence["aligned"].astype(np.float32))
            self._set_titles(evidence)
        except Exception as e:
            log.debug("repainting the aligned-sum window failed: %s", e)

    def _set_titles(self, evidence: dict) -> None:
        gain = float(evidence.get("gain", float("nan")))
        titles = {"unaligned": "Unaligned",
                  "aligned": ("Aligned" if not np.isfinite(gain)
                              else f"Aligned · sharpness ×{gain:.2f}")}
        for key, title in titles.items():
            try:
                self._panels[key].set_title(title)
            except Exception as e:
                log.debug("titling the %s panel failed: %s", key, e)

    def close(self) -> None:
        """Drop the panels and let go of the loader's handle on this window."""
        self.closed = True
        self._panels = {}
        if getattr(self.state, "aligned_window", None) is self:
            self.state.aligned_window = None


def _show_aligned_sum(session, state: MultiAngleLoaderState) -> None:
    """Open or repaint the window showing the solved real-space alignment.

    Called on the main thread, once a real-space solve has landed. Silent
    rather than noisy when it cannot be drawn: it is evidence for a stage that
    has already reported its own result, so a failure here must not read as a
    failure of the alignment.
    """
    if not state.real.solved:
        return
    images = [_member_image(state, member) for member in state.members]
    if not images or any(image is None for image in images):
        return
    try:
        model = state.build_model(
            dp_offsets=np.zeros((len(images), 2), dtype=np.int64))
        evidence = _alignment_evidence(images, model)
    except Exception as e:
        log.debug("building the aligned sum failed: %s", e)
        return
    if evidence is None:
        return

    # Keyed on the offsets it was drawn for, like the member thumbnails are on
    # theirs: re-running a solve that lands on the same answer produces the
    # same two pictures, and encoding them again is what the cache exists to
    # stop. A solve that MOVED a member re-encodes, because it must.
    gain = float(evidence.get("gain", float("nan")))
    offsets = np.asarray(model.nav_offsets).tolist()
    current = state.real_evidence
    if current is None or current.get("offsets") != offsets:
        state.real_evidence = {
            "unaligned": _thumbnail(evidence["unaligned"]),
            "aligned": _thumbnail(evidence["aligned"]),
            "gain": None if not np.isfinite(gain) else gain,
            "offsets": offsets,
        }
        _emit_state(state)

    window = state.aligned_window
    if window is not None and not window.closed:
        window.update(evidence)
        return
    window = AlignedSumWindow(session, session.next_window_id(), state)
    if not window.build(evidence):
        return
    session.register_window_controller(window.window_id, window)
    state.aligned_window = window


# ── the solves ───────────────────────────────────────────────────────────────

def _solver_parameters(params) -> dict:
    """The subset of *params* the translation solver accepts."""
    params = dict(params or {})
    unknown = sorted(set(params) - set(_SOLVER_PARAMETERS))
    if unknown:
        log.debug("ignoring solver parameters this stage does not take: %s",
                  unknown)
    return {name: params[name] for name in _SOLVER_PARAMETERS
            if name in params}


def beam_region(roi, shape) -> tuple[slice, slice] | None:
    """``(rows, columns)`` for the ``{cy, cx, half}`` *roi*, clipped to *shape*.

    ``None`` when there is no region, which means the whole pattern — what the
    stage did before a region could be given at all.
    """
    if not roi:
        return None
    try:
        cy = float(roi["cy"])
        cx = float(roi["cx"])
        half = float(roi["half"])
    except (KeyError, TypeError, ValueError):
        log.debug("ignoring a beam region that is not {cy, cx, half}: %r", roi)
        return None
    if half <= 0:
        return None
    height, width = int(shape[0]), int(shape[1])
    rows = slice(max(0, int(round(cy - half))),
                 min(height, int(round(cy + half)) + 1))
    columns = slice(max(0, int(round(cx - half))),
                    min(width, int(round(cx + half)) + 1))
    if rows.stop - rows.start < 2 or columns.stop - columns.start < 2:
        log.debug("a beam region of %r leaves nothing of a %r pattern to "
                  "search; using the whole pattern", roi, shape)
        return None
    return rows, columns


def _beam_position(pattern, params: dict, roi=None) -> tuple[float, float]:
    """One pattern's direct beam as ``(x, y)``, pyxem's order and convention.

    ``get_direct_beam_position`` reports ``centre − beam``: not the beam's
    coordinate but the shift that would centre it. That is fine for every
    caller here, because each one only ever takes DIFFERENCES between two
    members' values, and the frame centre — whatever pyxem takes it to be —
    cancels in the difference.

    *roi* is the region the beam is looked for in. Cropping here rather than
    passing ``half_square_width`` down is what lets the region sit ANYWHERE:
    that argument crops a square about the frame's own middle, and a descanned
    zero beam is not there. It matters more than it sounds — the default method
    is a centre of mass, so over a whole pattern it is pulled bodily by
    whichever reflections a tilt happens to excite, and members that differ
    only in excitation come back tens of pixels apart.
    """
    import hyperspy.api as hs

    pattern = np.asarray(pattern, dtype=np.float32)
    region = beam_region(roi, pattern.shape)
    searched = pattern if region is None else pattern[region[0], region[1]]

    signal = hs.signals.Signal2D(searched)
    signal.set_signal_type("electron_diffraction")
    arguments = {"method": str(params.get("beam_method",
                                          DEFAULTS["beam_method"])),
                 "lazy_output": False}
    half_square_width = int(params.get("half_square_width",
                                       DEFAULTS["half_square_width"]) or 0)
    if half_square_width > 0:
        arguments["half_square_width"] = half_square_width
    position = np.asarray(signal.get_direct_beam_position(**arguments).data,
                          dtype=np.float64).ravel()
    x, y = float(position[0]), float(position[1])
    if region is not None:
        # Back into the FULL pattern's frame. The answer is centre − beam, so a
        # crop moves it by the difference of the two centres less the crop's
        # own origin; both centres follow the same convention, so whatever
        # pyxem takes "centre" to mean cancels out of the difference.
        rows, columns = region
        x += (pattern.shape[1] - searched.shape[1]) / 2.0 - columns.start
        y += (pattern.shape[0] - searched.shape[0]) / 2.0 - rows.start
    return x, y


def _offsets_from_positions(positions, reference: int):
    """``(offsets, residuals)`` from one ``(x, y)`` beam position per member.

    The correction bringing member *i* onto the reference is that member's
    position less the reference's. Returned ``(dy, dx)``, the order
    :mod:`spyde.multiangle.model` states offsets in.
    """
    shifts = np.asarray([(y, x) for x, y in positions], dtype=np.float64)
    shifts = shifts - shifts[int(reference)]
    offsets = np.rint(shifts).astype(np.int64)
    return offsets, (shifts - offsets).astype(np.float32)


def _beam_offsets(patterns, reference: int, params: dict, roi=None):
    """Reciprocal offsets from the beam fitted in each member's MEAN pattern."""
    return _offsets_from_positions(
        [_beam_position(pattern, params, roi) for pattern in patterns],
        reference)


def _corner_beam_position(member: LoaderMember, params: dict, roi=None
                          ) -> tuple[float, float]:
    """One member's beam position, from the four scan corners alone.

    The beam is located in each corner's summed pattern, a plane is fitted
    through the four positions, and the plane is read at the middle of the
    scan. The plane is what makes four corners into one number: a beam that
    walks across the scan (instrument descan) is a ramp, and the middle of a
    ramp is the member's representative position — exactly the reading the DPC
    wizard's corner seed takes, out of the same :mod:`spyde.corners`.

    The corner blocks may have four different extents, which the fit handles
    without being told: each block is written into the scan-shaped field at its
    own position and size, so a wider block simply weighs more and pulls the
    plane towards where it actually was.
    """
    nav_shape = tuple(int(size) for size in member.scan_shape)
    # NaN everywhere but the corners, which is how the fit is told to ignore
    # the rest — `plane_through` keeps only the finite points.
    field = np.full(nav_shape + (2,), np.nan, dtype=np.float64)
    for corner, pattern in enumerate(member.corner_patterns):
        rows, columns = corner_slice(nav_shape, corner,
                                     member.corner_extents[corner])
        field[rows, columns] = _beam_position(pattern, params, roi)
    plane = plane_through(field, None)
    middle = plane[nav_shape[0] // 2, nav_shape[1] // 2]
    return float(middle[0]), float(middle[1])


def _corner_offsets(members, reference: int, params: dict, roi=None):
    """Reciprocal offsets from every member's four scan corners."""
    return _offsets_from_positions(
        [_corner_beam_position(member, params, roi) for member in members],
        reference)


def _start_stage(session, state, *, attribute: str, generation_key: str,
                 name: str, message: str, work, on_solved=None) -> None:
    """Run one alignment stage on a worker and announce it, start and end.

    The generation guard is what makes a superseded run harmless: a second
    click, or closing the dialog, bumps the counter, and the earlier run's
    result is dropped on arrival instead of overwriting a newer one.

    *name* is the SPACE that was aligned ("Real space"), not the stage: both
    the readout and the failure line read it as a subject.

    *on_solved* runs on the main thread once a CURRENT result has landed — the
    stage's own evidence window. Behind the guard, so a superseded solve cannot
    open one.
    """
    generation = bump_generation(state, generation_key)
    state.busy = True
    state.message = message
    emit_status(message)
    _emit_state(state)

    def _done(result):
        if not is_current(state, generation_key, generation):
            return
        offsets, residuals = result
        stage = LoaderStage(offsets=np.asarray(offsets, dtype=np.int64),
                            residuals=np.asarray(residuals, dtype=np.float32))
        setattr(state, attribute, stage)
        # A stage that reduced the members has given the tableau its first
        # pictures; one that did not finds every preview already current.
        _refresh_previews(state)
        state.busy = False
        state.message = (f"{name} aligned · max residual "
                         f"{stage.max_residual or 0.0:.2f} px")
        emit_status(state.message)
        _emit_state(state)
        if on_solved is not None:
            try:
                on_solved()
            except Exception as e:
                log.debug("the %s evidence window failed: %s", name, e)

    def _failed(error):
        emit_error(f"{name} alignment failed: {error}")

        def _apply():
            if not is_current(state, generation_key, generation):
                return
            state.busy = False
            state.message = f"{name} alignment failed: {error}"
            _emit_state(state)

        _on_main(session, _apply)

    run_on_worker(session, work, name=f"maped-align-{attribute}",
                  on_done=_done, on_error=_failed)


# ── the staged actions ───────────────────────────────────────────────────────

def _loader_state(session) -> MultiAngleLoaderState:
    """The open loader's state, started if the dialog never announced itself.

    Creating one here rather than refusing: a file list arriving before the
    open is an ordering bug in the renderer, and answering it with an empty
    snapshot would look to the user like the files vanished.
    """
    state = getattr(session, "_multiangle_loader", None)
    if state is None:
        state = MultiAngleLoaderState()
        session._multiangle_loader = state
    return state


def maped_open_loader(session, plot, payload) -> None:
    """Start the loader dialog — or reset one already open."""
    previous = getattr(session, "_multiangle_loader", None)
    if previous is not None:
        previous.cancel_in_flight()
    state = MultiAngleLoaderState(
        reader_options=dict((payload or {}).get("reader_options") or {}))
    session._multiangle_loader = state
    _emit_state(state)


def maped_close_loader(session, plot, payload) -> None:
    """Drop the loader state and stop anything still solving for it."""
    session.discard_multiangle_loader()
    _emit_state(None)


def _parse_angles(raw, count: int):
    """``(angles, complaint)`` — one ``{tilt, azimuth}`` or ``None`` per path.

    Positional, so an unequal list cannot be honoured at all: there would be no
    saying which file each angle belonged to, and guessing would place members
    at angles nobody chose.
    """
    if raw is None:
        return [None] * count, None
    angles = list(raw)
    if len(angles) != count:
        return [None] * count, (
            f"{len(angles)} angles for {count} file"
            f"{'s' if count != 1 else ''}; angles go with paths by position. "
            "Files added without them.")
    for entry in angles:
        if entry is not None and not isinstance(entry, dict):
            return [None] * count, (
                f"An angle is {{tilt, azimuth}} or null; got {entry!r}. "
                "Files added without them.")
    return angles, None


def _place_member(member: LoaderMember, angles) -> None:
    """Put *member* at the angles it was dropped on.

    Whichever of the two is given, as :func:`maped_set_member` does — a caller
    that knows only the azimuth of a spot should not have to invent a tilt.
    """
    if not angles:
        return
    try:
        if angles.get("tilt") is not None:
            member.tilt = float(angles["tilt"])
        if angles.get("azimuth") is not None:
            member.azimuth = float(angles["azimuth"])
            member.azimuth_is_auto = False
    except (TypeError, ValueError) as e:
        log.debug("%s was dropped on an angle that is not a number: %s",
                  member.name, e)


def maped_add_files(session, plot, payload) -> None:
    """Probe each path and append it as a member. Reads no frames.

    The ROWS are appended here, synchronously, so the list order is fixed by
    the order the calls arrived in and never by which worker finishes first.
    Only the probing itself moves to a worker: opening a member is a header
    read, but on a slow or networked store it is not instant, and the backend's
    event loop is the wrong place to find that out.

    So this is the one action that answers with TWO snapshots — the rows with
    ``busy: true``, then the probed rows. The last one is authoritative, as it
    is for the alignment stages.

    ``angles`` is an OPTIONAL list parallel to ``paths``, each entry
    ``{"tilt": …, "azimuth": …}`` or ``null``. It exists because a file dropped
    onto a spot on the polar tableau already has its angles, and this action
    hands back nothing addressable — without it a caller has to wait for a
    snapshot and find its own file in it BY PATH, which fails silently and
    invisibly the first time a path is normalised on the way through (a
    resolved symlink, an expanded ``~``, a case-canonicalised drive letter).
    Applied as the row is appended, so even the ``busy`` snapshot carries them.

    A supplied angle is the user's placement and outlives the probe: a tilt in
    the file's metadata fills a row that was dropped without one, but does not
    move one that was dropped ON an angle. ``null`` leaves that member exactly
    as the plain ``{paths}`` form does — the file's metadata, else the loader's
    even spread around the ring.
    """
    state = _loader_state(session)
    paths = [str(path) for path in ((payload or {}).get("paths") or []) if path]
    if not paths:
        _emit_state(state)
        return

    reader_options = dict(state.reader_options)
    reader_options.update((payload or {}).get("reader_options") or {})
    state.reader_options = reader_options
    angles, complaint = _parse_angles((payload or {}).get("angles"), len(paths))
    if complaint:
        emit_error(complaint)

    # Sized on the worker, not here. A directory store is measured by walking
    # every file in it, and a frame-chunked .zspy is one file per frame —
    # 65,579 of them per member, about 35 s each on a real acquisition. Doing
    # that before the first snapshot froze the whole backend for over two
    # minutes with the dialog still reading "No datasets".
    pending = [LoaderMember(path=path,
                            name=os.path.basename(path.rstrip(os.sep)))
               for path in paths]
    for member, placed_at in zip(pending, angles):
        _place_member(member, placed_at)
    state.members.extend(pending)
    # The list already describes a different acquisition from the one the
    # offsets were solved for, whatever the probe goes on to find.
    state.invalidate_alignments()
    generation = bump_generation(state, "_probe_generation")
    state.busy = True
    state.message = f"Opening {len(pending)} file{'s' if len(pending) != 1 else ''}…"
    _emit_state(state)

    def _work():
        from spyde.backend.heavy_imports import ensure_heavy_imports

        # Format registrations must land before the first open, for the reason
        # _load_file_thread gives: a `.csb` (or any runtime-registered format)
        # is otherwise not a format hyperspy knows about yet.
        ensure_heavy_imports()
        # Serialised, so two file drops in quick succession fill their rows one
        # batch after the other rather than interleaving.
        with state._probe_lock:
            for member, placed_at in zip(pending, angles):
                _probe_member(session, member, reader_options,
                              state.scan_shape)
                # After the probe, not instead of it: the file's metadata
                # fills a row dropped without an angle, and the drop wins on
                # one that has it.
                _place_member(member, placed_at)
            # Last, because it is the slowest thing here and the least worth
            # waiting for: the shape and dtype decide whether a member can be
            # used at all, its size only labels the row.
            for member in pending:
                member.size_bytes = int(_dataset_size_bytes(member.path))

    def _done(_result):
        if not is_current(state, "_probe_generation", generation):
            return
        _resettle(state)
        _default_beam_roi(state)
        state.message = _member_summary(state)
        _emit_state(state)
        # Straight on into reading the members for the tableau, WITHOUT letting
        # go of `busy` — the rows are up, and the pictures that go in them are
        # the next thing the user is waiting for, not a separate request.
        _start_preview_fill(session, state)

    def _failed(error):
        emit_error(f"Opening the multi-angle members failed: {error}")

        def _apply():
            if not is_current(state, "_probe_generation", generation):
                return
            state.busy = False
            state.message = f"Opening the members failed: {error}"
            _emit_state(state)

        _on_main(session, _apply)

    run_on_worker(session, _work, name="maped-probe",
                  on_done=_done, on_error=_failed)


def _parse_scan_shape(raw):
    """``(columns, rows)``, ``None`` to clear, or a complaint to show."""
    if raw is None:
        return None, None
    try:
        columns, rows = (int(size) for size in raw)
    except (TypeError, ValueError):
        return None, f"A scan shape is two whole numbers (x, y); got {raw!r}."
    if columns < 1 or rows < 1:
        return None, f"Scan shape must be positive; got {columns} × {rows}."
    return (columns, rows), None


def maped_set_scan_shape(session, plot, payload) -> None:
    """Give the members that record no scan grid one — or take it away again.

    ONE shape for the acquisition: its members share a scan grid, so a field
    per row would be N copies of the same number, each able to drift from the
    others.

    Only the members that need it are re-opened. A file that named its own
    scan grid — a Direct Electron ``.mrc`` and its ``_info.txt`` sidecar is
    the normal case — is left exactly as it was, because a number typed into a
    dialog must not override one the acquisition recorded.

    ``scan_shape: null`` clears it and re-probes those members back to their
    unaided result, so a shape entered wrongly can be undone rather than
    having to rebuild the list.
    """
    state = _loader_state(session)
    scan_shape, complaint = _parse_scan_shape((payload or {}).get("scan_shape"))
    if complaint:
        emit_error(complaint)
        _emit_state(state)
        return

    state.scan_shape = scan_shape
    described = (f"{scan_shape[0]} × {scan_shape[1]}" if scan_shape else "none")
    # Every member that has no scan grid of its own — including the ones a
    # previous scan shape rescued, which are exactly what clearing must undo.
    stale = [member for member in state.members if member.needs_scan_shape]
    if not stale:
        state.message = (
            f"Scan shape {described} · no member needs one"
            if scan_shape else "Scan shape cleared")
        _emit_state(state)
        return

    reader_options = dict(state.reader_options)
    generation = bump_generation(state, "_probe_generation")
    state.busy = True
    state.message = (
        f"Re-opening {len(stale)} member"
        f"{'s' if len(stale) != 1 else ''} with a {described} scan…"
        if scan_shape else
        f"Re-opening {len(stale)} member"
        f"{'s' if len(stale) != 1 else ''} without a scan shape…")
    _emit_state(state)

    def _work():
        from spyde.backend.heavy_imports import ensure_heavy_imports

        ensure_heavy_imports()
        with state._probe_lock:
            for member in stale:
                _probe_member(session, member, reader_options, scan_shape,
                              read_angles=False)

    def _done(_result):
        if not is_current(state, "_probe_generation", generation):
            return
        _resettle(state)
        failed = [member.name for member in stale if member.error]
        state.message = (
            f"Scan shape {described}"
            + (f" · {len(failed)} still unreadable: {', '.join(failed)}"
               if failed else
               f" · {len(stale)} member"
               f"{'s' if len(stale) != 1 else ''} re-opened"))
        _emit_state(state)
        # A re-opened member's scan grid changed, so its tiles were thrown away
        # with the rest of the probe. Refill them, and any a previous pass was
        # interrupted before reaching, without letting go of `busy`.
        _start_preview_fill(session, state)

    def _failed(error):
        emit_error(f"Applying the scan shape failed: {error}")

        def _apply():
            if not is_current(state, "_probe_generation", generation):
                return
            state.busy = False
            state.message = f"Applying the scan shape failed: {error}"
            _emit_state(state)

        _on_main(session, _apply)

    run_on_worker(session, _work, name="maped-scan-shape",
                  on_done=_done, on_error=_failed)


def maped_remove_member(session, plot, payload) -> None:
    """Drop one member from the list."""
    state = _loader_state(session)
    index = int((payload or {}).get("index", -1))
    if not 0 <= index < len(state.members):
        emit_error(f"No member {index} to remove.")
        _emit_state(state)
        return
    removed = state.members.pop(index)
    _resettle(state)
    state.message = f"Removed {removed.name}"
    _emit_state(state)


def maped_set_member(session, plot, payload) -> None:
    """Set one member's tilt and/or azimuth.

    Neither invalidates a solved alignment: the offsets are measured from the
    members' images, and the angles say where the member sat, not how it lines
    up. They do regroup the shells, which the snapshot recomputes.
    """
    state = _loader_state(session)
    payload = payload or {}
    index = int(payload.get("index", -1))
    if not 0 <= index < len(state.members):
        emit_error(f"No member {index} to set.")
        _emit_state(state)
        return
    member = state.members[index]
    if payload.get("tilt") is not None:
        member.tilt = float(payload["tilt"])
    if payload.get("azimuth") is not None:
        member.azimuth = float(payload["azimuth"])
        member.azimuth_is_auto = False
    state.message = (f"{member.name}: {member.tilt:g}° tilt, "
                     f"{member.azimuth:g}° azimuth")
    _emit_state(state)


def maped_set_reference(session, plot, payload) -> None:
    """Choose the member the others are aligned to.

    Both solves are dropped: an offset is a correction TOWARDS the reference,
    so every row of both arrays means something else now.
    """
    state = _loader_state(session)
    index = int((payload or {}).get("index", -1))
    if not 0 <= index < len(state.members):
        emit_error(f"No member {index} to use as the reference.")
        _emit_state(state)
        return
    if not state.members[index].usable:
        emit_error(f"{state.members[index].name} did not open.")
        _emit_state(state)
        return
    state.reference = index
    state.invalidate_alignments()
    state.message = f"Reference: {state.members[index].name}"
    _emit_state(state)


def maped_set_virtual_image(session, plot, payload) -> None:
    """Choose which virtual image the real-space stage aligns on.

    ``name: null`` means compute one — the sum over the whole detector, which
    is the only image available for a file that shipped none.

    The real-space solve is dropped, because its offsets were measured by
    registering a different set of pictures. The reciprocal solve is NOT: it
    measured the detector, which this does not touch.
    """
    state = _loader_state(session)
    name = (payload or {}).get("name")
    name = None if name is None else str(name)
    if name is not None and name not in state.available_virtual_images:
        emit_error(f"No virtual image {name!r} common to every member.")
        _emit_state(state)
        return

    state.virtual_image = name
    state.virtual_image_is_auto = False
    _refresh_previews(state)
    state.invalidate_real()
    state.message = f"Virtual image: {name or 'computed'}"
    _emit_state(state)
    # Choosing the computed image on members that were never reduced leaves the
    # tableau with nothing to show, so it is refilled here rather than waiting
    # for a solve — the same rule as after a probe. A no-op when every member's
    # new picture was already to hand, which is the case for every named one.
    _start_preview_fill(session, state)


def _parse_corner(payload) -> tuple[int, float | None, str | None]:
    """``(corner, extent, complaint)`` from a ``maped_set_corner_extent``."""
    try:
        corner = int(payload.get("corner"))
    except (TypeError, ValueError):
        return -1, None, (f"A corner is one of 0–{N_CORNERS - 1} "
                          f"({', '.join(CORNER_NAMES)}); "
                          f"got {payload.get('corner')!r}.")
    if not 0 <= corner < N_CORNERS:
        return -1, None, (f"Corner must be 0–{N_CORNERS - 1}; "
                          f"got {corner}.")
    raw = payload.get("extent")
    if raw is None:
        return corner, None, None
    try:
        extent = float(raw)
    except (TypeError, ValueError):
        return corner, None, (
            f"A corner extent is a fraction of the scan; got {raw!r}.")
    if not 0.0 < extent <= 0.5:
        return corner, None, (
            f"Corner extent must be between 0 and 0.5; got {extent:g}.")
    return corner, extent, None


def maped_set_corner_extent(session, plot, payload) -> None:
    """Resize one scan corner, on one member or on all of them.

    ``member: null`` sets every member, which is the normal gesture: the
    members of an acquisition are the same scan of the same sample, so a corner
    that sits on vacuum in one sits on vacuum in all of them.

    ``extent`` is a fraction of the scan axis, and omitting it changes nothing
    — that call just asks for whatever corner sums are missing, which is how
    the tableau gets its first pictures without solving anything.

    A changed extent drops that corner's sum, and with it the reciprocal
    alignment (see :meth:`MultiAngleLoaderState.invalidate_reciprocal`). The
    corners are then re-summed on a worker, so this answers TWICE like the
    other stages that read data.
    """
    state = _loader_state(session)
    payload = payload or {}
    corner, extent, complaint = _parse_corner(payload)
    if complaint:
        emit_error(complaint)
        _emit_state(state)
        return

    raw_member = payload.get("member")
    if raw_member is None:
        members = list(state.members)
    else:
        index = int(raw_member)
        if not 0 <= index < len(state.members):
            emit_error(f"No member {index} to set a corner on.")
            _emit_state(state)
            return
        members = [state.members[index]]

    if extent is not None:
        moved = [member for member in members
                 if member.corner_extents[corner] != extent]
        for member in moved:
            member.corner_extents[corner] = extent
            member.forget_corners(corner)
        # Only when a corner actually MOVED. A slider re-sending its resting
        # value is the normal thing for a renderer to do, and throwing a solved
        # alignment away for it would make the stage impossible to keep.
        if moved:
            state.invalidate_reciprocal()

    usable = [member for member in members if member.usable]
    if not usable:
        state.message = (f"{CORNER_NAMES[corner].capitalize()} corner: "
                         f"{extent:g} of the scan" if extent is not None
                         else "No member can be read yet")
        _emit_state(state)
        return

    generation = bump_generation(state, "_corner_generation")
    state.busy = True
    state.message = (
        f"Summing the {CORNER_NAMES[corner]} corner of {len(usable)} member"
        f"{'s' if len(usable) != 1 else ''}…" if extent is not None
        else f"Summing the scan corners of {len(usable)} member"
             f"{'s' if len(usable) != 1 else ''}…")
    _emit_state(state)

    def _work():
        from spyde.backend.heavy_imports import ensure_heavy_imports

        ensure_heavy_imports()
        _ensure_corner_sums(state, usable)

    def _done(_result):
        if not is_current(state, "_corner_generation", generation):
            return
        state.busy = False
        state.message = (
            f"{CORNER_NAMES[corner].capitalize()} corner: {extent:g} of the "
            f"scan" if extent is not None else "Scan corners summed")
        _emit_state(state)

    def _failed(error):
        emit_error(f"Summing the scan corners failed: {error}")

        def _apply():
            if not is_current(state, "_corner_generation", generation):
                return
            state.busy = False
            state.message = f"Summing the scan corners failed: {error}"
            _emit_state(state)

        _on_main(session, _apply)

    run_on_worker(session, _work, name="maped-corners",
                  on_done=_done, on_error=_failed)


def _not_ready(state: MultiAngleLoaderState) -> str | None:
    """Why the members cannot be aligned yet, or None if they can."""
    opening = [member.name for member in state.members if member.pending]
    if opening:
        return "Still opening: " + ", ".join(opening)
    if len(state.members) < 2:
        return "Needs at least two members."
    unusable = [member.name for member in state.members if not member.usable]
    if unusable:
        return "Remove or replace: " + ", ".join(unusable)
    if state.reference is None:
        return "Choose a reference member."
    return None


def maped_set_beam_roi(session, plot, payload) -> None:
    """Place the region the reciprocal stage looks for the zero beam in.

    ``{"cy": …, "cx": …, "half": …}`` in detector pixels, or ``null`` to search
    the whole pattern again. One region for the acquisition — see
    :attr:`MultiAngleLoaderState.beam_roi`.

    Nothing is recomputed here. Moving the region invalidates the reciprocal
    solve and stops, because the corner sums it would be applied to are
    unchanged: only the search inside them moves, and re-running that is the
    Align button's job, not a side effect of dragging.
    """
    state = _loader_state(session)
    raw = (payload or {}).get("beam_roi", (payload or {}).get("roi"))

    if raw is None:
        roi = None
    else:
        try:
            roi = {"cy": float(raw["cy"]), "cx": float(raw["cx"]),
                   "half": float(raw["half"])}
        except (KeyError, TypeError, ValueError):
            emit_error("A beam region is {cy, cx, half} in detector pixels; "
                       f"got {raw!r}.")
            _emit_state(state)
            return
        if roi["half"] <= 0:
            emit_error(f"A beam region needs a positive half-width; got "
                       f"{roi['half']:g}.")
            _emit_state(state)
            return

    if roi != state.beam_roi:
        state.beam_roi = roi
        # Only on a real move: a renderer re-sending a resting value is
        # ordinary, and throwing a solved alignment away for it would make the
        # stage impossible to hold on to.
        state.invalidate_reciprocal()
    state.message = ("Searching the whole pattern for the zero beam"
                     if roi is None else
                     f"Zero beam searched within {roi['half']:g} px of "
                     f"({roi['cx']:.0f}, {roi['cy']:.0f})")
    _emit_state(state)


def _default_beam_roi(state: MultiAngleLoaderState) -> None:
    """Put a region on the middle of the detector, once its size is known.

    A default rather than nothing, because nothing is the broken case: the
    finder is a centre of mass, and over a whole pattern it reads wherever the
    excited reflections happen to lie. An eighth of the detector about the
    middle holds a descanned zero beam and excludes the first reflections at
    this camera length; it is a starting point to drag, not a measurement.
    """
    if state.beam_roi is not None:
        return
    shapes = [member.detector_shape for member in state.members
              if member.detector_shape is not None]
    if not shapes:
        return
    height, width = (int(size) for size in shapes[0][:2])
    state.beam_roi = {"cy": height / 2.0, "cx": width / 2.0,
                      "half": max(8.0, round(min(height, width) / 8.0))}


def maped_align_real(session, plot, payload) -> None:
    """Solve the real-space alignment on a worker, from the virtual images.

    Which picture is registered is :attr:`MultiAngleLoaderState.virtual_image`:
    a name every member carries, or ``None`` for the computed one — the sum
    over the whole detector, which is what a member has when its file shipped
    no virtual image at all. Only the ``None`` case reads the members.

    ``params["max_shift"]`` (px, default 32) is the dialog's one solver knob:
    a correlation peak implying a larger shift is rejected, which is what stops
    a crystalline sample locking onto the wrong lattice translation. The other
    keys of :data:`_SOLVER_PARAMETERS` are accepted from a script.
    """
    from spyde.multiangle.align import best_real_space

    state = _loader_state(session)
    problem = _not_ready(state)
    if problem:
        emit_error(problem)
        _emit_state(state)
        return
    parameters = _solver_parameters((payload or {}).get("params"))
    reference = int(state.reference)
    # The list as it is NOW: the offsets this produces are indexed against it,
    # and a file added while it runs must not change what row 3 means.
    members = list(state.members)
    chosen = state.virtual_image

    def _work():
        if chosen is None:
            _ensure_reductions(state, members)
        images = np.stack([_member_image(state, member)
                           for member in members])

        def score(offsets):
            """How much sharper the members are summed at *offsets*.

            The same ratio the dialog shows, so what is optimised here is what
            the user is shown — and a run that cannot improve on the solver
            keeps the solver's answer.
            """
            trial = state.build_model(
                nav_offsets=np.asarray(offsets, dtype=np.int64),
                dp_offsets=np.zeros((len(images), 2), dtype=np.int64))
            evidence = _alignment_evidence(images, trial)
            if evidence is None:
                return float("-inf")
            gain = float(evidence.get("gain", float("nan")))
            return gain if np.isfinite(gain) else float("-inf")

        offsets, residuals, report = best_real_space(
            images, reference=reference, score=score, **parameters)
        for candidate, gain, failure in report["attempts"]:
            log.debug("real-space candidate %s -> %s", candidate,
                      failure or f"sharpness x{gain:.3f}")
        log.info("real-space alignment: sharpness x%.2f", report["gain"])
        return offsets, residuals

    named = "computed" if chosen is None else chosen
    _start_stage(session, state, attribute="real",
                 generation_key="_real_generation",
                 name="Real space",
                 message=f"Aligning {len(state.members)} members in real "
                         f"space ({named})…",
                 work=_work,
                 on_solved=lambda: _show_aligned_sum(session, state))


def maped_align_reciprocal(session, plot, payload) -> None:
    """Solve the reciprocal alignment on a worker, three ways.

    ``method="corners"`` (the default) sums the patterns over four small corner
    blocks of each scan, locates the beam in those four sums and reads the
    plane through them at the middle of the scan. It is the fast one — a
    fraction of a percent of each member, where the other two read a sample of
    every member — and it is the only one whose corners the user can move.

    ``method="beam"`` fits the beam in the member's mean pattern, which is the
    steadier measurement when the corners are not representative.
    ``method="correlate"`` registers the mean patterns against one another and
    needs no beam at all, which is what a pattern with a blocked or diffuse
    centre leaves you.
    """
    from spyde.multiangle import solve_reciprocal

    state = _loader_state(session)
    problem = _not_ready(state)
    if problem:
        emit_error(problem)
        _emit_state(state)
        return

    payload = payload or {}
    method = str(payload.get("method", DEFAULTS["method"]))
    if method not in RECIPROCAL_METHODS:
        emit_error(f"Unknown reciprocal alignment method {method!r}; "
                   f"expected one of {', '.join(RECIPROCAL_METHODS)}.")
        _emit_state(state)
        return
    parameters = dict(payload.get("params") or {})
    # From the state, not the payload: the region is something the user placed
    # and can see, so a solve started from anywhere has to use the one on
    # screen rather than whichever the caller happened to pass.
    roi = dict(state.beam_roi) if state.beam_roi else None
    reference = int(state.reference)
    members = list(state.members)          # see maped_align_real

    def _work():
        if method == "corners":
            _ensure_corner_sums(state, members)
            return _corner_offsets(members, reference, parameters, roi)
        _ensure_reductions(state, members)
        patterns = [member.pattern for member in members]
        if method == "beam":
            return _beam_offsets(patterns, reference, parameters, roi)
        return solve_reciprocal(np.stack(patterns), reference=reference,
                                **_solver_parameters(parameters))

    _start_stage(session, state, attribute="reciprocal",
                 generation_key="_reciprocal_generation",
                 name="Reciprocal space",
                 message=f"Aligning {len(state.members)} members in "
                         f"reciprocal space ({method})…",
                 work=_work)


def maped_commit(session, plot, payload) -> None:
    """Build the signal tree from the solved model.

    Everything past this point is
    :func:`~spyde.backend._session_multiangle.compose_multiangle_tree` — the
    same function the one-shot loader calls, given the same kind of model. The
    two doors differ in how the model was arrived at and in nothing else.
    """
    state = _loader_state(session)
    if not state.can_commit:
        emit_error("Needs two members and both alignments solved.")
        _emit_state(state)
        return

    generation = bump_generation(state, "_commit_generation")
    model = state.build_model()
    members = [member.signal for member in state.members]
    # The SAME images the real-space stage registered, virtual or computed —
    # they become the composed stack's per-member navigator planes, so a tree
    # drawn from one picture and aligned by another would be a tree whose
    # navigator disagrees with its own offsets.
    images = [_member_image(state, member) for member in state.members]

    state.busy = True
    state.message = f"Opening {model.n_members} multi-angle members…"
    emit_status(state.message)
    ipc.emit({"type": "loading", "busy": True, "text": state.message})
    _emit_state(state)

    def _work():
        session._await_dask()
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()
        # Belt and braces: the member rows already refuse a shape that does
        # not match the others, so this can only fire if that gate is ever
        # loosened — and a mismatch reaching the composition surfaces as an
        # array shape error naming neither file.
        _check_members(members, model.paths)
        compose_multiangle_tree(session, members, model, images)

    def _done(_result):
        # The alignment window was evidence for a decision that has now been
        # made, and it is a BARE figure: left open it is the active plot, so
        # the Plot Control dock shows no workflow and the acquisition the user
        # just opened is not the one in focus.
        _close_aligned_window(session, state)
        _report_commit(session, state, generation,
                       "Multi-angle acquisition opened.")

    def _failed(error):
        ipc.emit({"type": "loading", "busy": False, "text": ""})
        # BOTH channels, deliberately: the status bar reaches a user whose
        # dialog is already gone, and the snapshot reaches a dialog that is
        # still up. Either alone leaves one of the two cases silent.
        emit_error(f"Failed to open the multi-angle acquisition: {error}")
        _on_main(session, lambda: _report_commit(
            session, state, generation,
            f"Failed to open the acquisition: {error}"))

    run_on_worker(session, _work, name="maped-commit",
                  on_done=_done, on_error=_failed)


def _close_aligned_window(session, state) -> None:
    """Take down the real-space alignment window, if one is up.

    Shares the ``_forget_window`` route with ``discard_multiangle_loader`` so
    the renderer removes it and the figure is released, rather than the handle
    simply being dropped. Logged at WARNING on failure rather than debug: the
    symptom is a window left on screen owning the Plot Control dock, which is
    not something to find out about from a log level nobody runs at.
    """
    window = getattr(state, "aligned_window", None)
    if window is None:
        return
    try:
        session._forget_window(window.window_id)
    except Exception:
        log.warning("closing the real-space alignment window failed; it will "
                    "stay on screen.", exc_info=True)
    state.aligned_window = None


def _report_commit(session, state, generation: int, message: str) -> None:
    """Announce how the commit ended — open dialog or not.

    Every other result here is dropped when its generation is stale, because a
    superseded result describes a state nobody is in. A commit is the
    exception: the renderer sends it and CLOSES the dialog, so the ordinary
    rule would make a FAILED commit silent — the one outcome the user cannot
    do without. A closed loader gets the outcome on the EMPTY snapshot, which
    says what happened without putting the dismissed member list back.
    """
    open_here = (is_current(state, "_commit_generation", generation)
                 and getattr(session, "_multiangle_loader", None) is state)
    if open_here:
        state.busy = False
        state.message = message
        _emit_state(state)
        return
    ipc.emit({**state_message(None), "message": message})
