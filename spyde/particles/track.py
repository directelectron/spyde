"""
track.py — frame-to-frame linking and the event stream. Plan steps C1 and C2.

One pass over the frames, one `linear_sum_assignment` per frame pair, and the
assignment's **leftovers are the physics**: a detection nothing was assigned to is
a birth, a track nothing was assigned from is a death, and the two-body cases
(merge, split) are read off those leftovers by a post-pass.

Why `scipy.optimize.linear_sum_assignment` and not trackpy
----------------------------------------------------------
trackpy's model (centroid distance, gated by a search radius, optional memory) is
the right one and is what this implements — but scipy is already a core dependency,
and the whole linker plus the event post-pass is under 400 lines of code. Adding a
dependency to get a Hungarian solve we already ship is not a trade worth making.

Units — read this before setting `max_dist`
-------------------------------------------
:mod:`spyde.particles.measure` writes centroids in **calibrated units** (pixels x
``scale``), the same units as ``area``, ``equiv_diameter`` and everything else in
the property row. So every distance here — :attr:`LinkParams.max_dist`,
``merge_dist``, ``split_dist``, the trajectories on :class:`LinkResult` — is in
``particles.units``, never pixels. A :class:`~spyde.drift.model.DriftModel`,
however, is in **pixels** (it is a shift applied to an image), so
:func:`sample_frame_positions` converts to pixels, asks the model, and converts
back. That conversion is in exactly one place for the same reason ``measure.py``
calibrates in exactly one place.

Why the gate is a maximum-cardinality trick rather than a big square matrix
---------------------------------------------------------------------------
The textbook formulation (Jaqaman 2008) pads the cost matrix with dummy
rows/columns priced at the gate, giving a ``(n_t + n_d)`` square problem in which
an unmatched track and an unmatched detection are both first-class outcomes. That
is what makes "a detection may go unmatched rather than be forced into a bad pair"
true, and it is the property that matters.

The rectangular matrix used here has the same optimum for strictly less work.
Infeasible pairs (distance above the gate) get a constant sentinel large enough
that the solver minimises the *number* of them before it looks at any real cost;
those pairs are then discarded. Both formulations therefore compute a maximum-
cardinality matching over the feasible pairs and, among those, the minimum total
cost — the padded version can never prefer two dummies to a feasible pair, because
a feasible pair costs less than the gate while two dummies cost twice it. The
rectangular problem is ``n_t x n_d`` instead of ``(n_t + n_d)^2``, i.e. 4x smaller
at 500 particles per frame, and `linear_sum_assignment` is superlinear in the
matrix size.

Cost, and what the gate applies to
----------------------------------
Cost is centroid distance, optionally plus a property-similarity penalty
(:attr:`LinkParams.property_weight`, **off by default**). The **gate is on
distance alone** — the penalty only re-orders pairs that are already admissible.
Letting the penalty push a pair over the gate would silently turn ``max_dist``
into "max_dist minus however dissimilar these two happen to look", which is not
what a user setting a search radius means.

The penalty is off by default because on real data a particle's measured area
fluctuates frame to frame by far more than its centroid moves — the area of a
threshold-defined region swings with noise, the centroid barely does — so
weighting area *adds* noise to a cost that was already the reliable signal. It
earns its keep only when positions are genuinely ambiguous (dense fields, fast
motion), which is why it is a knob and not a constant.

Cost at scale
-------------
Per frame this is ``O(n_t * n_d)`` to build the matrix and up to ``O(n^3)`` for the
assignment, with *n* the particles in ONE frame — never in the movie. Measured on
this box, 50 frames of synthetic tracks (link only, no segmentation):

===================  ============  ========================
particles per frame  ms per frame  extrapolated 3000 frames
===================  ============  ========================
50                           0.23                    0.7 s
100                          0.47                    1.4 s
200                          1.19                    3.6 s
500                          9.5                    28.6 s
800                         23.8                    71 s
===================  ============  ========================

So the plan's target (3000 frames, ~500 particles each) is **~29 s**, negligible
beside segmenting 3000 frames of 2048^2 — but the growth is clearly superlinear
(~n^2.2 over that range), so a field of several thousand particles per frame would
need the cost matrix restricted to near neighbours (a KD-tree query inside
``max_dist``, then a sparse assignment) rather than built dense. That is not done
here: it is unnecessary at the stated scale and would add a second code path with no
data to validate it against. For reference the whole 24-frame fixture links in
1.8 ms (74 us/frame at ~6 particles), against 486 ms to segment and measure it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from spyde.signals.particles import COL, SpyDEParticles

log = logging.getLogger(__name__)

#: The four event types. Order is the navigator lane order (plan C2) — green
#: birth, red death, mauve merge, yellow split.
EVENT_KINDS: tuple[str, ...] = ("birth", "death", "merge", "split")

#: Properties the optional similarity term compares. Both are in the property row
#: for every engine; ``intensity_mean`` is NaN when ``measure_frame`` ran without
#: an intensity image, which is handled as "unknown", not as "very different".
DEFAULT_SIMILARITY_PROPERTIES: tuple[str, ...] = ("area", "intensity_mean")

# Floor for the ADAPTIVE merge/split radius, in pixels (scaled to calibrated units
# at use). The adaptive radius is the particle's own equivalent diameter, which is
# the right scale — a big particle absorbs a neighbour from further away than a
# small one does — but a 1-2 px detection has an equivalent diameter near zero,
# and a zero radius means its merges are never detected. Two pixels is the
# smallest radius at which "these two detections became one" is even meaningful.
_ADAPTIVE_RADIUS_FLOOR_PX = 2.0


# ── events ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParticleEvent:
    """One thing that happened to a track, at a frame.

    Frozen because the event stream is handed to the navigator lane, the Events
    table and the report embed; a record three surfaces share must not be
    mutable in one of them.

    Parameters
    ----------
    frame
        The frame the event is **observed at**. For a birth that is the track's
        first detected frame. For a death it is ``last_detected + 1`` — the first
        frame the particle is *gone*, which is the convention the synthetic
        fixture's ``death`` column uses and the frame a user would point at and
        call the dissolution.
    kind
        One of :data:`EVENT_KINDS`.
    tracks
        The track ids involved. Birth/death: one. Merge: ``(absorbed, survivor)``.
        Split: ``(parent, fragment)``.
    particles
        Global particle indices into ``SpyDEParticles.flat_buffer``, in the same
        order as *tracks* — so the Events table can jump straight to a row and the
        overlay can highlight the exact detections.
    """

    frame: int
    kind: str
    tracks: tuple[int, ...]
    particles: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-safe dict — this is what crosses the IPC boundary."""
        return {
            "frame": int(self.frame),
            "kind": str(self.kind),
            "tracks": [int(t) for t in self.tracks],
            "particles": [int(i) for i in self.particles],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParticleEvent":
        return cls(
            frame=int(d["frame"]),
            kind=str(d["kind"]),
            tracks=tuple(int(t) for t in d.get("tracks", ())),
            particles=tuple(int(i) for i in d.get("particles", ())),
        )


def events_to_records(events: Sequence[ParticleEvent]) -> list[dict[str, Any]]:
    """The event stream as JSON-safe records, ready for IPC or CSV."""
    return [e.to_dict() for e in events]


def events_from_records(records: Sequence[dict[str, Any]]) -> list[ParticleEvent]:
    """Inverse of :func:`events_to_records`."""
    return [ParticleEvent.from_dict(r) for r in records]


def event_counts(events: Sequence[ParticleEvent], n_frames: int) -> dict[str, np.ndarray]:
    """``{kind: (n_frames,) float32}`` counts — the navigator's event lane (C2).

    A dict of separate traces rather than one stacked array because each kind gets
    its own colour and its own row in the lane; a caller that wants the total sums
    them.
    """
    out = {k: np.zeros(int(n_frames), dtype=np.float32) for k in EVENT_KINDS}
    for e in events:
        if 0 <= e.frame < int(n_frames) and e.kind in out:
            out[e.kind][e.frame] += 1.0
    return out


# ── parameters ───────────────────────────────────────────────────────────────

@dataclass
class LinkParams:
    """Linker parameters. Distances are in the particles' calibrated units.

    Parameters
    ----------
    max_dist
        Search radius: a track and a detection further apart than this are never
        linked. **In ``particles.units``**, not pixels (see the module docstring).
        The default is deliberately generous — on the synthetic fixture
        (``scale=0.5`` nm/px) the fastest particle moves 2.2 px = 1.1 nm per frame,
        so 10.0 is ~9x the true step. The asymmetry justifies it: a gate that is
        too tight **fragments** a track, and a fragmented trajectory cannot be
        repaired downstream, while a gate that is too loose only matters when two
        particles are within it of each other — and there the assignment still
        picks the globally cheapest pairing. Measured on the fixture, every gate
        from 4 px to 80 px recovers exactly the same 7 tracks and the same events;
        at 2 px the tracks fragment (8 tracks, 3 spurious deaths) and at 1.2 px it
        collapses (26 tracks, 19 deaths, 4 phantom splits). The usable window is
        wide upward and sharp downward, so err high. Dense fields should tighten it.
    memory
        Frames a track may go undetected and still be re-linked afterwards. This is
        what makes a blinking detection **one** track instead of several. 0 means a
        track ends the moment it is missed. Note the gate is measured from the
        track's last *seen* position and is not widened by the gap, so a fast
        particle that blinks may still fall outside it.
    property_weight
        Weight of the property-similarity penalty, expressed as a multiple of
        *max_dist* so it is scale-free. 0 (default) is distance only; see the
        module docstring for why that is the default.
    properties
        Which columns the similarity term compares.
    merge_dist, split_dist
        Radius for the merge / split post-pass. ``None`` (default) is **adaptive**:
        the particle's own ``equiv_diameter``, floored at 2 px. Adaptive is right
        because the signature of a merge is a centroid jumping to the join of two
        bodies, and that jump is a fraction of the body size — a fixed radius that
        works for 50 px particles invents merges among 5 px ones.
    initial_births
        Whether detections in frame 0 emit ``birth`` events. Default True: it makes
        the event stream a *complete* description of the assignment (every track has
        exactly one birth, so ``len(births) == n_tracks``), and a consumer that
        wants only nucleations filters ``frame > 0`` — see
        :meth:`LinkResult.events_of`. The reverse is not recoverable: drop them and
        nothing downstream can tell which tracks were present from the start.
    """

    max_dist: float = 10.0
    memory: int = 0
    property_weight: float = 0.0
    properties: tuple[str, ...] = DEFAULT_SIMILARITY_PROPERTIES
    merge_dist: float | None = None
    split_dist: float | None = None
    initial_births: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_dist) or self.max_dist <= 0:
            raise ValueError(f"max_dist must be finite and > 0; got {self.max_dist}")
        if self.memory < 0:
            raise ValueError(f"memory must be >= 0; got {self.memory}")
        if self.property_weight < 0:
            raise ValueError(
                f"property_weight must be >= 0; got {self.property_weight}")
        for name in self.properties:
            if name not in COL:
                raise KeyError(f"unknown property column {name!r} in properties")
        for name in ("merge_dist", "split_dist"):
            v = getattr(self, name)
            if v is not None and (not np.isfinite(v) or v <= 0):
                raise ValueError(f"{name} must be None or finite and > 0; got {v}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_dist": float(self.max_dist),
            "memory": int(self.memory),
            "property_weight": float(self.property_weight),
            "properties": list(self.properties),
            "merge_dist": None if self.merge_dist is None else float(self.merge_dist),
            "split_dist": None if self.split_dist is None else float(self.split_dist),
            "initial_births": bool(self.initial_births),
        }


# ── result ───────────────────────────────────────────────────────────────────

@dataclass
class LinkResult:
    """Track ids, the event stream, and the positions the linking actually used.

    ``track_id`` is returned as a **parallel array** rather than written straight
    into the buffer, so a link can be inspected, compared against another
    parameter choice, or thrown away without having mutated the particle table.
    :meth:`apply` performs the write when you want it.

    Parameters
    ----------
    track_id
        ``(n_particles,)`` int32, contiguous from 0. Every particle belongs to
        exactly one track, so there is no -1 in a completed link.
    events
        Chronological (by ``frame``, then by track id).
    positions
        ``(n_particles, 2)`` float64 ``(y, x)`` in **calibrated units**, in
        :attr:`reference`'s frame — i.e. what the cost matrix saw. Kept because a
        trajectory read from the buffer would be in the lab frame even when the
        link ran drift-corrected, and silently mixing the two is the bug this
        field exists to prevent.
    frame_index
        ``(n_particles,)`` int32 frame of each particle, taken from the CSR row
        pointers (the authoritative frame index) rather than the float ``t``
        column.
    reference
        ``"lab"`` or ``"sample"`` — which frame of reference :attr:`positions`
        and every trajectory are in.
    track_first_frame, track_last_frame, track_first_index, track_last_index
        ``(n_tracks,)`` per-track endpoints. Computed for the event pass and
        exposed because the kymograph, the table dock and the trails overlay all
        want them and re-deriving them is a scan of the whole buffer.
    """

    track_id: np.ndarray
    events: list[ParticleEvent]
    positions: np.ndarray
    frame_index: np.ndarray
    reference: str
    track_first_frame: np.ndarray
    track_last_frame: np.ndarray
    track_first_index: np.ndarray
    track_last_index: np.ndarray
    n_frames: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.track_id = np.ascontiguousarray(self.track_id, dtype=np.int32)
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        self.frame_index = np.ascontiguousarray(self.frame_index, dtype=np.int32)
        # Chronological order WITHIN each track, in one stable sort: the buffer is
        # sorted by frame, so a stable sort on track id leaves each track's rows in
        # increasing frame order. This is what makes `trajectory` a slice.
        self._order = np.argsort(self.track_id, kind="stable")
        counts = np.bincount(self.track_id, minlength=self.n_tracks) \
            if self.track_id.size else np.zeros(0, np.int64)
        self._starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    @property
    def n_tracks(self) -> int:
        return int(self.track_first_frame.size)

    @property
    def n_particles(self) -> int:
        return int(self.track_id.size)

    # ── writing back ─────────────────────────────────────────────────────────

    def apply(self, particles: SpyDEParticles) -> SpyDEParticles:
        """Write :attr:`track_id` into the buffer's ``track_id`` column, in place.

        The column is float32, which represents integers exactly up to 2**24 — 16.7M
        tracks, against a target scale of 1.5M particles (plan §0.5), so an id can
        never be rounded to a neighbouring track's.
        """
        if len(particles.flat_buffer) != self.track_id.size:
            raise ValueError(
                f"this result is for {self.track_id.size} particles but the table "
                f"has {len(particles.flat_buffer)} — it came from a different link"
            )
        particles.flat_buffer[:, COL["track_id"]] = self.track_id.astype(np.float32)
        return particles

    # ── per-track access ─────────────────────────────────────────────────────

    def track_indices(self, track: int) -> np.ndarray:
        """Global particle indices of one track, in chronological order."""
        t = int(track)
        if not 0 <= t < self.n_tracks:
            raise IndexError(f"track {t} outside 0..{self.n_tracks - 1}")
        return self._order[self._starts[t]:self._starts[t + 1]]

    def trajectory(self, track: int) -> np.ndarray:
        """``(k, 3)`` ``[frame, y, x]`` for one track, in :attr:`reference`'s frame.

        Positions are in calibrated units. Frames are not necessarily contiguous —
        with ``memory > 0`` a track can skip frames, and the gap is visible as a
        jump in the first column rather than being interpolated over. Inventing a
        position for a frame the particle was not detected in would put a
        measurement in the table that nothing measured.
        """
        idx = self.track_indices(track)
        out = np.empty((idx.size, 3), dtype=np.float64)
        out[:, 0] = self.frame_index[idx]
        out[:, 1:] = self.positions[idx]
        return out

    def track_lengths(self) -> np.ndarray:
        """``(n_tracks,)`` number of frames each track was DETECTED in.

        Not ``last - first + 1``: with ``memory > 0`` those differ, and the
        difference is exactly the QC signal (a track detected in 4 of 30 frames is
        probably noise) that plan C3's lifetime sort is for.
        """
        return np.diff(self._starts).astype(np.int64)

    def track_at(self, frame: int) -> np.ndarray:
        """Track ids present in *frame*, in buffer order.

        ``searchsorted``, not a boolean scan: the overlay calls this on every
        navigator move, and ``frame_index`` is non-decreasing (the buffer is sorted
        by frame), so an O(log N) slice is available where an O(N) comparison over
        1.5M rows would otherwise run per frame.
        """
        f = int(frame)
        lo, hi = np.searchsorted(self.frame_index, [f, f + 1])
        return self.track_id[lo:hi]

    # ── events ───────────────────────────────────────────────────────────────

    def events_of(self, kind: str | None = None, *,
                  exclude_initial: bool = False) -> list[ParticleEvent]:
        """Events of one *kind*, optionally dropping frame-0 births.

        *exclude_initial* is the "only real nucleations" filter — see
        :attr:`LinkParams.initial_births` for why frame-0 births are recorded at
        all.
        """
        if kind is not None and kind not in EVENT_KINDS:
            raise ValueError(
                f"unknown event kind {kind!r}; expected one of {', '.join(EVENT_KINDS)}")
        out = [e for e in self.events if kind is None or e.kind == kind]
        if exclude_initial:
            out = [e for e in out if not (e.kind == "birth" and e.frame == 0)]
        return out

    def event_counts(self) -> dict[str, np.ndarray]:
        """Per-frame counts per kind — see :func:`event_counts`."""
        return event_counts(self.events, self.n_frames)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary. Arrays stay out — save those with the particles."""
        return {
            "n_tracks": self.n_tracks,
            "n_particles": self.n_particles,
            "n_frames": int(self.n_frames),
            "reference": self.reference,
            "params": dict(self.params),
            "events": events_to_records(self.events),
        }

    def __repr__(self) -> str:
        counts = {k: len(self.events_of(k)) for k in EVENT_KINDS}
        return (
            f"LinkResult({self.n_tracks} tracks over {self.n_frames} frames, "
            f"reference={self.reference!r}, events={counts})"
        )


# ── coordinates ──────────────────────────────────────────────────────────────

def frame_indices(particles: SpyDEParticles) -> np.ndarray:
    """``(n_particles,)`` int32 frame index per particle, from the CSR pointers.

    The ``t`` column carries the same number, but the row pointers are the
    *definition* of which frame a row belongs to — and they are integers, whereas
    ``t`` is a float32 that a caller could in principle have written a
    non-integral value into.
    """
    counts = np.diff(particles.t_offsets)
    return np.repeat(np.arange(particles.n_frames, dtype=np.int32),
                     counts).astype(np.int32)


def sample_frame_positions(particles: SpyDEParticles, drift) -> np.ndarray:
    """Lab-frame centroids mapped into the drift-corrected (sample) frame.

    ``(n_particles, 2)`` float64 ``(y, x)``, **still in calibrated units** — the
    return value is directly comparable with the stored centroids.

    The conversion to and from pixels is what this function exists for.
    :class:`~spyde.drift.model.DriftModel` is defined on image pixels (its shifts
    are what you pass to ``scipy.ndimage.shift``) while the property row is
    calibrated, so a caller adding ``model.shifts[t]`` straight onto ``y``/``x`` is
    off by a factor of ``scale`` — an over- or under-correction that still yields a
    smooth, visibly-improved trajectory and so does not announce itself. Only the
    ``scale == 1`` case, where nobody looks, comes out right.

    Going through ``to_sample_frame`` rather than adding the shifts here also means
    a future non-rigid model, whose mapping is not a plain addition, needs no change
    on this side.
    """
    if drift.n_frames < particles.n_frames:
        raise ValueError(
            f"drift model covers {drift.n_frames} frames but the particles span "
            f"{particles.n_frames} — a shorter model would silently reuse the "
            "last shift for every frame beyond it"
        )
    scale = float(particles.scale) or 1.0
    pos_px = particles.flat_buffer[:, [COL["y"], COL["x"]]].astype(np.float64) / scale
    idx = frame_indices(particles).astype(np.intp)
    if pos_px.size == 0:
        return pos_px
    return np.asarray(drift.to_sample_frame(pos_px, idx), dtype=np.float64) * scale


# ── the linker ───────────────────────────────────────────────────────────────

def link(
    particles: SpyDEParticles,
    params: LinkParams | None = None,
    *,
    drift=None,
    apply: bool = False,
    **kwargs: Any,
) -> LinkResult:
    """Link detections into tracks and extract the event stream.

    Parameters
    ----------
    particles
        The CSR table from segment + measure. Not modified unless *apply*.
    params
        :class:`LinkParams`. Individual fields may instead be passed as keyword
        arguments (``link(p, max_dist=4, memory=1)``), which is what the wizard's
        parameter dict does.
    drift
        Optional :class:`~spyde.drift.model.DriftModel`. When given, linking and
        every reported trajectory run in the **sample** frame — the stage's motion
        removed, so a static particle's trajectory is a point. When ``None``, the
        lab frame, i.e. the raw centroids as measured. Both are valid answers to
        different questions ("did the particle move, or did the stage?", plan A9),
        which is why the choice is the caller's and is recorded on the result.
    apply
        Write the ids into ``particles.flat_buffer``'s ``track_id`` column as well
        as returning them. Off by default so a link is a pure computation.

    Returns
    -------
    LinkResult

    Notes
    -----
    Cost per frame is ``O(n_t * n_d)`` to build the matrix and up to
    ``O(n^3)`` for the assignment, with *n* the number of particles in ONE frame —
    never the whole movie. Nothing here reads pixel data at all.
    """
    if kwargs:
        base = params.to_dict() if params is not None else LinkParams().to_dict()
        unknown = set(kwargs) - set(base)
        if unknown:
            raise TypeError(
                f"unknown link parameter(s): {', '.join(sorted(unknown))}")
        base.update(kwargs)
        base["properties"] = tuple(base["properties"])
        params = LinkParams(**base)
    p = params or LinkParams()

    n_frames = particles.n_frames
    n_particles = particles.n_particles
    fidx = frame_indices(particles)

    if drift is not None:
        positions = sample_frame_positions(particles, drift)
        reference = "sample"
    else:
        positions = particles.flat_buffer[:, [COL["y"], COL["x"]]].astype(np.float64)
        reference = "lab"

    track_id = _assign_tracks(particles, positions, p)

    n_tracks = int(track_id.max()) + 1 if track_id.size else 0
    if n_tracks:
        # `track_id` is contiguous from 0 and the buffer is sorted by frame, so a
        # STABLE sort groups each track's rows in chronological order — the endpoints
        # are then the group's first and last element, with no per-track scan.
        order = np.argsort(track_id, kind="stable")
        starts = np.concatenate(
            [[0], np.cumsum(np.bincount(track_id, minlength=n_tracks))])
        first_index = order[starts[:-1]].astype(np.int64)
        last_index = order[starts[1:] - 1].astype(np.int64)
        first_frame = fidx[first_index].astype(np.int32)
        last_frame = fidx[last_index].astype(np.int32)
    else:
        first_index = last_index = np.zeros(0, np.int64)
        first_frame = last_frame = np.zeros(0, np.int32)

    events = _extract_events(
        particles, positions, track_id, n_frames,
        first_frame=first_frame, last_frame=last_frame,
        first_index=first_index, last_index=last_index, p=p,
    )

    result = LinkResult(
        track_id=track_id,
        events=events,
        positions=positions,
        frame_index=fidx,
        reference=reference,
        track_first_frame=first_frame,
        track_last_frame=last_frame,
        track_first_index=first_index,
        track_last_index=last_index,
        n_frames=n_frames,
        params=p.to_dict(),
    )
    log.info("[track] %d particles over %d frames -> %d tracks (%s frame), "
             "%d events", n_particles, n_frames, result.n_tracks, reference,
             len(events))
    if apply:
        result.apply(particles)
    return result


def _assign_tracks(particles: SpyDEParticles, positions: np.ndarray,
                   p: LinkParams) -> np.ndarray:
    """The forward pass: one assignment per frame pair. Returns ``(N,)`` int32 ids.

    Ids are handed out in (frame, row-within-frame) order, which is why they are
    stable: the same table linked twice produces the same numbering, and there is
    no dictionary iteration or set ordering anywhere in the loop.
    """
    from scipy.optimize import linear_sum_assignment

    n_particles = particles.n_particles
    track_id = np.full(n_particles, -1, dtype=np.int32)
    if n_particles == 0:
        return track_id

    prop = _similarity_matrix_source(particles, p)

    # Live-track registry. Lists, not arrays: they are appended to once per new
    # track and read once per frame, and the per-frame cost is dominated by the
    # assignment. `active` holds only tracks still inside the memory window.
    last_frame: list[int] = []
    last_index: list[int] = []
    active: list[int] = []

    def start_track(gi: int, t: int) -> None:
        tid = len(last_frame)
        last_frame.append(t)
        last_index.append(gi)
        track_id[gi] = tid
        active.append(tid)

    for t in range(particles.n_frames):
        det = particles.indices_at(t)

        if t > 0 and active:
            # Retire anything outside the memory window. `memory + 1` because a
            # track last seen at t-1 has a gap of 1 and must always be eligible.
            active[:] = [tid for tid in active
                         if t - last_frame[tid] <= p.memory + 1]

        if det.size and active:
            src = np.fromiter((last_index[tid] for tid in active),
                              dtype=np.int64, count=len(active))
            cost, feasible = _cost_matrix(positions, src, det, prop, p)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if not feasible[r, c]:
                    continue                  # a sentinel pair — see the module docstring
                tid = active[r]
                gi = int(det[c])
                track_id[gi] = tid
                last_frame[tid] = t
                last_index[tid] = gi

        for gi in det:
            if track_id[gi] < 0:
                start_track(int(gi), t)

    return track_id


def _similarity_matrix_source(particles: SpyDEParticles,
                              p: LinkParams) -> np.ndarray | None:
    """``(N, k)`` float64 of the compared properties, or None when weight is 0."""
    if p.property_weight <= 0 or not p.properties:
        return None
    cols = [COL[name] for name in p.properties]
    return particles.flat_buffer[:, cols].astype(np.float64)


def _cost_matrix(positions: np.ndarray, src: np.ndarray, det: np.ndarray,
                 prop: np.ndarray | None,
                 p: LinkParams) -> tuple[np.ndarray, np.ndarray]:
    """``(cost, feasible)`` for one frame pair.

    Infeasible entries get a constant sentinel chosen so the solver minimises
    their *count* before it looks at any real cost — which is what makes discarding
    them afterwards equivalent to the padded square formulation (module docstring).
    The sentinel is derived from the actual matrix rather than being a magic
    literal, so it cannot be outgrown by a large ``max_dist`` or a big property
    penalty.
    """
    a = positions[src]                                  # (n_t, 2)
    b = positions[det]                                  # (n_d, 2)
    d = np.hypot(a[:, 0, None] - b[None, :, 0],
                 a[:, 1, None] - b[None, :, 1])

    # The gate is on DISTANCE ONLY. See the module docstring.
    feasible = d <= float(p.max_dist)

    cost = d.copy()
    if prop is not None:
        cost += float(p.property_weight) * float(p.max_dist) * \
            _property_dissimilarity(prop[src], prop[det])

    if feasible.any():
        big = (min(len(src), len(det)) + 1) * float(cost[feasible].max()) + 1.0
    else:
        big = 1.0
    cost = np.where(feasible, cost, big)
    return cost, feasible


def _property_dissimilarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(n_t, n_d)`` mean symmetric relative difference in 0..1.

    ``|a - b| / (|a| + |b|)`` rather than ``|a - b| / a``: it is bounded, symmetric,
    and needs no reference value, so one expression works for area (thousands of
    nm^2) and for a normalised intensity (fractions of one) without per-column
    scaling.

    A NaN on either side is **dropped from the mean**, not counted as zero.
    ``intensity_mean`` is NaN whenever ``measure_frame`` ran without an intensity
    image, and both alternatives are wrong in a way that is hard to see from the
    outside: counting it as maximally different makes the term a uniform offset on
    every pair, and counting it as zero *dilutes* the columns that were measured —
    with one of two properties missing, a requested weight of 1.0 would silently
    act as 0.5. Averaging over the known columns keeps the weight meaning what the
    caller asked for.
    """
    valid = np.isfinite(a)[:, None, :] & np.isfinite(b)[None, :, :]
    num = np.abs(a[:, None, :] - b[None, :, :])
    den = np.abs(a[:, None, :]) + np.abs(b[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        # den == 0 means both values are 0, i.e. genuinely identical.
        rel = np.where(den > 0, num / den, 0.0)
    rel = np.where(valid & np.isfinite(rel), rel, 0.0)
    n = valid.sum(axis=2)
    return np.where(n > 0, rel.sum(axis=2) / np.maximum(n, 1), 0.0)


# ── the event post-pass ──────────────────────────────────────────────────────

def _extract_events(particles, positions, track_id, n_frames, *,
                    first_frame, last_frame, first_index, last_index,
                    p: LinkParams) -> list[ParticleEvent]:
    """Turn the assignment's leftovers into the event stream.

    Every track contributes at most two events: one for its start (``birth`` or
    ``split``) and one for its end (``death`` or ``merge``). Merge and split
    **replace** the death and birth they explain rather than accompanying them —
    a particle that was absorbed did not dissolve, and reporting both would put a
    red dissolution flag on the navigator lane at a frame where nothing dissolved.
    The cost of that choice is that the event stream no longer closes the count
    arithmetic by itself; the accounting is
    ``count(t) - count(t-1) = births + splits - deaths - merges``.

    A track still detected in the FINAL frame gets no end event: the movie running
    out is not a dissolution.

    Walks FRAMES, not tracks, so only two ``{track: row}`` maps are ever resident.
    A single ``frame -> {track: row}`` index over the whole movie would be 1.5M dict
    entries at the target scale (plan §0.5) — hundreds of MB of Python objects to
    answer a question that only ever spans two adjacent frames.
    """
    n_tracks = int(first_frame.size)
    if n_tracks == 0 or n_frames == 0:
        return []

    starts_at: list[list[int]] = [[] for _ in range(n_frames)]
    ends_at: list[list[int]] = [[] for _ in range(n_frames)]
    for tid in range(n_tracks):
        starts_at[int(first_frame[tid])].append(tid)
        t1 = int(last_frame[tid])
        if t1 < n_frames - 1:
            # The event frame is the first frame the particle is GONE.
            ends_at[t1 + 1].append(tid)

    equiv = particles.flat_buffer[:, COL["equiv_diameter"]].astype(np.float64)
    floor = _ADAPTIVE_RADIUS_FLOOR_PX * (float(particles.scale) or 1.0)

    def radius(gi: int, override: float | None) -> float:
        if override is not None:
            return float(override)
        v = equiv[gi]
        # An unmeasured diameter (no masks, or a degenerate region) must not become
        # a zero radius that silently disables the whole post-pass.
        return max(float(v) if np.isfinite(v) else 0.0, floor)

    events: list[ParticleEvent] = []
    prev_map: dict[int, int] = {}

    for t in range(n_frames):
        # The maps are only ever read by a candidate event at t (which needs both t
        # and t-1), so a frame with no candidate at t and none at t+1 needs no map at
        # all. On a long stable movie that skips nearly every frame; when it does
        # build one, the next iteration is guaranteed to want it.
        wants = bool(ends_at[t]) or (t > 0 and bool(starts_at[t]))
        nxt = t + 1
        wants_next = nxt < n_frames and (bool(ends_at[nxt]) or bool(starts_at[nxt]))
        cur_map = ({int(track_id[gi]): int(gi) for gi in particles.indices_at(t)}
                   if (wants or wants_next) else {})

        for tid in starts_at[t]:
            gi = int(first_index[tid])
            if t == 0:
                if p.initial_births:
                    events.append(ParticleEvent(0, "birth", (tid,), (gi,)))
                continue
            # SPLIT: a track that continues across t-1 -> t, whose position BEFORE
            # the split is within its own body radius of this newcomer.
            parent = _nearest_continuing(
                cur_map, prev_map, positions, probe=positions[gi], exclude=tid,
                measure_at=prev_map,
                radius_of=lambda pgi: radius(pgi, p.split_dist),
            )
            if parent is None:
                events.append(ParticleEvent(t, "birth", (tid,), (gi,)))
            else:
                ptid, pgi = parent
                events.append(ParticleEvent(t, "split", (ptid, tid), (pgi, gi)))

        for tid in ends_at[t]:
            gi = int(last_index[tid])
            merge_r = radius(gi, p.merge_dist)
            # MERGE: a track that continues across t-1 -> t whose position AT t is
            # within the dying particle's body radius of where the dying particle
            # was at t-1 — i.e. a centroid that jumped to the join of two bodies.
            survivor = _nearest_continuing(
                cur_map, prev_map, positions, probe=positions[gi], exclude=tid,
                measure_at=cur_map, radius_of=lambda _pgi, r=merge_r: r,
            )
            if survivor is None:
                events.append(ParticleEvent(t, "death", (tid,), (gi,)))
            else:
                stid, sgi = survivor
                events.append(ParticleEvent(t, "merge", (tid, stid), (gi, sgi)))

        prev_map = cur_map

    events.sort(key=lambda e: (e.frame, EVENT_KINDS.index(e.kind), e.tracks))
    return events


def _nearest_continuing(cur_map, prev_map, positions, *, probe, exclude,
                        measure_at, radius_of):
    """Nearest track present in BOTH frames, measured on *measure_at*'s side.

    This is the whole merge/split rule, and the "both frames" requirement is what
    keeps it honest. A one-to-one assignment cannot express two-to-one, so the
    two-body events have to be inferred from a died track sitting next to a
    surviving one — and the only way to tell a *survivor* from another newcomer is
    that the survivor was already there the frame before.

    *measure_at* selects which side is compared with *probe*: ``cur_map`` for a
    merge (where did the surviving centroid jump TO) and ``prev_map`` for a split
    (where was the parent BEFORE it broke up). *radius_of* is passed the global row
    index on the measured side, so a split can size its gate by the parent's own
    body while a merge sizes it by the dying particle's.

    Known limits, stated rather than hidden:

    * **It cannot see a merge earlier than the segmenter does.** Measured on the
      synthetic fixture, whose two converging discs geometrically touch at frame
      14: with watershed splitting ON they remain two separate detections until
      **frame 18**, and with it OFF their thresholded blobs already connect at
      **frame 12**. The linker reports the frame the detections became one, which
      is a property of segmentation, not of linking, and no post-pass can recover
      the other frame from the table alone.
    * **A three-body coincidence is misread.** A track that genuinely dissolves
      within one body-diameter of a surviving neighbour is reported as a merge.
    * **A fragmentation in which the parent's OWN track ends is reported as two
      births, not a split**, because then neither newcomer was present the frame
      before. Requiring the parent to continue is what stops every ordinary birth
      that happens to appear beside an unrelated particle from being called a
      split; the trade is deliberate.
    * **Merge and split are exclusive per track end/start.** Three tracks
      collapsing into one in a single frame produce two merge events sharing one
      survivor, which is the honest reading, but a track cannot be recorded as both
      merging and splitting at the same frame.
    * **Which track survives a SYMMETRIC merge is a genuine tie**, decided by the
      assignment rather than by this rule. Two equal bodies merging put their joint
      centroid the same distance from both, so either can be the survivor. Measured
      on the fixture: linking in the lab frame reports ``tracks=(5, 4)`` and linking
      the same table drift-corrected reports ``(4, 5)`` — same merge, same frame,
      roles swapped, identical set of tracks and track lengths. Do not read meaning
      into which id survives.
    """
    best = None
    best_d = np.inf
    for tid, gi in cur_map.items():
        if tid == exclude or tid not in prev_map:
            continue
        mgi = int(measure_at[tid])
        d = float(np.hypot(*(probe - positions[mgi])))
        if d <= radius_of(mgi) and d < best_d:
            best, best_d = (tid, gi), d
    return best
