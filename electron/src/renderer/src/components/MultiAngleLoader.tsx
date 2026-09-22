/**
 * MultiAngleLoader.tsx — assemble a multi-angle 4-D STEM acquisition.
 *
 * An acquisition here is N separate 4-D datasets ("members"), each taken at its
 * own (tilt, azimuth) about one point, grouped into SHELLS by tilt magnitude —
 * six members at 1.0° plus four at 0.5° is two shells, ten members. Opening one
 * is therefore three steps, and this dialog is those three tabs: choose the
 * files and say what angle each is, align them in real space, align them in
 * reciprocal space. The later tabs stay locked until their prerequisite exists.
 *
 * EVERY TAB IS THE SAME PICTURE. An acquisition is a set of angles about a
 * point, so it is drawn the way it physically happened — one ring per tilt, one
 * slot per azimuth at `(tilt·sin θ, tilt·cos θ)`, exactly the polar navigator
 * the opened acquisition gets (spyde/actions/multiangle_navigator.py). A
 * missing angle is then a GAP in a ring rather than a row nobody wrote, and a
 * shift that follows the azimuth round the ring reads as a tilt-axis error
 * rather than ten unrelated numbers. Load fills the slots, real-space alignment
 * puts each member's virtual image in its slot, and reciprocal alignment opens
 * each member into its four scan-corner diffraction sums.
 *
 * THE EMPTY SLOTS ARE THE RENDERER'S OWN. The user says what the acquisition
 * looks like — a ring is a tilt and a number of angles — and that scaffolding
 * exists only here; the backend never hears about a slot nobody has filled. A
 * ring the DATA implies (a shell at a tilt no scaffold ring covers) is drawn
 * too, from its members' own azimuths, so a member can never be off-picture.
 *
 * THE BACKEND OWNS EVERYTHING ELSE. Every `maped_*` action is answered with one
 * `maped_state` message carrying the whole snapshot, so this component renders
 * from the last snapshot it received and never patches a local copy — a local
 * edit would be a second, divergent source of truth for shells, residuals and
 * whether the acquisition can be opened.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { TabRow, Field, NumInput, Info, PrimaryButton } from './WizardShell'
import { formatBytes } from '../kernel/format'
import { Dropdown } from './Dropdown'
import {
  useKeyedDebounce, useWizardEvent, useWizardLifecycle, type SendAction,
} from './wizardHooks'
import {
  MEMBER_DRAG_MIME, pathsFromDrop, peekMemberDrag, stashMemberDrag,
} from '../kernel/dnd'

const ACCENT = '#89b4fa'
const WARN = '#f9e2af'
const ERROR = '#f38ba8'
const RING_COLOR = '#585b70'
const LABEL_COLOR = '#9aa4b2'

/** A residual this close to half a pixel means the integer offset is the
 *  rounding of a value that could have gone either way — the alignment is a
 *  coin toss at that member, and the user has to know rather than read a
 *  confident-looking integer. */
const AMBIGUOUS_RESIDUAL_PX = 0.4

/** Smallest half-width the zero-beam search square may be shrunk to. Below a
 *  few detector pixels it can no longer hold a disk, so it would only ever
 *  report the centre it started from. */
const MIN_BEAM_ROI_HALF = 3

/** Two tilts this close are the same ring. A shell groups tilts within a
 *  tolerance, so a scaffold ring at 1.0° must still claim a member the backend
 *  probed as 0.999°. The SAME number as `DEFAULT_SHELL_TOLERANCE` in
 *  spyde/multiangle/model.py, which decides the shells the status line counts:
 *  at 0.05 here the picture drew two members on one ring that the text called
 *  two shells. A test parses this line to keep them equal. */
const TILT_TOLERANCE_DEG = 0.01

/** Corner order is row-major over the scan: the backend's `previews[0]` is the
 *  scan's first position, `previews[3]` its last. */
const CORNER_LABELS = ['Top-left', 'Top-right', 'Bottom-left', 'Bottom-right']
const CORNER_SHORT = ['TL', 'TR', 'BL', 'BR']

export interface MapedMember {
  index: number
  path: string
  name: string
  scan_shape: number[] | null
  detector_shape: number[] | null
  dtype: string | null
  size_bytes: number | null
  tilt: number | null
  azimuth: number | null
  shell: number | null
  error: string | null
  /** A data: URI thumbnail of this member's chosen virtual image, or null while
   *  the backend has not made one. */
  preview: string | null
  /** The virtual images this member carries by name. */
  virtual_images: string[]
}

export interface MapedShell {
  shell: number
  tilt: number
  members: number[]
}

/** The four scan-corner diffraction sums for one member, and the extent (in
 *  scan pixels) each was summed over. */
export interface MapedCorners {
  previews: (string | null)[]
  extents: (number | null)[]
}

export interface MapedSolve {
  solved: boolean
  offsets: number[][] | null
  residuals: number[] | null
  max_residual: number | null
  /** Reciprocal only: member index (as a string key) → its corner panels. */
  corners: Record<string, MapedCorners>
  /** Real only: the members summed both ways, and how much sharper aligning
   *  made them. Null until the solve has landed. */
  evidence: MapedEvidence | null
  /** Real only: which axes the specimen actually determined. */
  confidence: MapedConfidence | null
  /** Real only: what the SOLVER answered, kept so a member moved by hand can
   *  be put back. Null before a solve. */
  solver_offsets: number[][] | null
  /** Real only: the open pairwise comparison, or null. */
  pair: MapedPair | null
}

/** One member against the reference: both pictures and their overlay. */
export interface MapedPair {
  index: number
  image: string | null
  member: string | null
  reference: string | null
  overlay: string | null
  gain: number | null
}

/**
 * How much overlapping patches of the field agreed about each axis.
 *
 * A solve always returns two numbers. On a layered specimen only one of them
 * means anything — across the layers the patches agree, along them the
 * specimen is uniform and there is nothing to register on — so the dialog has
 * to say which, or it is claiming knowledge it does not have.
 */
export interface MapedConfidence {
  agreement: { y: number | null; x: number | null }
  determined: { y: boolean; x: boolean }
  votes: number
}

/** The real-space solve's evidence, as pictures. */
export interface MapedEvidence {
  unaligned: string | null
  aligned: string | null
  gain: number | null
  /** What the SOLVER's answer scored, kept once a member has been moved by
   *  hand so the two can be compared. */
  solver_gain: number | null
}

/** A square search region on the detector, by its centre and half-width. */
export interface BeamRoi { cy: number; cx: number; half: number }

export interface MapedState {
  members: MapedMember[]
  reference: number | null
  shells: MapedShell[]
  /** The scan grid forced on every member, [x, y], or null for "whatever each
   *  file says". One grid for the whole acquisition: its members share it. */
  scan_shape: number[] | null
  /** The virtual image every member's preview is made from, by name, or null
   *  for "computed" — the backend works one out. */
  virtual_image: string | null
  /** Names every member carries, so one can be chosen for all of them. */
  available_virtual_images: string[]
  /** Where the reciprocal stage looks for the zero beam, in DETECTOR pixels,
   *  or null for the whole pattern. One region for the acquisition: the
   *  members are the same detector at the same camera length. */
  beam_roi: BeamRoi | null
  real: MapedSolve
  reciprocal: MapedSolve
  busy: boolean
  message: string
  can_commit: boolean
}

const EMPTY_SOLVE: MapedSolve = {
  solved: false, offsets: null, residuals: null, max_residual: null, corners: {},
  evidence: null, confidence: null, solver_offsets: null, pair: null,
}

export const EMPTY_MAPED_STATE: MapedState = {
  members: [], reference: null, shells: [], scan_shape: null,
  virtual_image: null, available_virtual_images: [], beam_roi: null,
  real: EMPTY_SOLVE, reciprocal: EMPTY_SOLVE,
  busy: false, message: '', can_commit: false,
}

/** Default guard on the real-space correlation, in scan pixels. */
const DEFAULT_MAX_SHIFT_PX = 32.0

const num = (v: unknown): number | null =>
  v == null || !Number.isFinite(Number(v)) ? null : Number(v)

const numList = (v: unknown): number[] | null =>
  Array.isArray(v) ? v.map((n) => Number(n)) : null

const strList = (v: unknown): string[] =>
  Array.isArray(v) ? v.map((s) => String(s)) : []

/** A data: URI, or null. Anything else is dropped rather than handed to an
 *  <img src>, where a stray value would fire a network request. */
const dataUri = (v: unknown): string | null =>
  typeof v === 'string' && v.startsWith('data:') ? v : null

function parseCorners(raw: unknown): Record<string, MapedCorners> {
  const out: Record<string, MapedCorners> = {}
  if (!raw || typeof raw !== 'object') return out
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    const entry = (value ?? {}) as Record<string, unknown>
    const previews = Array.isArray(entry.previews) ? entry.previews : []
    const extents = Array.isArray(entry.extents) ? entry.extents : []
    out[String(key)] = {
      previews: CORNER_LABELS.map((_l, i) => dataUri(previews[i])),
      extents: CORNER_LABELS.map((_l, i) => num(extents[i])),
    }
  }
  return out
}

function parseSolve(raw: unknown): MapedSolve {
  const d = (raw ?? {}) as Record<string, unknown>
  return {
    solved: Boolean(d.solved),
    offsets: Array.isArray(d.offsets)
      ? (d.offsets as unknown[]).map((row) => numList(row) ?? [])
      : null,
    residuals: numList(d.residuals),
    max_residual: num(d.max_residual),
    corners: parseCorners(d.corners),
    evidence: parseEvidence(d.evidence),
    confidence: parseConfidence(d.confidence),
    solver_offsets: parseOffsets(d.solver_offsets),
    pair: parsePair(d.pair),
  }
}

/** The pairwise comparison, or null when none is open. */
function parsePair(raw: unknown): MapedPair | null {
  if (!raw || typeof raw !== 'object') return null
  const d = raw as Record<string, unknown>
  const index = num(d.index)
  if (index == null) return null
  const png = (value: unknown): string | null =>
    typeof value === 'string' ? value : null
  return {
    index,
    image: typeof d.image === 'string' ? d.image : null,
    member: png(d.member), reference: png(d.reference),
    overlay: png(d.overlay), gain: num(d.gain),
  }
}

/** ``[[dy, dx], ...]`` or null — the same shape `offsets` already comes in. */
function parseOffsets(raw: unknown): number[][] | null {
  if (!Array.isArray(raw)) return null
  const rows = raw
    .map((row) => (Array.isArray(row)
      ? [num(row[0]), num(row[1])] : [null, null]))
    .filter((row): row is number[] => row[0] != null && row[1] != null)
  return rows.length === raw.length ? rows : null
}

/** The per-axis agreement, or null when the solve has not reported one. */
function parseConfidence(raw: unknown): MapedConfidence | null {
  if (!raw || typeof raw !== 'object') return null
  const d = raw as Record<string, unknown>
  const agreement = (d.agreement ?? {}) as Record<string, unknown>
  const determined = (d.determined ?? {}) as Record<string, unknown>
  const y = num(agreement.y)
  const x = num(agreement.x)
  if (y == null && x == null) return null
  return {
    agreement: { y, x },
    determined: { y: determined.y === true, x: determined.x === true },
    votes: num(d.votes) ?? 0,
  }
}

/** The two summed pictures, or null — including when only one came through. */
function parseEvidence(raw: unknown): MapedEvidence | null {
  if (!raw || typeof raw !== 'object') return null
  const d = raw as Record<string, unknown>
  const unaligned = typeof d.unaligned === 'string' ? d.unaligned : null
  const aligned = typeof d.aligned === 'string' ? d.aligned : null
  if (!unaligned && !aligned) return null
  return { unaligned, aligned, gain: num(d.gain),
           solver_gain: num(d.solver_gain) }
}

/** Read one `maped_state` message. Defensive because a half-built member (a
 *  file that failed to open) legitimately has nulls everywhere but `error`. */
export function parseMapedState(detail: Record<string, unknown>): MapedState {
  const members = Array.isArray(detail.members) ? detail.members : []
  return {
    members: members.map((raw, i) => {
      const m = (raw ?? {}) as Record<string, unknown>
      return {
        index: num(m.index) ?? i,
        path: String(m.path ?? ''),
        name: String(m.name ?? m.path ?? `member ${i}`),
        scan_shape: numList(m.scan_shape),
        detector_shape: numList(m.detector_shape),
        dtype: m.dtype == null ? null : String(m.dtype),
        size_bytes: num(m.size_bytes),
        tilt: num(m.tilt),
        azimuth: num(m.azimuth),
        shell: num(m.shell),
        error: m.error == null ? null : String(m.error),
        preview: dataUri(m.preview),
        virtual_images: strList(m.virtual_images),
      }
    }),
    reference: num(detail.reference),
    shells: (Array.isArray(detail.shells) ? detail.shells : []).map((raw, i) => {
      const s = (raw ?? {}) as Record<string, unknown>
      return {
        shell: num(s.shell) ?? i,
        tilt: num(s.tilt) ?? 0,
        members: numList(s.members) ?? [],
      }
    }),
    scan_shape: numList(detail.scan_shape),
    virtual_image: detail.virtual_image == null ? null : String(detail.virtual_image),
    available_virtual_images: strList(detail.available_virtual_images),
    beam_roi: parseBeamRoi(detail.beam_roi),
    real: parseSolve(detail.real),
    reciprocal: parseSolve(detail.reciprocal),
    busy: Boolean(detail.busy),
    message: String(detail.message ?? ''),
    can_commit: Boolean(detail.can_commit),
  }
}

/** The search region, or null — including when it is there but unreadable. */
function parseBeamRoi(raw: unknown): BeamRoi | null {
  if (!raw || typeof raw !== 'object') return null
  const roi = raw as Record<string, unknown>
  const cy = Number(roi.cy), cx = Number(roi.cx), half = Number(roi.half)
  if (![cy, cx, half].every(Number.isFinite) || half <= 0) return null
  return { cy, cx, half }
}

/** "1°", "0.5°", "1.25°" — the shortest exact form, to two decimals. The SAME
 *  rule as `_format_degrees` in spyde/actions/multiangle_navigator.py, which
 *  labels the rings of the navigator this picture is a preview of: one ring
 *  reading "1.0°" here and "1°" there is two names for one shell. A test parses
 *  this line to keep them equal. */
function formatDegrees(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return '—'
  const text = value.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')
  return `${text || '0'}°`
}

/** An azimuth label, where a trailing ".0" is noise: "60°", "17.5°". */
function formatAzimuth(value: number): string {
  const rounded = Math.round(value * 10) / 10
  return `${Number.isInteger(rounded) ? rounded : rounded.toFixed(1)}°`
}

/** "angle07.mrc" → "angle07". A caption inside a 56-pixel tile has room for the
 *  part that differs between members and none for the extension they share; the
 *  whole name is a hover and a double-click away. */
const stem = (name: string): string => name.replace(/\.[^./\\]+$/, '')

const formatShape = (shape: number[] | null): string =>
  shape && shape.length ? shape.join('×') : '?'

const memberFacts = (member: MapedMember): string => [
  `${formatShape(member.scan_shape)} scan`,
  `${formatShape(member.detector_shape)} det`,
  member.dtype ?? '',
  formatBytes(member.size_bytes),
].filter(Boolean).join(' · ')

/** What the acquisition IS, in one line: "10 angles · 2 shells · 1.0°, 0.5°". */
function summarise(state: MapedState): string {
  if (state.members.length === 0) return 'No datasets'
  const angles = `${state.members.length} angle${state.members.length === 1 ? '' : 's'}`
  if (state.shells.length === 0) return `${angles} · no shells yet (set a tilt on each member)`
  const shells = `${state.shells.length} shell${state.shells.length === 1 ? '' : 's'}`
  return `${angles} · ${shells} · ${state.shells.map((s) => formatDegrees(s.tilt)).join(', ')}`
}

// ─────────────────────────────────────────────────────────────────────────────
// The ring picture
// ─────────────────────────────────────────────────────────────────────────────

/** One ring the user asked for: a tilt, and how many angles sit on it. */
interface RingSpec { id: number; tilt: number; count: number }

interface Slot {
  key: string
  azimuth: number
  member: MapedMember | null
  /** True for a slot that exists only because a member is standing on it — it
   *  came from the data, not from the ring the user laid out. */
  fromData: boolean
}

interface Ring {
  key: string
  tilt: number
  /** Ring radius as a fraction of the outermost one. */
  radiusFraction: number
  slots: Slot[]
  /** The scaffold ring this came from, or null for a ring the data implied. */
  specId: number | null
}

/** Shortest signed distance between two azimuths, in degrees. */
function azimuthDistance(a: number, b: number): number {
  const delta = (((a - b) % 360) + 540) % 360 - 180
  return Math.abs(delta)
}

/**
 * The azimuth with the most room around it — where a ring's tilt label goes.
 *
 * A label at a fixed azimuth lands on top of whichever member happens to sit
 * there, and at azimuth 0 it always does, since an acquisition that starts at
 * 0° is the normal case. In the widest gap it only ever collides when the ring
 * is genuinely full. The SAME rule as `_widest_gap` in
 * multiangle_navigator.py, down to where an EMPTY ring puts its label — a test
 * parses that default to keep the two pictures alike.
 */
function widestGapAzimuth(azimuths: number[]): number {
  const sorted = azimuths.map((a) => ((a % 360) + 360) % 360).sort((x, y) => x - y)
  if (sorted.length === 0) return 0
  if (sorted.length === 1) return sorted[0] + 180
  let best = 0
  let widest = -1
  for (let i = 0; i < sorted.length; i += 1) {
    const next = i === sorted.length - 1 ? sorted[0] + 360 : sorted[i + 1]
    const gap = next - sorted[i]
    if (gap > widest) { widest = gap; best = sorted[i] + gap / 2 }
  }
  return best
}

/**
 * Lay the acquisition out as rings of slots.
 *
 * Scaffold rings come first and claim every member within `TILT_TOLERANCE_DEG`
 * of their tilt, each member taking the free slot nearest its azimuth. A member
 * that matches no slot closely enough — or that arrives on a full ring — gets a
 * slot of its own on that ring, and a tilt with no scaffold ring at all gets a
 * whole ring made of its members' azimuths. So a member is ALWAYS somewhere on
 * the picture: the one thing this layout must never do is drop one.
 */
function layoutRings(specs: RingSpec[], members: MapedMember[]): {
  rings: Ring[]
  unplaced: MapedMember[]
} {
  const placeable = members.filter(
    (m) => m.tilt != null && Number.isFinite(m.tilt)
      && m.azimuth != null && Number.isFinite(m.azimuth))
  const unplaced = members.filter((m) => !placeable.includes(m))

  const claimed = new Set<number>()
  const rings: Ring[] = []

  const membersNear = (tilt: number): MapedMember[] =>
    placeable
      .filter((m) => !claimed.has(m.index)
        && Math.abs((m.tilt as number) - tilt) <= TILT_TOLERANCE_DEG)
      .sort((a, b) => (a.azimuth as number) - (b.azimuth as number))

  for (const spec of specs) {
    const count = Math.max(1, Math.round(spec.count))
    const slots: Slot[] = Array.from({ length: count }, (_v, k) => ({
      key: `${spec.id}-${k}`,
      azimuth: (k * 360) / count,
      member: null,
      fromData: false,
    }))
    const spacing = 360 / count
    for (const member of membersNear(spec.tilt)) {
      let nearest = -1
      let best = Infinity
      slots.forEach((slot, k) => {
        if (slot.member || slot.fromData) return
        const distance = azimuthDistance(slot.azimuth, member.azimuth as number)
        if (distance < best) { best = distance; nearest = k }
      })
      if (nearest >= 0 && best <= spacing / 2 + 1e-6) {
        slots[nearest].member = member
        // The DATA wins over the scaffolding: a member acquired at 57° is drawn
        // at 57°, not snapped to the 45° spot it was nearest to. The slot was
        // only ever a guess at where an angle would land.
        slots[nearest].azimuth = member.azimuth as number
      } else {
        // Off the lattice, or a ring laid out with too few angles: give it its
        // own spot rather than leaving it off the picture.
        slots.push({
          key: `${spec.id}-extra-${member.index}`,
          azimuth: member.azimuth as number,
          member,
          fromData: true,
        })
      }
      claimed.add(member.index)
    }
    rings.push({
      key: `spec-${spec.id}`,
      tilt: spec.tilt,
      radiusFraction: 1,
      slots: slots.sort((a, b) => a.azimuth - b.azimuth),
      specId: spec.id,
    })
  }

  // Whatever is left groups itself into rings by tilt.
  const leftovers = placeable.filter((m) => !claimed.has(m.index))
  const byTilt = new Map<string, MapedMember[]>()
  for (const member of leftovers) {
    const key = (Math.round((member.tilt as number) * 1000) / 1000).toFixed(3)
    const bucket = byTilt.get(key)
    if (bucket) bucket.push(member)
    else byTilt.set(key, [member])
  }
  for (const [key, bucket] of byTilt) {
    rings.push({
      key: `data-${key}`,
      tilt: Number(key),
      radiusFraction: 1,
      slots: bucket
        .sort((a, b) => (a.azimuth as number) - (b.azimuth as number))
        .map((member) => ({
          key: `data-${key}-${member.index}`,
          azimuth: member.azimuth as number,
          member,
          fromData: true,
        })),
      specId: null,
    })
  }

  // Radius is the tilt, as on the navigator — with a floor the navigator does
  // not need, because a slot here is a fixed-size TILE: a 0.05° inner shell
  // beside a 2° outer one would be a dot at the centre with its slots on top
  // of each other, where the navigator's ring is just a small circle.
  rings.sort((a, b) => a.tilt - b.tilt)
  const maxTilt = rings.reduce((acc, r) => Math.max(acc, r.tilt), 0)
  const floor = rings.length > 1 ? 0.34 : 1
  for (const ring of rings) {
    const proportional = maxTilt > 0 ? ring.tilt / maxTilt : 1
    ring.radiusFraction = Math.min(1, Math.max(proportional, floor))
  }
  return { rings, unplaced }
}

type Tab = 'Load' | 'Align real space' | 'Align reciprocal space'
const TABS: readonly Tab[] = ['Load', 'Align real space', 'Align reciprocal space']
const TAB_TESTID: Record<Tab, string> = {
  'Load': 'maped-tab-load',
  'Align real space': 'maped-tab-real',
  'Align reciprocal space': 'maped-tab-reciprocal',
}

type ReciprocalMethod = 'corners' | 'beam' | 'correlate'
const RECIPROCAL_METHODS: readonly { value: ReciprocalMethod; label: string }[] = [
  { value: 'corners', label: 'Corners (fast)' },
  { value: 'beam', label: 'Direct beam' },
  { value: 'correlate', label: 'Cross-correlate mean patterns' },
]

/** The dropdown value standing for "no named image — the backend computes one",
 *  since a Dropdown's value is a string and the action's is null. */
const COMPUTE_VI = '__compute__'

/** The virtual images on offer, plus computing one — the same list everywhere a
 *  virtual image is chosen (the real-space tab, and the pair view). */
const virtualImageOptions = (
  images: string[],
): readonly { value: string; label: string }[] => [
  ...images.map((name) => ({ value: name, label: name })),
  { value: COMPUTE_VI, label: 'Compute from the data' },
]

/** …and back again, for the action that takes null. */
const namedImage = (value: string): string | null =>
  (value === COMPUTE_VI ? null : value)

const FILE_FILTER = {
  name: 'EM Data',
  extensions: ['hspy', 'zspy', 'mrc', 'tif', 'tiff', 'de5'],
}

/**
 * What an enlarged panel is showing — named, not captured.
 *
 * A member is held by INDEX rather than as a copy of its row, because the
 * enlarged view carries live controls (its angles, the reference radio): a
 * captured member would go on showing the angles it had when it was clicked
 * while the backend echoed new ones underneath, which is the divergent local
 * copy this whole dialog exists without.
 */
type ZoomTarget =
  | { kind: 'member'; index: number }
  /** `detector` is set for a DIFFRACTION panel, which is what makes the
   *  enlarged view able to carry the zero-beam region: on a tableau tile one
   *  screen pixel is about five detector pixels, so a drag there is far too
   *  coarse to place a 24 px region on a disk. */
  | { kind: 'panel'; src: string | null; caption: string; detector?: number[] | null }

/** Start an internal slot-to-slot move: the member index on the drag, and the
 *  same index kept in-process for a drop whose `getData()` comes back empty
 *  (see dnd.ts — a real OS drag in the packaged app can do that). */
function startMemberDrag(e: React.DragEvent, index: number): void {
  e.dataTransfer.setData(MEMBER_DRAG_MIME, String(index))
  e.dataTransfer.effectAllowed = 'move'
  stashMemberDrag(index)
}

/** The member a drop carries, or null for anything else. */
function droppedMember(e: React.DragEvent): number | null {
  const carried = e.dataTransfer.getData(MEMBER_DRAG_MIME)
  if (carried !== '') return Number(carried)
  return e.dataTransfer.types.includes(MEMBER_DRAG_MIME) ? peekMemberDrag() : null
}

export function MultiAngleLoader({ sendAction, onClose }: {
  sendAction: SendAction
  onClose: () => void
}) {
  const [state, setState] = useState<MapedState>(EMPTY_MAPED_STATE)
  const [tab, setTab] = useState<Tab>('Load')
  const [method, setMethod] = useState<ReciprocalMethod>('corners')
  const [maxShift, setMaxShift] = useState(DEFAULT_MAX_SHIFT_PX)
  const [rings, setRings] = useState<RingSpec[]>([])
  const [zoom, setZoom] = useState<ZoomTarget | null>(null)
  //: Which member the arrow keys move. A view preference, not a fact about
  //: the data, so it lives here and never goes to the backend.
  const [selected, setSelected] = useState<number | null>(null)
  const nudgeRef = useRef<HTMLDivElement | null>(null)
  // The pair view's own pad. Sharing the grid's ref left whichever pad
  // mounted last holding it, so the arrow keys could land on the member
  // the OTHER pad was showing.
  const pairNudgeRef = useRef<HTMLDivElement | null>(null)
  const evidenceRef = useRef<HTMLDivElement | null>(null)
  // A drop whose files resolve to no OS path is the one failure a drop handler
  // can have SILENTLY (Electron 44 removed File.path; the preload bridge's
  // webUtils.getPathForFile is what replaces it). Say so rather than no-op.
  const [dropNote, setDropNote] = useState('')
  const debounce = useKeyedDebounce(250)

  // `sendAction` is a fresh closure on every provider render, so the
  // snapshot-gated effect below reads it through a ref rather than capturing
  // one render's value.
  const send = useRef(sendAction)
  send.current = sendAction
  // Set by the Open button: the commit must NOT be followed by a close that
  // could tear the loader session down underneath it.
  const committed = useRef(false)

  // A file dropped on a SLOT is two actions, and the second needs an index the
  // backend has not issued yet — so the angles wait here by path until the
  // snapshot that carries the new member arrives. The renderer cannot know the
  // index in advance; this is the whole reason the map exists.
  const awaitingAngles = useRef(new Map<string, { tilt: number; azimuth: number }>())

  // Mount → open, unmount → close, the same staged pair every wizard caret
  // speaks — with no window id, because the loader is a modal over the whole
  // app rather than a caret on one window.
  useWizardLifecycle({
    sendAction,
    openAction: 'maped_open_loader',
    closeAction: 'maped_close_loader',
    skipClose: () => committed.current,
  })

  useWizardEvent('spyde:maped_state', undefined, (detail) =>
    setState(parseMapedState(detail)))

  // The second half of a slot drop: the member the backend just added gets the
  // angles of the slot it was dropped on.
  useEffect(() => {
    if (awaitingAngles.current.size === 0) return
    for (const member of state.members) {
      for (const [path, angles] of awaitingAngles.current) {
        // The backend may hand back an absolute path where a relative one went
        // in, so a tail match stands in for equality.
        if (member.path !== path && !member.path.endsWith(path)
          && !path.endsWith(member.path)) continue
        awaitingAngles.current.delete(path)
        send.current('maped_set_member', { index: member.index, ...angles })
        break
      }
    }
  }, [state.members])

  useEffect(() => {
    if (!zoom) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setZoom(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [zoom])

  // Its own listener, because the zoom's is gated on a zoom being open and
  // the comparison is not one.
  const pairOpen = state.real.pair != null
  useEffect(() => {
    if (!pairOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') sendAction('maped_set_pair', { index: null })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [pairOpen, sendAction])

  // Counted over members that actually OPENED: three bare MRCs that all failed
  // to probe are three slots, but nothing an alignment could be solved on, and
  // an enabled tab there just moves the failure one click further in.
  const realLocked = state.members.filter((m) => !m.error).length < 2
  const reciprocalLocked = !state.real.solved
  const locked = (t: Tab): boolean =>
    (t === 'Align real space' && realLocked) ||
    (t === 'Align reciprocal space' && reciprocalLocked)

  // Removing a member can re-lock the tab the user is standing on.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { if (locked(tab)) setTab('Load') }, [realLocked, reciprocalLocked])

  const addPaths = (paths: string[] | undefined): void => {
    if (paths && paths.length) sendAction('maped_add_files', { paths })
  }

  /** Add one file AT a slot's angles — the two-step the backend's indices force. */
  const addAt = (path: string, tilt: number, azimuth: number): void => {
    awaitingAngles.current.set(path, { tilt, azimuth })
    sendAction('maped_add_files', { paths: [path] })
  }

  /** The OS paths behind a drop, saying on screen why there were none. */
  const droppedPaths = (e: React.DragEvent): string[] => {
    const { paths, note } = pathsFromDrop(e)
    setDropNote(note)
    return paths
  }

  const setMember = (index: number, patch: { tilt?: number; azimuth?: number }): void =>
    debounce(`member-${index}`, () => sendAction('maped_set_member', { index, ...patch }))

  const layout = useMemo(() => layoutRings(rings, state.members), [rings, state.members])

  const failed = state.members.filter((m) => m.error)

  // The scan grid is a FALLBACK, not a step: a Direct Electron MRC normally
  // carries its grid in an `_info.txt` sidecar, and a probe that found one
  // should not be second-guessed by an empty field on screen. So it appears
  // only when a member failed a probe that could not work out a grid — or once
  // a grid has been forced, since that has to stay undoable.
  const needsScanShape =
    state.scan_shape != null ||
    state.members.some((m) => m.error != null && m.scan_shape == null)

  const byIndex = useMemo(() => {
    const map = new Map<number, MapedMember>()
    for (const m of state.members) map.set(m.index, m)
    return map
  }, [state.members])

  /** What an enlarged member shows below its image: the facts a 58-pixel spot
   *  cannot carry, and the controls that have nowhere else to live — its
   *  angles, the reference choice, and removal. */
  const memberDetail = (member: MapedMember): React.ReactNode => (
    <div style={styles.zoomDetail}>
      <div style={styles.zoomFacts}>{memberFacts(member)}</div>
      {member.error && (
        <div style={styles.zoomError}>⚠ {member.error}</div>
      )}
      <div style={styles.zoomAngles}>
        <NumInput testid={`maped-tilt-${member.index}`} label="tilt" suffix="°"
          width={58} value={member.tilt}
          onChange={(tilt) => setMember(member.index, { tilt })} />
        <NumInput testid={`maped-azimuth-${member.index}`} label="az" suffix="°"
          width={58} value={member.azimuth}
          onChange={(azimuth) => setMember(member.index, { azimuth })} />
        <label style={styles.zoomReference}>
          <input type="radio" name="maped-reference-zoom"
            data-testid={`maped-reference-${member.index}`}
            checked={state.reference === member.index}
            onChange={() => sendAction('maped_set_reference', { index: member.index })} />
          reference
        </label>
        <button data-testid={`maped-remove-${member.index}`} style={styles.zoomRemove}
          onClick={() => {
            sendAction('maped_remove_member', { index: member.index })
            setZoom(null)
          }}>Remove</button>
      </div>
    </div>
  )

  /** The enlarged panel, resolved against the CURRENT snapshot. */
  const zoomed = ((): React.ReactNode => {
    if (!zoom) return null
    if (zoom.kind === 'panel') {
      return <PanelZoom src={zoom.src} caption={zoom.caption}
        roi={zoom.detector ? state.beam_roi : null} detector={zoom.detector}
        onRoi={(roi) => debounce('beam-roi',
          () => sendAction('maped_set_beam_roi', { beam_roi: roi }))}
        onClose={() => setZoom(null)} />
    }
    const member = byIndex.get(zoom.index)
    // The member it was opened on is gone — so is the panel.
    if (!member) return null
    return (
      <PanelZoom src={member.preview} caption={member.name}
        detail={memberDetail(member)} onClose={() => setZoom(null)} />
    )
  })()

  /** One slot on the Load ring: a drop target, a picker, or a filled member. */
  const loadSlot = (slot: Slot, ring: Ring): React.ReactNode => (
    <SlotTile
      slot={slot} tilt={ring.tilt} size={SLOT_SIZE_LOAD}
      reference={slot.member != null && state.reference === slot.member.index}
      onOpenPicker={async () => {
        const paths = await window.electron.pickFiles(FILE_FILTER)
        if (paths && paths.length) addAt(paths[0], ring.tilt, slot.azimuth)
      }}
      onDropFiles={(e) => {
        const paths = droppedPaths(e)
        if (paths.length) addAt(paths[0], ring.tilt, slot.azimuth)
      }}
      onDropMember={(index) =>
        sendAction('maped_set_member', { index, tilt: ring.tilt, azimuth: slot.azimuth })}
      onZoom={() => slot.member && setZoom({ kind: 'member', index: slot.member.index })}
    />
  )

  /** The same slot on the real-space tab: the virtual image, and nothing to
   *  drop — the acquisition is already assembled by then. */
  const virtualImageSlot = (slot: Slot, ring: Ring): React.ReactNode => (
    <SlotTile
      slot={slot} tilt={ring.tilt} size={SLOT_SIZE_REAL} readOnly
      reference={slot.member != null && state.reference === slot.member.index}
      missing={slot.member != null && state.virtual_image != null
        && slot.member.virtual_images.length > 0
        && !slot.member.virtual_images.includes(state.virtual_image)}
      onZoom={() => {
        if (!slot.member) return
        // The reference has nothing to be compared against, so it keeps the
        // plain enlargement.
        if (state.reference === slot.member.index) {
          setZoom({ kind: 'member', index: slot.member.index })
          return
        }
        setSelected(slot.member.index)
        sendAction('maped_set_pair', { index: slot.member.index })
      }}
      selected={slot.member != null && selected === slot.member.index}
      onSelect={slot.member != null && state.reference !== slot.member.index
        ? () => {
            setSelected(slot.member!.index)
            nudgeRef.current?.focus()
            // The BOX, not the pad: the pad is already in view when the
            // thumbnails below it are not, so scrolling the pad does nothing.
            evidenceRef.current?.scrollIntoView({ block: 'end' })
          }
        : undefined}
    />
  )

  return (
    <div style={styles.overlay} data-testid="multiangle-loader"
      onDragOver={(e) => e.preventDefault()} onDrop={(e) => e.preventDefault()}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <div style={styles.head}>
          <h3 style={styles.title}>Multi-Angle 4D STEM</h3>
          <button data-testid="maped-close" style={styles.close} onClick={onClose}>✕</button>
        </div>
        <div style={styles.summary}>
          <span data-testid="maped-summary">{summarise(state)}</span>
          {/* A failed member is one small tile among ten, so its state has to
              be said in words up here as well as drawn down there. */}
          {failed.length > 0 && (
            <span data-testid="maped-error-count" style={styles.summaryError}>
              {' · '}{failed.length} failed to open
            </span>
          )}
        </div>

        <TabRow<Tab>
          tabs={TABS} active={tab} onSelect={setTab} locked={locked}
          testid={(t) => TAB_TESTID[t]}
        />

        <div style={styles.body}>
          {tab === 'Load' && (
            <>
              <RingEditor rings={rings} layout={layout.rings} onChange={setRings} />

              {/* Above the picture, not below it: a file that will not open
                  cannot be on any ring, so the ring is 370 pixels of scrolling
                  between the user and the only control that would fix it. */}
              {needsScanShape && (
                <ScanShapeField
                  value={state.scan_shape}
                  onChange={(scan_shape) => debounce('scan-shape', () =>
                    sendAction('maped_set_scan_shape', { scan_shape }))}
                />
              )}

              {failed.length > 0 && <ProblemList members={failed} />}

              <AngleTableau
                testid="maped-tableau-load" rings={layout.rings}
                size={TABLEAU_SIZE} slotSize={SLOT_SIZE_LOAD} renderSlot={loadSlot}
                empty="Add a ring"
              />

              {/* Directly under the picture it is dragged into. */}
              {layout.unplaced.length > 0 && (
                <UnplacedTray members={layout.unplaced}
                  onZoom={(m) => setZoom({ kind: 'member', index: m.index })} />
              )}

              <div
                data-testid="maped-dropzone"
                onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }}
                onDrop={(e) => {
                  e.preventDefault(); e.stopPropagation(); addPaths(droppedPaths(e))
                }}
                style={styles.dropzone}
              >
                Drop files here to add them without an angle, or
                <button data-testid="maped-add-files" style={styles.linkButton}
                  onClick={async () => addPaths(await window.electron.pickFiles(FILE_FILTER))}>
                  add datasets…
                </button>
                <span style={styles.dropzoneSep}>·</span>
                <button data-testid="maped-add-folders" style={styles.linkButton}
                  onClick={async () => addPaths(await window.electron.pickFolders())}>
                  add .zspy/.zarr folders…
                </button>
              </div>
              {dropNote && (
                <div data-testid="maped-drop-note" style={styles.dropNote}>{dropNote}</div>
              )}

              {state.members.length > 0 && (
                <div data-testid="maped-reference-line" style={styles.referenceLine}>
                  Reference member:{' '}
                  <strong style={{ color: '#cdd6f4' }}>
                    {byIndex.get(state.reference ?? -1)?.name ?? 'none chosen'}
                  </strong>
                  {' '}— every alignment is solved against it. Double-click a spot
                  to enlarge it, rename its angles or make it the reference.
                </div>
              )}
            </>
          )}

          {tab === 'Align real space' && (
            <>
              <Field label="Virtual image">
                <Dropdown<string>
                  value={state.virtual_image ?? COMPUTE_VI}
                  options={virtualImageOptions(state.available_virtual_images)}
                  onChange={(v) => sendAction('maped_set_virtual_image',
                    { name: namedImage(v) })}
                  testid="maped-virtual-image" width={240}
                />
              </Field>

              <AngleGrid
                testid="maped-tableau-real" rings={layout.rings}
                slotSize={SLOT_SIZE_REAL} renderSlot={virtualImageSlot}
                empty="No members placed on the ring yet."
              />

              {/* One row: every pixel above the evidence box is a pixel the
                  aligned picture does not get, and that picture is the whole
                  point of the panel below. */}
              <div style={styles.runRow}>
                <Field label={<>Max shift (px) <Info width={300} testid="maped-info-max-shift" text={INFO.maxShift} /></>}>
                  <NumInput value={maxShift} onChange={setMaxShift}
                    step="1" width={72} testid="maped-max-shift" />
                </Field>
                <PrimaryButton
                  testid="maped-run-real" busy={state.busy}
                  label={state.real.solved ? 'Re-run' : 'Run'}
                  onClick={() => sendAction('maped_align_real', { params: { max_shift: maxShift } })}
                />
              </div>
              {state.real.evidence && (
                <AlignmentEvidence evidence={state.real.evidence}
                  confidence={state.real.confidence}
                  solve={state.real} members={state.members}
                  reference={state.reference} selected={selected}
                  padRef={nudgeRef} boxRef={evidenceRef}
                  onSelect={(index) => {
                    setSelected(index)
                    nudgeRef.current?.focus()
                    evidenceRef.current?.scrollIntoView({ block: 'end' })
                  }}
                  onSet={(index, offset) => sendAction(
                    'maped_set_real_offset', { index, offset })} />
              )}
              <SolveReport testid="maped-real" solve={state.real} members={state.members} />
            </>
          )}

          {tab === 'Align reciprocal space' && (
            <>
              <Field label="Method">
                <Dropdown<ReciprocalMethod>
                  value={method} options={RECIPROCAL_METHODS}
                  onChange={setMethod} testid="maped-reciprocal-method" width={240}
                />
              </Field>

              <CornerTableau
                state={state}
                onExtent={(member, corner, extent) => debounce(
                  `corner-${member ?? 'all'}-${corner}`,
                  () => sendAction('maped_set_corner_extent', { member, corner, extent }))}
                onZoom={setZoom}
                onBeamRoi={(roi) => debounce('beam-roi',
                  () => sendAction('maped_set_beam_roi', { beam_roi: roi }))}
              />

              <PrimaryButton
                testid="maped-run-reciprocal" busy={state.busy}
                label={state.reciprocal.solved
                  ? 'Re-run'
                  : 'Run'}
                onClick={() => sendAction('maped_align_reciprocal', { method, params: {} })}
              />
              <SolveReport testid="maped-reciprocal" solve={state.reciprocal} members={state.members} />
            </>
          )}
        </div>

        <div data-testid="maped-status" style={styles.status}>
          {state.busy ? (state.message || 'Working…') : state.message}
        </div>

        <div style={styles.footer}>
          <button data-testid="maped-cancel" style={styles.cancel} onClick={onClose}>
            Cancel
          </button>
          <button
            data-testid="maped-open"
            style={{ ...styles.confirm, opacity: state.can_commit ? 1 : 0.5 }}
            disabled={!state.can_commit}
            title={state.can_commit ? '' : 'Align first'}
            onClick={() => {
              committed.current = true
              sendAction('maped_commit', {})
              onClose()
            }}
          >
            Open
          </button>
        </div>
      </div>

      {zoomed}
      {state.real.pair && (
        <PairView
          pair={state.real.pair} members={state.members}
          reference={state.reference}
          images={state.available_virtual_images}
          offsets={state.real.offsets} solverOffsets={state.real.solver_offsets}
          weak={NUDGE_AXES
            .filter((axis) => state.real.confidence
              && state.real.confidence.votes > 0
              && !state.real.confidence.determined[axis.key])
            .map((axis) => axis.label)}
          padRef={pairNudgeRef}
          onSet={(index, offset) =>
            sendAction('maped_set_real_offset', { index, offset })}
          onImage={(name) => sendAction('maped_set_pair',
            { index: state.real.pair?.index, image: name })}
          onClose={() => sendAction('maped_set_pair', { index: null })}
        />
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// The tableau
// ─────────────────────────────────────────────────────────────────────────────

const TABLEAU_SIZE = 372
const SLOT_SIZE_LOAD = 56
//: Twice the ring's tile. On the real-space tab the pictures ARE the content
//: — the user is judging whether they line up — and the ring's geometry says
//: nothing there that the grid's caption does not say in words.
const SLOT_SIZE_REAL = 128
const GRID_GAP = 12
/** Room outside the outermost ring for its tilt label. */
const RING_MARGIN = 10

/**
 * The acquisition as a picture: concentric rings at each tilt, a slot at each
 * azimuth, azimuth 0° pointing up and growing clockwise.
 *
 * Positioned divs rather than an SVG because every slot is an image, a drop
 * target and a button — all three of which an `<img>` inside a positioned box
 * gives for free and an SVG node does not.
 */
/**
 * The members as a grid of pictures, for the tab that is about the pictures.
 *
 * The ring cannot be made to fill its square: the radius encodes the tilt, so
 * bigger tiles mean fewer of them before they overlap — at 372 px the inner
 * ring holds four. The Load tab needs that geometry, because dropping a file
 * on an azimuth is how an acquisition is assembled there. By the real-space
 * tab the acquisition exists and the tiles are read-only evidence, so the
 * angles go in the caption and the space goes to the images.
 */
function AngleGrid({ rings, slotSize, renderSlot, testid, empty }: {
  rings: Ring[]
  slotSize: number
  renderSlot: (slot: Slot, ring: Ring) => React.ReactNode
  testid: string
  empty: string
}) {
  const placed = rings.flatMap((ring) => ring.slots
    .filter((slot) => slot.member != null)
    .map((slot) => ({ slot, ring })))
  if (placed.length === 0) {
    return <div data-testid={`${testid}-empty`} style={styles.tableauEmpty}>{empty}</div>
  }
  return (
    <div data-testid={testid} style={styles.grid}>
      {placed.map(({ slot, ring }) => (
        <div key={slot.key} style={styles.gridCell}>
          {renderSlot(slot, ring)}
          <div style={styles.gridCaption}>
            {`${formatDegrees(ring.tilt)} · ${formatAzimuth(slot.azimuth)}`}
          </div>
        </div>
      ))}
    </div>
  )
}


function AngleTableau({ rings, size, slotSize, renderSlot, testid, empty }: {
  rings: Ring[]
  size: number
  slotSize: number
  renderSlot: (slot: Slot, ring: Ring) => React.ReactNode
  testid: string
  empty: string
}) {
  const centre = size / 2
  const outer = centre - slotSize / 2 - RING_MARGIN

  if (rings.length === 0) {
    return <div data-testid={`${testid}-empty`} style={styles.tableauEmpty}>{empty}</div>
  }

  return (
    <div data-testid={testid} style={{ ...styles.tableau, width: size, height: size }}>
      {rings.map((ring) => {
        const radius = outer * ring.radiusFraction
        const labelAt = widestGapAzimuth(ring.slots.map((s) => s.azimuth))
        const labelRadians = (labelAt * Math.PI) / 180
        return (
          <React.Fragment key={ring.key}>
            <div style={{
              ...styles.ringCircle,
              left: centre - radius, top: centre - radius,
              width: radius * 2, height: radius * 2,
            }} />
            <div style={{
              ...styles.ringLabel,
              left: centre + radius * Math.sin(labelRadians),
              top: centre - radius * Math.cos(labelRadians),
            }}>{formatDegrees(ring.tilt)}</div>
            {ring.slots.map((slot) => {
              const radians = (slot.azimuth * Math.PI) / 180
              return (
                <div key={slot.key} style={{
                  ...styles.slotAnchor,
                  left: centre + radius * Math.sin(radians),
                  top: centre - radius * Math.cos(radians),
                }}>
                  {renderSlot(slot, ring)}
                </div>
              )
            })}
          </React.Fragment>
        )
      })}
    </div>
  )
}

/**
 * One spot on the ring.
 *
 * Empty it is an outlined drop target that opens the file picker when clicked;
 * filled it is the member's own thumbnail, draggable onto another spot (which
 * is how a member gets re-angled without typing), and double-clickable for the
 * large view. A member that failed to open keeps its spot and turns red — the
 * one thing this must never do is let an error read as an absence.
 */
function SlotTile({
  slot, tilt, size, reference, missing, readOnly, selected,
  onOpenPicker, onDropFiles, onDropMember, onZoom, onSelect,
}: {
  slot: Slot
  tilt: number
  size: number
  reference: boolean
  missing?: boolean
  readOnly?: boolean
  /** Real-space tab only: this member is the one the arrow keys move. A
   *  single click selects, which is free because zooming is a DOUBLE click. */
  selected?: boolean
  onSelect?: () => void
  onOpenPicker?: () => void
  onDropFiles?: (e: React.DragEvent) => void
  onDropMember?: (index: number) => void
  onZoom?: () => void
}) {
  const [over, setOver] = useState(false)
  const member = slot.member
  const testid = member
    ? `maped-member-${member.index}`
    : `maped-slot-${tilt.toFixed(2)}-${Math.round(slot.azimuth)}`

  const accept = (e: React.DragEvent): void => {
    e.preventDefault()
    e.stopPropagation()
    setOver(false)
    const dragged = droppedMember(e)
    if (dragged != null && onDropMember) { onDropMember(dragged); return }
    onDropFiles?.(e)
  }

  const tone = member?.error
    ? styles.slotErrored
    : member
      ? styles.slotFilled
      : styles.slotEmpty

  return (
    <div
      data-testid={testid}
      title={member
        ? `${member.name}\n${formatDegrees(tilt)} tilt · ${formatAzimuth(slot.azimuth)} azimuth`
        : `Empty — ${formatDegrees(tilt)} tilt · ${formatAzimuth(slot.azimuth)} azimuth`}
      draggable={!readOnly && member != null}
      onDragStart={(e) => member && startMemberDrag(e, member.index)}
      onDragOver={readOnly ? undefined : (e) => {
        e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; setOver(true)
      }}
      onDragLeave={readOnly ? undefined : () => setOver(false)}
      onDrop={readOnly ? undefined : accept}
      onClick={() => {
        if (member && onSelect) { onSelect(); return }
        if (!member && !readOnly) onOpenPicker?.()
      }}
      onDoubleClick={() => onZoom?.()}
      style={{
        ...styles.slot, ...tone,
        width: size, height: size,
        ...(over ? styles.slotOver : null),
        ...(reference ? styles.slotReference : null),
        ...(selected ? styles.slotSelected : null),
        cursor: member && onSelect ? 'pointer'
          : member ? 'grab' : (readOnly ? 'default' : 'pointer'),
      }}
      data-selected={selected ? 'true' : undefined}
      aria-selected={onSelect ? Boolean(selected) : undefined}
    >
      {member?.preview
        ? <img src={member.preview} alt="" style={styles.slotImage} draggable={false} />
        : (
          <span style={styles.slotGlyph}>
            {member?.error ? '⚠' : member ? '▦' : '+'}
          </span>
        )}
      {reference && <span style={styles.referenceBadge}>★</span>}
      {missing && (
        <span data-testid={`maped-slot-missing-${member?.index}`}
          style={styles.missingBadge} title="Missing this virtual image">
          ?
        </span>
      )}
      {member?.error && (
        <span data-testid={`maped-slot-error-${member.index}`} style={styles.slotErrorBadge}
          title={member.error}>!</span>
      )}
      <span style={styles.slotCaption}>
        {member ? stem(member.name) : formatAzimuth(slot.azimuth)}
      </span>
    </div>
  )
}

/**
 * The ring scaffolding editor — the one thing on the Load tab the backend knows
 * nothing about.
 *
 * A ring is a tilt and a number of angles, which is how an acquisition is
 * actually planned ("six at one degree, four at a half"), and laying it out
 * before any file exists is what turns loading from a list into filling in a
 * picture. A ring the DATA implied is listed too, but cannot be removed — it is
 * a report of where its members are, not a choice.
 */
function RingEditor({ rings, layout, onChange }: {
  rings: RingSpec[]
  layout: Ring[]
  onChange: (rings: RingSpec[]) => void
}) {
  const [tilt, setTilt] = useState('1.0')
  const [count, setCount] = useState('6')
  const nextId = useRef(0)

  const add = (): void => {
    const t = Number(tilt)
    const n = Number(count)
    if (!Number.isFinite(t) || t <= 0 || !Number.isInteger(n) || n < 1) return
    nextId.current += 1
    onChange([...rings, { id: nextId.current, tilt: t, count: n }])
  }

  return (
    <div data-testid="maped-ring-editor" style={styles.ringEditor}>
      <div style={styles.ringEditorRow}>
        <span style={styles.ringEditorLabel}>Add a ring</span>
        <input data-testid="maped-ring-tilt" type="number" step="any" min="0"
          value={tilt} onChange={(e) => setTilt(e.target.value)}
          style={styles.ringInput} title="Tilt of the ring, in degrees" />
        <span style={styles.ringEditorTimes}>° ×</span>
        <input data-testid="maped-ring-count" type="number" step="1" min="1"
          value={count} onChange={(e) => setCount(e.target.value)}
          style={styles.ringInput} title="How many angles sit on the ring" />
        <span style={styles.ringEditorTimes}>angles</span>
        <button data-testid="maped-ring-add" style={styles.ringAdd}
          onClick={add}>Add</button>
      </div>
      <div style={styles.ringChips}>
        {layout.map((ring) => (
          <span key={ring.key} data-testid={`maped-ring-chip-${ring.key}`} style={styles.ringChip}>
            {formatDegrees(ring.tilt)} × {ring.slots.length}
            {ring.specId == null
              ? <span style={styles.ringChipFromData} title="From the data">from data</span>
              : (
                <button data-testid={`maped-ring-remove-${ring.specId}`}
                  style={styles.ringChipRemove} title="Remove this ring"
                  onClick={() => onChange(rings.filter((r) => r.id !== ring.specId))}>×</button>
              )}
          </span>
        ))}
        {layout.length === 0 && (
          <span style={styles.hint}>No rings</span>
        )}
      </div>
    </div>
  )
}

/** Members the picture cannot place: no tilt, or no azimuth. Dragging one onto
 *  a spot is what gives it both at once. */
function UnplacedTray({ members, onZoom }: {
  members: MapedMember[]
  onZoom: (member: MapedMember) => void
}) {
  return (
    <div data-testid="maped-unplaced" style={styles.tray}>
      <div style={styles.trayHead}>
        {members.length} dataset{members.length === 1 ? '' : 's'} with no angle yet —
        drag one onto a spot, or double-click to type its angles.
      </div>
      <div style={styles.trayRow}>
        {members.map((member) => (
          <div key={member.index} data-testid={`maped-unplaced-${member.index}`}
            draggable title={member.path}
            onDragStart={(e) => startMemberDrag(e, member.index)}
            onDoubleClick={() => onZoom(member)}
            style={{ ...styles.trayChip, ...(member.error ? styles.trayChipError : null) }}>
            {member.preview
              ? <img src={member.preview} alt="" style={styles.trayThumb} draggable={false} />
              : <span style={styles.trayGlyph}>{member.error ? '⚠' : '▦'}</span>}
            <span style={styles.trayName}>{member.name}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/** Every member that failed to open, with what went wrong — one place, always
 *  rendered, so a failure can never be just a small red tile. */
function ProblemList({ members }: { members: MapedMember[] }) {
  return (
    <div data-testid="maped-problems" style={styles.problems}>
      {members.map((member) => (
        <div key={member.index} data-testid={`maped-member-error-${member.index}`}
          style={styles.problemRow}>
          <span style={styles.problemName}>{member.name}</span>
          <span style={styles.problemText}>⚠ {member.error}</span>
        </div>
      ))}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Reciprocal space: the corners
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Each member's four scan corners, and the extent summed into each.
 *
 * Centring the direct beam does not need every pattern in the scan — four
 * corner sums say where the beam sits and how the flat field tilts across the
 * scan, which is why this is the fast method. The extents are INPUTS, so the
 * panels are here before anything has run, with the previews filling in
 * afterwards.
 */
/**
 * The real-space solve's evidence: the members summed, aligned and not.
 *
 * In the dialog rather than in a window behind it. The dialog is a full-screen
 * modal, so a figure opened in the workspace cannot be looked at or reached
 * while it is up — and "the residual was 0.25 px" is not evidence that the
 * members landed on each other. Two pictures are.
 *
 * The sharpness ratio is stated plainly, including when it is ~1.00: a solve
 * that did not sharpen the sum is the case worth noticing, and a number that
 * only ever appears when it flatters the result is not evidence either.
 */
/**
 * Which axes the specimen determined, from patches of the field voting.
 *
 * Stated per axis because they are routinely not alike: on a layered specimen
 * the patches agree closely across the layers and barely at all along them,
 * where the specimen is uniform and there is nothing to register on. Both
 * offsets are still numbers, and without this the dialog presents them as
 * equally good.
 */
function AxisConfidence({ confidence, onNudge }: {
  confidence: MapedConfidence, onNudge?: () => void }) {
  const axes = [
    { key: 'x' as const, label: 'across' },
    { key: 'y' as const, label: 'down' },
  ]
  // No votes is NOT a finding. A scan too small to divide into patches was
  // never checked, and saying "the specimen does not fix this" about it would
  // be inventing a result — the opposite of what this panel is for.
  if (confidence.votes <= 0) {
    return (
      <div data-testid="maped-real-confidence" style={styles.confidenceRow}>
        <span style={{ color: '#6c7086' }}>
          too few scan positions to check the axes separately
        </span>
      </div>
    )
  }
  const weak = axes.filter((axis) => !confidence.determined[axis.key])
  return (
    <div data-testid="maped-real-confidence" style={styles.confidenceRow}>
      <span style={{ color: '#a6adc8' }}>
        {`${confidence.votes} patches agreed:`}
      </span>
      {axes.map((axis) => {
        const value = confidence.agreement[axis.key]
        const firm = confidence.determined[axis.key]
        return (
          <span key={axis.key}
            data-testid={`maped-real-confidence-${axis.key}`}
            style={{ color: firm ? '#a6e3a1' : '#f9e2af' }}>
            {`${axis.label} ${value == null ? '—' : `${Math.round(value * 100)}%`}`}
          </span>
        )
      })}
      {weak.length > 0 && (
        <span data-testid="maped-real-unconstrained" style={styles.confidenceWarning}>
          {`— the specimen does not fix ${weak.map((a) => a.label).join(' or ')}: `
           + 'that offset is a guess. '}
          {onNudge && (
            <button data-testid="maped-nudge-hint" style={styles.confidenceLink}
              onClick={onNudge}>
              Set it by eye ↓
            </button>
          )}
        </span>
      )}
    </div>
  )
}

/**
 * One member against the reference, and the two overlaid.
 *
 * The tab's evidence is the sum over EVERY member, where one member's
 * contribution is a fraction and moving it by a pixel changes almost nothing
 * visible. Judging an alignment needs the two pictures that are supposed to
 * coincide and nothing else in the way.
 *
 * The overlay is the member in red over the reference in cyan. Misaligned,
 * every edge carries a coloured fringe and the side the fringe falls on says
 * which way to press; aligned, the colours cancel to grey — a state the eye
 * reads without being told what score to expect.
 */
function PairView({ pair, members, reference, images, offsets, solverOffsets,
                   weak, padRef, onSet, onImage, onClose }: {
  pair: MapedPair
  members: MapedMember[]
  reference: number | null
  images: string[]
  offsets: number[][] | null
  solverOffsets: number[][] | null
  weak: string[]
  padRef: React.MutableRefObject<HTMLDivElement | null>
  onSet: (index: number, offset: number[] | null) => void
  onImage: (name: string | null) => void
  onClose: () => void
}) {
  useEffect(() => { padRef.current?.focus() }, [pair.index, padRef])
  const member = members.find((m) => m.index === pair.index) ?? null
  const fixed = members.find((m) => m.index === reference) ?? null
  const strip = (name: string): string => name.replace(/\.[^.]+$/, '')
  const gain = pair.gain

  return (
    <div data-testid="maped-pair" style={styles.zoomOverlay}>
      <div style={styles.pairBox}>
        <div style={styles.pairHead}>
          <span style={{ color: '#cdd6f4', fontWeight: 600 }}>
            {`${strip(member?.name ?? '?')} ↔ ${strip(fixed?.name ?? '?')}`}
            <span style={{ color: '#6c7086', fontWeight: 400 }}> (reference)</span>
          </span>
          <span style={{ flex: 1 }} />
          <span style={{ color: '#a6adc8', fontSize: 11 }}>Image</span>
          <Dropdown<string>
            value={pair.image ?? COMPUTE_VI}
            options={virtualImageOptions(images)}
            onChange={(v) => onImage(namedImage(v))}
            testid="maped-pair-image" width={200}
          />
          <button data-testid="maped-pair-close" style={styles.nudgeReset}
            onClick={onClose}>Close</button>
        </div>

        <div style={styles.pairRow}>
          <div style={styles.pairSide}>
            {([['maped-pair-member', strip(member?.name ?? ''), pair.member],
               ['maped-pair-reference', `${strip(fixed?.name ?? '')} ★`,
                pair.reference]] as const).map(([testid, label, src]) => (
              <figure key={testid} style={styles.pairFigure}>
                {src && <img data-testid={testid} src={src} alt=""
                  style={styles.pairSmall} draggable={false} />}
                <figcaption style={styles.evidenceCaption}>{label}</figcaption>
              </figure>
            ))}
          </div>
          {pair.overlay && (
            <figure style={styles.pairFigure}>
              <img data-testid="maped-pair-overlay" src={pair.overlay} alt=""
                style={styles.pairBig} draggable={false} />
              <figcaption style={styles.evidenceCaption}>
                {`red = ${strip(member?.name ?? '')} · `
                 + `cyan = ${strip(fixed?.name ?? '')} · grey = aligned`}
              </figcaption>
            </figure>
          )}
        </div>

        <NudgePad
          members={members} reference={reference} selected={pair.index}
          offsets={offsets} solverOffsets={solverOffsets} weak={weak}
          padRef={padRef} onSelect={() => undefined} onSet={onSet}
        />

        <div style={styles.pairFoot}>
          <span data-testid="maped-pair-gain">
            {gain == null ? 'pair sharpness —'
              : `pair sharpness x${gain.toFixed(2)}`}
          </span>
          <span style={{ flex: 1 }} />
          <span style={{ color: '#6c7086' }}>Esc closes</span>
        </div>
      </div>
    </div>
  )
}


const NUDGE_AXES = [
  { key: 'x' as const, label: 'across' },
  { key: 'y' as const, label: 'down' },
]

/** How far one press moves a member, and how far with Shift held. */
const NUDGE_STEP = 1
const NUDGE_BIG_STEP = 5

/**
 * Move one member by hand, with the arrow keys.
 *
 * Here because the solve reports an offset per axis whether or not the
 * specimen determined one, and where it did not the number came from noise.
 * Nobody can register a feature that is not there — but someone looking at
 * the two pictures below this can place it, and this is how they say so.
 *
 * The keys are handled on THIS element, never on the window: the tab also has
 * number fields and a dropdown, and taking arrows away from those to serve a
 * pad that may not even be in view would be a poor trade.
 */
function NudgePad({ members, reference, selected, offsets, solverOffsets,
                   weak, padRef, onSelect, onSet }: {
  members: MapedMember[]
  reference: number | null
  selected: number | null
  offsets: number[][] | null
  solverOffsets: number[][] | null
  weak: string[]
  padRef: React.MutableRefObject<HTMLDivElement | null>
  onSelect: (index: number) => void
  onSet: (index: number, offset: number[] | null) => void
}) {
  const [live, setLive] = useState(false)
  const movable = members.filter((m) => m.index !== reference && !m.error)
  const index = selected != null && movable.some((m) => m.index === selected)
    ? selected
    : (movable[0]?.index ?? null)
  const member = members.find((m) => m.index === index) ?? null
  // The tableau labels a member without its extension; two names for one
  // thing in one panel reads like two things.
  const label = (member?.name ?? '').replace(/\.[^.]+$/, '')
  const current = index != null ? offsets?.[index] ?? null : null
  const solver = index != null ? solverOffsets?.[index] ?? null : null
  const edited = current != null && solver != null
    && (current[0] !== solver[0] || current[1] !== solver[1])

  // Where the last press asked the member to be. Two presses inside one
  // round trip would otherwise both add to the same snapshot and the second
  // would overwrite the first, so a held key would move at snapshot rate and
  // quick taps would vanish.
  const pending = useRef<number[] | null>(null)
  useEffect(() => {
    const sent = pending.current
    if (sent && current && sent[0] === current[0] && sent[1] === current[1]) {
      pending.current = null
    }
  }, [current?.[0], current?.[1]])
  useEffect(() => { pending.current = null }, [index])

  const move = (dy: number, dx: number): void => {
    const base = pending.current ?? current
    if (index == null || base == null) return
    // ABSOLUTE, not a step: a held key can drop every message but the last
    // and still land in the right place.
    const next = [base[0] + dy, base[1] + dx]
    pending.current = next
    onSet(index, next)
  }

  const onKeyDown = (e: React.KeyboardEvent): void => {
    const step = e.shiftKey ? NUDGE_BIG_STEP : NUDGE_STEP
    const moves: Record<string, [number, number]> = {
      ArrowUp: [-step, 0], ArrowDown: [step, 0],
      ArrowLeft: [0, -step], ArrowRight: [0, step],
    }
    const delta = moves[e.key]
    if (!delta) return          // Escape still closes the zoom, Tab still tabs
    e.preventDefault()
    e.stopPropagation()
    move(delta[0], delta[1])
  }

  if (index == null) return null
  return (
    <div
      data-testid="maped-nudge" ref={padRef} tabIndex={0}
      onKeyDown={onKeyDown}
      onFocus={() => setLive(true)} onBlur={() => setLive(false)}
      style={{ ...styles.nudgePad, ...(live ? styles.nudgePadLive : null) }}
    >
      <div style={styles.nudgeRow}>
        {movable.length > 1 && (
          <Dropdown<string>
            value={String(index)}
            options={movable.map((m) => ({ value: String(m.index), label: stem(m.name) }))}
            onChange={(v) => onSelect(Number(v))}
            testid="maped-nudge-member" width={150}
          />
        )}
        {movable.length <= 1 && (
          <span data-testid="maped-nudge-member" style={{ color: '#cdd6f4' }}>
            {label}
          </span>
        )}
        <span data-testid="maped-nudge-offset"
          style={{ color: edited ? '#f9e2af' : '#cdd6f4' }}>
          {current ? `(${current[0]}, ${current[1]}) px` : '—'}
        </span>
        {edited && solver && (
          <span data-testid="maped-nudge-solver" style={{ color: '#6c7086' }}>
            {`solver (${solver[0]}, ${solver[1]})`}
          </span>
        )}
        {([['maped-nudge-up', '↑', -1, 0],
           ['maped-nudge-down', '↓', 1, 0],
           ['maped-nudge-left', '←', 0, -1],
           ['maped-nudge-right', '→', 0, 1]] as const).map(
          ([testid, glyph, dy, dx]) => (
            <button key={testid} data-testid={testid} style={styles.nudgeKey}
              onClick={() => { move(dy, dx); padRef.current?.focus() }}>
              {glyph}
            </button>
          ))}
        <button
          data-testid="maped-nudge-reset"
          style={{ ...styles.nudgeReset, ...(edited ? null : styles.nudgeSpent) }}
          disabled={!edited}
          onClick={() => {
            pending.current = null
            onSet(index, null)
            padRef.current?.focus()
          }}
        >
          Reset
        </button>
      </div>
      <div data-testid="maped-nudge-help"
        style={{ ...styles.nudgeHelp, color: live ? '#cdd6f4' : '#6c7086' }}>
        {live ? '' : 'click here, then '}
        {`arrow keys move ${label || 'it'} · Shift for ${NUDGE_BIG_STEP} px`}
        {weak.length > 0 && ` · ${weak.join(' and ')} `
          + `${weak.length > 1 ? 'were' : 'was'} not measured`}
      </div>
    </div>
  )
}

function AlignmentEvidence({ evidence, confidence, solve, members, reference,
                            selected, padRef, boxRef, onSelect, onSet }: {
  evidence: MapedEvidence
  confidence: MapedConfidence | null
  solve: MapedSolve
  members: MapedMember[]
  reference: number | null
  selected: number | null
  padRef: React.MutableRefObject<HTMLDivElement | null>
  boxRef: React.MutableRefObject<HTMLDivElement | null>
  onSelect: (index: number) => void
  onSet: (index: number, offset: number[] | null) => void
}) {
  const weakAxes = NUDGE_AXES
    .filter((axis) => confidence && !confidence.determined[axis.key]
                      && confidence.votes > 0)
    .map((axis) => axis.label)
  const gain = evidence.gain
  // Has anyone been moved by hand? Then the solver's verdict is a judgement
  // of somebody else's decision, and stating it as if it were about their
  // move is the misleading case.
  const edited = solve.offsets != null && solve.solver_offsets != null
    && solve.offsets.some((row, index) => {
      const answer = solve.solver_offsets?.[index]
      return !answer || row[0] !== answer[0] || row[1] !== answer[1]
    })
  const solverGain = evidence.solver_gain
  const verdict = gain == null ? null
    : edited && solverGain != null
      ? {
          text: `solver x${solverGain.toFixed(2)}`,
          tone: gain > solverGain * 1.02 ? '#a6e3a1'
            : gain < solverGain * 0.98 ? '#f38ba8' : '#f9e2af',
        }
    : edited ? { text: 'moved by hand', tone: '#f9e2af' }
    : gain >= 1.15 ? { text: 'the members stack', tone: '#a6e3a1' }
    : gain >= 1.05 ? { text: 'a little sharper', tone: '#f9e2af' }
    : { text: 'aligning barely changed the sum — check it', tone: '#f38ba8' }
  return (
    <div data-testid="maped-real-evidence" style={styles.evidenceBox}
      ref={boxRef}>
      <div style={styles.evidenceColumns}>
      <div style={styles.evidenceText}>
      {/* The number first: the panel sits at the bottom of a scrolling tab, so
          anything below the pictures is the part a user does not see. */}
      {gain != null && (
        <div style={styles.evidenceHeader}>
          <span data-testid="maped-real-gain" style={{ color: verdict?.tone }}>
            {`sharpness x${gain.toFixed(2)}`}
          </span>
          {verdict && (
            <span data-testid="maped-real-verdict" style={{ color: verdict.tone }}>
              {` · ${verdict.text}`}
            </span>
          )}
        </div>
      )}
      {confidence && (
        <AxisConfidence confidence={confidence}
          onNudge={() => onSelect(
            members.find((m) => m.index !== reference && !m.error)?.index ?? 0)}
        />
      )}
      <NudgePad
        members={members} reference={reference} selected={selected}
        offsets={solve.offsets} solverOffsets={solve.solver_offsets}
        weak={weakAxes} padRef={padRef} onSelect={onSelect} onSet={onSet}
      />
      </div>
      {/* Beside the controls, not under them: the tab body scrolls, and a
          picture below the fold cannot show the user what their key press
          just did — which is the only reason the pad exists. */}
      <div style={styles.evidenceRow}>
        {([['Unaligned', evidence.unaligned],
           ['Aligned', evidence.aligned]] as const).map(([label, src]) => (
          <figure key={label} style={styles.evidenceFigure}>
            {src
              ? <img data-testid={`maped-real-evidence-${label.toLowerCase()}`}
                  src={src} alt="" style={styles.evidenceImage} draggable={false} />
              : <div style={styles.evidenceMissing}>not drawn</div>}
            <figcaption style={styles.evidenceCaption}>{label}</figcaption>
          </figure>
        ))}
      </div>
      </div>
    </div>
  )
}

/**
 * The zero-beam search region, drawn on a corner panel and draggable on it.
 *
 * The region is what stops the beam finder reading a reflection instead of the
 * beam: it is a centre of mass, so over a whole pattern it goes wherever the
 * excited reflections are. Placing it is therefore a measurement decision, and
 * a measurement decision belongs on the picture rather than in a number field —
 * though the field is there too, because "24" is easier to repeat than a drag.
 *
 * Drag the middle to move it, a corner to resize it. The panel maps the whole
 * detector onto a square, so panel fractions convert straight to detector
 * pixels. One region for the acquisition, so dragging it on any member's panel
 * moves the one every member is searched with.
 */
function BeamRegionBox({ roi, detector, onChange }: {
  roi: BeamRoi
  detector: number[] | null
  onChange: (roi: BeamRoi) => void
}) {
  const host = React.useRef<HTMLDivElement | null>(null)
  const height = detector?.[0] ?? 0
  const width = detector?.[1] ?? 0
  if (!height || !width) return null

  const left = ((roi.cx - roi.half) / width) * 100
  const top = ((roi.cy - roi.half) / height) * 100
  const size = ((roi.half * 2) / Math.max(width, height)) * 100

  /** Drag in panel pixels, applied in detector pixels.
   *
   * No `preventDefault` and no pointer capture on the way in: both stop the
   * browser synthesising the click, and a DOUBLE-click on the region is how
   * the panel under it gets enlarged — which is the only place the region can
   * be aimed properly. The listeners go on the window instead, so a fast drag
   * that leaves the box still tracks, and a two-pixel dead zone keeps the
   * jitter of a double-click from moving anything.
   */
  const drag = (event: React.PointerEvent, mode: 'move' | 'size') => {
    event.stopPropagation()
    const panel = host.current?.parentElement
    if (!panel) return
    const box = panel.getBoundingClientRect()
    const perPixelX = width / box.width
    const perPixelY = height / box.height
    const startX = event.clientX, startY = event.clientY
    const start = { ...roi }

    const clamp = (next: BeamRoi): BeamRoi => {
      const half = Math.max(3, Math.min(next.half, Math.min(width, height) / 2))
      return {
        half,
        cx: Math.max(half, Math.min(width - half, next.cx)),
        cy: Math.max(half, Math.min(height - half, next.cy)),
      }
    }
    const onMove = (move: PointerEvent) => {
      if (Math.abs(move.clientX - startX) < 2 && Math.abs(move.clientY - startY) < 2) return
      const dx = (move.clientX - startX) * perPixelX
      const dy = (move.clientY - startY) * perPixelY
      onChange(clamp(mode === 'move'
        ? { ...start, cx: start.cx + dx, cy: start.cy + dy }
        : { ...start, half: start.half + (dx + dy) / 2 }))
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  return (
    <div ref={host} data-testid="maped-beam-roi"
      onPointerDown={(e) => drag(e, 'move')}
      /* The double-click is deliberately NOT swallowed: it enlarges the panel,
         and the region sits over the middle of it — exactly where someone
         double-clicks to get a closer look at the beam they are aiming at.
         A drag only moves the region once the pointer moves, so the two
         gestures do not collide. */
      style={{ ...styles.beamRoi, left: `${left}%`, top: `${top}%`,
               width: `${size}%`, height: `${size}%` }}
    >
      <div data-testid="maped-beam-roi-handle" style={styles.beamRoiHandle}
        onPointerDown={(e) => drag(e, 'size')} />
    </div>
  )
}

function CornerTableau({ state, onExtent, onZoom, onBeamRoi }: {
  state: MapedState
  onExtent: (member: number | null, corner: number, extent: number) => void
  onZoom: (target: ZoomTarget) => void
  onBeamRoi: (roi: BeamRoi) => void
}) {
  const members = state.members.filter((m) => !m.error)
  const corners = state.reciprocal.corners

  if (members.length === 0) {
    return <div data-testid="maped-corners-empty" style={styles.hint}>
      No members
    </div>
  }

  /** The extent shared by every member at this corner, or null when they
   *  differ — a blank box is honest where one member's number would not be. */
  const commonExtent = (corner: number): number | null => {
    const values = members.map((m) => corners[String(m.index)]?.extents[corner] ?? null)
    const first = values[0]
    return values.every((v) => v === first) ? first : null
  }

  return (
    <div style={styles.corners}>
      <div data-testid="maped-corner-all" style={styles.cornerAll}>
        <span style={styles.cornerAllLabel}>Extent for every member (scan px)</span>
        {CORNER_SHORT.map((short, corner) => (
          <label key={short} style={styles.cornerAllField}>
            <span style={styles.cornerAllShort}>{short}</span>
            <ExtentInput
              testid={`maped-corner-extent-all-${corner}`}
              value={commonExtent(corner)} width={48}
              onChange={(extent) => onExtent(null, corner, extent)}
            />
          </label>
        ))}
        {state.beam_roi && (
          <label style={styles.cornerAllField}
            title="The zero beam is looked for inside this square. Drag the
 green box on any panel to place it.">
            <span style={styles.cornerAllLabel}>Zero-beam search ±px</span>
            <NumInput
              testid="maped-beam-roi-half"
              value={Math.round(state.beam_roi.half)}
              step="1" min={MIN_BEAM_ROI_HALF}
              accept={(half) => half >= MIN_BEAM_ROI_HALF}
              onChange={(half) => state.beam_roi
                && onBeamRoi({ ...state.beam_roi, half })}
              width={56} style={styles.extentInput}
            />
          </label>
        )}
      </div>

      <div style={styles.cornerCards}>
        {members.map((member) => {
          const entry = corners[String(member.index)]
          return (
            <div key={member.index} data-testid={`maped-corners-${member.index}`}
              style={styles.cornerCard}>
              <div style={styles.cornerCardHead} title={member.path}>
                <span style={styles.cornerCardName}>{member.name}</span>
                <span style={styles.cornerCardAngles}>
                  {formatDegrees(member.tilt)}
                  {member.azimuth == null ? '' : ` · ${formatAzimuth(member.azimuth)}`}
                </span>
              </div>
              <div style={styles.cornerGrid}>
                {CORNER_LABELS.map((label, corner) => {
                  const src = entry?.previews[corner] ?? null
                  return (
                    <div key={label} style={styles.cornerCell}>
                      <div
                        data-testid={`maped-corner-${member.index}-${corner}`}
                        title={`${label} corner of ${member.name}`}
                        onDoubleClick={() => onZoom({
                          kind: 'panel', src,
                          caption: `${member.name} — ${label} corner`,
                          detector: member.detector_shape,
                        })}
                        style={{ ...styles.cornerPanel, ...(src ? null : styles.cornerPanelEmpty) }}
                      >
                        {src
                          ? <img src={src} alt="" style={styles.cornerImage} draggable={false} />
                          : <span style={styles.cornerGlyph}>{CORNER_SHORT[corner]}</span>}
                        {src && state.beam_roi && (
                          <BeamRegionBox roi={state.beam_roi}
                            detector={member.detector_shape}
                            onChange={onBeamRoi} />
                        )}
                      </div>
                      <ExtentInput
                        testid={`maped-corner-extent-${member.index}-${corner}`}
                        value={entry?.extents[corner] ?? null}
                        onChange={(extent) => onExtent(member.index, corner, extent)}
                      />
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/** A corner's extent in scan pixels: a positive whole number of them. */
const isPositiveInteger = (n: number): boolean => Number.isInteger(n) && n > 0

/** A corner's extent box — a `NumInput` sized to the panel it sits under. The
 *  shared row gives a width in pixels; a per-corner box fills its panel. */
function ExtentInput({ value, onChange, testid, width }: {
  value: number | null
  onChange: (v: number) => void
  testid: string
  width?: number
}) {
  return (
    <NumInput
      testid={testid} value={value} onChange={onChange}
      step="1" min={1} accept={isPositiveInteger} placeholder="—"
      width={width ?? '100%'} style={styles.extentInput}
    />
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// The enlarged panel
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Any panel, large — the one behaviour shared by all three tableaux.
 *
 * A thumbnail sized to fit ten of them on a ring is too small to judge an
 * alignment by, which is the whole job of the later tabs, so every panel
 * double-clicks into this. Escape or a click anywhere dismisses it; the detail
 * box swallows its own clicks so the controls inside it stay usable.
 */
function PanelZoom({ src, caption, detail, roi, detector, onRoi, onClose }: {
  src: string | null
  caption: string
  detail?: React.ReactNode
  roi?: BeamRoi | null
  detector?: number[] | null
  onRoi?: (roi: BeamRoi) => void
  onClose: () => void
}) {
  return (
    <div data-testid="maped-zoom" style={styles.zoomOverlay} onClick={onClose}>
      <div style={styles.zoomBox}>
        <div data-testid="maped-zoom-caption" style={styles.zoomCaption}>{caption}</div>
        {src
          ? <div style={styles.zoomImageBox} onClick={(e) => e.stopPropagation()}>
              <img data-testid="maped-zoom-image" src={src} alt=""
                style={styles.zoomImage} draggable={false} />
              {roi && detector && onRoi && (
                <BeamRegionBox roi={roi} detector={detector} onChange={onRoi} />
              )}
            </div>
          : <div data-testid="maped-zoom-blank" style={styles.zoomBlank}>
              Nothing rendered for this panel yet.
            </div>}
        {detail && (
          <div onClick={(e) => e.stopPropagation()}>{detail}</div>
        )}
        <div style={styles.zoomHint}>Esc to close</div>
      </div>
    </div>
  )
}

/**
 * The scan grid for the whole acquisition, when a file could not supply one.
 *
 * An MRC records how many frames it holds and how big the detector is, but not
 * how those frames were laid out on the specimen — so a bare MRC probes as a
 * 3-D stack and cannot be a 4-D member until someone says 32 × 32. Setting it
 * re-probes every member that needs it; clearing BOTH boxes sends null, which
 * re-probes back to whatever the files say on their own, so a wrong guess is
 * always undoable.
 *
 * A half-typed pair (one box filled, one empty) sends nothing — it is neither a
 * grid nor a request to forget one.
 */
function ScanShapeField({ value, onChange }: {
  value: number[] | null
  onChange: (v: number[] | null) => void
}) {
  // The two boxes are ONE value, so what is typed in either has to outlive the
  // other being filled in: a half-typed pair sends nothing, so no snapshot can
  // hand the first box its own number back, and a box that dropped its draft on
  // blur would leave the user watching what they just typed disappear as they
  // tabbed across. The typed pair is therefore held HERE — each `NumInput`'s
  // own draft only covers the box being typed in — and the snapshot is adopted
  // only when the grid it carries actually CHANGES (a re-probe, or the echo of
  // a grid someone else set).
  const gridKey = value == null ? '' : `${value[0]}x${value[1]}`
  const [typed, setTyped] = useState<{ x: number | null; y: number | null }>(
    () => ({ x: value?.[0] ?? null, y: value?.[1] ?? null }))
  const seen = useRef(gridKey)
  useEffect(() => {
    if (gridKey === seen.current) return
    seen.current = gridKey
    setTyped({ x: value?.[0] ?? null, y: value?.[1] ?? null })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gridKey])

  const edit = (axis: 'x' | 'y', n: number | null): void => {
    const next = { ...typed, [axis]: n }
    setTyped(next)
    if (next.x == null && next.y == null) { onChange(null); return }
    // A half-typed pair is neither a grid nor a request to forget one.
    if (next.x != null && next.y != null) onChange([next.x, next.y])
  }

  const axis = (key: 'x' | 'y') => (
    <NumInput
      testid={`maped-scan-${key}`} value={typed[key]}
      step="1" min={1} accept={isPositiveInteger} placeholder={key}
      width={62} onChange={(n) => edit(key, n)} onClear={() => edit(key, null)}
    />
  )

  return (
    <div data-testid="maped-scan-shape" style={styles.scanShape}>
      <Field label={<>Scan grid <Info width={300} testid="maped-info-scan-shape" text={INFO.scanShape} /></>}>
        <span style={styles.scanInputs}>
          {axis('x')}
          <span style={styles.scanTimes}>×</span>
          {axis('y')}
        </span>
      </Field>
      <div style={styles.scanHint}>
        {value == null
          ? 'Not recorded in the file. Set it here.'
          : `Reading every member as ${value[0]} × ${value[1]}.`}
      </div>
    </div>
  )
}

/** The explanations, behind ⓘ — a paragraph that would otherwise sit on the
 *  face of a tab whose subject is controls, not prose. */
const INFO = {
  maxShift: 'Largest shift to search, in scan pixels. On a periodic sample '
    + 'an unbounded search can lock onto the wrong lattice translation. Set it '
    + 'a little above the drift you expect.',
  scanShape: 'Scan grid, x × y. MRC records the frame count and detector size '
    + 'but not the scan grid. A Direct Electron _info.txt supplies it, so this '
    + 'is normally not needed.',
}

/** The answer, including how trustworthy it is. The integer offsets are what
 *  gets applied; the max sub-pixel residual is what says whether rounding to
 *  them threw anything away. */
function SolveReport({ solve, members, testid }: {
  solve: MapedSolve
  members: MapedMember[]
  testid: string
}) {
  if (!solve.solved) {
    return <div data-testid={`${testid}-empty`} style={styles.hint}>Not solved</div>
  }
  const offsets = solve.offsets ?? []
  const ambiguous =
    solve.max_residual != null && solve.max_residual >= AMBIGUOUS_RESIDUAL_PX
  return (
    <div data-testid={`${testid}-result`} style={styles.result}>
      <div data-testid={`${testid}-residual`}
        style={{ ...styles.residual, color: ambiguous ? WARN : '#a6e3a1' }}>
        Max residual{' '}
        {solve.max_residual == null ? '—' : `${solve.max_residual.toFixed(3)} px`}
      </div>
      {ambiguous && (
        <div data-testid={`${testid}-residual-warning`} style={styles.residualWarning}>
          Near half a pixel — check these members.
        </div>
      )}
      <div style={styles.offsetList}>
        {offsets.map((offset, i) => {
          const residual = solve.residuals?.[i]
          const rowAmbiguous = residual != null && residual >= AMBIGUOUS_RESIDUAL_PX
          return (
            <div key={i} data-testid={`${testid}-offset-${i}`} style={styles.offsetRow}>
              <span style={styles.offsetName}>{members[i]?.name ?? `member ${i}`}</span>
              <span style={styles.offsetValue}>({offset.join(', ')})</span>
              <span style={{ ...styles.offsetResidual, color: rowAmbiguous ? WARN : '#6c7086' }}>
                {residual == null ? '' : `${residual.toFixed(3)} px`}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 9500,
    background: 'rgba(17,17,27,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  dialog: {
    width: 760, maxHeight: '88vh',
    display: 'flex', flexDirection: 'column',
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 10,
    padding: 18, color: '#cdd6f4',
    boxShadow: '0 16px 40px rgba(0,0,0,0.55)', fontSize: 13,
  },
  head: { display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
  title: { margin: 0, fontSize: 16, fontWeight: 600 },
  close: {
    background: 'none', border: 'none', color: '#6c7086',
    cursor: 'pointer', fontSize: 14,
  },
  summary: { margin: '4px 0 12px', fontSize: 12.5, color: '#a6adc8' },
  summaryError: { color: ERROR, fontWeight: 600 },
  body: {
    display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'stretch',
    overflowY: 'auto', padding: '12px 2px', minHeight: 200,
  },

  // ── the ring picture ──
  tableau: {
    position: 'relative', alignSelf: 'center', flex: '0 0 auto',
    background: '#11111b', border: '1px solid #313244', borderRadius: 12,
  },
  grid: {
    display: 'flex', flexWrap: 'wrap', gap: GRID_GAP,
    alignContent: 'flex-start',
  },
  gridCell: { display: 'flex', flexDirection: 'column', gap: 3 },
  gridCaption: { fontSize: 10, color: '#6c7086', textAlign: 'center' },
  tableauEmpty: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    border: '1px dashed #45475a', borderRadius: 12, padding: '34px 12px',
    color: '#6c7086', fontSize: 12, textAlign: 'center',
  },
  ringCircle: {
    position: 'absolute', borderRadius: '50%',
    border: `1px solid ${RING_COLOR}`, pointerEvents: 'none',
  },
  ringLabel: {
    position: 'absolute', transform: 'translate(-50%, -50%)',
    fontSize: 11, color: LABEL_COLOR, background: '#11111b',
    padding: '0 4px', pointerEvents: 'none',
  },
  slotAnchor: { position: 'absolute', transform: 'translate(-50%, -50%)' },
  slot: {
    position: 'relative', boxSizing: 'border-box',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    borderRadius: 10, overflow: 'visible',
  },
  // The name rides INSIDE the tile rather than below it: a caption hanging
  // under an outer-ring spot lands on the tile of the inner-ring spot at the
  // same azimuth, which is how the two rings were colliding.
  slotCaption: {
    position: 'absolute', left: 1, right: 1, bottom: 1,
    padding: '1px 3px', borderRadius: '0 0 8px 8px',
    background: 'rgba(17,17,27,0.78)',
    fontSize: 9, color: '#bac2de', textAlign: 'center',
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
    pointerEvents: 'none',
  },
  slotEmpty: {
    border: '1px dashed #585b70', background: '#181825', color: '#6c7086',
  },
  slotFilled: {
    border: `1px solid ${ACCENT}`, background: '#181825', color: '#cdd6f4',
  },
  slotErrored: {
    border: `1px solid ${ERROR}`, background: 'rgba(243,139,168,0.14)', color: ERROR,
  },
  slotOver: {
    borderColor: ACCENT, borderStyle: 'solid',
    boxShadow: `0 0 0 3px rgba(137,180,250,0.28)`,
  },
  slotSelected: {
    borderColor: '#89b4fa', borderWidth: 2,
    boxShadow: '0 0 0 2px rgba(137, 180, 250, 0.35)',
  },
  slotReference: { boxShadow: `0 0 0 2px ${WARN}` },
  slotImage: {
    width: '100%', height: '100%', objectFit: 'cover',
    borderRadius: 9, display: 'block',
  },
  slotGlyph: { fontSize: 16, lineHeight: 1, marginBottom: 8 },
  referenceBadge: {
    position: 'absolute', top: -6, left: -6, fontSize: 11, color: WARN,
    background: '#11111b', borderRadius: '50%', lineHeight: 1, padding: 1,
  },
  missingBadge: {
    position: 'absolute', top: -6, right: -6, fontSize: 10, fontWeight: 700,
    color: '#11111b', background: WARN, borderRadius: '50%',
    width: 14, height: 14, display: 'flex',
    alignItems: 'center', justifyContent: 'center',
  },
  slotErrorBadge: {
    position: 'absolute', top: -6, right: -6, fontSize: 10, fontWeight: 700,
    color: '#11111b', background: ERROR, borderRadius: '50%',
    width: 14, height: 14, display: 'flex',
    alignItems: 'center', justifyContent: 'center',
  },

  // ── ring editor ──
  ringEditor: {
    display: 'flex', flexDirection: 'column', gap: 6,
    background: '#181825', border: '1px solid #313244',
    borderRadius: 8, padding: '8px 12px',
  },
  ringEditorRow: { display: 'flex', alignItems: 'center', gap: 6 },
  ringEditorLabel: { fontSize: 11.5, color: '#a6adc8', marginRight: 2 },
  ringEditorTimes: { fontSize: 11, color: '#6c7086' },
  ringInput: {
    width: 58, background: '#11111b', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 4, padding: '3px 5px', fontSize: 11,
  },
  ringAdd: {
    background: 'transparent', border: `1px solid ${ACCENT}`, color: ACCENT,
    borderRadius: 5, padding: '3px 12px', fontSize: 11, cursor: 'pointer',
  },
  ringChips: { display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' },
  ringChip: {
    display: 'inline-flex', alignItems: 'center', gap: 5,
    background: '#11111b', border: '1px solid #313244', borderRadius: 9,
    padding: '1px 4px 1px 9px', fontSize: 11, color: '#cdd6f4',
  },
  ringChipFromData: { fontSize: 9.5, color: '#6c7086', padding: '0 5px' },
  ringChipRemove: {
    background: 'transparent', border: 'none', color: '#6c7086',
    fontSize: 14, lineHeight: 1, cursor: 'pointer', padding: '0 4px',
  },

  // ── drop strip, tray, problems ──
  dropzone: {
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
    flexWrap: 'wrap',
    border: '1px dashed #45475a', borderRadius: 10, padding: '10px 12px',
    color: '#a6adc8', fontSize: 12, textAlign: 'center',
  },
  dropzoneSep: { color: '#45475a' },
  linkButton: {
    background: 'none', border: 'none', color: ACCENT,
    cursor: 'pointer', fontSize: 12, padding: 0, textDecoration: 'underline',
  },
  dropNote: { fontSize: 11.5, color: WARN },
  tray: {
    display: 'flex', flexDirection: 'column', gap: 6,
    background: '#181825', border: '1px solid #313244',
    borderRadius: 8, padding: '8px 12px',
  },
  trayHead: { fontSize: 11, color: '#a6adc8', lineHeight: 1.4 },
  trayRow: { display: 'flex', flexWrap: 'wrap', gap: 6 },
  trayChip: {
    display: 'flex', alignItems: 'center', gap: 6,
    background: '#11111b', border: '1px solid #313244', borderRadius: 8,
    padding: '3px 8px 3px 3px', fontSize: 11, cursor: 'grab', maxWidth: 210,
  },
  trayChipError: { borderColor: ERROR, background: 'rgba(243,139,168,0.10)' },
  trayThumb: { width: 26, height: 26, objectFit: 'cover', borderRadius: 5 },
  trayGlyph: {
    width: 26, height: 26, display: 'flex',
    alignItems: 'center', justifyContent: 'center',
    background: '#181825', borderRadius: 5, fontSize: 12, color: '#6c7086',
  },
  trayName: {
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  problems: {
    display: 'flex', flexDirection: 'column', gap: 3,
    border: `1px solid ${ERROR}`, borderRadius: 8,
    background: 'rgba(243,139,168,0.08)', padding: '6px 10px',
  },
  problemRow: { display: 'flex', gap: 8, fontSize: 11, alignItems: 'baseline' },
  problemName: {
    minWidth: 110, color: '#cdd6f4',
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  problemText: { color: ERROR, flex: 1, lineHeight: 1.4 },
  referenceLine: { fontSize: 11.5, color: '#a6adc8', lineHeight: 1.45 },

  // ── corners ──
  corners: { display: 'flex', flexDirection: 'column', gap: 8 },
  cornerAll: {
    display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
    background: '#181825', border: '1px solid #313244',
    borderRadius: 8, padding: '7px 12px',
  },
  cornerAllLabel: { fontSize: 11.5, color: '#a6adc8' },
  cornerAllField: { display: 'flex', alignItems: 'center', gap: 4 },
  cornerAllShort: { fontSize: 10, color: '#6c7086', width: 16 },
  cornerCards: { display: 'flex', flexWrap: 'wrap', gap: 8 },
  cornerCard: {
    background: '#11111b', border: '1px solid #313244', borderRadius: 10,
    padding: 8, width: 224,
  },
  cornerCardHead: {
    display: 'flex', flexDirection: 'column', gap: 1, marginBottom: 6,
  },
  cornerCardName: {
    fontSize: 11.5, color: '#cdd6f4',
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  cornerCardAngles: { fontSize: 10, color: '#6c7086' },
  cornerGrid: {
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6,
  },
  cornerCell: { display: 'flex', flexDirection: 'column', gap: 3 },
  cornerPanel: {
    position: 'relative', width: '100%', aspectRatio: '1 / 1',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: '#181825', border: '1px solid #313244', borderRadius: 6,
    overflow: 'hidden', cursor: 'zoom-in',
  },
  cornerPanelEmpty: { borderStyle: 'dashed', borderColor: '#45475a' },
  evidenceBox: {
    display: 'flex', flexDirection: 'column', gap: 6,
    background: '#181825', border: '1px solid #313244',
    borderRadius: 8, padding: 8,
  },
  evidenceHeader: { fontSize: 12 },
  confidenceRow: {
    display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'baseline',
    fontSize: 11.5,
  },
  confidenceWarning: { color: '#f9e2af', flexBasis: '100%', fontSize: 11 },
  confidenceLink: {
    background: 'none', border: 'none', padding: 0, font: 'inherit',
    color: '#89b4fa', textDecoration: 'underline', cursor: 'pointer',
  },
  nudgePad: {
    display: 'flex', flexDirection: 'column', gap: 4,
    border: '1px solid #313244', borderRadius: 6, padding: '5px 7px',
    outline: '1px solid transparent',
  },
  nudgePadLive: { outline: '1px solid #89b4fa', borderColor: '#45475a' },
  nudgeRow: {
    display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap',
    fontSize: 11.5,
  },
  nudgeKey: {
    background: '#313244', color: '#cdd6f4', border: '1px solid #45475a',
    borderRadius: 4, cursor: 'pointer', fontSize: 12,
    width: 22, height: 20, lineHeight: '16px', padding: 0,
  },
  nudgeReset: {
    background: 'none', color: '#a6adc8', border: '1px solid #45475a',
    borderRadius: 4, cursor: 'pointer', fontSize: 11, padding: '1px 6px',
  },
  nudgeHelp: { fontSize: 10.5 },
  pairBox: {
    display: 'flex', flexDirection: 'column', gap: 8,
    background: '#181825', border: '1px solid #313244', borderRadius: 10,
    padding: 12, maxWidth: '100%', maxHeight: '100%',
  },
  pairHead: { display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 },
  pairRow: { display: 'flex', gap: 10, alignItems: 'flex-start' },
  pairSide: { display: 'flex', flexDirection: 'column', gap: 8 },
  pairFigure: { margin: 0, display: 'flex', flexDirection: 'column', gap: 3 },
  pairSmall: {
    width: 200, height: 200, objectFit: 'contain',
    imageRendering: 'pixelated', background: '#11111b', borderRadius: 4,
  },
  pairBig: {
    // The judging picture, so it gets the room: a one-pixel fringe has to be
    // a visible block, which needs integer upscaling and no smoothing.
    width: 412, height: 412, objectFit: 'contain',
    imageRendering: 'pixelated', background: '#11111b', borderRadius: 4,
  },
  pairFoot: { display: 'flex', gap: 8, fontSize: 11.5, color: '#a6adc8' },
  nudgeSpent: { opacity: 0.4, cursor: 'default' },
  runRow: { display: 'flex', gap: 14, alignItems: 'flex-end' },
  evidenceColumns: { display: 'flex', gap: 10, alignItems: 'flex-start' },
  evidenceText: {
    display: 'flex', flexDirection: 'column', gap: 6,
    flex: '1 1 380px', minWidth: 0,
  },
  evidenceRow: { display: 'flex', alignItems: 'flex-start', gap: 10 },
  evidenceFigure: { margin: 0, display: 'flex', flexDirection: 'column', gap: 4 },
  evidenceImage: {
    width: 132, height: 132, objectFit: 'contain',
    imageRendering: 'pixelated', background: '#11111b', borderRadius: 6,
  },
  evidenceMissing: {
    width: 132, height: 132, display: 'flex', alignItems: 'center',
    justifyContent: 'center', color: '#585b70', fontSize: 11,
    border: '1px dashed #45475a', borderRadius: 6,
  },
  evidenceCaption: { fontSize: 11, color: '#a6adc8', textAlign: 'center' },
  beamRoi: {
    position: 'absolute', boxSizing: 'border-box',
    border: '1.5px solid #a6e3a1', borderRadius: 3,
    background: 'rgba(166, 227, 161, 0.10)',
    cursor: 'move', touchAction: 'none',
  },
  beamRoiHandle: {
    position: 'absolute', right: -4, bottom: -4, width: 8, height: 8,
    background: '#a6e3a1', borderRadius: 2, cursor: 'nwse-resize',
    touchAction: 'none',
  },
  cornerImage: { width: '100%', height: '100%', objectFit: 'cover', display: 'block' },
  cornerGlyph: { fontSize: 11, color: '#585b70', letterSpacing: 1 },
  // What a `NumInput` in a corner panel looks like on top of its own styling:
  // smaller, centred, and sized to the cell rather than to its content.
  extentInput: {
    boxSizing: 'border-box', padding: '2px 4px', fontSize: 10.5, textAlign: 'center',
  },

  // ── enlarged panel ──
  zoomOverlay: {
    position: 'fixed', inset: 0, zIndex: 9600,
    background: 'rgba(17,17,27,0.92)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: 24,
  },
  zoomBox: {
    display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center',
    maxWidth: '100%', maxHeight: '100%',
  },
  zoomCaption: { fontSize: 13, color: '#cdd6f4', fontWeight: 600 },
  zoomImageBox: { position: 'relative', lineHeight: 0 },
  zoomImage: {
    // A WIDTH, not just a cap: these are thumbnails, so a max-size rule alone
    // leaves a 28-pixel preview drawn at 28 pixels and "enlarge" does nothing.
    // Upscaled without smoothing, because the pixels are the measurement.
    width: 'min(54vh, 58vw)', height: 'auto',
    maxHeight: '62vh', objectFit: 'contain',
    imageRendering: 'pixelated',
    border: '1px solid #45475a', borderRadius: 8, background: '#11111b',
  },
  zoomBlank: {
    width: 320, height: 240,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    border: '1px dashed #45475a', borderRadius: 8, color: '#6c7086', fontSize: 12,
  },
  zoomDetail: {
    display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center',
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 8,
    padding: '8px 14px',
  },
  zoomFacts: { fontSize: 11, color: '#a6adc8' },
  zoomError: { fontSize: 11, color: ERROR, maxWidth: 420, lineHeight: 1.4 },
  zoomAngles: { display: 'flex', alignItems: 'center', gap: 12 },
  zoomReference: {
    display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#a6adc8',
  },
  zoomRemove: {
    background: 'transparent', border: `1px solid ${ERROR}`, color: ERROR,
    borderRadius: 5, padding: '2px 10px', fontSize: 11, cursor: 'pointer',
  },
  zoomHint: { fontSize: 10.5, color: '#6c7086' },

  // ── shared form bits ──
  scanShape: {
    display: 'flex', flexDirection: 'column', gap: 4,
    background: '#181825', border: '1px solid #313244',
    borderRadius: 8, padding: '8px 12px',
  },
  scanInputs: { display: 'flex', alignItems: 'center', gap: 6 },
  scanTimes: { fontSize: 11, color: '#6c7086' },
  scanHint: { fontSize: 10.5, color: '#6c7086', lineHeight: 1.4 },
  hint: { fontSize: 11.5, color: '#6c7086', fontStyle: 'italic' },
  result: { display: 'flex', flexDirection: 'column', gap: 6 },
  residual: { fontSize: 12.5, fontWeight: 600 },
  residualWarning: { fontSize: 11.5, color: WARN, lineHeight: 1.4 },
  offsetList: { display: 'flex', flexDirection: 'column', gap: 2 },
  offsetRow: {
    display: 'flex', alignItems: 'center', gap: 10,
    background: '#11111b', borderRadius: 4, padding: '3px 8px',
  },
  offsetName: {
    flex: 1, fontSize: 11.5, whiteSpace: 'nowrap',
    overflow: 'hidden', textOverflow: 'ellipsis',
  },
  offsetValue: { fontSize: 11.5, fontFamily: 'monospace' },
  offsetResidual: { fontSize: 11, minWidth: 62, textAlign: 'right' },
  status: {
    fontSize: 11, color: '#a6adc8',
    borderTop: '1px solid #313244', padding: '8px 0', minHeight: 16,
  },
  footer: { display: 'flex', justifyContent: 'flex-end', gap: 8 },
  cancel: {
    background: 'transparent', border: '1px solid #313244', color: '#cdd6f4',
    borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 12,
  },
  confirm: {
    background: ACCENT, border: 'none', color: '#11111b', fontWeight: 600,
    borderRadius: 6, padding: '6px 18px', cursor: 'pointer', fontSize: 12,
  },
}
