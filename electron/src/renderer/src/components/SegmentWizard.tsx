/**
 * SegmentWizard.tsx — the Segment Particles caret (`seg_` staged actions,
 * backend: spyde/actions/particles_action.py; plan §B7).
 *
 * ONE KNOB AND A BUTTON. The caret used to show ~15 controls at once (a
 * sensitivity slider, min-size, three checkboxes, a `▸ more` block of ten more
 * parameters, a histogram, a stats line, the class list, three buttons and two
 * status lines) and the verdict on it was "information overload". The task is
 * "find the particles"; everything else is tuning for someone who already knows
 * the answer is wrong.
 *
 *   ┌ Segment Particles ──────── ✕ ┐
 *   │ [Classical] [Scribble] [Prompt]│
 *   │  Fewer ────●──── More          │
 *   │  Merge closer than ──●── 12 nm │
 *   │  Ignore smaller than ─●── off  │
 *   │  6 particles on this frame     │
 *   │  [    Find in all frames    ]  │
 *   │  ▸ Advanced                    │
 *   └────────────────────────────────┘
 *
 * Four things here are load-bearing, none of them cosmetic:
 *
 * 1. **The default face carries the task, and its two shared controls are
 *    PHYSICAL.** `merge_nm` and `min_nm` are distances the eye can check against
 *    the scale bar, and they act on the measured instances rather than on any one
 *    method's parameters — so the face is identical on all three engines, and only
 *    Classical adds the sensitivity slider above them. Their nm→px conversion
 *    needs the DISPLAYED signal's scale, which `SegmentWizard.set_params` stashes
 *    on every dispatch; without it the backend silently reads nanometres as pixels.
 *    Everything else is Advanced. Measured (plan §0.9), not taste: teaching the
 *    classifier faint contrast buys +1 true particle and 25 spurious ones, and
 *    `min_size=10` removes 24 of the 25. But the floor is applied by the BACKEND
 *    unconditionally, so the user does not have to know that — `min_size` is a
 *    recovery knob, not a tuning knob, and it lives in Advanced next to the floor
 *    warning that explains it.
 *
 * 2. **The EFFECTIVE `min_size` is what is shown.** The backend floors it and
 *    reports the floored value + a flag in every `seg_preview`; the caret snaps
 *    its field to that value rather than leaving the user's 0 on screen while a
 *    10 ran. Showing a number different from the one that ran is exactly the
 *    failure `SegmentParams` refuses for `local_size`. The warning that explains
 *    the snap renders INSIDE Advanced, immediately under the field it is about —
 *    as a block on the primary face it was a large orange alarm for a parameter
 *    nobody should normally touch.
 *
 * 3. **Per-class labelled-pixel counts are the point of the class list, and the
 *    class list is the SCRIBBLE tab's business.** Under-training a class is *the*
 *    failure mode and these counts are how you notice; a class below `LOW_PIXELS`
 *    is dimmed and flagged so "the preview got worse" pushes you toward painting
 *    another example instead of toward the sensitivity slider, which cannot fix a
 *    missing example. On the Classical tab there is nothing to train, so the list
 *    is not shown there.
 *
 * 4. **Nothing was deleted, only demoted — and Advanced is TWO COLUMNS.** Every
 *    control that left the primary face is inside `▸ Advanced` and sends the
 *    identical action with the identical payload — `params()` is unchanged and
 *    the backend schema is untouched. Stacked in one column that block reached
 *    907 px in an 805 px MDI area, and the caret cannot scroll (the Threshold
 *    menu is absolutely positioned and any `overflow:auto` ancestor clips it),
 *    so the histogram and Commit Frame were unreachable rather than merely
 *    cramped. Two columns — params you SET on the left, feedback the frame
 *    gives you on the right — is plan B7's own answer and the only one that
 *    neither deletes a control nor needs a scroller; the caret widens to
 *    `ADV_WIDTH` only while it is open. Advanced is collapsed by default and
 *    remembers its state for the session (module-scope, not per-window: it is a
 *    preference about the UI, not about a dataset). The one place it is
 *    tab-scoped is the classical
 *    MASK block (threshold / pre-blur / rolling ball / local window / dark), and
 *    that is correctness rather than tidiness: the scribble engine hands
 *    `split_instances` a probability map thresholded at 0.5 and never reads
 *    them, so on the Scribble tab they would be knobs that do nothing.
 *
 * The brush swatches / size / eraser are NOT here — they live on the floating
 * ClassStrip next to the plot (plan B0), because while painting you are looking
 * at the image. Class NAMES and counts stay here, which is the authoritative
 * list.
 */
import React from 'react'
import { WizardShell, TabRow, Slider, Select, Check, NumInput, Field, S } from './WizardShell'
import { useWizardLifecycle, useDebouncedAction, useWizardEvent, CommitButton } from './wizardHooks'
import type { SendAction } from './wizardHooks'
import { ClassStrip } from './ClassStrip'
import { useSpyDE } from '../kernel/SpyDEContext'
import type { SegClassInfo } from '../kernel/protocol'

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: SendAction
  onClose: () => void
  /** Absolute placement for the floating brush strip over the figure's
   *  top-left. Only FloatingToolbar knows the owning window's live rect. */
  stripPos?: React.CSSProperties
}

/** The three mask sources of plan §0.2 = `particles_action.METHODS`. */
type Method = 'classical' | 'scribble' | 'prompt'
const METHODS: readonly Method[] = ['classical', 'scribble', 'prompt']
// TabRow renders the tab VALUE as its label, so the tabs are the Title Case
// strings and the lowercase backend key is recovered for the action + testid.
type TabLabel = 'Classical' | 'Scribble' | 'Prompt'
const TABS: readonly TabLabel[] = ['Classical', 'Scribble', 'Prompt']
const METHOD_OF: Record<TabLabel, Method> = {
  Classical: 'classical', Scribble: 'scribble', Prompt: 'prompt',
}
const TAB_OF: Record<Method, TabLabel> = {
  classical: 'Classical', scribble: 'Scribble', prompt: 'Prompt',
}

/** `particles_action.THRESHOLD_METHODS`, in the backend's order. */
const THRESHOLDS = [
  'otsu', 'mean', 'minimum', 'yen', 'isodata', 'li',
  'local', 'local_otsu', 'niblack', 'sauvola',
] as const
type Threshold = typeof THRESHOLDS[number]
const THRESHOLD_OPTS = THRESHOLDS.map(v => ({ value: v, label: v }))

/** Below this many labelled pixels a class is flagged as under-trained. A
 *  scribble is a few hundred pixels per dab, so ~200 is "one dab or less". */
const LOW_PIXELS = 200

/** Mirrors `particles_action.DEFAULTS`. Kept in sync by shape, not by import —
 *  the backend re-coerces everything anyway and echoes the effective values. */
interface SegSaved {
  method: Method
  sensitivity: number
  minScore: number
  mergeNm: number
  minNm: number
  threshold: Threshold
  minSize: number
  maxSize: number
  watershed: boolean
  minSeparation: number
  markerSmooth: number
  gaussian: number
  rbKernel: number
  invert: boolean
  localSize: number
  clearBorder: boolean
  storeMasks: boolean
  track: boolean
  maxDist: number
  brush: number
  activeClass: number
  eraser: boolean
}
const DEFAULTS: SegSaved = {
  method: 'classical', sensitivity: 0.5, minScore: 0, mergeNm: 0, minNm: 0,
  threshold: 'otsu', minSize: 20,
  maxSize: 0, watershed: true, minSeparation: 3, markerSmooth: 1.0,
  gaussian: 0.0, rbKernel: 0, invert: false, localSize: 31, clearBorder: false,
  storeMasks: true, track: true, maxDist: 10.0, brush: 3.0,
  activeClass: 0, eraser: false,
}

// Tuned state kept OUTSIDE the component (same pattern as the FV/OM carets) so
// closing the caret to look at the frame doesn't lose the tuning.
const _segStore = new Map<number, SegSaved>()

// Whether `▸ Advanced` is open. Module scope, NOT per-window and NOT persisted
// to disk: a user who opened it once is mid-tuning and wants it open on the next
// caret too, but a fresh session starts calm again.
let _advancedOpen = false

interface Preview {
  frame: number
  count: number
  areas: number[]
  median: number
  units: string
  minSize: number
  floored: boolean
  elapsedMs: number
  /** `[y0, x0, h, w]` when the frame was too big to segment whole, else null —
   *  the count then describes that window, not the frame. */
  preview_box: [number, number, number, number] | null
  /** Fraction of the previewed window called foreground, and the backend's
   *  verdict that this is a failed threshold rather than a result. */
  coverage: number
  thresholdFailed: boolean
  /** What the two FACE sliders are in: 'nm' on a real-space axis, 'px' when
   *  the signal's axis is not a length (reciprocal space, mrad, uncalibrated). */
  faceUnits: string
  /** Monotonic per-caret preview counter. The COUNT is not a reliable "did it
   *  re-run" signal (two sensitivities can find the same number of particles),
   *  so the caret publishes this as `data-seq` for the e2e to poll instead. */
  seq: number
}

export function SegmentWizard({ caretPos, windowId, sendAction, onClose, stripPos }: Props) {
  const saved = _segStore.get(windowId) ?? DEFAULTS
  const [method, setMethod] = React.useState<Method>(saved.method)
  const [sensitivity, setSensitivity] = React.useState(saved.sensitivity)
  const [minScore, setMinScore] = React.useState(saved.minScore)
  const [mergeNm, setMergeNm] = React.useState(saved.mergeNm)
  const [minNm, setMinNm] = React.useState(saved.minNm)
  const [threshold, setThreshold] = React.useState<Threshold>(saved.threshold)
  const [minSize, setMinSize] = React.useState(saved.minSize)
  const [maxSize, setMaxSize] = React.useState(saved.maxSize)
  const [watershed, setWatershed] = React.useState(saved.watershed)
  const [minSeparation, setMinSeparation] = React.useState(saved.minSeparation)
  const [markerSmooth, setMarkerSmooth] = React.useState(saved.markerSmooth)
  const [gaussian, setGaussian] = React.useState(saved.gaussian)
  const [rbKernel, setRbKernel] = React.useState(saved.rbKernel)
  const [invert, setInvert] = React.useState(saved.invert)
  const [localSize, setLocalSize] = React.useState(saved.localSize)
  const [clearBorder, setClearBorder] = React.useState(saved.clearBorder)
  const [storeMasks, setStoreMasks] = React.useState(saved.storeMasks)
  const [track, setTrack] = React.useState(saved.track)
  const [maxDist, setMaxDist] = React.useState(saved.maxDist)
  const [brush, setBrush] = React.useState(saved.brush)
  const [activeClass, setActiveClass] = React.useState(saved.activeClass)
  const [eraser, setEraser] = React.useState(saved.eraser)
  const [advanced, setAdvanced] = React.useState(_advancedOpen)

  // Backend-owned state (never edited here, only rendered).
  const [classes, setClasses] = React.useState<SegClassInfo[]>([])
  const [labelledFrames, setLabelledFrames] = React.useState<number[]>([])
  const [trained, setTrained] = React.useState(false)
  const [frame, setFrame] = React.useState(0)
  const [preview, setPreview] = React.useState<Preview | null>(null)
  // The fit report is kept as its OWN line, not just a status message: the
  // backend follows `seg_trained` immediately with `_emit_state` + a re-preview,
  // whose "N particles on frame M" status overwrites it within milliseconds —
  // so a transient status is a report the user never gets to read.
  const [trainReport, setTrainReport] = React.useState<string | null>(null)
  const [status, setStatus] = React.useState('Drag Fewer / More, then find in all frames.')

  const vals = React.useRef<SegSaved>(saved)
  vals.current = {
    method, sensitivity, minScore, mergeNm, minNm, threshold, minSize, maxSize, watershed, minSeparation,
    markerSmooth, gaussian, rbKernel, invert, localSize, clearBorder,
    storeMasks, track, maxDist, brush, activeClass, eraser,
  }
  React.useEffect(() => { _segStore.set(windowId, vals.current) })

  // Mirror the disclosure into module scope from an EFFECT, never from inside
  // the state updater: React may invoke an updater more than once per dispatch
  // (StrictMode's double-invoke, the eager-bailout probe, a replayed queue), so
  // a write in there is not a "toggle once" — it is a toggle per invocation, and
  // the disclosure sticks open.
  React.useEffect(() => { _advancedOpen = advanced }, [advanced])
  const toggleAdvanced = () => setAdvanced(a => !a)

  /** The backend's parameter names (`particles_action.DEFAULTS` keys). Demoting
   *  a control into Advanced changes NOTHING here — same keys, same shape. */
  const params = (): Record<string, unknown> => {
    const v = vals.current
    return {
      method: v.method, sensitivity: v.sensitivity, min_score: v.minScore,
      merge_nm: v.mergeNm, min_nm: v.minNm,
      threshold: v.threshold,
      min_size: v.minSize, max_size: v.maxSize, watershed: v.watershed,
      min_separation: v.minSeparation, marker_smooth: v.markerSmooth,
      gaussian: v.gaussian, rb_kernel: v.rbKernel, invert: v.invert,
      local_size: v.localSize, clear_border: v.clearBorder,
      store_masks: v.storeMasks, track: v.track, max_dist: v.maxDist,
      brush: v.brush,
      // The brush WIDGET lives in Python, so the strip's state has to travel or
      // it cannot affect painting. These two were missing, and the symptoms were
      // exactly that: every stroke came out in class 0 ("I can only scribble one
      // colour") and the eraser never erased ("delete doesn't work"), because
      // the backend read `active_class`/`erase` from params that nothing set.
      active_class: v.activeClass,
      erase: v.eraser,
    }
  }

  // Mount → seg_open (previews the displayed frame), unmount → seg_close
  // (clears the overlay). StrictMode-safe: exactly one open reaches the backend.
  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'seg_open', openPayload: params, closeAction: 'seg_close',
  })

  // Debounced live tune → re-preview the CURRENT frame only. A pending tune is
  // cancelled on unmount so it cannot hit a torn-down preview.
  const sendTune = useDebouncedAction(sendAction, 'seg_tune', windowId)
  const tune = () => sendTune(params)
  const live = <T,>(set: (v: T) => void) => (v: T) => { set(v); tune() }

  // sendAction is recreated on EVERY provider render — it must never be an
  // effect dependency (see the verbatim note in ConsoleBar.tsx:226). Route the
  // stroke handler's send through a ref.
  const sendRef = React.useRef(sendAction)
  sendRef.current = sendAction

  useWizardEvent('spyde:seg_state', windowId, (d) => {
    if (Array.isArray(d.classes)) setClasses(d.classes as SegClassInfo[])
    if (Array.isArray(d.labelled_frames)) setLabelledFrames(d.labelled_frames as number[])
    if (typeof d.trained === 'boolean') setTrained(d.trained)
    if (typeof d.frame === 'number') setFrame(d.frame)
    // The engine can change WITHOUT the caret asking (seg_train switches to
    // scribble on success), so the tab follows the backend.
    if (typeof d.method === 'string' && METHODS.includes(d.method as Method)) {
      setMethod(d.method as Method)
    }
  })

  useWizardEvent('spyde:seg_preview', windowId, (d) => {
    const areas = Array.isArray(d.areas) ? (d.areas as number[]) : []
    const eff = Number(d.min_size ?? vals.current.minSize)
    const floored = Boolean(d.min_size_floored)
    setPreview(prev => ({
      frame: Number(d.frame ?? 0), count: Number(d.count ?? 0), areas,
      median: Number(d.median_area ?? 0), units: String(d.units ?? 'px'),
      minSize: eff, floored, elapsedMs: Number(d.elapsed_ms ?? 0),
      preview_box: (Array.isArray(d.preview_box) && d.preview_box.length === 4
        ? (d.preview_box.map(Number) as [number, number, number, number])
        : null),
      coverage: Number(d.coverage ?? 0),
      thresholdFailed: Boolean(d.threshold_failed),
      faceUnits: String(d.face_units ?? 'nm'),
      seq: (prev?.seq ?? 0) + 1,
    }))
    // Never leave a number on screen that is not the one that ran. Snapping is
    // loop-free: the backend coerces the snapped value to itself, so the next
    // tune round-trips unchanged.
    if (Number.isFinite(eff) && eff !== vals.current.minSize) {
      setMinSize(eff)
      vals.current = { ...vals.current, minSize: eff }
    }
    // The COUNT now has its own line above the button, so the footer carries
    // only what that line does not: which frame, and how long it took.
    setStatus(`Frame ${d.frame} · ${d.elapsed_ms} ms`)
  })

  useWizardEvent('spyde:seg_trained', windowId, (d) => {
    const r = (d.report ?? {}) as Record<string, unknown>
    const acc = typeof r.train_accuracy === 'number' ? r.train_accuracy.toFixed(3) : '—'
    const dev = typeof r.device === 'string' ? ` · ${r.device}` : ''
    // WHICH SPLIT ROUTE the training just selected, on the persistent line for
    // the reason spelled out where `trainReport` is declared: the backend also
    // says this in a status, and that status is overwritten by the re-preview
    // milliseconds later, so as a status it is a report nobody reads. It earns
    // the room because it is the difference between a 0.33 s and a 1.78 s split
    // at 4096², the user is the one who decides it by painting, and nothing else
    // on screen distinguishes the two.
    const route = r.has_boundary ? ' · seam split' : ' · watershed split'
    setTrainReport(
      `Trained on ${r.n_pixels ?? '?'} px, ${r.n_classes ?? '?'} classes` +
      ` · acc ${acc}${dev}${route}`)
    setStatus('Trained — re-previewing the frame…')
  })

  // ── brush strokes ─────────────────────────────────────────────────────────
  // The anyplotlib brush widget (plan B0) is not landed yet, so nothing emits
  // strokes in the app today. The wiring is here so the strip's active class /
  // eraser / size are not dead state: any figure widget event carrying a
  // `points` array from THIS window's figure is forwarded as one seg_paint
  // stroke. Points are IMAGE PIXELS with no scale/offset applied — plan trap 6,
  // and what `seg_paint` documents it expects — so nothing is converted.
  const { state } = useSpyDE()
  const figIds = React.useMemo(
    () => new Set((state.windows.get(windowId)?.figures ?? []).map(f => f.figId)),
    [state.windows, windowId])
  const figIdsRef = React.useRef(figIds)
  figIdsRef.current = figIds
  // The navigator's frame, read through a ref: the listener is registered once
  // per window and must see the CURRENT frame, not the one at registration.
  const frameRef = React.useRef(frame)
  frameRef.current = frame

  React.useEffect(() => {
    const onFigureEvent = (e: Event) => {
      const d = (e as CustomEvent).detail as { figId?: string; event?: Record<string, unknown> }
      if (!d?.figId || !figIdsRef.current.has(d.figId)) return
      const pts = d.event?.points
      if (!Array.isArray(pts) || pts.length === 0) return
      const v = vals.current
      sendRef.current('seg_paint', {
        frame: Number(d.event?.frame ?? frameRef.current),
        points: pts, class_id: v.activeClass, erase: v.eraser, brush: v.brush,
      }, windowId)
    }
    window.addEventListener('spyde:figure_event', onFigureEvent)
    return () => window.removeEventListener('spyde:figure_event', onFigureEvent)
  }, [windowId])

  // ── actions ───────────────────────────────────────────────────────────────

  const onMethod = (m: Method) => {
    setMethod(m)
    vals.current = { ...vals.current, method: m }
    // The backend re-previews on set_method, so this is NOT a tune.
    sendAction('seg_set_method', { method: m }, windowId)
  }

  const labelledPixels = classes.reduce((a, c) => a + (c.pixels || 0), 0)
  const isPrompt = method === 'prompt'
  const isScribble = method === 'scribble'
  const canTrain = labelledPixels > 0 && !isPrompt
  const canRun = !isPrompt && (method !== 'scribble' || trained)

  const train = () => {
    setStatus('Training…')
    sendAction('seg_train', {}, windowId)
  }
  const runAll = () => {
    setStatus('Segmenting the movie…')
    sendAction('seg_run', params(), windowId)
  }

  const areaUnits = preview ? `${preview.units}²` : 'px²'
  // The two face sliders are in nm ONLY when the signal's axis is a real-space
  // length. On a reciprocal-space signal (axis in nm⁻¹) there is no distance to
  // convert to, the backend falls back to pixels, and the label has to follow —
  // a slider reading "50 nm" that acts on 50 px is a claim about the scale bar
  // that is false.
  const faceUnits = preview?.faceUnits ?? 'nm'
  const fmtFace = (v: number) => (v ? `${v} ${faceUnits}` : 'off')
  // On a frame too large to segment whole the backend previews a centred crop,
  // so say "in this region" rather than "on this frame" — the number is true of
  // the box, not of the frame, and claiming otherwise would understate the count
  // by whatever fraction was skipped.
  const box = preview?.preview_box ?? null
  const countText = preview
    ? `${preview.count} particle${preview.count === 1 ? '' : 's'} ` +
      (box ? 'in this region' : 'on this frame')
    : 'no preview yet'

  return (
    <>
      <WizardShell
        testid="segment-wizard" title="Segment Particles" posStyle={caretPos}
        onClose={onClose} closeTestid="seg-close"
        status={status} statusTestid="seg-status"
        // The face stays NARROW; only Advanced widens (plan B7's "wide
        // 2-column caret, using WizardShell's existing width override"). A
        // caret that is 520 px wide while showing three controls would undo
        // the §0.9a calm the collapsed face is for.
        width={advanced ? ADV_WIDTH : 262}
      >
        <TabRow
          tabs={TABS} active={TAB_OF[method]}
          onSelect={(t) => onMethod(METHOD_OF[t])}
          testid={(t) => `seg-tab-${METHOD_OF[t]}`}
        />

        {isPrompt && (
          <div data-testid="seg-prompt-note" style={noteStyle}>
            Prompt segmentation is not installed yet — use Classical or Scribble.
          </div>
        )}

        {/* ── Classical: one knob ──────────────────────────────────────────── */}
        {method === 'classical' && (
          <div style={sensRowStyle}>
            <span style={endLabelStyle}>Fewer</span>
            {/* No numeric readout: "0.50" of what? The endpoints ARE the units.
                The value is still stored and sent unchanged. */}
            <input data-testid="seg-sensitivity" type="range"
              min={0} max={1} step={0.01} value={sensitivity}
              style={{ flex: 1, minWidth: 40 }}
              onChange={(e) => { const n = Number(e.target.value); setSensitivity(n); tune() }} />
            <span style={endLabelStyle}>More</span>
          </div>
        )}

        {/* ── The two face controls, both in NANOMETRES ────────────────────
            A distance in the image is something the eye can judge against the
            scale bar; a 0-1 "confidence" is not, which is why that control read
            as meaningless. Both are engine-independent — they act on the
            measured instances, not on any one method's parameters — so the
            caret's face is the same on Classical, Scribble and Prompt.

            `merge` answers "some of these should be one particle": pieces whose
            gap is under it are relabelled as one. `min` drops anything smaller
            than that across. Confidence is still available under Advanced. */}
        <Field label="Merge closer than">
          <div style={sensRowStyle}>
            <input data-testid="seg-merge-nm" type="range"
              min={0} max={100} step={1} value={mergeNm}
              style={{ flex: 1, minWidth: 40 }}
              onChange={(e) => { const n = Number(e.target.value); setMergeNm(n); tune() }} />
            <span style={endLabelStyle}>{fmtFace(mergeNm)}</span>
          </div>
        </Field>
        <Field label="Ignore smaller than">
          <div style={sensRowStyle}>
            <input data-testid="seg-min-nm" type="range"
              min={0} max={200} step={1} value={minNm}
              style={{ flex: 1, minWidth: 40 }}
              onChange={(e) => { const n = Number(e.target.value); setMinNm(n); tune() }} />
            <span style={endLabelStyle}>{fmtFace(minNm)}</span>
          </div>
        </Field>

        {/* ── Scribble: the class list + Train ─────────────────────────────── */}
        {isScribble && (
          <>
            {!trained && (
              <div data-testid="seg-scribble-note" style={noteStyle}>
                Paint with the strip on the image — include at least one FAINT
                particle — then Train.
              </div>
            )}
            {trainReport && (
              <div data-testid="seg-trained-note" style={okNoteStyle}>{trainReport}</div>
            )}
            <ClassList classes={classes} activeClass={activeClass}
              onSelect={(id) => {
                setEraser(false); setActiveClass(id)
                vals.current = { ...vals.current, activeClass: id, eraser: false }
                tune()
              }} />
            <button data-testid="seg-train" style={canTrain ? secondaryBtnStyle : disabledSecondaryStyle}
              disabled={!canTrain}
              title={canTrain ? 'Fit the scribble classifier on every labelled pixel'
                : 'Paint a few scribbles first'}
              onClick={train}>Train</button>
          </>
        )}

        {/* The train report is the direct answer to the Train button, so on any
            other tab it would be a report with no question. */}
        {!isScribble && trainReport && (
          <div data-testid="seg-trained-note" style={okNoteStyle}>{trainReport}</div>
        )}

        <div data-testid="seg-preview-stats" data-seq={preview?.seq ?? 0}
          data-count={preview?.count ?? -1}
          data-failed={preview?.thresholdFailed ? 'true' : 'false'}
          data-cropped={box ? 'true' : 'false'} style={countStyle}>
          {countText}
        </div>

        {/* ── the threshold-failed verdict ─────────────────────────────────
            A global threshold on a low-contrast frame has no bimodal histogram
            to find, so it lands inside the noise and the split shatters the
            support film into thousands of pieces. Reported as "14028 particles
            in this region" that reads as a real (if bad) answer, and the
            natural response is to reach for the sliders — none of which can fix
            it. Measured on a stand-in for the reported frame: min_size alone
            takes 4873 → 17 instances but coverage only 39% → 7%, and the
            settings that DO yield 8 instances cover 52% of the frame, i.e. the
            8 bodies are the film.

            So the caret names the failure and points at the engine that does
            work on this data (plan §0.9: the learned classifier is the primary
            path, not threshold tuning). It does NOT silently re-tune. */}
        {preview?.thresholdFailed && (
          <div data-testid="seg-threshold-failed" style={warnStyle}>
            The threshold landed inside the noise — {preview.count.toLocaleString()}
            {' '}pieces covering {Math.round(preview.coverage * 100)}% of the
            {box ? ' region' : ' frame'}. That is the film being segmented, not
            particles.{isScribble ? ' Paint a few examples and Train.'
              : ' Try Scribble: paint a few examples and Train.'}
          </div>
        )}
        {box && (
          <div data-testid="seg-preview-cropped" style={hintStyle}
            title={`Previewing a ${box[3]}x${box[2]} px window at full resolution `
              + `from (${box[1]}, ${box[0]}). Segmenting a whole 4k frame costs `
              + `~8 s, which makes every adjustment feel like a hang. The full `
              + `run uses the entire frame.`}>
            preview window {box[3]}x{box[2]} px · full run uses every pixel
          </div>
        )}

        <button data-testid="seg-run" style={canRun ? primaryWideStyle : disabledWideStyle}
          disabled={!canRun}
          title={canRun ? 'Segment every frame into a new particle dataset'
            : (isPrompt ? 'Prompt segmentation is not installed'
              : 'Train the scribble classifier first')}
          onClick={runAll}>Find in all frames</button>

        {/* ── everything else ──────────────────────────────────────────────── */}
        <button data-testid="seg-advanced-toggle" style={discloseStyle}
          aria-expanded={advanced} onClick={toggleAdvanced}>
          {advanced ? '▾ Advanced' : '▸ Advanced'}
        </button>

        {/* ── Advanced: TWO COLUMNS (plan B7) ──────────────────────────────
            Single-column, this ran 907 px tall in an 805 px MDI area, and the
            caret has no scroller of its own (the Threshold menu is absolutely
            positioned, so an `overflow:auto` ancestor clips it) — so the
            histogram and Commit Frame were not merely awkward, they were
            unreachable. Two columns is plan B7's own answer to that, and it
            halves the height without demoting anything further or deleting
            anything, which §0.9a rules out.

            The split is params | feedback: the left column is everything you
            SET, the right is everything the frame TELLS you plus the button
            that keeps it. Balance matters as much as the grouping — the
            classical-only `detection` block is the tallest thing here, and
            with it on the left the two columns come out close to even. */}
        {advanced && (
          <div data-testid="seg-advanced" style={advStyle}>
            <div style={advColsStyle}>
              <div data-testid="seg-advanced-params" style={advColStyle}>
                <div style={colHeadStyle}>params</div>

                {/* ── Confidence ─────────────────────────────────────────
                    The per-instance contrast-to-noise filter. It came OFF the
                    default face when the two nm controls replaced it — a 0-1
                    "confidence" is not something the eye can judge against the
                    scale bar, while a distance is — but it is the only control
                    that cuts over-split support-film texture, which is small
                    AND round and so survives every size and shape filter here.
                    It is demoted, not deleted (plan §0.9a); it was briefly
                    BOTH, which left `min_score` pinned at 0 with no control
                    able to move it.

                    It acts on the measured OUTPUT, so it means the same thing
                    on all three engines and dragging re-filters an existing
                    result instead of re-segmenting. */}
                <div style={S.hint}>confidence</div>
                <div style={sensRowStyle}>
                  <span style={endLabelStyle}>All</span>
                  <input data-testid="seg-min-score" type="range"
                    min={0} max={0.99} step={0.01} value={minScore}
                    style={{ flex: 1, minWidth: 40 }}
                    onChange={(e) => { const n = Number(e.target.value); setMinScore(n); tune() }} />
                  <span style={endLabelStyle}>
                    {minScore ? `${Math.round(minScore * 100)}%` : 'off'}
                  </span>
                </div>

                <div style={secStyle}>size filter</div>
                <Row label="Min size (px)">
                  <NumInput testid="seg-min-size" value={minSize} step="1" width={54}
                    onChange={live(setMinSize)} />
                </Row>
                {/* The floor warning belongs HERE, under the field it explains
                    — on the primary face it was a large orange alarm about a
                    parameter the backend already fixed on the user's behalf. */}
                {preview?.floored && (
                  <div data-testid="seg-min-size-floor" style={warnStyle}>
                    floored to {preview.minSize} px — at 0 the split returns
                    background speckle as particles
                  </div>
                )}
                <Row label="Max size (0=off)">
                  <NumInput testid="seg-max-size" value={maxSize} step="1" width={54}
                    onChange={live(setMaxSize)} />
                </Row>

                <div style={secStyle}>splitting</div>
                <Check testid="seg-watershed" checked={watershed} onChange={live(setWatershed)}
                  label="Split touching" />
                <Row label="Min separation">
                  <NumInput testid="seg-min-separation" value={minSeparation} step="1" width={54}
                    onChange={live(setMinSeparation)} />
                </Row>
                <Cell label="Marker smoothing">
                  <Slider testid="seg-marker-smooth" value={markerSmooth} min={0} max={10} step={0.1}
                    onChange={live(setMarkerSmooth)} fmt={(n) => n.toFixed(1)} />
                </Cell>
                <Check testid="seg-clear-border" checked={clearBorder} onChange={live(setClearBorder)}
                  label="Drop edge particles" />

                {/* CLASSICAL ONLY, and not merely for space: these build the
                    classical MASK. The scribble engine hands `split_instances`
                    a probability map thresholded at 0.5, so it never reads
                    threshold / sensitivity / gaussian / rb_kernel / invert /
                    local_size — see
                    `spyde/particles/classical.py::split_instances`. Rendering
                    them on the Scribble tab would be six knobs that do nothing,
                    which is the overload complaint in miniature. They keep
                    their stored values and are still sent in every payload. */}
                {method === 'classical' && (
                  <>
                    <div style={secStyle}>detection</div>
                    <Cell label="Threshold">
                      <Select testid="seg-threshold" value={threshold} options={THRESHOLD_OPTS}
                        onChange={live(setThreshold)} />
                    </Cell>
                    <Cell label="Pre-blur σ (px)">
                      <Slider testid="seg-gaussian" value={gaussian} min={0} max={10} step={0.1}
                        onChange={live(setGaussian)} fmt={(n) => n.toFixed(1)} />
                    </Cell>
                    <Row label="Rolling ball">
                      <NumInput testid="seg-rb-kernel" value={rbKernel} step="1" width={54}
                        onChange={live(setRbKernel)} />
                    </Row>
                    <Row label="Local window">
                      <NumInput testid="seg-local-size" value={localSize} step="2" width={54}
                        onChange={live(setLocalSize)} />
                    </Row>
                    <Check testid="seg-invert" checked={invert} onChange={live(setInvert)}
                      label="Dark particles" />
                  </>
                )}
              </div>

              <div data-testid="seg-advanced-feedback" style={advColStyle}>
                <div style={colHeadStyle}>feedback</div>

                <div style={S.hint}>size {areaUnits}</div>
                <SizeHistogram areas={preview?.areas ?? []} />
                <div data-testid="seg-size-stats" style={statsStyle}>
                  {preview ? `med ${fmtArea(preview.median)} ${areaUnits}` : '—'}
                </div>
                <div data-testid="seg-counts" style={S.hint}>
                  {labelledFrames.length} frames labelled · {classes.length} classes
                  {labelledPixels > 0 ? ` · ${labelledPixels.toLocaleString()} px` : ''}
                  {` · frame ${frame}`}
                </div>

                {/* `output` describes what the RUN produces, so it belongs with
                    the button that produces one rather than among the detection
                    knobs. It is also what keeps the two columns near even once
                    `detection` renders on the left. */}
                <div style={secStyle}>output</div>
                <Check testid="seg-store-masks" checked={storeMasks} onChange={live(setStoreMasks)}
                  label="Store outlines" />
                <Check testid="seg-track" checked={track} onChange={live(setTrack)}
                  label="Link tracks" />
                <Row label="Link radius">
                  <NumInput testid="seg-max-dist" value={maxDist} step="0.1" width={54}
                    onChange={live(setMaxDist)} />
                </Row>
                <CommitButton wizardKey="seg" windowId={windowId} sendAction={sendAction}
                  label="Commit Frame" />
              </div>
            </div>
          </div>
        )}
      </WizardShell>

      {/* Painting controls go NEXT TO THE PLOT, not in the caret (plan B0).
          Rendered AFTER the shell so FloatingToolbar's placement effect still
          measures the caret (it reads the wrapper's firstElementChild). */}
      {stripPos && classes.length > 0 && method === 'scribble' && (
        <ClassStrip
          classes={classes} activeId={activeClass}
          // Push to the backend, don't just set React state: the widget that
          // does the painting is Python-side.
          onSelect={(id) => {
            setActiveClass(id); setEraser(false)
            vals.current = { ...vals.current, activeClass: id, eraser: false }
            tune()
          }}
          brush={brush}
          onBrush={(b) => { setBrush(b); vals.current = { ...vals.current, brush: b }; tune() }}
          eraser={eraser}
          onEraser={(on) => {
            setEraser(on)
            vals.current = { ...vals.current, eraser: on }
            tune()
          }}
          posStyle={stripPos}
        />
      )}
    </>
  )
}

// ── the class list ───────────────────────────────────────────────────────────

/**
 * Class NAMES + per-class labelled-pixel counts — the authoritative list (the
 * ClassStrip is swatches only). Under-training a class is THE failure mode, so
 * a starved class is dimmed and flagged rather than merely listed.
 */
function ClassList({ classes, activeClass, onSelect }: {
  classes: SegClassInfo[]; activeClass: number; onSelect: (id: number) => void
}) {
  return (
    <div data-testid="seg-class-list" style={classListStyle}>
      {classes.length === 0 && <div style={S.hint}>—</div>}
      {classes.map(c => {
        const low = c.pixels < LOW_PIXELS
        return (
          <button
            key={c.id}
            data-testid={`seg-class-${c.id}`}
            data-active={c.id === activeClass ? 'true' : 'false'}
            data-low={low ? 'true' : 'false'}
            title={low
              ? `${c.name}: only ${c.pixels} labelled px — paint another example`
              : `${c.name}: ${c.pixels.toLocaleString()} labelled px`}
            onClick={() => onSelect(c.id)}
            style={{
              ...classRowStyle,
              ...(c.id === activeClass ? classRowActiveStyle : {}),
              // Under-training is THE failure mode; a starved class is dimmed
              // so it reads as a problem, not as a row.
              opacity: low ? 0.55 : 1,
            }}
          >
            <span style={{ ...classDotStyle, background: c.colour }} />
            <span style={{
              ...classNameStyle,
              // The active row must read as active next to a UA focus ring on
              // some other control — colour alone is too close to the panel
              // background at this size.
              fontWeight: c.id === activeClass ? 700 : 400,
            }}>{c.name}</span>
            <span data-testid={`seg-class-pixels-${c.id}`}
              style={{ ...classPxStyle, color: low ? '#f9e2af' : '#cdd6f4' }}>
              {low && '! '}{c.pixels.toLocaleString()}
            </span>
          </button>
        )
      })}
      {/* No `+ add class` button here on purpose. The backend has no
          `seg_add_class` verb (LabelStore.add_class exists but nothing routes to
          it), and a permanently disabled control is pure noise on a face this
          feature has just spent a redesign emptying out — it advertises
          something that cannot happen. Bring it back WITH the backend verb, not
          before. The four default classes — particle, support film, vacuum and
          boundary — cover every case the engine currently distinguishes.

          `boundary` is the one worth knowing about: painting the JOINS between
          touching particles lets the backend split them by connected components
          and skip the distance transform and watershed entirely (measured 1.78 s
          -> 0.33 s on a 4096 frame). Paint the seam between two bodies, never
          the outline of one — a head taught outlines shrinks everything and
          splits nothing. Leaving it unpainted is safe: the split falls back to
          the watershed. */}
    </div>
  )
}

// ── the size histogram ───────────────────────────────────────────────────────

const N_BINS = 18

/**
 * A tiny inline SVG sparkline of the per-instance area distribution — no
 * charting dependency, and it re-renders on every preview so you SEE the
 * distribution shift as sensitivity is dragged instead of guessing from one
 * count. It lives in Advanced: it answers "why is the count wrong", which is a
 * question you only ask once the count already looks wrong.
 *
 * Binned to the 98th percentile rather than the max: one 50× outlier (two
 * merged particles, or the support film caught as one body) otherwise squashes
 * every real bar into the first bin and the histogram says nothing.
 */
function SizeHistogram({ areas }: { areas: number[] }) {
  const bins = React.useMemo(() => {
    if (!areas.length) return new Array<number>(N_BINS).fill(0)
    const sorted = [...areas].sort((a, b) => a - b)
    const hi = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))] || 1
    const out = new Array<number>(N_BINS).fill(0)
    for (const a of areas) {
      const k = Math.min(N_BINS - 1, Math.max(0, Math.floor((a / hi) * N_BINS)))
      out[k] += 1
    }
    return out
  }, [areas])

  const peak = Math.max(1, ...bins)
  const w = 100 / N_BINS
  return (
    <svg data-testid="seg-histogram"
      data-nonzero={bins.filter(b => b > 0).length}
      viewBox="0 0 100 32" preserveAspectRatio="none"
      style={{ width: '100%', height: 32, display: 'block' }}>
      <rect x={0} y={0} width={100} height={32} fill="#11111b" />
      {bins.map((b, i) => {
        const h = (b / peak) * 30
        return <rect key={i} x={i * w + 0.4} y={31 - h} width={w - 0.8} height={h}
          fill="#89b4fa" />
      })}
    </svg>
  )
}

function fmtArea(v: number): string {
  if (!Number.isFinite(v)) return '—'
  if (v === 0) return '0'
  if (v >= 100) return v.toFixed(0)
  if (v >= 10) return v.toFixed(1)
  return v.toPrecision(2)
}

// Module-scope, NOT inline: a component defined inside the render body is a new
// type every render, so React remounts the sliders on each keystroke and they
// lose their drag ("sliders don't work") — see FindVectorsWizard's Cell.
function Cell({ label, children }: { label: string; children: React.ReactNode }) {
  return <div style={cellStyle}><label style={S.lbl}>{label}</label>{children}</div>
}
function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={S.fieldRow}>
      <label style={S.lbl}>{label}</label>
      {children}
    </div>
  )
}

const cellStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0,
}
const sensRowStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 7, padding: '2px 0',
}
const endLabelStyle: React.CSSProperties = {
  fontSize: 10, color: '#a6adc8', flex: '0 0 auto', whiteSpace: 'nowrap',
}
const countStyle: React.CSSProperties = {
  fontSize: 11, color: '#cdd6f4', fontVariantNumeric: 'tabular-nums',
}
const statsStyle: React.CSSProperties = {
  fontSize: 10, color: '#cdd6f4', fontVariantNumeric: 'tabular-nums',
}
/** The caret's width while Advanced is open — two ~236 px columns plus the
 *  gap, divider and the shell's own padding. */
const ADV_WIDTH = 520

const advStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 5, minWidth: 0,
  // NO overflow here: the Threshold Dropdown's menu is absolutely positioned
  // and any overflow:auto ancestor clips it (PlotControlDock.tsx:730). That
  // constraint is exactly why this block is two COLUMNS rather than a scroller
  // — see the comment at the JSX.
  borderTop: '1px solid #313244', paddingTop: 6,
}
const advColsStyle: React.CSSProperties = {
  display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12,
  alignItems: 'start', minWidth: 0,
}
const advColStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 5,
  // `minWidth: 0` on a grid child is what lets the Threshold trigger and the
  // histogram shrink to the column instead of forcing it wider than 1fr.
  minWidth: 0,
}
const colHeadStyle: React.CSSProperties = {
  fontSize: 9.5, color: '#6c7086', textTransform: 'uppercase',
  letterSpacing: 0.6, paddingBottom: 2, borderBottom: '1px solid #313244',
}
const secStyle: React.CSSProperties = {
  ...S.hint, borderTop: '1px solid #313244', paddingTop: 5, marginTop: 1,
}
const classListStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 2,
  maxHeight: 132, overflowY: 'auto',
}
const classRowStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 5, width: '100%',
  background: 'none', border: '1px solid transparent', borderRadius: 4,
  padding: '2px 4px', cursor: 'pointer', color: '#cdd6f4', textAlign: 'left',
}
const classRowActiveStyle: React.CSSProperties = {
  background: '#313244',
  // The full `border` SHORTHAND, never a `borderColor` longhand on top of the
  // base row's shorthand. React removes a dropped longhand by clearing that one
  // property, which leaves the shorthand's width/style with a reset colour — a
  // row that had been active kept a stale WHITE 1px border and read as selected
  // alongside the real selection (caught in a screenshot, not by a test).
  border: '1px solid #89b4fa',
}
const classDotStyle: React.CSSProperties = {
  width: 9, height: 9, borderRadius: 2, flex: '0 0 auto',
}
const classNameStyle: React.CSSProperties = {
  fontSize: 10, flex: 1, minWidth: 0, overflow: 'hidden',
  textOverflow: 'ellipsis', whiteSpace: 'nowrap',
}
const classPxStyle: React.CSSProperties = {
  fontSize: 10, fontVariantNumeric: 'tabular-nums', flex: '0 0 auto',
}
const hintStyle: React.CSSProperties = {
  fontSize: 9.5, color: '#6c7086', fontStyle: 'italic', marginTop: -2,
}
const primaryWideStyle: React.CSSProperties = {
  ...S.primary, alignSelf: 'stretch', textAlign: 'center',
}
const disabledWideStyle: React.CSSProperties = {
  ...primaryWideStyle, background: '#313244', color: '#6c7086', cursor: 'not-allowed',
}
// Train is a step ON THE WAY to the primary button, not a second primary — two
// equally-blue buttons is exactly the "which one do I press" the redesign is
// removing.
const secondaryBtnStyle: React.CSSProperties = {
  background: 'none', color: '#cdd6f4', border: '1px solid #45475a',
  borderRadius: 5, padding: '4px 10px', fontSize: 11, cursor: 'pointer',
  alignSelf: 'flex-start',
}
const disabledSecondaryStyle: React.CSSProperties = {
  ...secondaryBtnStyle, color: '#6c7086', cursor: 'not-allowed',
}
const discloseStyle: React.CSSProperties = {
  background: 'none', border: 'none', color: '#89b4fa', fontSize: 10,
  cursor: 'pointer', padding: 0, textAlign: 'left', alignSelf: 'flex-start',
}
const noteStyle: React.CSSProperties = {
  fontSize: 10, color: '#f9e2af', background: 'rgba(249,226,175,0.08)',
  border: '1px solid rgba(249,226,175,0.25)', borderRadius: 4, padding: '3px 5px',
}
const warnStyle: React.CSSProperties = { ...noteStyle, color: '#fab387' }
const okNoteStyle: React.CSSProperties = {
  fontSize: 10, color: '#a6e3a1', background: 'rgba(166,227,161,0.08)',
  border: '1px solid rgba(166,227,161,0.25)', borderRadius: 4, padding: '3px 5px',
}
