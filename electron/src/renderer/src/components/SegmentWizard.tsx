/**
 * SegmentWizard.tsx — the Segment Particles caret (`seg_` staged actions,
 * backend: spyde/actions/particles_action.py; plan §B7).
 *
 * A WIDE 2-COLUMN caret (330 px). Left column = the parameters you turn, right
 * column = the feedback that tells you whether turning them helped:
 *
 *   ┌ Segment Particles ─────────────── ✕ ┐
 *   │ [Classical] [Scribble] [Prompt]      │
 *   ├── params ──────────┬── feedback ─────┤
 *   │ Sensitivity ▓▓▓▓░  │ SIZE nm² histo  │
 *   │ Min size      24   │ ▁▃▅█▆▃▁         │
 *   │ Split          on  │ 212 · med 96    │
 *   │ Store masks   off  ├── classes ──────┤
 *   │                    │ ■ particle 1204 │
 *   ├──────────────────────────────────────┤
 *   │ [Train] [Run all]                    │
 *   └──────────────────────────────────────┘
 *
 * Three things here are load-bearing, none of them cosmetic:
 *
 * 1. **Sensitivity is the headline control and `min_size` sits next to it.**
 *    Measured (plan §0.9), not taste: teaching the classifier faint contrast
 *    buys +1 true particle and 25 spurious ones, and `min_size=10` removes 24
 *    of the 25. The classifier is not what buys specificity — the size filter
 *    is. Two coupled knobs in separate tabs would be tuned against each other
 *    blind, so they are adjacent and both above the fold.
 *
 * 2. **The EFFECTIVE `min_size` is what is shown.** The backend floors it and
 *    reports the floored value + a flag in every `seg_preview`; the caret snaps
 *    its field to that value rather than leaving the user's 0 on screen while a
 *    10 ran. Showing a number different from the one that ran is exactly the
 *    failure `SegmentParams` refuses for `local_size`.
 *
 * 3. **Per-class labelled-pixel counts are the point of the class list.**
 *    Under-training a class is *the* failure mode and these counts are how you
 *    notice; a class below `LOW_PIXELS` is dimmed and flagged so "the preview
 *    got worse" pushes you toward painting another example instead of toward
 *    the sensitivity slider, which cannot fix a missing example.
 *
 * The brush swatches / size / eraser are NOT here — they live on the floating
 * ClassStrip next to the plot (plan B0), because while painting you are looking
 * at the image. Class NAMES and counts stay here, which is the authoritative
 * list.
 */
import React from 'react'
import { WizardShell, TabRow, Slider, Select, Check, NumInput, S } from './WizardShell'
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
  method: 'classical', sensitivity: 0.5, threshold: 'otsu', minSize: 20,
  maxSize: 0, watershed: true, minSeparation: 3, markerSmooth: 1.0,
  gaussian: 0.0, rbKernel: 0, invert: false, localSize: 31, clearBorder: false,
  storeMasks: true, track: true, maxDist: 10.0, brush: 3.0,
  activeClass: 0, eraser: false,
}

// Tuned state kept OUTSIDE the component (same pattern as the FV/OM carets) so
// closing the caret to look at the frame doesn't lose the tuning.
const _segStore = new Map<number, SegSaved>()

interface Preview {
  frame: number
  count: number
  areas: number[]
  median: number
  units: string
  minSize: number
  floored: boolean
  elapsedMs: number
}

export function SegmentWizard({ caretPos, windowId, sendAction, onClose, stripPos }: Props) {
  const saved = _segStore.get(windowId) ?? DEFAULTS
  const [method, setMethod] = React.useState<Method>(saved.method)
  const [sensitivity, setSensitivity] = React.useState(saved.sensitivity)
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
  const [more, setMore] = React.useState(false)

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
  const [status, setStatus] = React.useState(
    'Tune on the displayed frame, then Run all.')

  const vals = React.useRef<SegSaved>(saved)
  vals.current = {
    method, sensitivity, threshold, minSize, maxSize, watershed, minSeparation,
    markerSmooth, gaussian, rbKernel, invert, localSize, clearBorder,
    storeMasks, track, maxDist, brush, activeClass, eraser,
  }
  React.useEffect(() => { _segStore.set(windowId, vals.current) })

  /** The backend's parameter names (`particles_action.DEFAULTS` keys). */
  const params = (): Record<string, unknown> => {
    const v = vals.current
    return {
      method: v.method, sensitivity: v.sensitivity, threshold: v.threshold,
      min_size: v.minSize, max_size: v.maxSize, watershed: v.watershed,
      min_separation: v.minSeparation, marker_smooth: v.markerSmooth,
      gaussian: v.gaussian, rb_kernel: v.rbKernel, invert: v.invert,
      local_size: v.localSize, clear_border: v.clearBorder,
      store_masks: v.storeMasks, track: v.track, max_dist: v.maxDist,
      brush: v.brush,
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
    setPreview({
      frame: Number(d.frame ?? 0), count: Number(d.count ?? 0), areas,
      median: Number(d.median_area ?? 0), units: String(d.units ?? 'px'),
      minSize: eff, floored, elapsedMs: Number(d.elapsed_ms ?? 0),
    })
    // Never leave a number on screen that is not the one that ran. Snapping is
    // loop-free: the backend coerces the snapped value to itself, so the next
    // tune round-trips unchanged.
    if (Number.isFinite(eff) && eff !== vals.current.minSize) {
      setMinSize(eff)
      vals.current = { ...vals.current, minSize: eff }
    }
    setStatus(`${d.count} particles on frame ${d.frame} · ${d.elapsed_ms} ms`)
  })

  useWizardEvent('spyde:seg_trained', windowId, (d) => {
    const r = (d.report ?? {}) as Record<string, unknown>
    const acc = typeof r.train_accuracy === 'number' ? r.train_accuracy.toFixed(3) : '—'
    const dev = typeof r.device === 'string' ? ` · ${r.device}` : ''
    setTrainReport(
      `Trained on ${r.n_pixels ?? '?'} px, ${r.n_classes ?? '?'} classes` +
      ` · acc ${acc}${dev}`)
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

  return (
    <>
      <WizardShell
        testid="segment-wizard" title="Segment Particles" posStyle={caretPos}
        onClose={onClose} closeTestid="seg-close"
        status={status} statusTestid="seg-status" width={330}
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
        {method === 'scribble' && !trained && (
          <div data-testid="seg-scribble-note" style={noteStyle}>
            Paint with the strip on the image — include at least one FAINT
            particle — then Train.
          </div>
        )}
        {trainReport && (
          <div data-testid="seg-trained-note" style={okNoteStyle}>{trainReport}</div>
        )}

        <div style={colsStyle}>
          {/* ── left: params ─────────────────────────────────────────────── */}
          <div style={colStyle}>
            <div style={S.hint}>params</div>
            <Cell label="Sensitivity">
              <Slider testid="seg-sensitivity" value={sensitivity} min={0} max={1} step={0.01}
                onChange={live(setSensitivity)} fmt={(n) => n.toFixed(2)} />
            </Cell>
            <Row label="Min size (px)">
              <NumInput testid="seg-min-size" value={minSize} step="1" width={54}
                onChange={live(setMinSize)} />
            </Row>
            {preview?.floored && (
              <div data-testid="seg-min-size-floor" style={warnStyle}>
                floored to {preview.minSize} px — at 0 the split returns
                background speckle as particles
              </div>
            )}
            <Check testid="seg-watershed" checked={watershed} onChange={live(setWatershed)}
              label="Split touching" />
            <Check testid="seg-store-masks" checked={storeMasks} onChange={live(setStoreMasks)}
              label="Store outlines" />
            <Check testid="seg-track" checked={track} onChange={live(setTrack)}
              label="Link tracks" />

            <button data-testid="seg-more" style={moreStyle} onClick={() => setMore(m => !m)}>
              {more ? '▾ fewer' : '▸ more'}
            </button>
            {more && (
              <div style={colStyle}>
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
                <Row label="Max size (0=off)">
                  <NumInput testid="seg-max-size" value={maxSize} step="1" width={54}
                    onChange={live(setMaxSize)} />
                </Row>
                <Row label="Min separation">
                  <NumInput testid="seg-min-separation" value={minSeparation} step="1" width={54}
                    onChange={live(setMinSeparation)} />
                </Row>
                <Cell label="Marker smoothing">
                  <Slider testid="seg-marker-smooth" value={markerSmooth} min={0} max={10} step={0.1}
                    onChange={live(setMarkerSmooth)} fmt={(n) => n.toFixed(1)} />
                </Cell>
                <Row label="Link radius">
                  <NumInput testid="seg-max-dist" value={maxDist} step="0.1" width={54}
                    onChange={live(setMaxDist)} />
                </Row>
                <Check testid="seg-invert" checked={invert} onChange={live(setInvert)}
                  label="Dark particles" />
                <Check testid="seg-clear-border" checked={clearBorder} onChange={live(setClearBorder)}
                  label="Drop edge particles" />
              </div>
            )}
          </div>

          {/* ── right: feedback + classes ────────────────────────────────── */}
          <div style={colStyle}>
            <div style={S.hint}>size {areaUnits}</div>
            <SizeHistogram areas={preview?.areas ?? []} />
            <div data-testid="seg-preview-stats" style={statsStyle}>
              {preview
                ? `${preview.count} found · med ${fmtArea(preview.median)}`
                : 'no preview yet'}
            </div>

            <div style={{ ...S.hint, borderTop: '1px solid #313244', paddingTop: 4 }}>
              classes
            </div>
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
                    onClick={() => { setEraser(false); setActiveClass(c.id) }}
                    style={{
                      ...classRowStyle,
                      ...(c.id === activeClass ? classRowActiveStyle : {}),
                      // Under-training is THE failure mode; a starved class is
                      // dimmed so it reads as a problem, not as a row.
                      opacity: low ? 0.55 : 1,
                    }}
                  >
                    <span style={{ ...classDotStyle, background: c.colour }} />
                    <span style={{
                      ...classNameStyle,
                      // The active row must read as active next to a UA focus
                      // ring on some other control — colour alone is too close
                      // to the panel background at this size.
                      fontWeight: c.id === activeClass ? 700 : 400,
                    }}>{c.name}</span>
                    <span data-testid={`seg-class-pixels-${c.id}`}
                      style={{ ...classPxStyle, color: low ? '#f9e2af' : '#cdd6f4' }}>
                      {low && '! '}{c.pixels.toLocaleString()}
                    </span>
                  </button>
                )
              })}
              {/* The backend has no `seg_add_class` staged verb (LabelStore
                  .add_class exists but nothing routes to it), so this is an
                  affordance with the reason on it rather than a button that
                  silently does nothing. */}
              <button data-testid="seg-add-class" style={addClassStyle} disabled
                title="Adding a class needs a seg_add_class backend verb — not wired yet">
                + add class
              </button>
            </div>
          </div>
        </div>

        <div style={btnRowStyle}>
          <button data-testid="seg-train" style={canTrain ? S.primary : disabledBtnStyle}
            disabled={!canTrain}
            title={canTrain ? 'Fit the scribble classifier on every labelled pixel'
              : 'Paint a few scribbles first (Scribble engine)'}
            onClick={train}>Train</button>
          <button data-testid="seg-run" style={canRun ? S.primary : disabledBtnStyle}
            disabled={!canRun}
            title={canRun ? 'Segment every frame into a new particle dataset'
              : 'Train the scribble classifier first'}
            onClick={runAll}>Run All</button>
          <CommitButton wizardKey="seg" windowId={windowId} sendAction={sendAction}
            label="Commit Frame" />
        </div>
        <div data-testid="seg-counts" style={S.hint}>
          {labelledFrames.length} frames labelled · {classes.length} classes
          {labelledPixels > 0 ? ` · ${labelledPixels.toLocaleString()} px` : ''}
          {` · frame ${frame}`}
        </div>
      </WizardShell>

      {/* Painting controls go NEXT TO THE PLOT, not in the caret (plan B0).
          Rendered AFTER the shell so FloatingToolbar's placement effect still
          measures the caret (it reads the wrapper's firstElementChild). */}
      {stripPos && classes.length > 0 && (
        <ClassStrip
          classes={classes} activeId={activeClass}
          onSelect={setActiveClass}
          brush={brush}
          onBrush={(b) => { setBrush(b); vals.current = { ...vals.current, brush: b }; tune() }}
          eraser={eraser} onEraser={setEraser}
          posStyle={stripPos}
        />
      )}
    </>
  )
}

// ── the size histogram ───────────────────────────────────────────────────────

const N_BINS = 18

/**
 * A tiny inline SVG sparkline of the per-instance area distribution — no
 * charting dependency, and it re-renders on every preview so you SEE the
 * distribution shift as sensitivity is dragged instead of guessing from one
 * count.
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

const colsStyle: React.CSSProperties = {
  display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 10,
  // NO overflow here: the Threshold Dropdown's menu is absolutely positioned
  // and any overflow:auto ancestor clips it (PlotControlDock.tsx:730).
  alignItems: 'start',
}
const colStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 5, minWidth: 0,
}
const cellStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0,
}
const statsStyle: React.CSSProperties = {
  fontSize: 10, color: '#cdd6f4', fontVariantNumeric: 'tabular-nums',
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
const addClassStyle: React.CSSProperties = {
  background: 'none', border: '1px dashed #45475a', borderRadius: 4,
  color: '#6c7086', fontSize: 10, padding: '2px 4px', cursor: 'not-allowed',
  textAlign: 'left',
}
const btnRowStyle: React.CSSProperties = {
  display: 'flex', gap: 6, flexWrap: 'wrap', borderTop: '1px solid #313244',
  paddingTop: 6,
}
const disabledBtnStyle: React.CSSProperties = {
  ...S.primary, background: '#313244', color: '#6c7086', cursor: 'not-allowed',
}
const moreStyle: React.CSSProperties = {
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
