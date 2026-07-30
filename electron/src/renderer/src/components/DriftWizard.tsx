/**
 * DriftWizard.tsx — the Drift Correction caret (`drift_` staged actions,
 * backend: spyde/actions/drift_action.py; plan §A8 + §0.9a).
 *
 * **Two toggles and a button.** The first version of this caret had thirteen
 * controls on its face — three model tabs, four numeric fields, three
 * checkboxes, Solve/Apply/Cancel — and the review was "way too complicated.
 * Too many options. Information overload." Plan §0.9a is the rule that came out
 * of it: the default face carries the TASK, not the algorithm. Reference mode,
 * sub-pixel factor, max shift, interpolation order and the model tabs all still
 * exist, all still reach the backend, and all still land in provenance — they
 * live behind the collapsed `Advanced` disclosure, because drift's parameters
 * have one right answer we already know.
 *
 * **What the caret does NOT show.** The dy/dx curve used to be a 40 px inline
 * SVG here; it is now its own figure window (`Drift dy/dx`), opened by
 * `drift_run` and filled progressively from the solver's `on_shift` stream. A
 * sparkline could show that the stage crept 30 px; only a real plot shows WHICH
 * frame jumped. The before/after sums stay in the `Drift Check` window, whose
 * bottom row is the discovery pair.
 *
 * **Discovery, not configuration.** The backend puts a draggable box on the
 * movie the moment this mounts, aligns ~20 frames sampled across the whole
 * movie on that box alone, and reports how much sharper the sum got. That
 * number (`drift_preview.gain`) is what the readout under the toggles shows —
 * drag the box onto a landmark and watch it rise, drag it onto empty film and
 * watch it fall below 1. `Use ROI for alignment` is then the commitment: the
 * full solve correlates on that same rectangle. It is OFF by default because
 * a guessed box is not automatically better than the whole frame (measured:
 * 1.03 px vs 0.25 px against ground truth on the test movie) — the preview is
 * how you find out whether yours is.
 *
 * Only `rigid` has a solver. `rigid+affine` and `non-rigid` are shown LOCKED
 * inside Advanced with the backend's own reason rather than silently falling
 * back — a rigid solve under a caret claiming "rigid+affine" puts a wrong
 * `kind` into the model's provenance, which is worse than the missing feature.
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Select, Check, S } from './WizardShell'
import { useWizardLifecycle, useDebouncedAction, useWizardEvent, CommitButton } from './wizardHooks'
import type { SendAction } from './wizardHooks'

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: SendAction
  onClose: () => void
}

/** `drift_action.METHODS`. */
type Method = 'rigid' | 'rigid_affine' | 'nonrigid'
type TabLabel = 'Rigid' | 'Rigid+Affine' | 'Non-rigid'
const TABS: readonly TabLabel[] = ['Rigid', 'Rigid+Affine', 'Non-rigid']
const METHOD_OF: Record<TabLabel, Method> = {
  'Rigid': 'rigid', 'Rigid+Affine': 'rigid_affine', 'Non-rigid': 'nonrigid',
}
const TAB_OF: Record<Method, TabLabel> = {
  rigid: 'Rigid', rigid_affine: 'Rigid+Affine', nonrigid: 'Non-rigid',
}
/** Verbatim from `drift_action._UNAVAILABLE` — the reason the backend gives. */
const UNAVAILABLE: Partial<Record<Method, string>> = {
  rigid_affine: 'the affine drift search (plan A4) is not implemented in spyde.drift yet',
  nonrigid: 'non-rigid warping (plan A5) is not implemented in spyde.drift yet',
}

type Reference = 'running' | 'sequential' | 'first'
const REFERENCES: readonly { value: Reference; label: string }[] = [
  { value: 'running', label: 'Running average' },
  { value: 'sequential', label: 'Previous frame' },
  { value: 'first', label: 'First frame' },
]

/** Mirrors `drift_action.DEFAULTS`. */
interface DriftSaved {
  useRoi: boolean
  rejectOutliers: boolean
  method: Method
  reference: Reference
  upsample: number
  maxShift: number
  apodize: boolean
  normalize: boolean
  order: number
  previewFrames: number
}
const DEFAULTS: DriftSaved = {
  useRoi: false, rejectOutliers: true, method: 'rigid', reference: 'running',
  upsample: 8, maxShift: 32, apodize: true, normalize: true, order: 1,
  previewFrames: 20,
}
const _driftStore = new Map<number, DriftSaved>()

interface Preview { roi: number[] | null; frames: number; gain: number }
interface Result { maxShift: number; gain: number; rejected: number; cancelled: boolean }

export function DriftWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const saved = _driftStore.get(windowId) ?? DEFAULTS
  const [useRoi, setUseRoi] = React.useState(saved.useRoi)
  const [rejectOutliers, setRejectOutliers] = React.useState(saved.rejectOutliers)
  const [method, setMethod] = React.useState<Method>(saved.method)
  const [reference, setReference] = React.useState<Reference>(saved.reference)
  const [upsample, setUpsample] = React.useState(saved.upsample)
  const [maxShift, setMaxShift] = React.useState(saved.maxShift)
  const [apodize, setApodize] = React.useState(saved.apodize)
  const [normalize, setNormalize] = React.useState(saved.normalize)
  const [order, setOrder] = React.useState(saved.order)
  const [previewFrames, setPreviewFrames] = React.useState(saved.previewFrames)

  const [advanced, setAdvanced] = React.useState(false)
  const [nFrames, setNFrames] = React.useState(0)
  const [solved, setSolved] = React.useState(false)
  const [running, setRunning] = React.useState(false)
  const [progress, setProgress] = React.useState<{ done: number; total: number } | null>(null)
  const [preview, setPreview] = React.useState<Preview | null>(null)
  const [result, setResult] = React.useState<Result | null>(null)
  const [status, setStatus] = React.useState('Drag the box onto a landmark to test it.')

  const vals = React.useRef<DriftSaved>(saved)
  vals.current = {
    useRoi, rejectOutliers, method, reference, upsample, maxShift, apodize,
    normalize, order, previewFrames,
  }
  React.useEffect(() => { _driftStore.set(windowId, vals.current) })

  /** The backend's parameter names (`drift_action.DEFAULTS` keys). */
  const params = (): Record<string, unknown> => {
    const v = vals.current
    return {
      use_roi: v.useRoi, reject_outliers: v.rejectOutliers, method: v.method,
      reference: v.reference, upsample: v.upsample, max_shift: v.maxShift,
      apodize: v.apodize, normalize: v.normalize, order: v.order,
      preview_frames: v.previewFrames,
    }
  }

  // Mount → drift_open (Drift Check window + the alignment box + the first
  // discovery preview; nothing SOLVES — plan A8 is explicit that drift
  // correction never runs on load). Unmount → drift_close. StrictMode-safe.
  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'drift_open', openPayload: params, closeAction: 'drift_close',
  })

  // A toggle/parameter change re-runs the ~20-frame discovery preview. Only
  // debounced HERE — the backend deliberately doesn't debounce drift_tune
  // again (it debounces the ROI DRAG, whose events arrive at frame rate).
  const sendTune = useDebouncedAction(sendAction, 'drift_tune', windowId)
  const tune = () => sendTune(params)
  const live = <T,>(set: (v: T) => void) => (v: T) => { set(v); tune() }

  useWizardEvent('spyde:drift_state', windowId, (d) => {
    if (typeof d.n_frames === 'number') setNFrames(d.n_frames)
    if (typeof d.solved === 'boolean') {
      setSolved(d.solved)
      if (!d.solved) setResult(null)
    }
    // The backend refuses an unimplemented model and stays on rigid, so the
    // tab follows what it actually selected — never what was clicked.
    const m = String(d.method ?? '') as Method
    if (m in TAB_OF) setMethod(m)
  })

  useWizardEvent('spyde:drift_preview', windowId, (d) => {
    const gain = Number(d.gain)
    setPreview({
      roi: Array.isArray(d.roi) ? (d.roi as number[]).map(Number) : null,
      frames: Number(d.frames ?? 0),
      gain: Number.isFinite(gain) ? gain : NaN,
    })
  })

  useWizardEvent('spyde:drift_progress', windowId, (d) => {
    const done = Number(d.done ?? 0), total = Number(d.total ?? 0)
    const live = total > 0 && done < total
    setProgress(live ? { done, total } : null)
    if (live) setRunning(true)
  })

  useWizardEvent('spyde:drift_result', windowId, (d) => {
    const gain = Number(d.gain)
    setResult({
      maxShift: Number(d.max_abs_shift ?? 0),
      gain: Number.isFinite(gain) ? gain : NaN,
      rejected: Number(d.rejected ?? 0),
      cancelled: Boolean(d.cancelled),
    })
    setProgress(null)
    setRunning(false)
    setSolved(true)
    setStatus(d.cancelled ? 'Stopped — partial model' : 'Solved.')
  })

  const onMethod = (t: TabLabel) => {
    const m = METHOD_OF[t]
    setMethod(m)
    vals.current = { ...vals.current, method: m }
    sendAction('drift_set_method', { method: m }, windowId)
  }

  const solve = () => {
    setResult(null)
    setRunning(true)
    setStatus(`Correcting drift over ${nFrames || '…'} frames`)
    sendAction('drift_run', params(), windowId)
  }

  const discard = () => {
    setRunning(false)
    setProgress(null)
    setResult(null)
    setSolved(false)
    setStatus('Discarded.')
    sendAction('drift_discard', {}, windowId)
  }

  const locked = UNAVAILABLE[method]
  const pct = progress ? Math.round((progress.done / progress.total) * 100) : 0

  return (
    <WizardShell
      testid="drift-wizard" title="Correct Drift" posStyle={caretPos}
      onClose={onClose} closeTestid="drift-close"
      status={status} statusTestid="drift-status"
    >
      {/* The whole default face: two toggles, one number, one button. */}
      <Check testid="drift-use-roi" checked={useRoi} onChange={live(setUseRoi)}
        label="Use ROI for alignment" />
      <Check testid="drift-reject" checked={rejectOutliers}
        onChange={live(setRejectOutliers)} label="Ignore bad frames" />

      <RoiReadout preview={preview} useRoi={useRoi} />

      <button data-testid="drift-solve" style={S.primary}
        onClick={running ? discard : solve}>
        {running ? 'Stop' : 'Correct Drift'}
      </button>

      {progress && (
        <div data-testid="drift-progress" data-percent={pct} style={progressOuter}>
          <div style={{ ...progressInner, width: `${pct}%` }} />
          <span style={progressLabel}>{progress.done}/{progress.total}</span>
        </div>
      )}

      {result && (
        <>
          <div data-testid="drift-result" style={resultStyle}>
            {result.cancelled ? '◐' : '✓'}{' '}
            {Number.isFinite(result.gain) ? `${result.gain.toFixed(1)}x sharper · ` : ''}
            {result.maxShift.toFixed(1)} px drift
            {result.rejected ? ` · ${result.rejected} bad frames` : ''}
          </div>
          <div style={btnRowStyle}>
            {/* Apply adds the LAZY corrected node (map_blocks over the source's
                own chunking) — nothing is copied, so this is cheap even on a
                multi-GB movie. */}
            <CommitButton wizardKey="drift" windowId={windowId}
              sendAction={sendAction} label="Apply" />
            <button data-testid="drift-discard" style={ghostStyle}
              onClick={discard}>Discard</button>
          </div>
        </>
      )}

      <Advanced open={advanced} onToggle={() => setAdvanced(v => !v)}>
        <TabRow
          tabs={TABS} active={TAB_OF[method]} onSelect={onMethod}
          locked={(t) => Boolean(UNAVAILABLE[METHOD_OF[t]])}
          testid={(t) => `drift-tab-${METHOD_OF[t]}`}
        />
        {/* Both stubs are locked, so this names them rather than waiting for a
            click that cannot happen. Text is the backend's own wording. */}
        <div data-testid="drift-unavailable" style={S.hint}>
          {locked ?? 'Rigid+Affine and Non-rigid are not implemented in spyde.drift yet.'}
        </div>
        <Field label="Reference">
          <Select testid="drift-reference" value={reference} options={REFERENCES}
            onChange={live(setReference)} />
        </Field>
        <Field label="Sub-pixel factor">
          <NumInput testid="drift-upsample" value={upsample} step="1" width={56}
            onChange={live(setUpsample)} />
        </Field>
        <Field label="Max shift (px)">
          <NumInput testid="drift-max-shift" value={maxShift} step="1" width={56}
            onChange={live(setMaxShift)} />
        </Field>
        <Field label="Interp. order">
          <NumInput testid="drift-order" value={order} step="1" width={56}
            onChange={live(setOrder)} />
        </Field>
        <Field label="Preview frames">
          <NumInput testid="drift-preview-frames" value={previewFrames} step="1"
            width={56} onChange={live(setPreviewFrames)} />
        </Field>
        <Check testid="drift-apodize" checked={apodize} onChange={live(setApodize)}
          label="Edge taper" />
        <Check testid="drift-normalize" checked={normalize}
          onChange={live(setNormalize)} label="Phase correlation" />
        <div data-testid="drift-frames" style={S.hint}>
          {nFrames ? `${nFrames} frames` : 'reading the movie…'}
          {solved ? ' · solved' : ''}
        </div>
      </Advanced>
    </WizardShell>
  )
}

// ── the discovery readout ────────────────────────────────────────────────────

/**
 * The one number that answers "is this box any good?".
 *
 * Gradient energy of the box's drift-corrected sum over its raw sum (backend
 * `_gradient_energy`, measured on the pixels both sums cover so an aligned
 * frame's NaN border cannot inflate it). Above ~1.5 the box is a usable
 * landmark; at or below 1 aligning on it changes nothing — either the region is
 * featureless or the movie does not drift. Colour-coded rather than left as a
 * bare number, because the whole point is a glanceable verdict while dragging.
 */
function RoiReadout({ preview, useRoi }: { preview: Preview | null; useRoi: boolean }) {
  if (!preview) {
    return <div data-testid="drift-roi-readout" style={S.hint}>testing the box…</div>
  }
  const { gain, roi, frames } = preview
  const good = Number.isFinite(gain) && gain >= 1.5
  const size = roi ? `${roi[3]}x${roi[2]} px` : 'whole frame'
  return (
    <div data-testid="drift-roi-readout" data-gain={Number.isFinite(gain) ? gain.toFixed(3) : ''}
      style={{ ...readoutStyle, color: good ? '#a6e3a1' : '#f9e2af' }}>
      {size} · {Number.isFinite(gain) ? `${gain.toFixed(1)}x sharper` : 'no result'}
      <span style={S.hint}>{' '}over {frames} frames{useRoi ? '' : ' (preview only)'}</span>
    </div>
  )
}

// ── the Advanced disclosure ──────────────────────────────────────────────────

/**
 * Collapsed by default (plan §0.9a). Deliberately local to this file rather
 * than added to WizardShell: the shell is shared with carets another change is
 * editing right now, and a disclosure is six lines.
 */
function Advanced({ open, onToggle, children }: {
  open: boolean; onToggle: () => void; children: React.ReactNode
}) {
  return (
    <div style={advancedWrap}>
      <button data-testid="drift-advanced-toggle" style={discloseStyle} onClick={onToggle}>
        {open ? '▾' : '▸'} Advanced
      </button>
      {open && (
        <div data-testid="drift-advanced" style={S.page}>{children}</div>
      )}
    </div>
  )
}

const readoutStyle: React.CSSProperties = {
  fontSize: 10, fontVariantNumeric: 'tabular-nums',
}
const resultStyle: React.CSSProperties = {
  fontSize: 11, color: '#a6e3a1', fontVariantNumeric: 'tabular-nums',
}
const progressOuter: React.CSSProperties = {
  position: 'relative', height: 12, background: '#11111b',
  border: '1px solid #313244', borderRadius: 3, overflow: 'hidden',
}
const progressInner: React.CSSProperties = {
  position: 'absolute', inset: 0, right: 'auto', background: '#89b4fa',
}
const progressLabel: React.CSSProperties = {
  position: 'absolute', inset: 0, fontSize: 9, lineHeight: '12px',
  textAlign: 'center', color: '#cdd6f4', fontVariantNumeric: 'tabular-nums',
}
const btnRowStyle: React.CSSProperties = {
  display: 'flex', gap: 6, flexWrap: 'wrap',
}
const ghostStyle: React.CSSProperties = {
  background: '#313244', color: '#cdd6f4', border: '1px solid #45475a',
  borderRadius: 5, padding: '6px 10px', fontSize: 12, cursor: 'pointer',
}
const advancedWrap: React.CSSProperties = {
  borderTop: '1px solid #313244', paddingTop: 4,
  display: 'flex', flexDirection: 'column', gap: 6,
}
const discloseStyle: React.CSSProperties = {
  background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer',
  fontSize: 10, padding: 0, textAlign: 'left', alignSelf: 'flex-start',
}
