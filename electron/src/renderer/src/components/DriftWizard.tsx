/**
 * DriftWizard.tsx — the Drift Correction caret (`drift_` staged actions,
 * backend: spyde/actions/drift_action.py; plan §A8).
 *
 * A NARROW 240 px caret, deliberately: **the verification surface is a separate
 * window, not this one.** `drift_open` opens a bare-figure "Drift Check" window
 * holding the raw and corrected sum images side by side — an aligned stack sums
 * sharp, a misaligned one blurs, and judging that is the whole point of the
 * check. A 240 px caret cannot show a sum image at a size where sharpness is
 * judgeable, so it holds only the model tabs, the solver parameters, progress
 * and Apply.
 *
 * The inline trace here is therefore a SUMMARY, not the verification: dy/dx vs
 * frame at caret scale, enough to see "the stage crept 30 px to the right" or
 * "frame 12 is an outlier" without moving your eyes off the controls.
 *
 * **Where the trace comes from (this differs from the plan text).** Plan A8 says
 * the trace fills incrementally as the solve progresses, and the caret spec said
 * to build it from streaming `drift_preview` messages. The backend does not (and
 * currently cannot) do that, and says so in its own docstring:
 * `solve_translation` returns its shifts only when it finishes — `progress(done,
 * total)` carries no partial trace and the array is local to the solver. So:
 *   • `drift_progress` drives the PROGRESS BAR during the solve;
 *   • `drift_result` delivers the whole `shifts` array at the end and is what
 *     draws the trace;
 *   • `drift_preview` is the `drift_tune` reply and carries ONE pair's dy/dx
 *     (the first-pair re-solve), shown as the readout above the trace.
 * Samples from `drift_preview` are still appended to the trace, so the day
 * `spyde/drift/translation.py` grows an `on_shift(i, dy, dx)` callback the caret
 * fills in progressively with no change here.
 *
 * Only `rigid` has a solver. `rigid+affine` and `non-rigid` are declared by the
 * backend so the tabs can render, and both are shown LOCKED with the backend's
 * own reason rather than silently falling back — a rigid solve under a caret
 * claiming "rigid+affine" puts a wrong `kind` into the model's provenance,
 * which is worse than the missing feature.
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
  method: Method
  reference: Reference
  upsample: number
  maxShift: number
  apodize: boolean
  normalize: boolean
  rejectOutliers: boolean
  order: number
}
const DEFAULTS: DriftSaved = {
  method: 'rigid', reference: 'running', upsample: 8, maxShift: 32,
  apodize: true, normalize: true, rejectOutliers: true, order: 1,
}
const _driftStore = new Map<number, DriftSaved>()

export function DriftWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const saved = _driftStore.get(windowId) ?? DEFAULTS
  const [method, setMethod] = React.useState<Method>(saved.method)
  const [reference, setReference] = React.useState<Reference>(saved.reference)
  const [upsample, setUpsample] = React.useState(saved.upsample)
  const [maxShift, setMaxShift] = React.useState(saved.maxShift)
  const [apodize, setApodize] = React.useState(saved.apodize)
  const [normalize, setNormalize] = React.useState(saved.normalize)
  const [rejectOutliers, setRejectOutliers] = React.useState(saved.rejectOutliers)
  const [order, setOrder] = React.useState(saved.order)

  const [nFrames, setNFrames] = React.useState(0)
  const [solved, setSolved] = React.useState(false)
  const [progress, setProgress] = React.useState<{ done: number; total: number } | null>(null)
  const [shifts, setShifts] = React.useState<[number, number][]>([])
  const [pair, setPair] = React.useState<{ dy: number; dx: number } | null>(null)
  const [status, setStatus] = React.useState('Tune the solver, then Solve.')

  const vals = React.useRef<DriftSaved>(saved)
  vals.current = { method, reference, upsample, maxShift, apodize, normalize, rejectOutliers, order }
  React.useEffect(() => { _driftStore.set(windowId, vals.current) })

  /** The backend's parameter names (`drift_action.DEFAULTS` keys). */
  const params = (): Record<string, unknown> => {
    const v = vals.current
    return {
      method: v.method, reference: v.reference, upsample: v.upsample,
      max_shift: v.maxShift, apodize: v.apodize, normalize: v.normalize,
      reject_outliers: v.rejectOutliers, order: v.order,
    }
  }

  // Mount → drift_open (opens the Drift Check window with the RAW sum; nothing
  // solves — plan A8 is explicit that drift correction never runs on load).
  // Unmount → drift_close tears the check window down. StrictMode-safe.
  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'drift_open', openPayload: params, closeAction: 'drift_close',
  })

  // Debounced tune → re-solves the FIRST PAIR only (two FFTs), which answers
  // the only question a tune can answer cheaply: are max_shift and upsample in
  // the right range for this movie.
  const sendTune = useDebouncedAction(sendAction, 'drift_tune', windowId)
  const tune = () => sendTune(params)
  const live = <T,>(set: (v: T) => void) => (v: T) => { set(v); tune() }

  useWizardEvent('spyde:drift_state', windowId, (d) => {
    if (typeof d.n_frames === 'number') setNFrames(d.n_frames)
    if (typeof d.solved === 'boolean') setSolved(d.solved)
    // The backend refuses an unimplemented model and stays on rigid, so the
    // tab follows what it actually selected — never what was clicked.
    const m = String(d.method ?? '') as Method
    if (m in TAB_OF) setMethod(m)
  })

  useWizardEvent('spyde:drift_preview', windowId, (d) => {
    const dy = Number(d.dy), dx = Number(d.dx)
    if (!Number.isFinite(dy) || !Number.isFinite(dx)) return
    setPair({ dy, dx })
    // Only meaningful as a trace if the solver ever streams; see the header.
    // Until then the first-pair sample is the whole "trace" before a solve.
    setShifts(s => (s.length > 1 ? s : [[0, 0], [dy, dx]]))
    setStatus(`First pair: dy ${dy.toFixed(2)} · dx ${dx.toFixed(2)} px`)
  })

  useWizardEvent('spyde:drift_progress', windowId, (d) => {
    const done = Number(d.done ?? 0), total = Number(d.total ?? 0)
    setProgress(total > 0 && done < total ? { done, total } : null)
  })

  useWizardEvent('spyde:drift_result', windowId, (d) => {
    const raw = Array.isArray(d.shifts) ? (d.shifts as unknown[]) : []
    setShifts(raw.map(r => {
      const p = r as [number, number]
      return [Number(p?.[0]), Number(p?.[1])] as [number, number]
    }))
    setProgress(null)
    setSolved(true)
    const max = Number(d.max_abs_shift ?? 0)
    const rejected = Number(d.rejected ?? 0)
    setStatus(d.cancelled
      ? `Cancelled — partial model, max shift ${max.toFixed(2)} px`
      : `Solved: max shift ${max.toFixed(2)} px`
        + (rejected ? ` · ${rejected} frames rejected` : ''))
  })

  const onMethod = (t: TabLabel) => {
    const m = METHOD_OF[t]
    setMethod(m)
    vals.current = { ...vals.current, method: m }
    sendAction('drift_set_method', { method: m }, windowId)
  }

  const solve = () => {
    setShifts([])
    setStatus(`Solving drift over ${nFrames || '…'} frames`)
    sendAction('drift_run', params(), windowId)
  }

  const locked = UNAVAILABLE[method]
  const pct = progress ? Math.round((progress.done / progress.total) * 100) : 0

  return (
    <WizardShell
      testid="drift-wizard" title="Drift Correction" posStyle={caretPos}
      onClose={onClose} closeTestid="drift-close"
      status={status} statusTestid="drift-status"
    >
      <TabRow
        tabs={TABS} active={TAB_OF[method]} onSelect={onMethod}
        locked={(t) => Boolean(UNAVAILABLE[METHOD_OF[t]])}
        testid={(t) => `drift-tab-${METHOD_OF[t]}`}
      />
      {/* Both stubs are locked, so this names them rather than waiting for a
          click that cannot happen. Text is the backend's own wording. One block,
          not two stacked paragraphs — the caret's height decides whether it can
          sit BELOW the window or gets pushed to the side, onto the Drift Check
          window it just opened. */}
      <div data-testid="drift-unavailable" style={S.hint}>
        {locked ?? 'Rigid+Affine and Non-rigid are not implemented in spyde.drift yet.'}
        {' Check the result in the Drift Check window — an aligned stack sums sharp.'}
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
      <Check testid="drift-apodize" checked={apodize} onChange={live(setApodize)}
        label="Edge taper" />
      <Check testid="drift-normalize" checked={normalize} onChange={live(setNormalize)}
        label="Phase correlation" />
      <Check testid="drift-reject" checked={rejectOutliers} onChange={live(setRejectOutliers)}
        label="Reject bad frames" />

      {pair && (
        <div data-testid="drift-preview-readout" style={readoutStyle}>
          first pair · dy {pair.dy.toFixed(2)} · dx {pair.dx.toFixed(2)} px
        </div>
      )}

      <ShiftTrace shifts={shifts} />

      {progress && (
        <div data-testid="drift-progress" data-percent={pct} style={progressOuter}>
          <div style={{ ...progressInner, width: `${pct}%` }} />
          <span style={progressLabel}>{progress.done}/{progress.total}</span>
        </div>
      )}

      <div style={btnRowStyle}>
        <button data-testid="drift-solve" style={S.primary} onClick={solve}>Solve</button>
        {/* Apply adds the LAZY corrected node (map_blocks over the source's own
            chunking) — nothing is copied, so this is cheap even on a multi-GB
            movie. Disabled until there is a model to apply. */}
        <CommitButton wizardKey="drift" windowId={windowId} sendAction={sendAction}
          label={solved ? 'Apply' : 'Apply (solve first)'} />
        <button data-testid="drift-cancel" style={cancelStyle} onClick={onClose}>Cancel</button>
      </div>
      <div data-testid="drift-frames" style={S.hint}>
        {nFrames ? `${nFrames} frames` : 'reading the movie…'}
        {solved ? ' · solved' : ''}
      </div>
    </WizardShell>
  )
}

// ── the inline shift-vs-time trace ───────────────────────────────────────────

/**
 * dy(t) and dx(t) over the movie as one small SVG — no charting dependency.
 * Both series share ONE symmetric y-scale so their relative magnitude is
 * readable (drift is usually anisotropic and two auto-scaled axes would hide
 * that); a zero line is drawn because "did it drift at all" is the first
 * question. NaN rows (frames a cancelled solve never reached) break the
 * polyline rather than being drawn as zeros, which would read as "no drift".
 */
function ShiftTrace({ shifts }: { shifts: [number, number][] }) {
  const { dyPath, dxPath, span } = React.useMemo(() => {
    const n = shifts.length
    if (n < 2) return { dyPath: '', dxPath: '', span: 0 }
    let lim = 0
    for (const [dy, dx] of shifts) {
      if (Number.isFinite(dy)) lim = Math.max(lim, Math.abs(dy))
      if (Number.isFinite(dx)) lim = Math.max(lim, Math.abs(dx))
    }
    lim = lim || 1
    const build = (col: 0 | 1): string => {
      const parts: string[] = []
      let pen = false
      shifts.forEach((s, i) => {
        const v = s[col]
        if (!Number.isFinite(v)) { pen = false; return }
        const x = (i / (n - 1)) * 100
        const y = 20 - (v / lim) * 18
        parts.push(`${pen ? 'L' : 'M'}${x.toFixed(2)} ${y.toFixed(2)}`)
        pen = true
      })
      return parts.join(' ')
    }
    return { dyPath: build(0), dxPath: build(1), span: lim }
  }, [shifts])

  return (
    <div>
      {/* Legend inlined into the label rather than a row of its own — the caret
          is height-constrained (see the tab-row comment). */}
      <div style={S.hint}>
        shift vs frame · <span style={{ color: '#89b4fa' }}>dy</span>
        {' '}<span style={{ color: '#f38ba8' }}>dx</span>
        {span ? ` · ±${span.toFixed(1)} px` : ''}
      </div>
      <svg data-testid="drift-trace" data-points={shifts.length}
        viewBox="0 0 100 40" preserveAspectRatio="none"
        style={{ width: '100%', height: 40, display: 'block' }}>
        <rect x={0} y={0} width={100} height={40} fill="#11111b" />
        <line x1={0} y1={20} x2={100} y2={20} stroke="#313244" strokeWidth={0.5} />
        <path d={dyPath} fill="none" stroke="#89b4fa" strokeWidth={1}
          vectorEffect="non-scaling-stroke" />
        <path d={dxPath} fill="none" stroke="#f38ba8" strokeWidth={1}
          vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  )
}

const readoutStyle: React.CSSProperties = {
  fontSize: 10, color: '#cdd6f4', fontVariantNumeric: 'tabular-nums',
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
  display: 'flex', gap: 6, flexWrap: 'wrap', borderTop: '1px solid #313244',
  paddingTop: 6,
}
const cancelStyle: React.CSSProperties = {
  background: '#313244', color: '#cdd6f4', border: '1px solid #45475a',
  borderRadius: 5, padding: '6px 10px', fontSize: 12, cursor: 'pointer',
}
