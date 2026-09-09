/**
 * SegmentWizard.tsx — the Segment caret (`seg_` staged actions, backend:
 * spyde/actions/segment_action.py).
 *
 * The face carries the task, not the algorithm: pick a class, paint on the
 * image (Shift+drag — the brush itself lives in the anyplotlib figure), Train,
 * look at the mask, Run. The instance-split parameters sit behind Advanced.
 *
 * The strokes never pass through here. anyplotlib's brush widget delivers a
 * finished stroke straight to the backend, which paints it into its label
 * store and answers with `seg_state`; the per-class pixel counts shown beside
 * each class button are that answer, and are how a user notices a class is
 * under-trained.
 */
import React from 'react'
import { WizardShell, Field, NumInput, Check, S } from './WizardShell'
import { useWizardLifecycle, useDebouncedAction, useWizardEvent } from './wizardHooks'
import type { SendAction } from './wizardHooks'
import type { SegClassInfo } from '../kernel/protocol'

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: SendAction
  onClose: () => void
}

/** Mirrors `segment_action.DEFAULTS`. */
interface SegSaved {
  radius: number
  activeClass: number
  erase: boolean
  minSize: number
  splitTouching: boolean
  minSeparation: number
}
const DEFAULTS: SegSaved = {
  radius: 4.0, activeClass: 0, erase: false,
  minSize: 20, splitTouching: true, minSeparation: 3,
}
const _segStore = new Map<number, SegSaved>()

interface Result { regions: number; fields: number; cancelled: boolean }

export function SegmentWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const saved = _segStore.get(windowId) ?? DEFAULTS
  const [radius, setRadius] = React.useState(saved.radius)
  const [activeClass, setActiveClass] = React.useState(saved.activeClass)
  const [erase, setErase] = React.useState(saved.erase)
  const [minSize, setMinSize] = React.useState(saved.minSize)
  const [splitTouching, setSplitTouching] = React.useState(saved.splitTouching)
  const [minSeparation, setMinSeparation] = React.useState(saved.minSeparation)

  const [advanced, setAdvanced] = React.useState(false)
  const [classes, setClasses] = React.useState<SegClassInfo[]>([])
  const [nFields, setNFields] = React.useState(0)
  const [space, setSpace] = React.useState('signal')
  const [trained, setTrained] = React.useState(false)
  const [accuracy, setAccuracy] = React.useState(0)
  const [running, setRunning] = React.useState(false)
  const [progress, setProgress] = React.useState<{ done: number; total: number } | null>(null)
  const [result, setResult] = React.useState<Result | null>(null)
  const [status, setStatus] = React.useState('Shift+drag on the image to paint.')

  const vals = React.useRef<SegSaved>(saved)
  vals.current = { radius, activeClass, erase, minSize, splitTouching, minSeparation }
  React.useEffect(() => { _segStore.set(windowId, vals.current) })

  /** The backend's parameter names (`segment_action.DEFAULTS` keys). */
  const params = (): Record<string, unknown> => {
    const v = vals.current
    return {
      radius: v.radius, active_class: v.activeClass, erase: v.erase,
      min_size: v.minSize, split_touching: v.splitTouching,
      min_separation: v.minSeparation,
    }
  }

  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'seg_open', openPayload: params, closeAction: 'seg_close',
  })

  const sendTune = useDebouncedAction(sendAction, 'seg_tune', windowId)
  const tune = () => sendTune(params)
  const live = <T,>(set: (v: T) => void) => (v: T) => { set(v); tune() }

  useWizardEvent('spyde:seg_state', windowId, (d) => {
    if (Array.isArray(d.classes)) setClasses(d.classes as SegClassInfo[])
    if (typeof d.n_fields === 'number') setNFields(d.n_fields)
    if (typeof d.space === 'string') setSpace(d.space)
    if (typeof d.trained === 'boolean') setTrained(d.trained)
    if (typeof d.train_accuracy === 'number') setAccuracy(d.train_accuracy)
    if (typeof d.running === 'boolean') {
      setRunning(d.running)
      if (!d.running) setProgress(null)
    }
  })

  useWizardEvent('spyde:progress', windowId, (d) => {
    if (d.label !== 'Segmenting') return
    const done = Number(d.done ?? 0), total = Number(d.total ?? 0)
    setProgress(total > 0 && done < total ? { done, total } : null)
  })

  useWizardEvent('spyde:seg_result', windowId, (d) => {
    setResult({
      regions: Number(d.n_regions ?? 0), fields: Number(d.n_fields ?? 0),
      cancelled: Boolean(d.cancelled),
    })
    setRunning(false)
    setProgress(null)
    setStatus(d.cancelled ? 'Stopped early — partial result opened.' : 'Result opened.')
  })

  const pickClass = (id: number) => {
    setActiveClass(id)
    setErase(false)
    vals.current = { ...vals.current, activeClass: id, erase: false }
    sendAction('seg_tune', params(), windowId)
  }
  const toggleErase = () => {
    const next = !erase
    setErase(next)
    vals.current = { ...vals.current, erase: next }
    sendAction('seg_tune', params(), windowId)
  }

  const painted = classes.reduce((sum, c) => sum + c.pixels, 0)
  const canTrain = classes.filter((c) => c.pixels > 0).length >= 2
  const pct = progress ? Math.round((progress.done / progress.total) * 100) : 0
  const fieldsLabel = nFields > 1 ? `${nFields} frames`
    : space === 'navigation' ? 'the scan positions' : 'this image'

  return (
    <WizardShell
      testid="seg-wizard" title="Segment" posStyle={caretPos}
      onClose={onClose} closeTestid="seg-close"
      status={status} statusTestid="seg-status"
    >
      {/* Class strip: the brush paints in the highlighted class. */}
      <div data-testid="seg-classes" style={stripStyle}>
        {classes.map((c) => (
          <button key={c.id} data-testid={`seg-class-${c.id}`}
            data-pixels={c.pixels} aria-pressed={!erase && activeClass === c.id}
            title={`${c.name}: ${c.pixels} px painted`}
            style={{
              ...classBtn, borderColor: c.colour,
              background: !erase && activeClass === c.id ? c.colour : 'transparent',
              color: !erase && activeClass === c.id ? '#111' : c.colour,
            }}
            onClick={() => pickClass(c.id)}>
            {c.name}
            <span style={pixelCount}>{c.pixels}</span>
          </button>
        ))}
        <button data-testid="seg-erase" aria-pressed={erase}
          style={{ ...classBtn, borderColor: '#cdd6f4',
            background: erase ? '#cdd6f4' : 'transparent',
            color: erase ? '#111' : '#cdd6f4' }}
          onClick={toggleErase}>
          erase
        </button>
      </div>

      <Field label="Brush (px)">
        <NumInput testid="seg-radius" value={radius} step="1" width={56}
          onChange={live(setRadius)} />
      </Field>

      <div style={btnRowStyle}>
        <button data-testid="seg-train" style={S.primary} disabled={!canTrain || running}
          onClick={() => { setStatus('Training…'); sendAction('seg_train', {}, windowId) }}>
          {trained ? 'Retrain' : 'Train'}
        </button>
        <button data-testid="seg-clear" style={ghostStyle} disabled={painted === 0 || running}
          onClick={() => { setResult(null); sendAction('seg_clear', {}, windowId) }}>
          Clear
        </button>
      </div>

      <div data-testid="seg-trained" data-accuracy={trained ? accuracy.toFixed(3) : ''}
        style={S.hint}>
        {trained
          ? `trained · accuracy ${accuracy.toFixed(3)} · new strokes retrain`
          : painted ? 'paint at least two classes, then Train' : 'paint a particle and some background'}
      </div>

      <button data-testid="seg-run" style={S.primary} disabled={!trained && !running}
        onClick={() => {
          if (running) { sendAction('seg_stop', {}, windowId); return }
          setResult(null)
          setRunning(true)
          setStatus(`Segmenting ${fieldsLabel}…`)
          sendAction('seg_run', params(), windowId)
        }}>
        {running ? 'Stop' : `Segment ${fieldsLabel}`}
      </button>

      {progress && (
        <div data-testid="seg-progress" data-percent={pct} style={progressOuter}>
          <div style={{ ...progressInner, width: `${pct}%` }} />
          <span style={progressLabel}>{progress.done}/{progress.total}</span>
        </div>
      )}

      {result && (
        <div data-testid="seg-result" data-regions={result.regions} style={resultStyle}>
          {result.cancelled ? '◐' : '✓'} {result.regions} region{result.regions === 1 ? '' : 's'}
          {result.fields > 1 ? ` in ${result.fields} frames` : ''}
        </div>
      )}

      <button data-testid="seg-advanced-toggle" style={disclosureStyle}
        onClick={() => setAdvanced((v) => !v)}>
        {advanced ? '▾' : '▸'} Advanced
      </button>
      {advanced && (
        <div data-testid="seg-advanced">
          <Field label="Min size (px)">
            <NumInput testid="seg-min-size" value={minSize} step="1" width={56}
              onChange={live(setMinSize)} />
          </Field>
          <Check testid="seg-split-touching" checked={splitTouching}
            onChange={live(setSplitTouching)} label="Split touching (watershed)" />
          <Field label="Min separation (px)">
            <NumInput testid="seg-min-separation" value={minSeparation} step="1" width={56}
              onChange={live(setMinSeparation)} />
          </Field>
          <div style={S.hint}>
            Paint the seam between touching particles as boundary and the split
            follows your strokes instead of the watershed.
          </div>
        </div>
      )}
    </WizardShell>
  )
}

const stripStyle: React.CSSProperties = {
  display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 6,
}
const classBtn: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: 5,
  padding: '2px 8px', borderRadius: 10, border: '1px solid',
  fontSize: 11, cursor: 'pointer', background: 'transparent',
}
const pixelCount: React.CSSProperties = { fontSize: 9, opacity: 0.8 }
const btnRowStyle: React.CSSProperties = { display: 'flex', gap: 6, marginTop: 4 }
const ghostStyle: React.CSSProperties = {
  ...S.primary, background: 'transparent', border: '1px solid #585b70', color: '#cdd6f4',
}
const disclosureStyle: React.CSSProperties = {
  background: 'none', border: 'none', color: '#9399b2', cursor: 'pointer',
  padding: '4px 0', fontSize: 11, textAlign: 'left',
}
const progressOuter: React.CSSProperties = {
  position: 'relative', height: 14, marginTop: 6, borderRadius: 4,
  background: '#313244', overflow: 'hidden',
}
const progressInner: React.CSSProperties = {
  position: 'absolute', left: 0, top: 0, bottom: 0, background: '#89b4fa',
}
const progressLabel: React.CSSProperties = {
  position: 'relative', fontSize: 9, lineHeight: '14px', paddingLeft: 4, color: '#cdd6f4',
}
const resultStyle: React.CSSProperties = {
  marginTop: 6, fontSize: 12, color: '#a6e3a1',
}
