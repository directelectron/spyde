/**
 * Motion.tsx — drift correction across a movie's frames.
 *
 * The old app called this S.T.A.C.K.; MOTION.md is the feature inventory and
 * records what was ported. The layout follows the original's three columns —
 * image, power spectrum, then the drift trajectory and per-frame table —
 * because the FFT beside the image is how you SEE that alignment worked, and
 * the trajectory is how you tell drift from a failed fit.
 *
 * The scrubber is the centre of the mode. Dragging it shows a raw frame;
 * the view chips switch to the sums once an alignment has run.
 */
import React from 'react'

import { FigureFrame, useFigureBridge } from '@de/shell-renderer'
import type { FigureMessage } from '@de/shell-renderer'

import { C, FONT_MONO, stateOf } from '../theme'
import { Btn, Field, FieldPair, Pill, Section, SegBtn, Segmented, Select } from '../ui'

export interface MotionState {
  loaded: boolean; busy: boolean
  filename: string | null
  n_frames: number; width: number; height: number
  frame: number; view: string
  gain: string | null
  orientation: number; orientations: string[]
  has_result: boolean; has_local: boolean
  image_window: number; fft_window: number
}

export interface MotionShifts {
  shifts_x_raw: number[]; shifts_y_raw: number[]
  shifts_x_smooth: number[]; shifts_y_smooth: number[]
  n_frames: number; throw: number
}

export const MOTION_INITIAL: MotionState = {
  loaded: false, busy: false, filename: null,
  n_frames: 0, width: 0, height: 0, frame: 0, view: 'raw',
  gain: null, orientation: 0, orientations: [],
  has_result: false, has_local: false, image_window: -1, fft_window: -1,
}

const BINS = ['1', '2', '4', '8']
const PATCHES = ['256', '512', '1024']
const REFERENCES = [
  { key: 'central', label: 'Central frame' },
  { key: 'first', label: 'First frame' },
  { key: 'average', label: 'Average' },
]

export function MotionMode({ state, shifts, figures, bridge, act }: {
  state: MotionState
  shifts: MotionShifts | null
  figures: Map<number, FigureMessage>
  bridge: ReturnType<typeof useFigureBridge>
  act: (action: string, payload?: Record<string, unknown>) => void
}) {
  const [bin, setBin] = React.useState('2')
  const [reference, setReference] = React.useState('central')
  const [throwN, setThrowN] = React.useState(0)
  const [local, setLocal] = React.useState(false)
  const [patch, setPatch] = React.useState('512')
  const [showFft, setShowFft] = React.useState(true)

  const openStack = async () => {
    const path = await window.groundcrew?.openFile?.(
      [{ name: 'Movie stack', extensions: ['mrc', 'mrcs', 'tif', 'tiff'] }])
    if (path) act('motion_open_stack', { path })
  }
  const openGain = async () => {
    const path = await window.groundcrew?.openFile?.(
      [{ name: 'Gain reference', extensions: ['mrc', 'mrcs', 'tif', 'tiff'] }])
    if (path) act('motion_open_gain', { path })
  }
  const save = async () => {
    const path = await window.groundcrew?.saveFile?.(
      [{ name: 'MRC', extensions: ['mrc'] }, { name: 'TIFF', extensions: ['tif'] }],
      'aligned_sum.mrc')
    if (path) act('motion_save', { path })
  }

  const imageFig = figures.get(state.image_window)
  const fftFig = figures.get(state.fft_window)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Toolbar
        state={state} bin={bin} setBin={setBin}
        reference={reference} setReference={setReference}
        throwN={throwN} setThrowN={setThrowN}
        local={local} setLocal={setLocal} patch={patch} setPatch={setPatch}
        showFft={showFft} setShowFft={setShowFft}
        onOpenStack={openStack} onOpenGain={openGain} onSave={save}
        act={act}
      />

      {!state.loaded ? (
        <Empty busy={state.busy} onOpen={openStack} />
      ) : (
        <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div style={{ display: 'flex', flex: 1, minHeight: 0, gap: 8, padding: 8 }}>
              <Pane fig={imageFig} bridge={bridge} label="Image" />
              {showFft && <Pane fig={fftFig} bridge={bridge} label="FFT" />}
            </div>
            <Scrubber state={state} act={act} />
          </div>

          <aside style={{
            width: 268, flex: '0 0 268px', padding: '14px 12px', overflowY: 'auto',
            background: C.panel, borderLeft: `1px solid ${C.border}`,
          }}>
            <Section title="Drift">
              <DriftPlot shifts={shifts} />
            </Section>
            <Section title="Per-frame shifts">
              <ShiftTable shifts={shifts} />
            </Section>
          </aside>
        </div>
      )}
    </div>
  )
}

function Toolbar(props: any) {
  const { state, act } = props
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
      padding: '9px 12px', borderBottom: `1px solid ${C.border}`,
    }}>
      <Btn onClick={props.onOpenStack} disabled={state.busy} testId="motion-open">
        Open stack
      </Btn>
      <Btn onClick={props.onOpenGain} disabled={state.busy} testId="motion-open-gain">
        Open gain
      </Btn>
      <span style={{ fontSize: 11.5, color: C.textMuted, maxWidth: 150,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {state.gain ?? 'no gain'}
      </span>
      {state.gain && (
        <>
          <select
            value={state.orientation} data-testid="motion-orientation"
            onChange={(e) => act('motion_set_orientation',
              { index: Number(e.target.value) })}
            style={{
              padding: '4px 6px', background: C.ctl,
              border: `1px solid ${C.ctlLine}`, borderRadius: 5,
              color: C.text, fontSize: 11.5, outline: 'none',
            }}>
            {state.orientations.map((o: string, i: number) =>
              <option key={o} value={i}>{o}</option>)}
          </select>
          <Btn onClick={() => act('motion_validate_gain')} disabled={state.busy}
            testId="motion-validate"
            title="Score all eight orientations and adopt the best">Validate</Btn>
        </>
      )}

      <span style={{ width: 1, height: 20, background: C.border, margin: '0 2px' }} />

      <Labelled label="Bin">
        <Chips options={BINS} value={props.bin} onChange={props.setBin}
          disabled={state.busy} testId="motion-bin" />
      </Labelled>
      <Labelled label="Ref">
        <select value={props.reference} data-testid="motion-ref"
          onChange={(e) => props.setReference(e.target.value)}
          disabled={state.busy}
          style={{
            padding: '4px 6px', background: C.ctl, border: `1px solid ${C.ctlLine}`,
            borderRadius: 5, color: C.text, fontSize: 11.5, outline: 'none',
          }}>
          {REFERENCES.map((r) => <option key={r.key} value={r.key}>{r.label}</option>)}
        </select>
      </Labelled>
      <Labelled label="Throw">
        {/* Early frames carry the beam-induced initial burst and drag the whole
            trajectory. MotionCor2's -Throw. */}
        <input type="number" min={0} max={50} value={props.throwN}
          data-testid="motion-throw" disabled={state.busy}
          onChange={(e) => props.setThrowN(Math.max(0, Number(e.target.value) || 0))}
          title="Discard this many leading frames before aligning"
          style={{
            width: 52, padding: '4px 6px', background: C.well,
            border: `1px solid ${C.ctlLine}`, borderRadius: 5, color: C.text,
            font: `12px ${FONT_MONO}`, outline: 'none',
          }} />
      </Labelled>

      <SegBtn on={props.local} disabled={state.busy} testId="motion-local"
        onClick={() => props.setLocal(!props.local)}
        title="Also fit a per-patch motion field after whole-frame alignment">
        Local
      </SegBtn>
      {props.local && (
        <Labelled label="Patch">
          <Chips options={PATCHES} value={props.patch} onChange={props.setPatch}
            disabled={state.busy} testId="motion-patch" />
        </Labelled>
      )}

      <span style={{ flex: 1 }} />

      {state.busy && <Pill state="warn">WORKING</Pill>}
      <Btn tone="go" disabled={!state.loaded || state.busy} testId="motion-align"
        onClick={() => act('motion_align', {
          bin_factor: Number(props.bin), reference: props.reference,
          throw: props.throwN, local: props.local,
          patch_size: Number(props.patch),
        })}>▶ Align</Btn>
      <Btn disabled={!state.busy} testId="motion-stop"
        onClick={() => act('motion_stop')}>■ Stop</Btn>
      <Btn disabled={!state.has_result || state.busy} testId="motion-save"
        onClick={props.onSave}>Save</Btn>
      <SegBtn on={props.showFft} onClick={() => props.setShowFft(!props.showFft)}
        testId="motion-fft-toggle" title="Show or hide the power spectrum">FFT</SegBtn>
    </div>
  )
}

function Labelled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
      <span style={{ fontSize: 11, color: C.textMuted }}>{label}</span>
      {children}
    </span>
  )
}

function Chips({ options, value, onChange, disabled, testId }: {
  options: string[]; value: string; onChange: (v: string) => void
  disabled?: boolean; testId?: string
}) {
  return (
    <span style={{ display: 'inline-flex', gap: 3 }} data-testid={testId}>
      {options.map((o) => (
        <button key={o} onClick={() => onChange(o)} disabled={disabled}
          data-on={o === value ? 'true' : 'false'}
          style={{
            padding: '3px 8px', borderRadius: 999, fontSize: 11,
            background: o === value ? C.accentSunken : C.ctl,
            border: `1px solid ${o === value ? C.accent : C.ctlLine}`,
            color: o === value ? C.accent : C.textDim,
            cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.5 : 1,
          }}>{o}</button>
      ))}
    </span>
  )
}

function Empty({ busy, onOpen }: { busy: boolean; onOpen: () => void }) {
  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', gap: 12, color: C.textMuted,
    }}>
      <div style={{ fontSize: 14, color: C.textDim }}>
        {busy ? 'Loading…' : 'No movie loaded'}
      </div>
      <p style={{ margin: 0, maxWidth: 420, textAlign: 'center', fontSize: 12.5,
        lineHeight: 1.6 }}>
        Open an MRC or TIFF movie stack to estimate and remove specimen drift
        across its frames.
      </p>
      <Btn onClick={onOpen} disabled={busy} testId="motion-open-empty">Open stack…</Btn>
    </div>
  )
}

function Pane({ fig, bridge, label }: {
  fig: FigureMessage | undefined
  bridge: ReturnType<typeof useFigureBridge>; label: string
}) {
  if (!fig) {
    return <div style={{
      flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
      border: `1px solid ${C.border}`, borderRadius: 7, color: C.textMuted,
      fontSize: 12,
    }}>{label}</div>
  }
  return (
    <div style={{ flex: 1, minWidth: 0, border: `1px solid ${C.border}`,
      borderRadius: 7, overflow: 'hidden' }}>
      <FigureFrame bridge={bridge} figId={fig.fig_id} html={fig.html}
        title={fig.title ?? label}
        onResize={(w, h) => window.groundcrew?.resizeFigure(fig.fig_id, w, h)}
        style={{ width: '100%', height: '100%', border: 'none' }}
        data-testid={`motion-${label.toLowerCase()}-frame`} />
    </div>
  )
}

/** Frame scrubber and the view selector. */
function Scrubber({ state, act }: {
  state: MotionState; act: (a: string, p?: Record<string, unknown>) => void
}) {
  const views: Array<[string, string, boolean]> = [
    ['raw', 'Raw frame', true],
    ['unaligned', 'Unaligned sum', state.has_result],
    ['aligned', 'Aligned sum', state.has_result],
    ['corrected', 'Local corrected', state.has_local],
  ]
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px',
      background: C.panel, borderTop: `1px solid ${C.border}`,
    }}>
      <Segmented>
        {views.filter(([, , ok]) => ok).map(([key, label]) => (
          <SegBtn key={key} on={state.view === key} testId={`motion-view-${key}`}
            onClick={() => act('motion_set_view', { view: key })}>{label}</SegBtn>
        ))}
      </Segmented>

      <input type="range" min={0} max={Math.max(0, state.n_frames - 1)}
        value={state.frame} data-testid="motion-scrub"
        disabled={state.busy || state.n_frames < 2}
        onChange={(e) => act('motion_set_frame', { frame: Number(e.target.value) })}
        style={{ flex: 1, accentColor: C.accent }} />

      <span data-testid="motion-frame-label" style={{
        font: `11.5px ${FONT_MONO}`, color: C.textDim, minWidth: 92,
        textAlign: 'right', fontVariantNumeric: 'tabular-nums',
      }}>
        Frame {state.n_frames ? state.frame + 1 : 0}/{state.n_frames}
      </span>
    </div>
  )
}

/**
 * X and Y drift against frame number, raw behind smoothed.
 *
 * Both are drawn deliberately: the smoothed curve is what gets applied, and
 * seeing it against the raw estimates is how you tell real drift from a
 * correlation that failed on a few frames.
 */
function DriftPlot({ shifts }: { shifts: MotionShifts | null }) {
  if (!shifts || !shifts.shifts_x_smooth.length) {
    return <div style={{ fontSize: 11.5, color: C.textMuted }}>
      Run an alignment to see the trajectory
    </div>
  }
  const W = 244, H = 110, PAD = 4
  const all = [...shifts.shifts_x_raw, ...shifts.shifts_y_raw,
               ...shifts.shifts_x_smooth, ...shifts.shifts_y_smooth]
  const lo = Math.min(...all), hi = Math.max(...all)
  const span = (hi - lo) || 1
  const n = shifts.shifts_x_smooth.length
  const px = (i: number) => PAD + (i / Math.max(n - 1, 1)) * (W - 2 * PAD)
  const py = (v: number) => H - PAD - ((v - lo) / span) * (H - 2 * PAD)
  const path = (vals: number[]) =>
    vals.map((v, i) => `${i ? 'L' : 'M'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ')

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} height={H} data-testid="motion-drift"
        style={{ width: '100%', background: C.bgSunken, borderRadius: 4,
          border: `1px solid ${C.border}`, display: 'block' }}>
        <line x1={PAD} y1={py(0)} x2={W - PAD} y2={py(0)} stroke={C.border} />
        <path d={path(shifts.shifts_x_raw)} fill="none" stroke="#8ab4ff" opacity={0.35} />
        <path d={path(shifts.shifts_y_raw)} fill="none" stroke="#f38ba8" opacity={0.35} />
        <path d={path(shifts.shifts_x_smooth)} fill="none" stroke="#8ab4ff" strokeWidth={1.6} />
        <path d={path(shifts.shifts_y_smooth)} fill="none" stroke="#f38ba8" strokeWidth={1.6} />
      </svg>
      <div style={{ display: 'flex', gap: 12, marginTop: 4, fontSize: 10.5,
        color: C.textMuted }}>
        <span style={{ color: '#8ab4ff' }}>■ dX</span>
        <span style={{ color: '#f38ba8' }}>■ dY</span>
        <span style={{ flex: 1 }} />
        <span style={{ fontFamily: FONT_MONO }}>
          {lo.toFixed(1)} … {hi.toFixed(1)} px
        </span>
      </div>
    </div>
  )
}

function ShiftTable({ shifts }: { shifts: MotionShifts | null }) {
  if (!shifts || !shifts.shifts_x_smooth.length) {
    return <div style={{ fontSize: 11.5, color: C.textMuted }}>—</div>
  }
  const rows = shifts.shifts_x_smooth.map((dx, i) => {
    const dy = shifts.shifts_y_smooth[i]
    return { frame: i + 1 + (shifts.throw || 0), dx, dy,
             mag: Math.hypot(dx, dy) }
  })
  return (
    <div style={{ maxHeight: 260, overflowY: 'auto' }} data-testid="motion-table">
      <table style={{ width: '100%', borderCollapse: 'collapse',
        font: `11px ${FONT_MONO}`, fontVariantNumeric: 'tabular-nums' }}>
        <thead>
          <tr>{['Frame', 'dX', 'dY', '|shift|'].map((h) => (
            <th key={h} style={{
              textAlign: h === 'Frame' ? 'left' : 'right', padding: '3px 6px',
              borderBottom: `1px solid ${C.border}`, color: C.textMuted,
              fontSize: 10, letterSpacing: '.04em', fontWeight: 600,
              position: 'sticky', top: 0, background: C.panel,
            }}>{h}</th>))}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.frame}>
              <td style={{ padding: '2px 6px', color: C.textMuted }}>{r.frame}</td>
              <td style={{ padding: '2px 6px', textAlign: 'right' }}>{r.dx.toFixed(2)}</td>
              <td style={{ padding: '2px 6px', textAlign: 'right' }}>{r.dy.toFixed(2)}</td>
              <td style={{ padding: '2px 6px', textAlign: 'right',
                color: r.mag > 10 ? stateOf('warn').fg : C.textDim }}>
                {r.mag.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
