/**
 * PhaseViewerCard.tsx — one phase's viewer: [master pattern] [unit cell]
 * [detector], and under them the microscope set-up it is simulated in.
 *
 * The card knows nothing about where it is mounted. The host passes the
 * phase's figure and set-up (both from the backend's phase viewer) and an
 * `onChange` that forwards one change back; SpyDE pops a stack of these out
 * of the sidebar, and the same card fits any host that can do those two.
 *
 * Numbers commit on Enter or blur rather than per keystroke: each commit
 * moves the detector, and a half-typed value is not a set-up anyone means.
 */
import React from 'react'
import type { PhaseSetupMessage, PhaseSetupField } from '../kernel/protocol'
import { SeamlessFigureFrame } from './ReportFigureCell'

/** The figure's size; matches phase_viewer.FIGURE_SIZE. */
const FIGURE_W = 744
const FIGURE_H = 248

export type PhaseSetupChange =
  | { field: string, value: number }
  | { master: 'bands' | 'full' }
  | { spots: boolean }

interface Props {
  phaseId: string
  setup?: PhaseSetupMessage
  figure?: { figId: string, filePath: string | null, title: string }
  iframeRefs: React.MutableRefObject<Map<string, HTMLIFrameElement>>
  replayState: (figId: string, target?: HTMLIFrameElement) => void
  onChange: (change: PhaseSetupChange) => void
  onClose: () => void
}

export function PhaseViewerCard({ phaseId, setup, figure, iframeRefs, replayState,
  onChange, onClose }: Props) {
  return (
    <section style={S.card} data-testid={`phase-viewer-${phaseId}`}
      aria-label={`Phase viewer: ${setup?.title ?? ''}`}>
      <header style={S.header}>
        <span style={S.title}>{setup?.title ?? 'Phase'}</span>
        {setup?.subtitle && <span style={S.subtitle}>{setup.subtitle}</span>}
        <span style={{ flex: 1 }} />
        <button style={S.close} onClick={onClose} aria-label="Close phase viewer"
          data-testid={`phase-viewer-close-${phaseId}`}>✕</button>
      </header>

      <div style={S.figure}>
        {figure
          ? <SeamlessFigureFrame figId={figure.figId} filePath={figure.filePath}
              title={figure.title} iframeRefs={iframeRefs} replayState={replayState} />
          : <span style={S.muted}>Simulating…</span>}
      </div>

      {setup && <SetupForm setup={setup} phaseId={phaseId} onChange={onChange} />}
    </section>
  )
}

function SetupForm({ setup, phaseId, onChange }: {
  setup: PhaseSetupMessage, phaseId: string, onChange: (change: PhaseSetupChange) => void
}) {
  const full = setup.full
  const fullHint = !full.available ? 'Needs de_ebsd installed'
    : !full.applies ? 'A backscatter master pattern: SEM only'
      : full.state === 'computing' ? 'Simulating — about a second on a GPU'
        : full.state === 'error' ? full.note : ''
  const [geometryOpen, setGeometryOpen] = React.useState(false)
  const guesses = setup.fields.filter(field => field.source === 'default').length
  return (
    <div style={S.form}>
      <div style={S.row}>
        <span style={S.label}>Master pattern</span>
        <Segmented testid={`phase-master-${phaseId}`} value={setup.master}
          options={[['bands', 'Kinematic bands'], ['full', 'Full simulation']]}
          disabled={{ full: !full.available || !full.applies || full.state === 'computing' }}
          onPick={(master) => onChange({ master: master as 'bands' | 'full' })} />
        {fullHint && <span style={S.muted} data-testid={`phase-full-note-${phaseId}`}>{fullHint}</span>}
      </div>

      {/* Spots form in transmission; an EBSD screen sees only the bands. */}
      {setup.spots_apply && (
        <label style={S.row}>
          <input type="checkbox" checked={setup.spots} style={S.check}
            onChange={(e) => onChange({ spots: e.currentTarget.checked })}
            data-testid={`phase-spots-${phaseId}`} />
          <span style={S.label}>Diffraction spots on the detector</span>
        </label>
      )}

      {/* The geometry is only needed to match real data, so it stays folded
          away; its summary line says what it is set to. */}
      <button style={S.disclosure} aria-expanded={geometryOpen}
        onClick={() => setGeometryOpen(open => !open)}
        data-testid={`phase-geometry-${phaseId}`}>
        <span>{geometryOpen ? '▾' : '▸'} {setup.microscope} geometry</span>
        <span style={S.summary}>{summarise(setup)}</span>
        {guesses > 0 && (
          <span style={SOURCE_TONE.default}
            title="Not in the metadata — defaults; check them before matching data">
            {guesses} default{guesses > 1 ? 's' : ''}
          </span>
        )}
      </button>
      {geometryOpen && (
        <>
          <div style={S.fields}>
            {setup.fields.filter(field => !isPc(field)).map(field => (
              <FieldInput key={field.key} field={field} phaseId={phaseId}
                onCommit={(value) => onChange({ field: field.key, value })} />
            ))}
          </div>
          {/* The pattern centre is one point; its three coordinates read as one row. */}
          {setup.fields.some(isPc) && (
            <div style={S.pcRow} data-testid={`phase-pc-${phaseId}`}>
              <span style={S.pcLabel}>Pattern centre</span>
              {setup.fields.filter(isPc).map(field => (
                <FieldInput key={field.key} field={field} phaseId={phaseId}
                  label={field.key.slice(3)}
                  onCommit={(value) => onChange({ field: field.key, value })} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

/** One line saying what the geometry is set to, for the folded header. */
function summarise(setup: PhaseSetupMessage): string {
  const v = Object.fromEntries(setup.fields.map(f => [f.key, f.value]))
  const n = (x: number, digits = 0) => x.toFixed(digits)
  if (setup.microscope === 'SEM') {
    return `${n(v.beam_energy, 1)} kV · tilt ${n(v.sample_tilt, 1)}° · `
      + `PC ${n(v.pc_x, 3)} ${n(v.pc_y, 3)} ${n(v.pc_z, 3)}`
  }
  return `${n(v.beam_energy)} kV · ${n(v.camera_length)} mm`
}

const isPc = (field: PhaseSetupField) => field.key.startsWith('pc_')

function FieldInput({ field, phaseId, onCommit, label }: {
  field: PhaseSetupField, phaseId: string, onCommit: (value: number) => void,
  label?: string
}) {
  const [text, setText] = React.useState(format(field.value))
  React.useEffect(() => { setText(format(field.value)) }, [field.value])
  const commit = () => {
    const value = Number(text)
    if (Number.isFinite(value) && value !== field.value) onCommit(value)
    else setText(format(field.value))
  }
  return (
    <label style={S.field} title={SOURCE_TITLE[field.source]}>
      <span style={label ? S.pcAxis : S.fieldLabel}>{label ?? field.label}</span>
      <span style={S.fieldValue}>
        <input type="number" value={text} step={field.step} style={S.input}
          onChange={(e) => setText(e.currentTarget.value)}
          onBlur={commit}
          onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
          data-testid={`phase-field-${phaseId}-${field.key}`} />
        <span style={S.unit}>{field.unit}</span>
      </span>
      <span style={{ ...S.source, ...SOURCE_TONE[field.source] }}>{field.source}</span>
    </label>
  )
}

function Segmented({ value, options, onPick, disabled = {}, testid }: {
  value: string, options: [string, string][], onPick: (value: string) => void,
  disabled?: Record<string, boolean>, testid: string
}) {
  return (
    <div role="radiogroup" style={S.segmented} data-testid={testid}>
      {options.map(([key, label]) => (
        <button key={key} role="radio" aria-checked={value === key}
          disabled={!!disabled[key]} onClick={() => value !== key && onPick(key)}
          style={value === key ? S.segmentOn : S.segment}
          data-testid={`${testid}-${key}`}>{label}</button>
      ))}
    </div>
  )
}

const format = (value: number) => String(Number(value.toPrecision(6)))

const SOURCE_TITLE: Record<string, string> = {
  metadata: 'Read from the signal’s metadata',
  data: 'Recorded on the data when it was made',
  default: 'Not in the metadata — a default; check it',
}
const SOURCE_TONE: Record<string, React.CSSProperties> = {
  metadata: { color: '#a6e3a1' },
  data: { color: '#89b4fa' },
  default: { color: '#f9e2af' },
}

const S: Record<string, React.CSSProperties> = {
  card: {
    width: FIGURE_W + 16, boxSizing: 'border-box', padding: 8,
    background: '#181825', border: '1px solid #313244', borderRadius: 8,
    boxShadow: '0 8px 24px rgba(0,0,0,0.45)', display: 'flex',
    flexDirection: 'column', gap: 8, color: '#cdd6f4',
  },
  header: { display: 'flex', alignItems: 'baseline', gap: 8 },
  title: { fontSize: 12, fontWeight: 700 },
  subtitle: { fontSize: 10, color: '#a6adc8' },
  close: {
    background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer',
    fontSize: 13, width: 28, height: 24,
  },
  figure: {
    position: 'relative', width: FIGURE_W, height: FIGURE_H, display: 'flex',
    alignItems: 'center', justifyContent: 'center', background: '#11111b',
    borderRadius: 4, overflow: 'hidden',
  },
  form: { display: 'flex', flexDirection: 'column', gap: 6 },
  row: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' },
  label: { fontSize: 10, color: '#a6adc8' },
  muted: { fontSize: 10, color: '#6c7086' },
  fields: {
    display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '4px 12px',
  },
  field: { display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 },
  pcRow: {
    display: 'grid', gridTemplateColumns: 'auto repeat(3, minmax(0, 1fr))',
    alignItems: 'center', gap: '4px 12px',
  },
  pcLabel: { fontSize: 10, color: '#a6adc8', whiteSpace: 'nowrap' },
  pcAxis: { fontSize: 10, color: '#a6adc8' },
  fieldLabel: { fontSize: 10, color: '#a6adc8', flex: 1, whiteSpace: 'nowrap' },
  fieldValue: { display: 'flex', alignItems: 'center', gap: 3 },
  input: {
    width: 64, background: '#1e1e2e', border: '1px solid #313244', borderRadius: 4,
    color: '#cdd6f4', fontSize: 11, padding: '2px 4px', fontFamily: 'ui-monospace, Consolas, monospace',
  },
  unit: { fontSize: 10, color: '#6c7086', width: 18 },
  source: { fontSize: 9, width: 46 },
  segmented: {
    display: 'flex', border: '1px solid #313244', borderRadius: 6, overflow: 'hidden',
  },
  segment: {
    background: '#1e1e2e', color: '#a6adc8', border: 'none', fontSize: 10,
    padding: '3px 10px', cursor: 'pointer',
  },
  segmentOn: {
    background: '#89b4fa', color: '#11111b', border: 'none', fontSize: 10,
    padding: '3px 10px', fontWeight: 700, cursor: 'pointer',
  },
  check: { accentColor: '#89b4fa', width: 14, height: 14, margin: 0 },
  disclosure: {
    display: 'flex', alignItems: 'center', gap: 10, background: 'none', border: 'none',
    borderTop: '1px solid #313244', padding: '6px 0 0', color: '#a6adc8', fontSize: 10,
    cursor: 'pointer', textAlign: 'left',
  },
  summary: { color: '#6c7086', fontFamily: 'ui-monospace, Consolas, monospace', flex: 1 },
}
