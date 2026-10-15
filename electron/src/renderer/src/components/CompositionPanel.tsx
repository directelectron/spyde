/**
 * CompositionPanel.tsx — the right-dock "Composition" section: the sample's
 * phases, with `&` between them and each phase's structure beside its
 * elements.
 *
 * A Zr-Nb alloy with Nb platelets in it is α-Zr AND β-Nb. Written flat
 * ("Zr, Nb") it would say one compound of both — which is why a sample is a
 * list of phases and not an element list. EELS and EDS fit every element of
 * every phase; the indexing wizards use each phase's structure.
 *
 * Clicking a phase's structure expands the phase to show its viewer — the
 * band sphere and the pattern it makes on the detector. That is part of the
 * phase's setup, not a dataset, so it is shown here rather than in a window.
 */
import React from 'react'
import type { Composition } from '../kernel/SpyDEContext'
import { useSpyDE } from '../kernel/SpyDEContext'
import { PeriodicTable, structureName, structureTone } from './PeriodicTable'
import { SeamlessFigureFrame } from './ReportFigureCell'

interface Props {
  activeId: number | null
  composition?: Composition
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}

export function CompositionPanel({ activeId, composition, sendAction }: Props) {
  const [editing, setEditing] = React.useState(false)
  const [expanded, setExpanded] = React.useState<string | null>(null)
  const { state, iframeRefs, replayState } = useSpyDE()
  const phases = composition?.phases ?? []

  const toggleViewer = (phaseId: string) => {
    if (expanded === phaseId) { setExpanded(null); return }
    setExpanded(phaseId)
    if (activeId != null) sendAction('show_phase', { phase: phaseId }, activeId)
  }

  return (
    <div style={S.section} data-testid="composition-section">
      <div style={S.head}>
        <span style={S.label}>Composition</span>
        {/* The empty state rides ON the header row rather than costing a line of
            its own — it says nothing an inline note can't. */}
        {phases.length === 0 && (
          <span style={{ ...S.empty, flex: 1, marginLeft: 6 }}
            data-testid="composition-empty">No elements set</span>
        )}
        <button data-testid="composition-edit" style={S.editBtn}
          onClick={() => setEditing(true)}>
          {phases.length ? 'Edit' : '＋ Elements and Phase'}
        </button>
      </div>

      {phases.map((phase, index) => (
        <React.Fragment key={phase.id}>
        <div style={S.phaseRow} data-testid={`composition-phase-${index}`}>
          {index > 0 && <span style={S.ampersand}>&amp;</span>}
          <div style={S.chips}>
            {phase.elements.length === 0
              ? <span style={S.empty}>no elements</span>
              : phase.elements.map(symbol => {
                const trace = phase.trace.includes(symbol)
                return (
                  <span key={symbol} style={trace ? S.traceChip : S.chip}
                    title={trace ? `${symbol} (trace)` : undefined}
                    data-trace={trace ? 'true' : undefined}
                    data-testid={`composition-chip-${index}-${symbol}`}>
                    <span style={S.symbol}>{symbol}</span>
                    {phase.percentages[symbol] != null
                      && <span style={S.percent}>{phase.percentages[symbol]}%</span>}
                  </span>
                )
              })}
          </div>
          {phase.cifPath && (
            <button style={{ ...S.structure, ...structureTone(phase) }}
              title={`${phase.cifPath}\nClick to show this phase's Kikuchi bands`}
              aria-expanded={expanded === phase.id}
              disabled={activeId == null}
              onClick={() => toggleViewer(phase.id)}
              data-testid={`composition-structure-${index}`}>
              {expanded === phase.id ? '▾ ' : '▸ '}{structureName(phase)}
            </button>
          )}
        </div>
        {expanded === phase.id && phase.cifPath && (
          <PhaseViewer figure={state.phaseFigures.get(phase.id)}
            iframeRefs={iframeRefs} replayState={replayState} index={index} />
        )}
        </React.Fragment>
      ))}

      {editing && activeId != null && (
        <PeriodicTable windowId={activeId} phases={phases} sendAction={sendAction}
          onClose={() => setEditing(false)} />
      )}
    </div>
  )
}

/** A phase's band sphere and detector preview, under its row. */
function PhaseViewer({ figure, iframeRefs, replayState, index }: {
  figure?: { figId: string, filePath: string | null, title: string }
  iframeRefs: React.MutableRefObject<Map<string, HTMLIFrameElement>>
  replayState: (figId: string, target?: HTMLIFrameElement) => void
  index: number
}) {
  return (
    <div style={S.viewer} data-testid={`composition-viewer-${index}`}>
      <div style={S.viewerFrame}>
        {figure
          ? <SeamlessFigureFrame figId={figure.figId} filePath={figure.filePath}
              title={figure.title} iframeRefs={iframeRefs} replayState={replayState} />
          : <span style={S.empty}>Simulating bands…</span>}
      </div>
      <span style={S.hint}>Drag the sphere to turn the crystal; the outline is the detector.</span>
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  // Metrics track PlotControlDock's `section`/`label` — this panel renders inline
  // among that dock's sections, so it has to compress with them.
  section: {
    padding: '6px 10px', borderTop: '1px solid #1e1e2e', flexShrink: 0,
  },
  head: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 4, marginBottom: 4 },
  label: { fontSize: 10, color: '#a6adc8', flex: 1 },
  editBtn: {
    background: 'none', border: '1px solid #313244', color: '#89b4fa', cursor: 'pointer',
    fontSize: 10, fontWeight: 600, padding: '1px 8px', borderRadius: 4,
  },
  empty: { fontSize: 10, color: '#6c7086' },
  phaseRow: { display: 'flex', alignItems: 'center', gap: 4, marginTop: 2 },
  ampersand: { fontSize: 11, fontWeight: 700, color: '#cba6f7' },
  chips: { display: 'flex', flexWrap: 'wrap', gap: 4 },
  chip: {
    display: 'flex', alignItems: 'center', gap: 3, background: '#1e1e2e',
    border: '1px solid #313244', borderRadius: 12, padding: '2px 8px',
  },
  traceChip: {
    display: 'flex', alignItems: 'center', gap: 3, background: 'none', opacity: 0.6,
    border: '1px dashed #585b70', borderRadius: 12, padding: '2px 8px',
  },
  symbol: { fontSize: 11, fontWeight: 700, color: '#cdd6f4' },
  percent: { fontSize: 9, color: '#a6adc8' },
  viewer: { display: 'flex', flexDirection: 'column', gap: 2, marginTop: 4 },
  // Matches phase_viewer.FIGURE_SIZE: a figure that is not a window is never
  // resized, so its host is exactly its size. The frame fills it absolutely.
  viewerFrame: {
    position: 'relative', width: 276, height: 150, display: 'flex',
    alignItems: 'center', justifyContent: 'center',
  },
  hint: { fontSize: 9, color: '#6c7086' },
  structure: {
    flex: 1, textAlign: 'right', fontSize: 9, background: 'none', border: 'none',
    padding: 0, cursor: 'pointer', textDecoration: 'underline dotted',
    textUnderlineOffset: 2,
  },
}
