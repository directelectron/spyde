/**
 * PhasesEditor.tsx — the sample's phases, composition and structure together.
 *
 * They used to be apart: the dock held a flat element list, the orientation
 * wizard held a list of .cif paths, and the COD search silently read the former
 * to fill the latter. That could not describe a two-phase sample at all. COD is
 * asked for a structure containing EXACTLY the elements given, so a Cu/Nb
 * sample searched as one composition asks for a Cu-Nb compound and gets nothing
 * back — while fcc Cu and bcc Nb are one query each.
 *
 * So a PHASE is one row here: what it is made of (the periodic table) and what
 * indexes it (a .cif, or a COD search scoped to THAT row's elements). The
 * sample owns the list, which is why the dock can show it and the wizard can
 * read it instead of keeping a private copy.
 *
 * COD results open INLINE under the row that asked for them rather than in a
 * modal of their own — the question "which phase am I filling?" should not need
 * remembering.
 */
import React from 'react'
import type { SamplePhase } from '../kernel/SpyDEContext'
import { PeriodicTable } from './PeriodicTable'

interface CodResult {
  id: string; formula: string; phase: string; sg: string
  a: number; b: number; c: number
  alpha: number | null; beta: number | null; gamma: number | null
  volume: number | null
}

interface Props {
  windowId: number
  phases: SamplePhase[]
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

const fmt = (v: number | null) => (v == null ? '–' : (Math.round(v * 1000) / 1000).toString())
const base = (p: string) => p.split(/[/\\]/).pop() || p

/** How a phase's structure reads where it is listed — green once it has one,
 *  greyed and italic while it does not. Exported because both orientation
 *  wizards list phases too, and they must not drift apart; it is NOT in
 *  WizardShell's shared `S`, which is typed `Record<string, CSSProperties>` and
 *  so turns a misspelled key into a silently missing style. */
export const PHASE_STYLE: Record<'set' | 'unset', React.CSSProperties> = {
  set: { fontSize: 9, fontWeight: 600, color: '#a6e3a1', overflow: 'hidden',
         textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  unset: { fontSize: 9, fontStyle: 'italic', color: '#6c7086' },
}

export function PhasesEditor({ windowId, phases, sendAction, onClose }: Props) {
  // Which row has the periodic table open, and which row's COD results are
  // showing — both are "one at a time", so an index is the whole state.
  const [editing, setEditing] = React.useState<number | null>(null)
  const [searching, setSearching] = React.useState<number | null>(null)
  const [results, setResults] = React.useState<CodResult[]>([])
  const [note, setNote] = React.useState('')

  React.useEffect(() => {
    const onResults = (e: Event) => {
      const d = (e as CustomEvent).detail as {
        window_id?: number; phase?: number | null
        results?: CodResult[]; error?: string; elements?: string[]
      }
      if (d.window_id != null && d.window_id !== windowId) return
      setResults(d.results ?? [])
      setNote(d.error
        ? d.error
        : (d.results?.length
          ? `${d.results.length} structure(s) for ${(d.elements ?? []).join('-')}`
          : `No structures for ${(d.elements ?? []).join('-')} — COD matches the elements exactly.`))
    }
    window.addEventListener('spyde:cod_results', onResults)
    return () => window.removeEventListener('spyde:cod_results', onResults)
  }, [windowId])

  const addPhase = () => {
    // The new row offers all three doors rather than forcing one: a phase you
    // already have a .cif for never needs its elements typed in, and opening
    // the periodic table over this modal uninvited would say otherwise.
    sendAction('add_phase', {}, windowId)
    setSearching(null)
  }
  const setElements = (index: number, elements: string[], percentages: Record<string, number>) => {
    sendAction('set_phase', { index, elements, percentages }, windowId)
    setEditing(null)
  }
  const pickCif = async (index: number) => {
    const path = await window.electron.pickFile({ name: 'Crystal (.cif)', extensions: ['cif'] })
    if (path) sendAction('set_phase_structure', { index, cif_path: path }, windowId)
  }
  const search = (index: number) => {
    setSearching(index); setResults([]); setNote('Searching COD…')
    sendAction('cod_search', { phase: index }, windowId)
  }
  const choose = (index: number, r: CodResult) => {
    sendAction('cod_pick', {
      phase: index, cod_id: r.id, label: `${r.formula} ${r.sg}`.trim(),
    }, windowId)
    setSearching(null)
  }

  return (
    <div style={S.backdrop} data-testid="phases-editor" onClick={onClose}>
      <div style={S.modal} onClick={e => e.stopPropagation()}>
        <div style={S.header}>
          <span style={S.title}>Sample phases</span>
          <button data-testid="phases-close" style={S.x} onClick={onClose}>✕</button>
        </div>
        <div style={S.hint}>
          One row per phase — what it is made of, and the structure that indexes it.
        </div>

        <div style={S.rows}>
          {phases.length === 0 && (
            <span style={S.empty} data-testid="phases-empty">
              No phases yet — add one to begin.
            </span>
          )}
          {phases.map((phase, index) => (
            <div key={index} style={S.row} data-testid={`phase-row-${index}`}>
              <div style={S.rowTop}>
                <span style={S.index}>{index + 1}</span>
                <div style={S.chips}>
                  {phase.elements.length === 0
                    ? <span style={S.empty}>no elements</span>
                    : phase.elements.map(el => (
                      <span key={el} style={S.chip} data-testid={`phase-${index}-el-${el}`}>
                        {el}
                        {phase.percentages[el] != null &&
                          <span style={S.pct}>{phase.percentages[el]}%</span>}
                      </span>
                    ))}
                </div>
                <button data-testid={`phase-${index}-remove`} style={S.x}
                  title="Remove this phase"
                  onClick={() => sendAction('remove_phase', { index }, windowId)}>✕</button>
              </div>

              <div style={S.rowBottom}>
                <button data-testid={`phase-${index}-elements`} style={S.btn}
                  onClick={() => { setEditing(index); setSearching(null) }}>
                  {phase.elements.length ? 'Elements' : '＋ Elements'}
                </button>
                <button data-testid={`phase-${index}-cif`} style={S.btn}
                  onClick={() => pickCif(index)}>From file</button>
                <button data-testid={`phase-${index}-cod`} style={S.btn}
                  disabled={phase.elements.length === 0}
                  title={phase.elements.length
                    ? `Search COD for ${phase.elements.join('-')} structures`
                    : 'Set this phase’s elements first'}
                  onClick={() => search(index)}>🔎 COD</button>
                <span style={phase.cifPath ? S.structure : S.noStructure}
                  data-testid={`phase-${index}-structure`}
                  title={phase.cifPath ?? undefined}>
                  {phase.label ?? (phase.cifPath ? base(phase.cifPath) : 'no structure')}
                </span>
              </div>

              {searching === index && (
                <div style={S.codBox} data-testid={`phase-${index}-cod-results`}>
                  {note && <div style={S.note}>{note}</div>}
                  {results.map(r => (
                    <button key={r.id} data-testid={`cod-row-${r.id}`} style={S.codRow}
                      title={`COD ${r.id}`} onClick={() => choose(index, r)}>
                      <div style={S.codTop}>
                        <span style={S.formula}>{r.formula}</span>
                        {r.phase && <span style={S.phaseName}>{r.phase}</span>}
                        <span style={S.sg}>{r.sg}</span>
                      </div>
                      <div style={S.cell}>
                        a {fmt(r.a)} · b {fmt(r.b)} · c {fmt(r.c)} Å &nbsp;
                        α {fmt(r.alpha)} · β {fmt(r.beta)} · γ {fmt(r.gamma)}°
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        <div style={S.footer}>
          <button data-testid="phases-add" style={S.add} onClick={addPhase}>＋ Add phase</button>
          <button data-testid="phases-done" style={S.done} onClick={onClose}>Done</button>
        </div>
      </div>

      {editing != null && (
        <PeriodicTable
          initial={phases[editing]?.elements ?? []}
          initialPct={phases[editing]?.percentages ?? {}}
          onApply={(els, pct) => setElements(editing, els, pct)}
          onClose={() => setEditing(null)}
        />
      )}
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  backdrop: {
    position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  modal: {
    background: '#181825', border: '1px solid #313244', borderRadius: 10, padding: 14,
    boxShadow: '0 12px 48px rgba(0,0,0,0.6)', width: 440, maxWidth: '94vw',
  },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  title: { color: '#cdd6f4', fontSize: 13, fontWeight: 600 },
  hint: { fontSize: 10, color: '#6c7086', margin: '2px 0 8px' },
  x: { background: 'none', border: 'none', color: '#f38ba8', cursor: 'pointer', fontSize: 13 },
  rows: { display: 'flex', flexDirection: 'column', gap: 6, maxHeight: '58vh', overflowY: 'auto' },
  row: {
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 6, padding: '6px 8px',
    display: 'flex', flexDirection: 'column', gap: 5,
  },
  rowTop: { display: 'flex', alignItems: 'center', gap: 6 },
  index: {
    color: '#6c7086', fontSize: 10, fontWeight: 700, minWidth: 12, textAlign: 'center',
  },
  chips: { display: 'flex', flexWrap: 'wrap', gap: 4, flex: 1 },
  chip: {
    display: 'inline-flex', alignItems: 'baseline', gap: 3, background: '#313244',
    border: '1px solid #45475a', borderRadius: 10, padding: '1px 7px',
    color: '#cdd6f4', fontSize: 10, fontWeight: 600,
  },
  pct: { color: '#a6adc8', fontSize: 9, fontWeight: 400 },
  rowBottom: { display: 'flex', alignItems: 'center', gap: 5 },
  btn: {
    background: '#11111b', border: '1px solid #313244', color: '#cdd6f4', cursor: 'pointer',
    fontSize: 10, padding: '2px 7px', borderRadius: 4, whiteSpace: 'nowrap',
  },
  structure: {
    flex: 1, color: '#a6e3a1', fontSize: 10, fontWeight: 600, textAlign: 'right',
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  noStructure: {
    flex: 1, color: '#6c7086', fontSize: 10, fontStyle: 'italic', textAlign: 'right',
  },
  empty: { color: '#6c7086', fontSize: 10, fontStyle: 'italic' },
  codBox: {
    display: 'flex', flexDirection: 'column', gap: 3, marginTop: 2,
    maxHeight: 220, overflowY: 'auto',
  },
  note: { fontSize: 10, color: '#a6adc8' },
  codRow: {
    display: 'flex', flexDirection: 'column', gap: 1, alignItems: 'flex-start',
    background: '#11111b', border: '1px solid #313244', borderRadius: 5,
    padding: '4px 7px', cursor: 'pointer', textAlign: 'left', width: '100%',
  },
  codTop: { display: 'flex', gap: 6, alignItems: 'baseline' },
  formula: { color: '#cdd6f4', fontSize: 11, fontWeight: 600 },
  phaseName: { color: '#f9e2af', fontSize: 10 },
  sg: { color: '#89b4fa', fontSize: 10 },
  cell: { color: '#6c7086', fontSize: 9 },
  footer: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    gap: 8, marginTop: 10,
  },
  add: {
    background: '#1e1e2e', border: '1px dashed #585b70', color: '#cba6f7', cursor: 'pointer',
    fontSize: 11, fontWeight: 600, padding: '4px 10px', borderRadius: 4,
  },
  done: {
    background: '#89b4fa', border: 'none', color: '#11111b', cursor: 'pointer',
    fontSize: 11, fontWeight: 700, padding: '4px 14px', borderRadius: 4,
  },
}
