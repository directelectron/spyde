/**
 * PeriodicTable.tsx — the popout that edits the sample's phases.
 *
 * A sample is a list of phases, each its elements (with optional atomic
 * percentages) and the structure that indexes it. There is always a SELECTED
 * phase, and the periodic table edits that one: its elements are lit, and a
 * click adds or removes an element from it and from nothing else. With no
 * phases yet, "Phase 1" is selected and the first click creates it.
 *
 * An element can be marked TRACE — the O in an Fe phase. It still counts for
 * EELS and EDS, but it is not what the structure is made of, so it is left out
 * of the COD search and kept when the structure is swapped.
 *
 * Phases are addressed by id, and every edit is sent as it happens: the popout
 * shows whatever the backend replies with, so the dock, the wizards and this
 * popout cannot disagree.
 *
 * A phase's structure is a .cif — from a file, a recent file, or the
 * Crystallography Open Database. COD matches the elements EXACTLY, so the
 * search uses the selected phase's non-trace elements only.
 */
import React from 'react'
import { createPortal } from 'react-dom'
import type { SamplePhase } from '../kernel/SpyDEContext'
import { useCifRecents, RecentCifs, fileName } from './CifRecents'

interface El { z: number; sym: string; row: number; col: number; cat: Cat }
type Cat = 'alkali' | 'alkaline' | 'tm' | 'post' | 'metalloid' | 'nonmetal'
  | 'halogen' | 'noble' | 'lanth' | 'act'

// Grid geometry. The f-block is conventionally drawn detached below the main
// table, so there is a thin SPACER row between them — and that row is a real
// grid row, which is what made the lanthanides render 8 px tall: they were
// placed on row 8, the spacer itself. Named here (and used to build
// gridTemplateRows in S.grid) so the two can't drift apart again.
const MAIN_ROWS = 7          // periods 1–7
const CELL_PX = 30
const SPACER_PX = 8
const SPACER_ROW = MAIN_ROWS + 1     // 8
const LANTH_ROW = SPACER_ROW + 1     // 9
const ACT_ROW = LANTH_ROW + 1        // 10

// One entry per element with its (row, col) on the grid above. The f-block
// occupies cols 3–17 of its two rows; period 6/7 group 3 is the gap it came out
// of.
const E = (z: number, sym: string, row: number, col: number, cat: Cat): El => ({ z, sym, row, col, cat })

const ELEMENTS: El[] = [
  E(1, 'H', 1, 1, 'nonmetal'), E(2, 'He', 1, 18, 'noble'),
  E(3, 'Li', 2, 1, 'alkali'), E(4, 'Be', 2, 2, 'alkaline'),
  E(5, 'B', 2, 13, 'metalloid'), E(6, 'C', 2, 14, 'nonmetal'), E(7, 'N', 2, 15, 'nonmetal'),
  E(8, 'O', 2, 16, 'nonmetal'), E(9, 'F', 2, 17, 'halogen'), E(10, 'Ne', 2, 18, 'noble'),
  E(11, 'Na', 3, 1, 'alkali'), E(12, 'Mg', 3, 2, 'alkaline'),
  E(13, 'Al', 3, 13, 'post'), E(14, 'Si', 3, 14, 'metalloid'), E(15, 'P', 3, 15, 'nonmetal'),
  E(16, 'S', 3, 16, 'nonmetal'), E(17, 'Cl', 3, 17, 'halogen'), E(18, 'Ar', 3, 18, 'noble'),
  E(19, 'K', 4, 1, 'alkali'), E(20, 'Ca', 4, 2, 'alkaline'),
  E(21, 'Sc', 4, 3, 'tm'), E(22, 'Ti', 4, 4, 'tm'), E(23, 'V', 4, 5, 'tm'), E(24, 'Cr', 4, 6, 'tm'),
  E(25, 'Mn', 4, 7, 'tm'), E(26, 'Fe', 4, 8, 'tm'), E(27, 'Co', 4, 9, 'tm'), E(28, 'Ni', 4, 10, 'tm'),
  E(29, 'Cu', 4, 11, 'tm'), E(30, 'Zn', 4, 12, 'tm'), E(31, 'Ga', 4, 13, 'post'), E(32, 'Ge', 4, 14, 'metalloid'),
  E(33, 'As', 4, 15, 'metalloid'), E(34, 'Se', 4, 16, 'nonmetal'), E(35, 'Br', 4, 17, 'halogen'), E(36, 'Kr', 4, 18, 'noble'),
  E(37, 'Rb', 5, 1, 'alkali'), E(38, 'Sr', 5, 2, 'alkaline'),
  E(39, 'Y', 5, 3, 'tm'), E(40, 'Zr', 5, 4, 'tm'), E(41, 'Nb', 5, 5, 'tm'), E(42, 'Mo', 5, 6, 'tm'),
  E(43, 'Tc', 5, 7, 'tm'), E(44, 'Ru', 5, 8, 'tm'), E(45, 'Rh', 5, 9, 'tm'), E(46, 'Pd', 5, 10, 'tm'),
  E(47, 'Ag', 5, 11, 'tm'), E(48, 'Cd', 5, 12, 'tm'), E(49, 'In', 5, 13, 'post'), E(50, 'Sn', 5, 14, 'post'),
  E(51, 'Sb', 5, 15, 'metalloid'), E(52, 'Te', 5, 16, 'metalloid'), E(53, 'I', 5, 17, 'halogen'), E(54, 'Xe', 5, 18, 'noble'),
  E(55, 'Cs', 6, 1, 'alkali'), E(56, 'Ba', 6, 2, 'alkaline'),
  E(72, 'Hf', 6, 4, 'tm'), E(73, 'Ta', 6, 5, 'tm'), E(74, 'W', 6, 6, 'tm'), E(75, 'Re', 6, 7, 'tm'),
  E(76, 'Os', 6, 8, 'tm'), E(77, 'Ir', 6, 9, 'tm'), E(78, 'Pt', 6, 10, 'tm'), E(79, 'Au', 6, 11, 'tm'),
  E(80, 'Hg', 6, 12, 'tm'), E(81, 'Tl', 6, 13, 'post'), E(82, 'Pb', 6, 14, 'post'), E(83, 'Bi', 6, 15, 'post'),
  E(84, 'Po', 6, 16, 'post'), E(85, 'At', 6, 17, 'halogen'), E(86, 'Rn', 6, 18, 'noble'),
  E(87, 'Fr', 7, 1, 'alkali'), E(88, 'Ra', 7, 2, 'alkaline'),
  E(104, 'Rf', 7, 4, 'tm'), E(105, 'Db', 7, 5, 'tm'), E(106, 'Sg', 7, 6, 'tm'), E(107, 'Bh', 7, 7, 'tm'),
  E(108, 'Hs', 7, 8, 'tm'), E(109, 'Mt', 7, 9, 'tm'), E(110, 'Ds', 7, 10, 'tm'), E(111, 'Rg', 7, 11, 'tm'),
  E(112, 'Cn', 7, 12, 'tm'), E(113, 'Nh', 7, 13, 'post'), E(114, 'Fl', 7, 14, 'post'), E(115, 'Mc', 7, 15, 'post'),
  E(116, 'Lv', 7, 16, 'post'), E(117, 'Ts', 7, 17, 'halogen'), E(118, 'Og', 7, 18, 'noble'),
  // f-block — BELOW the spacer row, cols 3–17.
  ...['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    .map((s, i) => E(57 + i, s, LANTH_ROW, 3 + i, 'lanth')),
  ...['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    .map((s, i) => E(89 + i, s, ACT_ROW, 3 + i, 'act')),
]

const CAT_COLOR: Record<Cat, string> = {
  alkali: '#f38ba8', alkaline: '#fab387', tm: '#89b4fa', post: '#94e2d5',
  metalloid: '#a6e3a1', nonmetal: '#f9e2af', halogen: '#cba6f7', noble: '#74c7ec',
  lanth: '#f5c2e7', act: '#eba0ac',
}
const colorOf = (symbol: string) =>
  CAT_COLOR[ELEMENTS.find(element => element.sym === symbol)?.cat ?? 'tm']

// ── how a phase reads wherever it is listed ──────────────────────────────────
/** A phase's elements without its trace ones — what its structure is made of. */
export const majorElements = (phase: SamplePhase) =>
  phase.elements.filter(symbol => !phase.trace.includes(symbol))

/** A phase's structure by name: its label, else its file, else "no structure". */
export const structureName = (phase: SamplePhase) =>
  phase.label ?? (phase.cifPath ? fileName(phase.cifPath) : 'no structure')

/** Green once a phase has a structure, greyed while it has none. Callers add
 *  their own size. */
export const structureTone = (phase: SamplePhase): React.CSSProperties => (phase.cifPath
  ? { color: '#a6e3a1', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }
  : { color: '#6c7086', fontStyle: 'italic' })

interface CodResult {
  id: string; formula: string; phase: string; sg: string
  a: number; b: number; c: number
  alpha: number | null; beta: number | null; gamma: number | null
  volume: number | null
}

type SendAction = (action: string, payload?: Record<string, unknown>, windowId?: number) => void

interface Props {
  windowId: number
  phases: SamplePhase[]
  sendAction: SendAction
  onClose: () => void
}

const formatNumber = (value: number | null) =>
  (value == null ? '–' : (Math.round(value * 1000) / 1000).toString())
const newPhaseId = () => Math.random().toString(36).slice(2, 14)

/** A percentage typed into a phase. Saved when the field loses focus (or on
 *  Enter) rather than per keystroke: saving "12." as 12 would make it
 *  impossible to type 12.5. */
function PercentInput({ value, onCommit, testid }: {
  value: number | undefined
  onCommit: (value: number | null) => void
  testid: string
}) {
  const [draft, setDraft] = React.useState<string | null>(null)
  // Show the typed text until the saved value comes back, not the old value.
  React.useEffect(() => { setDraft(null) }, [value])
  const commit = () => {
    if (draft == null) return
    const text = draft.trim()
    const number = text === '' ? null : parseFloat(text)
    if (Number.isNaN(number) || number === (value ?? null)) setDraft(null)
    else onCommit(number)
  }
  return (
    <input data-testid={testid} style={S.percentInput} placeholder="%"
      value={draft ?? (value ?? '')}
      onChange={event => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur() }} />
  )
}

export function PeriodicTable({ windowId, phases, sendAction, onClose }: Props) {
  // The phase the table edits. An id no phase has yet is the phase the next
  // edit creates, shown after the others.
  const [selectedId, setSelectedId] = React.useState(() => phases[0]?.id ?? newPhaseId())
  const selectedIndex = phases.findIndex(phase => phase.id === selectedId)
  const phase: SamplePhase | undefined = phases[selectedIndex]
  const position = selectedIndex >= 0 ? selectedIndex : phases.length
  const elements = phase?.elements ?? []
  const major = phase ? majorElements(phase) : []
  const elsewhere = new Set(phases.flatMap(other => (other === phase ? [] : other.elements)))

  const [codResults, setCodResults] = React.useState<CodResult[]>([])
  const [codNote, setCodNote] = React.useState('')
  const [searching, setSearching] = React.useState<string | null>(null)
  const { recents, remember } = useCifRecents()
  const act = (action: string, payload: Record<string, unknown>) =>
    sendAction(action, { phase: selectedId, ...payload }, windowId)

  React.useEffect(() => {
    const onResults = (event: Event) => {
      const detail = (event as CustomEvent).detail as {
        window_id?: number; phase?: string
        results?: CodResult[]; error?: string | null; elements?: string[]
      }
      if (detail.window_id != null && detail.window_id !== windowId) return
      // Only the phase that asked: a late reply for another must not replace
      // what is showing.
      if (detail.phase !== searching) return
      const formula = (detail.elements ?? []).join('-')
      setCodResults(detail.results ?? [])
      setCodNote(detail.error ?? (detail.results?.length
        ? `${detail.results.length} structure(s) for ${formula}`
        : `No structures for ${formula} — COD matches the elements exactly.`))
    }
    window.addEventListener('spyde:cod_results', onResults)
    return () => window.removeEventListener('spyde:cod_results', onResults)
  }, [windowId, searching])

  const select = (id: string) => { setSelectedId(id); setSearching(null) }
  const setStructure = (path: string) => {
    remember(path)
    act('set_phase_structure', { cif_path: path })
  }
  const evenPercentages = () => {
    const each = Math.round((100 / elements.length) * 10) / 10
    act('set_phase_percentages',
      { percentages: Object.fromEntries(elements.map(symbol => [symbol, each])) })
  }
  const removePhase = () => {
    act('remove_phase', {})
    select(phases[selectedIndex - 1]?.id ?? phases[selectedIndex + 1]?.id ?? newPhaseId())
  }

  // Rendered into document.body, not where it is written. A `position: fixed`
  // box is positioned against its nearest TRANSFORMED ancestor rather than the
  // viewport, and a wizard caret is placed with `transform: translateX(-50%)`.
  return createPortal((
    <div style={S.backdrop} data-testid="periodic-table" onClick={onClose}>
      <div style={S.modal} onClick={event => event.stopPropagation()}>
        <div style={S.header}>
          <span style={S.title}>Sample phases</span>
          <button data-testid="ptable-close" style={S.close} onClick={onClose}>✕</button>
        </div>

        <div style={S.strip} data-testid="ptable-phases">
          {phases.map((listed, index) => (
            <button key={listed.id} data-testid={`phase-btn-${index}`}
              data-active={listed === phase ? 'true' : undefined}
              title={structureName(listed)}
              style={listed === phase ? S.phaseButtonOn : S.phaseButton}
              onClick={() => select(listed.id)}>
              Phase {index + 1}
            </button>
          ))}
          {phase
            ? (
              <button data-testid="ptable-add-phase" style={S.addPhase}
                onClick={() => {
                  const id = newPhaseId()
                  sendAction('add_phase', { phase: id }, windowId)
                  select(id)
                }}>＋ Phase</button>
            )
            : (
              <button data-testid={`phase-btn-${position}`} data-active="true"
                title="new phase" style={S.phaseButtonOn}>
                Phase {position + 1}
              </button>
            )}
        </div>

        <div style={S.grid}>
          {ELEMENTS.map(element => {
            const on = elements.includes(element.sym)
            const color = CAT_COLOR[element.cat]
            const inOtherPhase = elsewhere.has(element.sym)
            return (
              <button
                key={element.sym}
                data-testid={`ptable-el-${element.sym}`}
                data-in-phase={on ? 'true' : undefined}
                onClick={() => act('toggle_phase_element', { element: element.sym })}
                title={`${element.sym} (${element.z})${inOtherPhase ? ' — in another phase' : ''}`}
                style={{
                  ...S.cell,
                  gridRow: element.row, gridColumn: element.col,
                  borderColor: color,
                  // Faintly tinted when another phase has it, so the whole
                  // sample stays visible while one phase is edited.
                  background: on ? color : inOtherPhase ? `${color}33` : 'transparent',
                  color: on ? '#11111b' : '#cdd6f4',
                }}
              >
                <span style={S.atomicNumber}>{element.z}</span>
                <span style={S.symbol}>{element.sym}</span>
              </button>
            )
          })}
        </div>

        {/* Keyed by phase: a field left mid-edit in one phase must not carry
            its text into the next phase's field for the same element. */}
        <div key={selectedId} style={S.phaseBox} data-testid={`phase-row-${position}`}>
          <div style={S.chips} data-testid="ptable-selected">
            {elements.length === 0
              ? <span style={S.hint}>Click elements above to build Phase {position + 1}, or load its structure.</span>
              : elements.map(symbol => {
                const trace = phase?.trace.includes(symbol) ?? false
                return (
                  <span key={symbol} style={S.chip} data-testid={`phase-${position}-el-${symbol}`}>
                    <span style={{ ...S.chipSymbol, color: colorOf(symbol), opacity: trace ? 0.6 : 1 }}>
                      {symbol}
                    </span>
                    <PercentInput testid={`phase-${position}-pct-${symbol}`}
                      value={phase?.percentages[symbol]}
                      onCommit={value => act('set_phase_percentages',
                        { percentages: { [symbol]: value } })} />
                    <button data-testid={`phase-${position}-trace-${symbol}`}
                      data-on={trace ? 'true' : undefined}
                      title={trace
                        ? 'Trace: counted for EELS and EDS, left out of the structure search'
                        : 'Mark as trace'}
                      style={trace ? S.traceOn : S.trace}
                      onClick={() => act('set_phase_trace', { element: symbol, trace: !trace })}>
                      trace
                    </button>
                  </span>
                )
              })}
            {elements.length > 1 && (
              <button data-testid={`phase-${position}-even`} style={S.ghost}
                onClick={evenPercentages}>Even %</button>
            )}
          </div>
          <div style={S.structureRow}>
            <span style={{ ...S.structure, ...(phase ? structureTone(phase) : S.noStructure) }}
              data-testid={`phase-${position}-structure`} title={phase?.cifPath ?? undefined}>
              {phase ? structureName(phase) : 'no structure'}
            </span>
            <button data-testid={`phase-${position}-cif`} style={S.ghost}
              onClick={async () => {
                const path = await window.electron.pickFile(
                  { name: 'Crystal (.cif)', extensions: ['cif'] })
                if (path) setStructure(path)
              }}>Load CIF</button>
            <button data-testid={`phase-${position}-cod`} style={S.ghost}
              disabled={major.length === 0}
              title={major.length
                ? `Search COD for ${major.join('-')} structures`
                : 'Give this phase a non-trace element first'}
              onClick={() => {
                setSearching(selectedId); setCodResults([]); setCodNote('Searching COD…')
                act('cod_search', {})
              }}>Search COD</button>
            {phase && (
              <button data-testid={`phase-${position}-remove`} style={S.remove}
                title="Remove this phase" onClick={removePhase}>
                Remove phase
              </button>
            )}
          </div>
          <RecentCifs recents={recents}
            exclude={phase?.cifPath ? [phase.cifPath] : []} onPick={setStructure} />
          {searching === selectedId && (
            <div style={S.codBox} data-testid={`phase-${position}-cod-results`}>
              {codNote && <div style={S.hint}>{codNote}</div>}
              {codResults.map(result => (
                <button key={result.id} data-testid={`cod-row-${result.id}`} style={S.codRow}
                  title={`COD ${result.id}`}
                  onClick={() => {
                    act('cod_pick', { cod_id: result.id,
                                      label: `${result.formula} ${result.sg}`.trim() })
                    setSearching(null)
                  }}>
                  <span style={S.codTop}>
                    <b style={S.formula}>{result.formula}</b>
                    {result.phase && <span style={S.mineral}>{result.phase}</span>}
                    <span style={S.spaceGroup}>{result.sg}</span>
                  </span>
                  <span style={S.cellDimensions}>
                    a {formatNumber(result.a)} · b {formatNumber(result.b)} · c {formatNumber(result.c)} Å &nbsp;
                    α {formatNumber(result.alpha)} · β {formatNumber(result.beta)} · γ {formatNumber(result.gamma)}°
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div style={S.actions}>
          <button data-testid="ptable-done" style={S.done} onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  ), document.body)
}

const S: Record<string, React.CSSProperties> = {
  backdrop: {
    position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  modal: {
    background: '#181825', border: '1px solid #313244', borderRadius: 10,
    padding: 14, boxShadow: '0 12px 48px rgba(0,0,0,0.6)', maxWidth: '92vw',
    display: 'flex', flexDirection: 'column', gap: 10,
  },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  title: { color: '#cdd6f4', fontSize: 13, fontWeight: 600 },
  close: { background: 'none', border: 'none', color: '#f38ba8', cursor: 'pointer', fontSize: 14 },

  strip: { display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 5 },
  phaseButton: {
    background: '#11111b', border: '1px solid #313244', color: '#cdd6f4',
    cursor: 'pointer', fontSize: 10, fontWeight: 600, padding: '2px 9px', borderRadius: 10,
  },
  phaseButtonOn: {
    background: '#89b4fa', border: '1px solid #89b4fa', color: '#11111b',
    cursor: 'pointer', fontSize: 10, fontWeight: 700, padding: '2px 9px', borderRadius: 10,
  },
  addPhase: {
    background: 'none', border: '1px dashed #585b70', color: '#cba6f7',
    cursor: 'pointer', fontSize: 10, fontWeight: 600, padding: '2px 9px', borderRadius: 10,
  },

  grid: {
    display: 'grid',
    gridTemplateColumns: `repeat(18, ${CELL_PX}px)`,
    // periods 1–7, the detached-f-block spacer, then the two f-block rows.
    gridTemplateRows:
      `repeat(${MAIN_ROWS}, ${CELL_PX}px) ${SPACER_PX}px repeat(2, ${CELL_PX}px)`,
    gap: 2,
  },
  cell: {
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    border: '1px solid', borderRadius: 4, cursor: 'pointer', padding: 0,
    lineHeight: 1, overflow: 'hidden',
  },
  atomicNumber: { fontSize: 6, opacity: 0.8 },
  symbol: { fontSize: 11, fontWeight: 700 },

  phaseBox: {
    display: 'flex', flexDirection: 'column', gap: 6,
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 6, padding: '6px 8px',
  },
  chips: { display: 'flex', flexWrap: 'wrap', gap: 6, minHeight: 26, alignItems: 'center' },
  chip: {
    display: 'flex', alignItems: 'center', gap: 4, background: '#11111b',
    border: '1px solid #313244', borderRadius: 12, padding: '2px 4px 2px 8px',
  },
  chipSymbol: { fontSize: 12, fontWeight: 700 },
  percentInput: {
    width: 38, background: '#181825', border: '1px solid #313244', borderRadius: 8,
    color: '#cdd6f4', fontSize: 10, padding: '2px 4px', textAlign: 'center',
  },
  trace: {
    background: 'none', border: '1px solid #313244', color: '#6c7086', cursor: 'pointer',
    fontSize: 9, padding: '1px 5px', borderRadius: 8,
  },
  traceOn: {
    background: '#45475a', border: '1px solid #585b70', color: '#f9e2af', cursor: 'pointer',
    fontSize: 9, fontWeight: 600, padding: '1px 5px', borderRadius: 8,
  },
  hint: { color: '#6c7086', fontSize: 11 },
  structureRow: { display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' },
  structure: { flex: 1, fontSize: 11, minWidth: 80 },
  noStructure: { color: '#6c7086', fontStyle: 'italic' },
  ghost: {
    background: 'none', border: '1px solid #313244', color: '#a6adc8',
    cursor: 'pointer', fontSize: 11, padding: '3px 9px', borderRadius: 6,
  },
  remove: {
    background: 'none', border: '1px solid #45475a', color: '#f38ba8',
    cursor: 'pointer', fontSize: 11, padding: '3px 9px', borderRadius: 6,
  },

  codBox: { display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 200, overflowY: 'auto' },
  codRow: {
    display: 'flex', flexDirection: 'column', gap: 1, alignItems: 'flex-start',
    background: '#11111b', border: '1px solid #313244', borderRadius: 5,
    padding: '4px 7px', cursor: 'pointer', textAlign: 'left', width: '100%',
  },
  codTop: { display: 'flex', gap: 6, alignItems: 'baseline' },
  formula: { color: '#cdd6f4', fontSize: 11 },
  mineral: { color: '#f9e2af', fontSize: 10 },
  spaceGroup: { color: '#89b4fa', fontSize: 10 },
  cellDimensions: { color: '#6c7086', fontSize: 9 },

  actions: { display: 'flex', justifyContent: 'flex-end' },
  done: {
    background: '#89b4fa', border: 'none', color: '#11111b', cursor: 'pointer',
    fontSize: 11, fontWeight: 700, padding: '4px 14px', borderRadius: 6,
  },
}
