/**
 * PeriodicTable.tsx — what the sample is made of, and which phases it is made
 * OF, in one popout.
 *
 * Clicking elements builds the sample's composition. Selecting a phase first
 * makes those clicks build THAT phase, so a phase is a subset of the sample
 * rather than a separate thing to type in twice — and an element can belong to
 * the sample without belonging to any phase, which is how you record the extra
 * oxygen that is not in either structure you are indexing against.
 *
 * Each phase also carries the structure that indexes it, loaded from a .cif or
 * found in the Crystallography Open Database. That search is scoped to the
 * phase's own elements: COD matches the elements EXACTLY, so asking a two-phase
 * Cu/Nb sample as one composition asks for a Cu-Nb compound and gets nothing,
 * while fcc Cu and bcc Nb are one query each.
 *
 * Composition and phases share this one popout deliberately. They were two, and
 * a phase's elements then had to be told to the periodic table while its
 * structure was told to a different modal on top of it.
 */
import React from 'react'
import { createPortal } from 'react-dom'
import type { SamplePhase } from '../kernel/SpyDEContext'

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

interface CodResult {
  id: string; formula: string; phase: string; sg: string
  a: number; b: number; c: number
  alpha: number | null; beta: number | null; gamma: number | null
  volume: number | null
}

interface Props {
  initial: string[]
  initialPct?: Record<string, number>
  onApply: (elements: string[], percentages: Record<string, number>) => void
  onClose: () => void
  /** The sample's phases, and the window to act on them through. Omitted by
   *  callers that only want the element picker. */
  phases?: SamplePhase[]
  windowId?: number | null
  sendAction?: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}

const fmt = (v: number | null) => (v == null ? '–' : (Math.round(v * 1000) / 1000).toString())
const base = (p: string) => p.split(/[/\\]/).pop() || p

export function PeriodicTable({ initial, initialPct = {}, onApply, onClose,
                                phases, windowId, sendAction }: Props) {
  const [sel, setSel] = React.useState<string[]>(initial)
  const [pct, setPct] = React.useState<Record<string, number>>(initialPct)
  // Which phase the clicks are building. `null` = the sample itself, which is
  // how an element gets in without belonging to a phase. A sample with exactly
  // ONE phase starts on it — there is nothing to disambiguate, and the common
  // case should not cost a click.
  const [activePhase, setActivePhase] = React.useState<number | null>(
    phases?.length === 1 ? 0 : null)
  // …and it stays selected when that phase ARRIVES rather than being there at
  // mount: adding a phase is a round trip through the backend, so the popout
  // opens empty and is told about it a moment later.
  const onlyPhase = (phases ?? []).length === 1
  React.useEffect(() => {
    if (onlyPhase) setActivePhase(prev => (prev == null ? 0 : prev))
  }, [onlyPhase])
  const [results, setResults] = React.useState<CodResult[]>([])
  const [searching, setSearching] = React.useState<number | null>(null)
  const [note, setNote] = React.useState('')
  const showPhases = !!phases && windowId != null && !!sendAction
  const phaseList = phases ?? []
  const act = (action: string, payload: Record<string, unknown>) => {
    if (sendAction && windowId != null) sendAction(action, payload, windowId)
  }

  React.useEffect(() => {
    const onResults = (e: Event) => {
      const d = (e as CustomEvent).detail as {
        window_id?: number; phase?: number | null
        results?: CodResult[]; error?: string; elements?: string[]
      }
      if (d.window_id != null && windowId != null && d.window_id !== windowId) return
      // Only the row that asked: a late reply from another row (or from another
      // caret listening on the same event) must not overwrite what is showing.
      if (d.phase != null && d.phase !== searching) return
      setResults(d.results ?? [])
      setNote(d.error ?? (d.results?.length
        ? `${d.results.length} structure(s) for ${(d.elements ?? []).join('-')}`
        : `No structures for ${(d.elements ?? []).join('-')} — COD matches the elements exactly.`))
    }
    window.addEventListener('spyde:cod_results', onResults)
    return () => window.removeEventListener('spyde:cod_results', onResults)
  }, [windowId, searching])

  // The backend is the truth about which elements the sample has, because it is
  // not only this popout that adds them: loading a .cif fills a phase's
  // elements in from the file. A local snapshot taken when the popout opened
  // would not know about those, and committing it on Apply would delete them.
  const initialKey = initial.join(',')
  React.useEffect(() => { setSel(initial) }, [initialKey])

  const toggle = (sym: string) => {
    // A click always affects the SAMPLE; with a phase selected it affects that
    // phase too, so a phase is a subset rather than a second thing to type in.
    const next = sel.includes(sym) ? sel.filter(s => s !== sym) : [...sel, sym]
    setSel(next)
    // Written through immediately, like every other edit in this popout — which
    // is what keeps the local list and the backend's from drifting apart.
    act('set_composition', { elements: next, percentages: pct })
    if (activePhase == null || !showPhases) return
    const phase = phaseList[activePhase]
    if (!phase) return
    const inPhase = phase.elements.includes(sym)
      ? phase.elements.filter(s => s !== sym)
      : [...phase.elements, sym]
    act('set_phase', { index: activePhase, elements: inPhase,
                       percentages: phase.percentages })
  }

  const setPctFor = (sym: string, v: string) => setPct(prev => {
    const n = { ...prev }
    if (v === '') delete n[sym]
    else { const f = parseFloat(v); if (!Number.isNaN(f)) n[sym] = f }
    return n
  })

  // Even-split the % across selected elements (a quick "atomic fraction" guess).
  const equalize = () => {
    if (!sel.length) return
    const each = Math.round((100 / sel.length) * 10) / 10
    setPct(Object.fromEntries(sel.map(s => [s, each])))
  }

  // Rendered into document.body, not where it is written. A `position: fixed`
  // box is positioned against its nearest TRANSFORMED ancestor rather than the
  // viewport, and the caret this can open from is placed with
  // `transform: translateX(-50%)` — so the popout centred itself on the caret
  // and hung off the left of the window.
  return createPortal((
    <div style={S.backdrop} data-testid="periodic-table" onClick={onClose}>
      <div style={S.modal} onClick={e => e.stopPropagation()}>
        <div style={S.header}>
          <span style={S.title}>
            {showPhases ? 'Sample composition and phases' : 'Sample composition — choose elements'}
          </span>
          <button data-testid="ptable-close" style={S.x} onClick={onClose}>✕</button>
        </div>

        <div style={S.grid}>
          {ELEMENTS.map(el => {
            const on = sel.includes(el.sym)
            return (
              <button
                key={el.sym}
                data-testid={`ptable-el-${el.sym}`}
                onClick={() => toggle(el.sym)}
                title={`${el.sym} (${el.z})`}
                style={{
                  ...S.cell,
                  gridRow: el.row, gridColumn: el.col,
                  borderColor: CAT_COLOR[el.cat],
                  background: on ? CAT_COLOR[el.cat] : 'transparent',
                  color: on ? '#11111b' : '#cdd6f4',
                }}
              >
                <span style={S.z}>{el.z}</span>
                <span style={S.sym}>{el.sym}</span>
              </button>
            )
          })}
        </div>

        {/* Selected elements + optional atomic %. */}
        <div style={S.footer}>
          <div style={S.selRow} data-testid="ptable-selected">
            {sel.length === 0 && <span style={S.hint}>Click elements above to add them.</span>}
            {sel.map(s => (
              <span key={s} style={S.selChip}>
                <span style={{ ...S.selSym, color: CAT_COLOR[ELEMENTS.find(e => e.sym === s)?.cat ?? 'tm'] }}>{s}</span>
                <input
                  data-testid={`ptable-pct-${s}`}
                  style={S.pctInput}
                  value={pct[s] ?? ''}
                  placeholder="%"
                  onChange={e => setPctFor(s, e.target.value)}
                />
              </span>
            ))}
          </div>
          {showPhases && (
            <div style={S.phases} data-testid="ptable-phases">
              <div style={S.phaseStrip}>
                <span style={S.phaseLabel}>Phases</span>
                {phaseList.map((phase, index) => (
                  <button key={index} data-testid={`phase-btn-${index}`}
                    data-active={activePhase === index ? 'true' : undefined}
                    title={phase.label ?? 'no structure yet'}
                    style={activePhase === index ? S.phaseBtnOn : S.phaseBtn}
                    onClick={() => setActivePhase(activePhase === index ? null : index)}>
                    Phase {index + 1}
                  </button>
                ))}
                <button data-testid="ptable-add-phase" style={S.addPhase}
                  onClick={() => {
                    act('add_phase', {})
                    // Select it: the next elements clicked are almost always
                    // the ones this phase is made of.
                    setActivePhase(phaseList.length)
                    setSearching(null)
                  }}>＋ Phase</button>
              </div>
              {activePhase != null && phaseList[activePhase] && (
                <div style={S.phaseRow} data-testid={`phase-row-${activePhase}`}>
                  <div style={S.phaseChips}>
                    {phaseList[activePhase].elements.length === 0
                      ? <span style={S.hint}>Click elements above to build this phase.</span>
                      : phaseList[activePhase].elements.map(s => (
                        <span key={s} style={S.selChip}
                          data-testid={`phase-${activePhase}-el-${s}`}>
                          <span style={{ ...S.selSym,
                            color: CAT_COLOR[ELEMENTS.find(e => e.sym === s)?.cat ?? 'tm'] }}>{s}</span>
                          <input
                            data-testid={`phase-${activePhase}-pct-${s}`}
                            style={S.pctInput}
                            value={phaseList[activePhase].percentages[s] ?? ''}
                            placeholder="%"
                            onChange={(e) => {
                              const phase = phaseList[activePhase]
                              const next = { ...phase.percentages }
                              const v = parseFloat(e.target.value)
                              if (e.target.value === '') delete next[s]
                              else if (!Number.isNaN(v)) next[s] = v
                              act('set_phase', { index: activePhase,
                                                 elements: phase.elements,
                                                 percentages: next })
                            }}
                          />
                        </span>
                      ))}
                  </div>
                  <button data-testid={`phase-${activePhase}-cif`} style={S.ghost}
                    onClick={async () => {
                      const path = await window.electron.pickFile(
                        { name: 'Crystal (.cif)', extensions: ['cif'] })
                      // A .cif knows its own elements; the backend fills them in
                      // when the phase has none.
                      if (path) act('set_phase_structure',
                        { index: activePhase, cif_path: path })
                    }}>Load CIF</button>
                  <button data-testid={`phase-${activePhase}-cod`} style={S.ghost}
                    disabled={phaseList[activePhase].elements.length === 0}
                    title={phaseList[activePhase].elements.length
                      ? `Search COD for ${phaseList[activePhase].elements.join('-')} structures`
                      : 'Give this phase some elements first'}
                    onClick={() => {
                      setSearching(activePhase); setResults([]); setNote('Searching COD…')
                      act('cod_search', { phase: activePhase })
                    }}>Search COD</button>
                  <span style={phaseList[activePhase].cifPath ? S.structure : S.noStructure}
                    data-testid={`phase-${activePhase}-structure`}
                    title={phaseList[activePhase].cifPath ?? undefined}>
                    {phaseList[activePhase].label
                      ?? (phaseList[activePhase].cifPath
                        ? base(phaseList[activePhase].cifPath as string) : 'no structure')}
                  </span>
                  <button data-testid={`phase-${activePhase}-remove`} style={S.x}
                    title="Remove this phase"
                    onClick={() => {
                      act('remove_phase', { index: activePhase })
                      setActivePhase(null); setSearching(null)
                    }}>✕</button>
                </div>
              )}
              {searching != null && searching === activePhase && (
                <div style={S.codBox} data-testid={`phase-${searching}-cod-results`}>
                  {note && <div style={S.note}>{note}</div>}
                  {results.map(r => (
                    <button key={r.id} data-testid={`cod-row-${r.id}`} style={S.codRow}
                      title={`COD ${r.id}`}
                      onClick={() => {
                        act('cod_pick', { phase: searching, cod_id: r.id,
                                          label: `${r.formula} ${r.sg}`.trim() })
                        setSearching(null)
                      }}>
                      <span style={S.codTop}>
                        <b style={S.formula}>{r.formula}</b>
                        {r.phase && <span style={S.phaseName}>{r.phase}</span>}
                        <span style={S.sg}>{r.sg}</span>
                      </span>
                      <span style={S.cellDims}>
                        a {fmt(r.a)} · b {fmt(r.b)} · c {fmt(r.c)} Å &nbsp;
                        α {fmt(r.alpha)} · β {fmt(r.beta)} · γ {fmt(r.gamma)}°
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          <div style={S.actions}>
            <button style={S.ghost} onClick={equalize} disabled={!sel.length}>Even %</button>
            <button style={S.ghost} onClick={() => { setSel([]); setPct({}) }}>Clear</button>
            <button data-testid="ptable-apply" style={S.apply}
              onClick={() => { onApply(sel, pct); onClose() }}>Apply</button>
          </div>
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
  },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  title: { color: '#cdd6f4', fontSize: 13, fontWeight: 600 },
  x: { background: 'none', border: 'none', color: '#f38ba8', cursor: 'pointer', fontSize: 14 },

  // ── phases ────────────────────────────────────────────────────────────────
  phases: {
    borderTop: '1px solid #313244', marginTop: 8, paddingTop: 8,
    display: 'flex', flexDirection: 'column', gap: 6,
  },
  phaseStrip: { display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 5 },
  phaseLabel: { fontSize: 10, color: '#a6adc8', marginRight: 2 },
  phaseBtn: {
    background: '#11111b', border: '1px solid #313244', color: '#cdd6f4',
    cursor: 'pointer', fontSize: 10, fontWeight: 600, padding: '2px 9px', borderRadius: 10,
  },
  phaseBtnOn: {
    background: '#89b4fa', border: '1px solid #89b4fa', color: '#11111b',
    cursor: 'pointer', fontSize: 10, fontWeight: 700, padding: '2px 9px', borderRadius: 10,
  },
  addPhase: {
    background: 'none', border: '1px dashed #585b70', color: '#cba6f7',
    cursor: 'pointer', fontSize: 10, fontWeight: 600, padding: '2px 9px', borderRadius: 10,
  },
  phaseRow: {
    display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap',
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 6, padding: '5px 8px',
  },
  phaseChips: { display: 'flex', flexWrap: 'wrap', gap: 4, flex: 1, minWidth: 120 },
  structure: { fontSize: 10, fontWeight: 600, color: '#a6e3a1', maxWidth: 190,
               overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  noStructure: { fontSize: 10, fontStyle: 'italic', color: '#6c7086' },
  codBox: { display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 200, overflowY: 'auto' },
  note: { fontSize: 10, color: '#a6adc8' },
  codRow: {
    display: 'flex', flexDirection: 'column', gap: 1, alignItems: 'flex-start',
    background: '#11111b', border: '1px solid #313244', borderRadius: 5,
    padding: '4px 7px', cursor: 'pointer', textAlign: 'left', width: '100%',
  },
  codTop: { display: 'flex', gap: 6, alignItems: 'baseline' },
  formula: { color: '#cdd6f4', fontSize: 11 },
  phaseName: { color: '#f9e2af', fontSize: 10 },
  sg: { color: '#89b4fa', fontSize: 10 },
  cellDims: { color: '#6c7086', fontSize: 9 },
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
  z: { fontSize: 6, opacity: 0.8 },
  sym: { fontSize: 11, fontWeight: 700 },
  footer: { marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 },
  selRow: { display: 'flex', flexWrap: 'wrap', gap: 6, minHeight: 26, alignItems: 'center' },
  hint: { color: '#6c7086', fontSize: 11 },
  selChip: {
    display: 'flex', alignItems: 'center', gap: 4, background: '#1e1e2e',
    border: '1px solid #313244', borderRadius: 12, padding: '2px 4px 2px 8px',
  },
  selSym: { fontSize: 12, fontWeight: 700 },
  pctInput: {
    width: 34, background: '#11111b', border: '1px solid #313244', borderRadius: 8,
    color: '#cdd6f4', fontSize: 10, padding: '2px 4px', textAlign: 'center',
  },
  actions: { display: 'flex', gap: 6, justifyContent: 'flex-end' },
  ghost: {
    background: 'none', border: '1px solid #313244', color: '#a6adc8',
    cursor: 'pointer', fontSize: 11, padding: '4px 10px', borderRadius: 6,
  },
  apply: {
    background: '#89b4fa', border: 'none', color: '#11111b', cursor: 'pointer',
    fontSize: 11, fontWeight: 700, padding: '4px 14px', borderRadius: 6,
  },
}

/** How a phase's structure reads where a phase is LISTED (the dock, the two
 *  orientation wizards) — green once it has one, greyed while it does not.
 *  Exported so those lists cannot drift apart; not in WizardShell's shared `S`,
 *  which is typed `Record<string, CSSProperties>` and so turns a misspelled key
 *  into a silently missing style. */
export const PHASE_STYLE: Record<'set' | 'unset', React.CSSProperties> = {
  set: { fontSize: 9, fontWeight: 600, color: '#a6e3a1', overflow: 'hidden',
         textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  unset: { fontSize: 9, fontStyle: 'italic', color: '#6c7086' },
}
