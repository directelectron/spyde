/**
 * PeriodicTable.tsx — a click-to-pick periodic table popout for choosing the
 * sample's elements (and optional atomic %). Toggling elements + Apply writes
 * them to the HyperSpy metadata via `set_composition`; the chosen composition
 * also drives the COD "easy CIF" structure search.
 */
import React from 'react'

interface El { z: number; sym: string; row: number; col: number; cat: Cat }
type Cat = 'alkali' | 'alkaline' | 'tm' | 'post' | 'metalloid' | 'nonmetal'
  | 'halogen' | 'noble' | 'lanth' | 'act'

// One entry per element with its (row, col) on the standard grid. The f-block
// sits in rows 8/9 (cols 3–17); period 6/7 group 3 is the gap.
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
  // f-block (rows 8/9, cols 3–17)
  ...['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    .map((s, i) => E(57 + i, s, 8, 3 + i, 'lanth')),
  ...['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    .map((s, i) => E(89 + i, s, 9, 3 + i, 'act')),
]

const CAT_COLOR: Record<Cat, string> = {
  alkali: '#f38ba8', alkaline: '#fab387', tm: '#89b4fa', post: '#94e2d5',
  metalloid: '#a6e3a1', nonmetal: '#f9e2af', halogen: '#cba6f7', noble: '#74c7ec',
  lanth: '#f5c2e7', act: '#eba0ac',
}

interface Props {
  initial: string[]
  initialPct?: Record<string, number>
  onApply: (elements: string[], percentages: Record<string, number>) => void
  onClose: () => void
}

export function PeriodicTable({ initial, initialPct = {}, onApply, onClose }: Props) {
  const [sel, setSel] = React.useState<string[]>(initial)
  const [pct, setPct] = React.useState<Record<string, number>>(initialPct)

  const toggle = (sym: string) => setSel(prev =>
    prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym])

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

  return (
    <div style={S.backdrop} data-testid="periodic-table" onClick={onClose}>
      <div style={S.modal} onClick={e => e.stopPropagation()}>
        <div style={S.header}>
          <span style={S.title}>Sample composition — choose elements</span>
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
          <div style={S.actions}>
            <button style={S.ghost} onClick={equalize} disabled={!sel.length}>Even %</button>
            <button style={S.ghost} onClick={() => { setSel([]); setPct({}) }}>Clear</button>
            <button data-testid="ptable-apply" style={S.apply}
              onClick={() => { onApply(sel, pct); onClose() }}>Apply</button>
          </div>
        </div>
      </div>
    </div>
  )
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
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(18, 30px)',
    gridTemplateRows: 'repeat(7, 30px) 8px repeat(2, 30px)',
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
