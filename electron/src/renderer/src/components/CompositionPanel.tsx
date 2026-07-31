/**
 * CompositionPanel.tsx — the right-dock "Composition" section: the sample's
 * elements (+ optional atomic %) stored in the HyperSpy metadata
 * (`Sample.elements` / `Sample.composition`). Click "Elements" to pick from a
 * periodic table popout. The composition also seeds the COD CIF search in the
 * orientation wizard.
 */
import React from 'react'
import type { Composition } from '../kernel/SpyDEContext'
import { PeriodicTable } from './PeriodicTable'

interface Props {
  activeId: number | null
  composition?: Composition
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}

export function CompositionPanel({ activeId, composition, sendAction }: Props) {
  const [open, setOpen] = React.useState(false)
  const elements = composition?.elements ?? []
  const pct = composition?.percentages ?? {}

  const apply = (els: string[], percentages: Record<string, number>) => {
    if (activeId == null) return
    sendAction('set_composition', { elements: els, percentages }, activeId)
  }

  return (
    <div style={S.section} data-testid="composition-section">
      <div style={S.head}>
        <span style={S.label}>Composition</span>
        {/* The empty state rides ON the header row rather than costing a line of
            its own — it says nothing an inline note can't. */}
        {elements.length === 0 && (
          <span style={{ ...S.empty, flex: 1, marginLeft: 6 }}
            data-testid="composition-empty">No elements set</span>
        )}
        <button data-testid="composition-edit" style={S.editBtn} onClick={() => setOpen(true)}>
          {elements.length ? 'Edit' : '＋ Elements'}
        </button>
      </div>

      {elements.length > 0 && (
        <div style={S.chips} data-testid="composition-chips">
          {elements.map(el => (
            <span key={el} style={S.chip} data-testid={`composition-chip-${el}`}>
              <span style={S.sym}>{el}</span>
              {pct[el] != null && <span style={S.pct}>{pct[el]}%</span>}
            </span>
          ))}
        </div>
      )}

      {open && (
        <PeriodicTable
          initial={elements}
          initialPct={pct}
          onApply={apply}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  // Metrics track PlotControlDock's `section`/`label` — this panel renders inline
  // among that dock's sections, so it has to compress with them.
  section: {
    padding: '6px 10px', borderTop: '1px solid #1e1e2e', flexShrink: 0,
  },
  head: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 },
  label: { fontSize: 10, color: '#a6adc8' },
  editBtn: {
    background: 'none', border: '1px solid #313244', color: '#89b4fa', cursor: 'pointer',
    fontSize: 10, fontWeight: 600, padding: '1px 8px', borderRadius: 4,
  },
  empty: { fontSize: 10, color: '#6c7086' },
  chips: { display: 'flex', flexWrap: 'wrap', gap: 4 },
  chip: {
    display: 'flex', alignItems: 'center', gap: 3, background: '#1e1e2e',
    border: '1px solid #313244', borderRadius: 12, padding: '2px 8px',
  },
  sym: { fontSize: 11, fontWeight: 700, color: '#cdd6f4' },
  pct: { fontSize: 9, color: '#a6adc8' },
}
