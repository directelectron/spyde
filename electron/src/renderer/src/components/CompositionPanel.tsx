/**
 * CompositionPanel.tsx — the right-dock "Composition" section: what the sample
 * is made of, PHASE by phase.
 *
 * A sample is not one element list. A Zr-Nb alloy with Nb platelets in it is
 * α-Zr AND β-Nb, and writing that flat ("Zr, Nb") says something different — it
 * says one compound of both, which is how the COD search came to ask for a
 * Cu-Nb structure and get nothing back. So the chips are clustered into phases
 * with `&` between them, and each phase shows the structure that indexes it
 * once one is chosen.
 *
 * Editing happens in ONE popout (PeriodicTable): clicking elements builds the
 * sample, and selecting a phase first makes those clicks build that phase, so a
 * phase is a SUBSET of the composition rather than a second thing to type in.
 *
 * The flat `Sample.elements` / `Sample.composition` metadata is unchanged
 * underneath — it is the union across phases, plus any element that belongs to
 * the sample without belonging to one, and it is what EELS edge suggestion and
 * EDS quantification read.
 */
import React from 'react'
import type { Composition, SamplePhase } from '../kernel/SpyDEContext'
import { PeriodicTable } from './PeriodicTable'

interface Props {
  activeId: number | null
  composition?: Composition
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}

const base = (p: string) => p.split(/[/\\]/).pop() || p

export function CompositionPanel({ activeId, composition, sendAction }: Props) {
  const [editing, setEditing] = React.useState(false)
  const elements = composition?.elements ?? []
  const phases: SamplePhase[] = composition?.phases?.length
    ? composition.phases
    // A signal whose composition predates phases still has one: itself.
    : (elements.length
      ? [{ elements, percentages: composition?.percentages ?? {},
           cifPath: null, label: null, codId: null }]
      : [])
  const pct = composition?.percentages ?? {}
  const claimed = new Set(phases.flatMap(p => p.elements))
  const unassigned = elements.filter(el => !claimed.has(el))

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
        {/* One door. Elements and phases are edited in the same popout, because
            a phase is a SUBSET of the composition — telling them to two
            different modals is what made a two-phase sample impossible to
            describe in the first place. */}
        <button data-testid="composition-edit" style={S.editBtn}
          onClick={() => setEditing(true)}>
          {phases.length ? 'Edit' : '＋ Elements and Phase'}
        </button>
      </div>

      {phases.map((phase, index) => (
        <div key={index} style={S.phaseRow} data-testid={`composition-phase-${index}`}>
          {index > 0 && <span style={S.amp}>&amp;</span>}
          <div style={S.chips} data-testid={index === 0 ? 'composition-chips' : undefined}>
            {phase.elements.length === 0
              ? <span style={S.empty}>no elements</span>
              : phase.elements.map(el => (
                <span key={el} style={S.chip} data-testid={`composition-chip-${el}`}>
                  <span style={S.sym}>{el}</span>
                  {phase.percentages[el] != null && <span style={S.pct}>{phase.percentages[el]}%</span>}
                </span>
              ))}
          </div>
          {/* The structure, once it is known — the other half of what a phase
              IS, and previously visible nowhere outside the wizard. */}
          {(phase.label || phase.cifPath) && (
            <span style={S.structure} title={phase.cifPath ?? undefined}
              data-testid={`composition-structure-${index}`}>
              {phase.label ?? base(phase.cifPath as string)}
            </span>
          )}
        </div>
      ))}

      {/* Elements the sample has that no phase claims — the extra oxygen. They
          are real composition and would otherwise be recorded but invisible. */}
      {unassigned.length > 0 && (
        <div style={S.phaseRow} data-testid="composition-unassigned">
          <span style={S.amp} title="in the sample, in no phase">+</span>
          <div style={S.chips}>
            {unassigned.map(el => (
              <span key={el} style={S.chip} data-testid={`composition-extra-${el}`}>
                <span style={S.sym}>{el}</span>
                {pct[el] != null && <span style={S.pct}>{pct[el]}%</span>}
              </span>
            ))}
          </div>
        </div>
      )}

      {editing && (
        <PeriodicTable
          // The SAMPLE's elements — the union across phases plus anything that
          // belongs to the sample without belonging to a phase (the extra
          // oxygen that is in neither structure you are indexing against).
          initial={elements}
          initialPct={composition?.percentages ?? {}}
          phases={phases}
          windowId={activeId}
          sendAction={sendAction}
          onApply={(els, percentages) => {
            if (activeId != null) {
              sendAction('set_composition', { elements: els, percentages }, activeId)
            }
            setEditing(false)
          }}
          onClose={() => setEditing(false)}
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
  head: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 4, marginBottom: 4 },
  label: { fontSize: 10, color: '#a6adc8', flex: 1 },
  editBtn: {
    background: 'none', border: '1px solid #313244', color: '#89b4fa', cursor: 'pointer',
    fontSize: 10, fontWeight: 600, padding: '1px 8px', borderRadius: 4,
  },
  empty: { fontSize: 10, color: '#6c7086' },
  phaseRow: { display: 'flex', alignItems: 'center', gap: 4, marginTop: 2 },
  amp: { fontSize: 11, fontWeight: 700, color: '#cba6f7' },
  chips: { display: 'flex', flexWrap: 'wrap', gap: 4 },
  chip: {
    display: 'flex', alignItems: 'center', gap: 3, background: '#1e1e2e',
    border: '1px solid #313244', borderRadius: 12, padding: '2px 8px',
  },
  sym: { fontSize: 11, fontWeight: 700, color: '#cdd6f4' },
  pct: { fontSize: 9, color: '#a6adc8' },
  structure: {
    flex: 1, textAlign: 'right', fontSize: 9, fontWeight: 600, color: '#a6e3a1',
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
}
