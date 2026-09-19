/**
 * SamplePhases.tsx — the sample's phases as the indexing wizards (Orientation
 * Mapping, Vector Orientation Mapping, EBSD) show them on their Load tab.
 *
 * The phases belong to the SAMPLE, not to a wizard: this is a list of them and
 * a button that opens the same popout the dock opens, so a structure chosen
 * here is the one the dock shows and every other wizard indexes with.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import type { SamplePhase } from '../kernel/SpyDEContext'
import { PeriodicTable, majorElements, structureName, structureTone } from './PeriodicTable'
import { S } from './WizardShell'

type SendAction = (action: string, payload?: Record<string, unknown>, windowId?: number) => void

export function useSamplePhases(windowId: number): SamplePhase[] {
  const { state } = useSpyDE()
  return state.composition.get(windowId)?.phases ?? []
}

/** The structure files of the phases that have one, in phase order. */
export const structuresOf = (phases: SamplePhase[]) =>
  phases.map(phase => phase.cifPath).filter((path): path is string => !!path)

/** A phase as a wizard names it: what its structure is made of. */
const phaseName = (phase: SamplePhase, index: number) =>
  majorElements(phase).join('-') || `phase ${index + 1}`

/** "Generating without Zr-O — no structure set." for the phases a library
 *  will leave out, or null when none are. */
export function missingStructures(phases: SamplePhase[]): string | null {
  const names = phases.flatMap((phase, index) => (phase.cifPath ? [] : [phaseName(phase, index)]))
  return names.length ? `Generating without ${names.join(', ')} — no structure set.` : null
}

export function SamplePhasesField({ windowId, sendAction, testidPrefix }: {
  windowId: number
  sendAction: SendAction
  /** `om` → `om-add-phase` / `om-cif-list`, and so on. */
  testidPrefix: string
}) {
  const phases = useSamplePhases(windowId)
  const [open, setOpen] = React.useState(false)
  return (
    <>
      <label style={S.lbl}>Sample phases</label>
      <button data-testid={`${testidPrefix}-add-phase`} style={S.primary}
        onClick={() => setOpen(true)}>
        {phases.length ? 'Edit phases' : '＋ Add phase'}
      </button>
      <div data-testid={`${testidPrefix}-cif-list`} style={S.cifList}>
        {phases.length === 0
          ? <span style={S.hint}>No phases yet — add at least one.</span>
          : phases.map((phase, index) => (
            <div key={phase.id} style={S.cifRow} title={phase.cifPath ?? phase.elements.join('-')}>
              <span style={S.cifName}>{phaseName(phase, index)}</span>
              <span style={{ fontSize: 9, ...structureTone(phase) }}>{structureName(phase)}</span>
            </div>
          ))}
      </div>
      {open && (
        <PeriodicTable windowId={windowId} phases={phases} sendAction={sendAction}
          onClose={() => setOpen(false)} />
      )}
    </>
  )
}
