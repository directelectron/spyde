/**
 * OrientationWizard.tsx — the staged Orientation-Mapping caret (Qt 4-tab parity).
 *
 *   1 Load    — the SAMPLE's phases (composition + structure, shared with the
 *               dock and the vector wizard) + accelerating voltage.
 *   2 Library — angle resolution + min intensity → "Generate Library"
 *               (`om_generate_library` builds the library + LIVE refine overlay).
 *   3 Refine  — gamma / min-intensity / normalize → `om_refine` (debounced); the
 *               matched template redraws under the crosshair.
 *   4 Run     — N best → "Compute Map" (`om_run`; reuses the cached library).
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Slider, Check, S } from './WizardShell'
import { useDebouncedAction } from './wizardHooks'
import { PeriodicTable, PHASE_STYLE } from './PeriodicTable'
import { useSpyDE } from '../kernel/SpyDEContext'

const TABS = ['Load', 'Library', 'Refine', 'Run'] as const
type Tab = typeof TABS[number]

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

// Per-window wizard state kept OUTSIDE the component so the built library isn't
// "lost" (forcing a ~1 min regenerate) when you step away and the caret unmounts.
interface OmSaved {
  tab: Tab; voltage: number; resolution: number; minInt: number
  gamma: number; refineMinInt: number; normalize: boolean; nBest: number; libReady: boolean
}
const _omStore = new Map<number, OmSaved>()

export function OrientationWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const saved = _omStore.get(windowId)
  const [tab, setTab] = React.useState<Tab>(saved?.tab ?? 'Load')
  // The phases come from the SAMPLE, not this caret: composition and structure
  // are one thing, so the dock and both orientation wizards read one list.
  const { state } = useSpyDE()
  const composition = state.composition.get(windowId)
  const phases = composition?.phases ?? []
  const [phasesOpen, setPhasesOpen] = React.useState(false)
  const [voltage, setVoltage] = React.useState(saved?.voltage ?? 200)
  const [resolution, setResolution] = React.useState(saved?.resolution ?? 1.0)
  const [minInt, setMinInt] = React.useState(saved?.minInt ?? 0.0001)
  const [gamma, setGamma] = React.useState(saved?.gamma ?? 1.0)
  const [refineMinInt, setRefineMinInt] = React.useState(saved?.refineMinInt ?? 0)
  const [normalize, setNormalize] = React.useState(saved?.normalize ?? false)
  const [nBest, setNBest] = React.useState(saved?.nBest ?? 5)
  const [libReady, setLibReady] = React.useState(saved?.libReady ?? false)
  const [status, setStatus] = React.useState(
    saved?.libReady ? 'Library ready — move the crosshair to refine, or Compute Map.'
                    : 'Add a phase to begin.')

  React.useEffect(() => {
    _omStore.set(windowId, { tab, voltage, resolution, minInt, gamma, refineMinInt, normalize, nBest, libReady })
  }, [windowId, tab, voltage, resolution, minInt, gamma, refineMinInt, normalize, nBest, libReady])

  // Debounced live refine — a pending refine is cancelled on unmount so
  // om_refine can't fire at a torn-down preview mid-debounce.
  const sendRefine = useDebouncedAction(sendAction, 'om_refine', windowId)
  const base = (p: string) => p.split(/[/\\]/).pop() || p

  const generate = () => {
    // A phase with no structure yet contributes no templates; name it rather
    // than silently building a library that is missing it.
    const missing = phases.filter(p => !p.cifPath)
    const paths = phases.map(p => p.cifPath).filter(Boolean) as string[]
    if (!paths.length) { setStatus('Give at least one phase a structure first.'); return }
    setStatus(missing.length
      ? `Generating without ${missing.map(p => p.elements.join('-') || 'a phase').join(', ')} — no structure set.`
      : 'Generating library…')
    sendAction('om_generate_library', {
      cif_paths: paths, accelerating_voltage: voltage, resolution, minimum_intensity: minInt,
    }, windowId)
    setLibReady(true)          // backend emits om_library_ready; optimistic unlock
    setTab('Refine')
  }

  // Debounced live refine — dispatch on slider settle so matches don't flood.
  const refine = (next: Partial<{ gamma: number; refineMinInt: number; normalize: boolean }>) => {
    const g = next.gamma ?? gamma, mi = next.refineMinInt ?? refineMinInt, nm = next.normalize ?? normalize
    sendRefine(() => ({ gamma: g, min_intensity: mi / 100, normalize_templates: nm }))
  }
  const compute = () => {
    setStatus('Computing orientation map…')
    sendAction('om_run', { n_best: nBest, gamma, normalize_templates: normalize }, windowId)
  }

  return (
    <WizardShell testid="orientation-wizard" title="Orientation Mapping" posStyle={caretPos}
      onClose={onClose} closeTestid="om-close" status={status} statusTestid="om-status">
      <TabRow tabs={TABS} active={tab} onSelect={setTab} testid={(t) => `om-tab-${t}`}
        locked={(t) => (t === 'Refine' || t === 'Run') && !libReady} />

      {tab === 'Load' && (
        <div style={S.page}>
          <label style={S.lbl}>Sample phases</label>
          {/* A door onto the sample's phases, shared with the dock and the
              vector wizard — not a private .cif list nothing else can see. */}
          <button data-testid="om-add-phase" style={S.primary}
            onClick={() => {
              // Clicking "Add phase" with none yet should land on a usable
              // row, not on an empty editor with a second Add phase in it.
              if (!phases.length) sendAction('add_phase', {}, windowId)
              setPhasesOpen(true)
            }}>{phases.length ? 'Phases' : '＋ Add phase'}</button>
          <div data-testid="om-cif-list" style={S.cifList}>
            {phases.length === 0
              ? <span style={S.hint}>No phases yet — add at least one.</span>
              : phases.map((phase, index) => (
                <div key={index} style={S.cifRow}
                  title={phase.cifPath ?? phase.elements.join('-')}>
                  <span style={S.cifName}>
                    {phase.elements.join('-') || `phase ${index + 1}`}
                  </span>
                  <span style={phase.cifPath ? PHASE_STYLE.set : PHASE_STYLE.unset}>
                    {phase.label ?? (phase.cifPath ? base(phase.cifPath) : 'no structure')}
                  </span>
                </div>
              ))}
          </div>
          <Field label="Voltage (kV)"><NumInput value={voltage} onChange={setVoltage} step="1" width={60} /></Field>
          {phasesOpen && (
            // The SAME popout the dock opens: a phase's elements and its
            // structure belong together, and the sample owns both.
            <PeriodicTable
              initial={composition?.elements ?? []}
              initialPct={composition?.percentages ?? {}}
              phases={phases}
              windowId={windowId}
              sendAction={sendAction}
              onApply={(els, percentages) => {
                sendAction('set_composition', { elements: els, percentages }, windowId)
                setPhasesOpen(false)
              }}
              onClose={() => setPhasesOpen(false)}
            />
          )}
        </div>
      )}

      {tab === 'Library' && (
        <div style={S.page}>
          <Field label="Angle res (°)"><NumInput value={resolution} onChange={setResolution} step="0.1" width={60} /></Field>
          <Field label="Min intensity"><NumInput value={minInt} onChange={setMinInt} step="0.0001" width={74} /></Field>
          <button data-testid="om-generate" style={S.primary} onClick={generate}>Generate Library</button>
        </div>
      )}

      {tab === 'Refine' && (
        <div style={S.page}>
          <div style={S.hint}>Move the crosshair on the navigator to preview the match.</div>
          <Field label="Gamma">
            <Slider testid="om-gamma" value={gamma} min={0.1} max={1.5} step={0.05}
              onChange={(n) => { setGamma(n); refine({ gamma: n }) }} />
          </Field>
          <Field label="Min int %">
            <Slider testid="om-minint" value={refineMinInt} min={0} max={100} step={1}
              onChange={(n) => { setRefineMinInt(n); refine({ refineMinInt: n }) }} />
          </Field>
          <Check testid="om-normalize" checked={normalize} label="Normalize templates"
            onChange={(b) => { setNormalize(b); refine({ normalize: b }) }} />
        </div>
      )}

      {tab === 'Run' && (
        <div style={S.page}>
          <Field label="N best"><NumInput value={nBest} onChange={setNBest} step="1" width={56} /></Field>
          <button data-testid="om-compute" style={S.primary} onClick={compute}>Compute Map</button>
        </div>
      )}
    </WizardShell>
  )
}
