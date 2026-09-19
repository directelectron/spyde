/**
 * VectorOrientationWizard.tsx — staged Vector-Orientation-Mapping caret.
 *
 * Fits orientation + STRAIN from the tree's diffraction vectors (sparse matcher):
 *   1 Load    — the SAMPLE's phases (composition + structure, shared with the
 *               dock) + accelerating voltage.
 *   2 Library — angle resolution + min intensity → `vom_generate_library`.
 *   3 Refine  — the matcher's pairing distance + excitation width re-fit the
 *               pattern under the crosshair live (`vom_refine`); the matched
 *               pattern (green) tracks the measured vectors (red) and the
 *               recovered strain/residual is shown. Both are Å⁻¹ on the slider
 *               and Å⁻¹ on the wire, as the schema declares them.
 *   4 Run     — fit every position with Refine's settings → `vom_run` (IPF-Z +
 *               εxx/εyy/εxy windows). No strain cap: the matcher solves the
 *               deformation in closed form and has nothing to bound.
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Slider, Check, S } from './WizardShell'
import { useDebouncedAction, useWizardEvent } from './wizardHooks'
import { SamplePhasesField, useSamplePhases, structuresOf, missingStructures } from './SamplePhases'

const TABS = ['Load', 'Library', 'Refine', 'Run'] as const
type Tab = typeof TABS[number]

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

interface VomFit {
  ok: boolean; exx?: number; eyy?: number; exy?: number
  residual?: number; friedel?: number | null; matched?: number
}

// Per-window wizard state, kept OUTSIDE the component so it survives the
// caret unmounting when you "step away" from the plot — the backend keeps the
// built template library on the tree, so we must NOT make the user regenerate it
// (a ~1 min rebuild) just because the React caret was torn down and remounted.
interface VomSaved {
  tab: Tab; voltage: number; resolution: number; minInt: number
  pairDistance: number; sigmaExcitation: number
  smooth: boolean; libReady: boolean
}
const _vomStore = new Map<number, VomSaved>()

export function VectorOrientationWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const saved = _vomStore.get(windowId)
  const [tab, setTab] = React.useState<Tab>(saved?.tab ?? 'Load')
  const phases = useSamplePhases(windowId)
  const [voltage, setVoltage] = React.useState(saved?.voltage ?? 200)
  const [resolution, setResolution] = React.useState(saved?.resolution ?? 1.0)
  const [minInt, setMinInt] = React.useState(saved?.minInt ?? 0.0001)
  const [pairDistance, setPairDistance] = React.useState(saved?.pairDistance ?? 0.05)        // Å⁻¹
  const [sigmaExcitation, setSigmaExcitation] = React.useState(saved?.sigmaExcitation ?? 0.04)  // Å⁻¹
  const [smooth, setSmooth] = React.useState(saved?.smooth ?? true)
  const [libReady, setLibReady] = React.useState(saved?.libReady ?? false)
  const [fit, setFit] = React.useState<VomFit | null>(null)
  const [status, setStatus] = React.useState(
    saved?.libReady ? 'Library ready — move the crosshair to refine, or Compute Maps.'
                    : 'Add a phase to begin.')

  // Persist the state for this window on every change so reopening restores it.
  React.useEffect(() => {
    _vomStore.set(windowId, { tab, voltage, resolution, minInt,
      pairDistance, sigmaExcitation, smooth, libReady })
  }, [windowId, tab, voltage, resolution, minInt, pairDistance,
      sigmaExcitation, smooth, libReady])

  // Debounced live refine — a pending refine is cancelled on unmount so
  // vom_refine can't fire at a torn-down preview mid-debounce.
  const sendRefine = useDebouncedAction(sendAction, 'vom_refine', windowId)

  // The library builds off-thread; its reply is what ends "Generating library…".
  useWizardEvent('spyde:vom_library_ready', windowId, (detail) => {
    if (detail.ok) {
      setStatus(`Library ready (${Number(detail.n_templates ?? 0).toLocaleString()} templates) — `
        + 'move the crosshair to refine, or Compute Maps.')
    } else {
      setLibReady(false)
      setTab('Library')
      setStatus(`Library failed: ${String(detail.error ?? 'unknown error')}`)
    }
  })

  // Live single-pattern fit readout streamed from the backend overlay.
  useWizardEvent('spyde:vom_fit', windowId, (d) => {
    setFit(d as unknown as VomFit)
  })

  const generate = () => {
    const paths = structuresOf(phases)
    if (!paths.length) { setStatus('Give at least one phase a structure first.'); return }
    // A phase with no structure contributes no templates; say which.
    setStatus(missingStructures(phases)
      ?? 'Generating library… (this can take ~1 min for a full library)')
    sendAction('vom_generate_library', {
      cif_paths: paths, accelerating_voltage: voltage, resolution, minimum_intensity: minInt,
    }, windowId)
    setLibReady(true)
    setTab('Refine')
  }

  // Debounced live refine. Both knobs are Å⁻¹ on the slider and Å⁻¹ on the
  // wire — the schema declares those units, so any host reading it sends the
  // same number. (The sliders this replaced showed percentages and dispatched
  // fractions, which only agreed because this caret did the division itself.)
  const refine = (next: Partial<{ pairDistance: number; sigmaExcitation: number }>) => {
    sendRefine(() => ({
      pair_distance: next.pairDistance ?? pairDistance,
      sigma_excitation: next.sigmaExcitation ?? sigmaExcitation,
    }))
  }
  const compute = () => {
    setStatus('Computing orientation + strain maps…')
    sendAction('vom_run', { pair_distance: pairDistance, sigma_excitation: sigmaExcitation, smooth }, windowId)
  }

  const pct = (v?: number) => (v === undefined ? '—' : `${(v * 100).toFixed(2)}%`)

  return (
    <WizardShell testid="vector-orientation-wizard" title="Vector Orientation Mapping" posStyle={caretPos}
      onClose={onClose} closeTestid="vom-close" status={status} statusTestid="vom-status">
      <TabRow tabs={TABS} active={tab} onSelect={setTab} testid={(t) => `vom-tab-${t}`}
        locked={(t) => (t === 'Refine' || t === 'Run') && !libReady} />

      {tab === 'Load' && (
        <div style={S.page}>
          <SamplePhasesField windowId={windowId} sendAction={sendAction} testidPrefix="vom" />
          <Field label="Voltage (kV)"><NumInput value={voltage} onChange={setVoltage} step="1" width={60} /></Field>
        </div>
      )}

      {tab === 'Library' && (
        <div style={S.page}>
          <Field label="Angle res (°)"><NumInput value={resolution} onChange={setResolution} step="0.1" width={60} /></Field>
          <Field label="Min intensity"><NumInput value={minInt} onChange={setMinInt} step="0.0001" width={74} /></Field>
          <button data-testid="vom-generate" style={S.primary} onClick={generate}>Generate Library</button>
        </div>
      )}

      {tab === 'Refine' && (
        <div style={S.page}>
          <div style={S.hint}>Move the crosshair on the navigator; the green
            matched pattern fits the red vectors.</div>
          <Field label="Pair distance (Å⁻¹)">
            <Slider testid="vom-pair-distance" value={pairDistance}
              min={0.01} max={0.15} step={0.005}
              onChange={(n) => { setPairDistance(n); refine({ pairDistance: n }) }} />
          </Field>
          <Field label="Excitation σ (Å⁻¹)">
            <Slider testid="vom-sigma-excitation" value={sigmaExcitation}
              min={0.01} max={0.12} step={0.005}
              onChange={(n) => { setSigmaExcitation(n); refine({ sigmaExcitation: n }) }} />
          </Field>
          <div style={S.hint}>Pair distance is how near a simulated reflection
            has to be to count as the same peak; too small and real peaks go
            unpaired, too large and one spot claims its neighbours. Excitation σ
            is how far off the Ewald sphere a reflection may sit and still be
            treated as excited. Defaults 0.05 and 0.04 Å⁻¹.</div>
          <div data-testid="vom-strain-readout" style={S.hint}>
            {fit && fit.ok
              ? `εxx=${pct(fit.exx)}  εyy=${pct(fit.eyy)}  εxy=${pct(fit.exy)}  ·  resid=${fit.residual?.toFixed(4)}  matched=${fit.matched}`
              : 'No fit yet — move the crosshair to a pattern with ≥5 vectors.'}
          </div>
        </div>
      )}

      {tab === 'Run' && (
        <div style={S.page}>
          <div style={S.hint}>Fits every position with the settings from Refine,
            then opens the orientation map and the strain maps.</div>
          <Check testid="vom-smooth" checked={smooth} onChange={setSmooth} label="Smooth strain (TV)" />
          <button data-testid="vom-compute" style={S.primary} onClick={compute}>Compute Maps</button>
        </div>
      )}
    </WizardShell>
  )
}
