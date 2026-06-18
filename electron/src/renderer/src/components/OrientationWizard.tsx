/**
 * OrientationWizard.tsx — the staged Orientation-Mapping caret (Qt 4-tab parity).
 *
 *   1 Load    — pick one or more .cif crystals (multi-phase) + accelerating voltage.
 *   2 Library — angle resolution + min intensity → "Generate Library"
 *               (`om_generate_library` builds the library + LIVE refine overlay).
 *   3 Refine  — gamma / min-intensity / normalize → `om_refine` (debounced); the
 *               matched template redraws under the crosshair.
 *   4 Run     — N best → "Compute Map" (`om_run`; reuses the cached library).
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Slider, Check, S } from './WizardShell'

const TABS = ['Load', 'Library', 'Refine', 'Run'] as const
type Tab = typeof TABS[number]

interface Props {
  openUp: boolean
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

export function OrientationWizard({ openUp, windowId, sendAction, onClose }: Props) {
  const [tab, setTab] = React.useState<Tab>('Load')
  const [cifs, setCifs] = React.useState<string[]>([])   // multi-phase: one per phase
  const [voltage, setVoltage] = React.useState(200)
  const [resolution, setResolution] = React.useState(1.0)
  const [minInt, setMinInt] = React.useState(0.0001)
  const [gamma, setGamma] = React.useState(0.5)
  const [refineMinInt, setRefineMinInt] = React.useState(0)
  const [normalize, setNormalize] = React.useState(false)
  const [nBest, setNBest] = React.useState(5)
  const [libReady, setLibReady] = React.useState(false)
  const [status, setStatus] = React.useState('Load a .cif crystal to begin.')

  const refineTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const base = (p: string) => p.split(/[/\\]/).pop() || p

  const pickCif = async () => {
    const path = await window.electron.pickFile({ name: 'Crystal (.cif)', extensions: ['cif'] })
    if (path) {
      setCifs(c => c.includes(path) ? c : [...c, path])
      setStatus('Crystal added — add more phases or generate the library.')
    }
  }
  const generate = () => {
    if (!cifs.length) { setStatus('Add a .cif first.'); return }
    setStatus('Generating library…')
    sendAction('om_generate_library', {
      cif_paths: cifs, accelerating_voltage: voltage, resolution, minimum_intensity: minInt,
    }, windowId)
    setLibReady(true)          // backend emits om_library_ready; optimistic unlock
    setTab('Refine')
  }

  // Debounced live refine — dispatch on slider settle so matches don't flood.
  const refine = (next: Partial<{ gamma: number; refineMinInt: number; normalize: boolean }>) => {
    const g = next.gamma ?? gamma, mi = next.refineMinInt ?? refineMinInt, nm = next.normalize ?? normalize
    if (refineTimer.current) clearTimeout(refineTimer.current)
    refineTimer.current = setTimeout(() => {
      sendAction('om_refine', { gamma: g, min_intensity: mi / 100, normalize_templates: nm }, windowId)
    }, 120)
  }
  const compute = () => {
    setStatus('Computing orientation map…')
    sendAction('om_run', { n_best: nBest, gamma, normalize_templates: normalize }, windowId)
  }

  return (
    <WizardShell testid="orientation-wizard" title="Orientation Mapping" openUp={openUp}
      onClose={onClose} closeTestid="om-close" status={status} statusTestid="om-status">
      <TabRow tabs={TABS} active={tab} onSelect={setTab} testid={(t) => `om-tab-${t}`}
        locked={(t) => (t === 'Refine' || t === 'Run') && !libReady} />

      {tab === 'Load' && (
        <div style={S.page}>
          <label style={S.lbl}>Crystal phases (.cif)</label>
          <button data-testid="om-pick-cif" style={S.fileBtn} onClick={pickCif}>＋ Add crystal (.cif)</button>
          <div data-testid="om-cif-list" style={S.cifList}>
            {cifs.length === 0
              ? <span style={S.hint}>No phases yet — add at least one.</span>
              : cifs.map(p => (
                <div key={p} style={S.cifRow} title={p}>
                  <span style={S.cifName}>{base(p)}</span>
                  <button data-testid={`om-cif-remove-${base(p)}`} style={S.close}
                    onClick={() => setCifs(c => c.filter(x => x !== p))}>✕</button>
                </div>
              ))}
          </div>
          <Field label="Voltage (kV)"><NumInput value={voltage} onChange={setVoltage} step="1" width={60} /></Field>
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
