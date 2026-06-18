/**
 * VectorOrientationWizard.tsx — staged Vector-Orientation-Mapping caret.
 *
 * Fits orientation + STRAIN from the tree's diffraction vectors (sparse matcher):
 *   1 Load    — pick a .cif crystal + accelerating voltage.
 *   2 Library — angle resolution + min intensity → `vom_generate_library`.
 *   3 Run     — strain cap + smoothing → `vom_run` (IPF-Z + εxx/εyy/εxy windows).
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Check, S } from './WizardShell'

const TABS = ['Load', 'Library', 'Run'] as const
type Tab = typeof TABS[number]

interface Props {
  openUp: boolean
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

export function VectorOrientationWizard({ openUp, windowId, sendAction, onClose }: Props) {
  const [tab, setTab] = React.useState<Tab>('Load')
  const [cif, setCif] = React.useState('')
  const [voltage, setVoltage] = React.useState(200)
  const [resolution, setResolution] = React.useState(1.0)
  const [minInt, setMinInt] = React.useState(0.0001)
  const [strainCap, setStrainCap] = React.useState(0.05)
  const [smooth, setSmooth] = React.useState(false)
  const [libReady, setLibReady] = React.useState(false)
  const [status, setStatus] = React.useState('Load a .cif crystal to begin.')

  const pickCif = async () => {
    const path = await window.electron.pickFile({ name: 'Crystal (.cif)', extensions: ['cif'] })
    if (path) { setCif(path); setStatus('Crystal loaded — generate the library.') }
  }
  const generate = () => {
    if (!cif) { setStatus('Choose a .cif first.'); return }
    setStatus('Generating library…')
    sendAction('vom_generate_library', {
      cif_path: cif, accelerating_voltage: voltage, resolution, minimum_intensity: minInt,
    }, windowId)
    setLibReady(true)
    setTab('Run')
  }
  const compute = () => {
    setStatus('Computing orientation + strain maps…')
    sendAction('vom_run', { strain_cap: strainCap, smooth }, windowId)
  }

  return (
    <WizardShell testid="vector-orientation-wizard" title="Vector Orientation Mapping" openUp={openUp}
      onClose={onClose} closeTestid="vom-close" status={status} statusTestid="vom-status">
      <TabRow tabs={TABS} active={tab} onSelect={setTab} testid={(t) => `vom-tab-${t}`}
        locked={(t) => t === 'Run' && !libReady} />

      {tab === 'Load' && (
        <div style={S.page}>
          <label style={S.lbl}>Crystal (.cif)</label>
          <button data-testid="vom-pick-cif" style={S.fileBtn} title={cif} onClick={pickCif}>
            {cif ? cif.split(/[/\\]/).pop() : 'Choose…'}
          </button>
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

      {tab === 'Run' && (
        <div style={S.page}>
          <Field label="Strain cap"><NumInput value={strainCap} onChange={setStrainCap} step="0.005" /></Field>
          <Check testid="vom-smooth" checked={smooth} onChange={setSmooth} label="Smooth strain (TV)" />
          <button data-testid="vom-compute" style={S.primary} onClick={compute}>Compute Maps</button>
        </div>
      )}
    </WizardShell>
  )
}
