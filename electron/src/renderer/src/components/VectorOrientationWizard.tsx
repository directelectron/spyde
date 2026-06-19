/**
 * VectorOrientationWizard.tsx — staged Vector-Orientation-Mapping caret.
 *
 * Fits orientation + STRAIN from the tree's diffraction vectors (sparse matcher):
 *   1 Load    — pick a .cif crystal + accelerating voltage.
 *   2 Library — angle resolution + min intensity → `vom_generate_library`.
 *   3 Refine  — strain cap + match tolerance sliders re-fit the pattern under the
 *               crosshair live (`vom_refine`); the fitted template (green) tracks
 *               the measured vectors (red) and the recovered strain/residual is
 *               shown — Qt "3 Refine" parity.
 *   4 Run     — strain cap + smoothing → `vom_run` (IPF-Z + εxx/εyy/εxy windows).
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Slider, Check, S } from './WizardShell'
import { useCifRecents, RecentCifs } from './CifRecents'
import { CodPicker } from './CodPicker'

const TABS = ['Load', 'Library', 'Refine', 'Run'] as const
type Tab = typeof TABS[number]

interface Props {
  openUp: boolean
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
  tab: Tab; cif: string; voltage: number; resolution: number; minInt: number
  strainCap: number; tolerance: number; smooth: boolean; libReady: boolean
}
const _vomStore = new Map<number, VomSaved>()

export function VectorOrientationWizard({ openUp, windowId, sendAction, onClose }: Props) {
  const saved = _vomStore.get(windowId)
  const [tab, setTab] = React.useState<Tab>(saved?.tab ?? 'Load')
  const [cif, setCif] = React.useState(saved?.cif ?? '')
  const [voltage, setVoltage] = React.useState(saved?.voltage ?? 200)
  const [resolution, setResolution] = React.useState(saved?.resolution ?? 1.0)
  const [minInt, setMinInt] = React.useState(saved?.minInt ?? 0.0001)
  const [strainCap, setStrainCap] = React.useState(saved?.strainCap ?? 5.0)   // %
  const [tolerance, setTolerance] = React.useState(saved?.tolerance ?? 4.0)   // % (sink bandwidth)
  const [smooth, setSmooth] = React.useState(saved?.smooth ?? true)
  const [libReady, setLibReady] = React.useState(saved?.libReady ?? false)
  const [fit, setFit] = React.useState<VomFit | null>(null)
  const [status, setStatus] = React.useState(
    saved?.libReady ? 'Library ready — move the crosshair to refine, or Compute Maps.'
                    : 'Load a .cif crystal to begin.')

  // Persist the state for this window on every change so reopening restores it.
  React.useEffect(() => {
    _vomStore.set(windowId, { tab, cif, voltage, resolution, minInt, strainCap, tolerance, smooth, libReady })
  }, [windowId, tab, cif, voltage, resolution, minInt, strainCap, tolerance, smooth, libReady])

  const refineTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const { recents, remember } = useCifRecents()

  // Live single-pattern fit readout streamed from the backend overlay.
  React.useEffect(() => {
    const onFit = (e: Event) => {
      const d = (e as CustomEvent).detail as Record<string, unknown>
      // Accept events with no window_id (null/undefined); only filter a mismatch.
      if (d.window_id != null && d.window_id !== windowId) return
      setFit(d as unknown as VomFit)
    }
    window.addEventListener('spyde:vom_fit', onFit)
    return () => window.removeEventListener('spyde:vom_fit', onFit)
  }, [windowId])

  const useCif = (path: string) => {
    setCif(path); remember(path); setStatus('Crystal loaded — generate the library.')
  }
  const pickCif = async () => {
    const path = await window.electron.pickFile({ name: 'Crystal (.cif)', extensions: ['cif'] })
    if (path) useCif(path)
  }
  const generate = () => {
    if (!cif) { setStatus('Choose a .cif first.'); return }
    setStatus('Generating library… (this can take ~1 min for a full library)')
    sendAction('vom_generate_library', {
      cif_path: cif, accelerating_voltage: voltage, resolution, minimum_intensity: minInt,
    }, windowId)
    setLibReady(true)
    setTab('Refine')
  }

  // Debounced live refine — strain cap & tolerance are sent as fractions.
  const refine = (next: Partial<{ strainCap: number; tolerance: number }>) => {
    const cap = next.strainCap ?? strainCap, tol = next.tolerance ?? tolerance
    if (refineTimer.current) clearTimeout(refineTimer.current)
    refineTimer.current = setTimeout(() => {
      sendAction('vom_refine', { strain_cap: cap / 100, sink_bw: tol / 100 }, windowId)
    }, 120)
  }
  const compute = () => {
    setStatus('Computing orientation + strain maps…')
    sendAction('vom_run', { strain_cap: strainCap / 100, sink_bw: tolerance / 100, smooth }, windowId)
  }

  const pct = (v?: number) => (v === undefined ? '—' : `${(v * 100).toFixed(2)}%`)

  return (
    <WizardShell testid="vector-orientation-wizard" title="Vector Orientation Mapping" openUp={openUp}
      onClose={onClose} closeTestid="vom-close" status={status} statusTestid="vom-status">
      <TabRow tabs={TABS} active={tab} onSelect={setTab} testid={(t) => `vom-tab-${t}`}
        locked={(t) => (t === 'Refine' || t === 'Run') && !libReady} />

      {tab === 'Load' && (
        <div style={S.page}>
          <label style={S.lbl}>Crystal (.cif)</label>
          <div style={{ display: 'flex', gap: 6, alignItems: 'stretch' }}>
            <button data-testid="vom-pick-cif" style={{ ...S.fileBtn, flex: 1, alignSelf: 'auto' }}
              title={cif} onClick={pickCif}>
              {cif ? cif.split(/[/\\]/).pop() : '＋ From file'}
            </button>
            <CodPicker windowId={windowId} sendAction={sendAction} onCif={useCif} />
          </div>
          <RecentCifs recents={recents} exclude={cif ? [cif] : []} onPick={useCif} />
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
          <div style={S.hint}>Move the crosshair on the navigator; the green template
            fits the red vectors. Tune the cap/tolerance to taste.</div>
          <Field label="Strain cap %">
            <Slider testid="vom-strain-cap" value={strainCap} min={0.5} max={10} step={0.1}
              onChange={(n) => { setStrainCap(n); refine({ strainCap: n }) }} />
          </Field>
          <Field label="Tolerance %">
            <Slider testid="vom-tolerance" value={tolerance} min={0.5} max={8} step={0.1}
              onChange={(n) => { setTolerance(n); refine({ tolerance: n }) }} />
          </Field>
          <div data-testid="vom-strain-readout" style={S.hint}>
            {fit && fit.ok
              ? `εxx=${pct(fit.exx)}  εyy=${pct(fit.eyy)}  εxy=${pct(fit.exy)}  ·  resid=${fit.residual?.toFixed(4)}  matched=${fit.matched}`
              : 'No fit yet — move the crosshair to a pattern with ≥4 vectors.'}
          </div>
        </div>
      )}

      {tab === 'Run' && (
        <div style={S.page}>
          <Field label="Strain cap %"><NumInput value={strainCap} onChange={setStrainCap} step="0.5" /></Field>
          <Check testid="vom-smooth" checked={smooth} onChange={setSmooth} label="Smooth strain (TV)" />
          <button data-testid="vom-compute" style={S.primary} onClick={compute}>Compute Maps</button>
        </div>
      )}
    </WizardShell>
  )
}
