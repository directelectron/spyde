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
import { useDebouncedAction } from './wizardHooks'
import { useCifRecents, RecentCifs } from './CifRecents'
import { CodPicker } from './CodPicker'

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
  tab: Tab; cifs: string[]; voltage: number; resolution: number; minInt: number
  gamma: number; refineMinInt: number; normalize: boolean; nBest: number; libReady: boolean
}
const _omStore = new Map<number, OmSaved>()

export function OrientationWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const saved = _omStore.get(windowId)
  const [tab, setTab] = React.useState<Tab>(saved?.tab ?? 'Load')
  const [cifs, setCifs] = React.useState<string[]>(saved?.cifs ?? [])   // multi-phase: one per phase
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
                    : 'Load a .cif crystal to begin.')

  React.useEffect(() => {
    _omStore.set(windowId, { tab, cifs, voltage, resolution, minInt, gamma, refineMinInt, normalize, nBest, libReady })
  }, [windowId, tab, cifs, voltage, resolution, minInt, gamma, refineMinInt, normalize, nBest, libReady])

  // Debounced live refine — a pending refine is cancelled on unmount so
  // om_refine can't fire at a torn-down preview mid-debounce.
  const sendRefine = useDebouncedAction(sendAction, 'om_refine', windowId)
  const { recents, remember } = useCifRecents()
  const base = (p: string) => p.split(/[/\\]/).pop() || p

  const addCif = (path: string) => {
    setCifs(c => c.includes(path) ? c : [...c, path])
    remember(path)
    setStatus('Crystal added — add more phases or generate the library.')
  }
  const pickCif = async () => {
    const path = await window.electron.pickFile({ name: 'Crystal (.cif)', extensions: ['cif'] })
    if (path) addCif(path)
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
          <label style={S.lbl}>Crystal phases (.cif)</label>
          <div style={{ display: 'flex', gap: 6, alignItems: 'stretch' }}>
            <button data-testid="om-pick-cif" style={{ ...S.fileBtn, flex: 1, alignSelf: 'auto' }}
              onClick={pickCif}>＋ From file</button>
            <CodPicker windowId={windowId} sendAction={sendAction} onCif={addCif} />
          </div>
          <RecentCifs recents={recents} exclude={cifs} onPick={addCif} />
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
