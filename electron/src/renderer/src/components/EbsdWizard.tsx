/**
 * EbsdWizard.tsx — the staged EBSD-Indexing caret.
 *
 * Deliberately the same four stages as OrientationWizard.tsx, because it is the
 * same job on a different signal — pick a crystal, build a library, check the
 * match under the crosshair, run the field:
 *
 *   1 Load     — .cif (or a space group) + voltage + the detector's projection
 *                centre. The PC defaults to whatever the data records.
 *   2 Library  — angle step + background correction → "Build Dictionary"
 *                (`ebsd_build_dictionary`; also switches the band overlay on).
 *   3 Refine   — how many bands to draw, zone axes, and PC nudges →
 *                `ebsd_refine` (debounced). The matched orientation's Kikuchi
 *                BANDS redraw on the pattern under the crosshair, and the live
 *                NCC + Euler readout streams back on `ebsd_match`.
 *   4 Run      — N best / refinement → "Index Map" (`ebsd_run`).
 *
 * Where the 4D-STEM caret draws matched SPOTS, this one draws matched LINES.
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Slider, Check, Select, S } from './WizardShell'
import { useDebouncedAction, useWizardEvent } from './wizardHooks'
import { useCifRecents, RecentCifs } from './CifRecents'
import { CodPicker } from './CodPicker'

const TABS = ['Load', 'Library', 'Refine', 'Run'] as const
type Tab = typeof TABS[number]

type BgMethod = 'dynamic' | 'static' | 'both' | 'none'
const BG_METHODS: readonly { value: BgMethod; label: string }[] = [
  { value: 'dynamic', label: 'Dynamic (per pattern)' },
  { value: 'static', label: 'Static (scan mean)' },
  { value: 'both', label: 'Both' },
  { value: 'none', label: 'None' },
]

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

// Per-window state kept OUTSIDE the component so stepping away and back doesn't
// discard a dictionary that took a minute to build (same reason OrientationWizard
// keeps an _omStore).
interface EbsdSaved {
  tab: Tab; cif: string; spaceGroup: number; voltage: number
  pcx: number; pcy: number; pcz: number
  step: number; minD: number; background: string; bgSigma: number
  nBands: number; zoneAxes: boolean
  keep: number; refine: boolean; refineSteps: number; dictReady: boolean
}
const _ebsdStore = new Map<number, EbsdSaved>()

export function EbsdWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const saved = _ebsdStore.get(windowId)
  const [tab, setTab] = React.useState<Tab>(saved?.tab ?? 'Load')
  const [cif, setCif] = React.useState(saved?.cif ?? '')
  const [spaceGroup, setSpaceGroup] = React.useState(saved?.spaceGroup ?? 225)
  const [voltage, setVoltage] = React.useState(saved?.voltage ?? 20)
  const [pcx, setPcx] = React.useState(saved?.pcx ?? 0.5)
  const [pcy, setPcy] = React.useState(saved?.pcy ?? 0.5)
  const [pcz, setPcz] = React.useState(saved?.pcz ?? 0.55)
  const [step, setStep] = React.useState(saved?.step ?? 4.0)
  const [minD, setMinD] = React.useState(saved?.minD ?? 0.7)
  const [background, setBackground] = React.useState<BgMethod>(
    (saved?.background as BgMethod) ?? 'dynamic')
  const [bgSigma, setBgSigma] = React.useState(saved?.bgSigma ?? 8.0)
  const [nBands, setNBands] = React.useState(saved?.nBands ?? 12)
  const [zoneAxes, setZoneAxes] = React.useState(saved?.zoneAxes ?? false)
  const [keep, setKeep] = React.useState(saved?.keep ?? 4)
  const [refine, setRefine] = React.useState(saved?.refine ?? true)
  const [refineSteps, setRefineSteps] = React.useState(saved?.refineSteps ?? 120)
  const [dictReady, setDictReady] = React.useState(saved?.dictReady ?? false)
  const [match, setMatch] = React.useState<null | { phi1: number; Phi: number; phi2: number; score: number }>(null)
  const [status, setStatus] = React.useState(
    saved?.dictReady ? 'Dictionary ready — move the crosshair to check the bands.'
                     : 'Pick a crystal (or keep the default cubic), then build the dictionary.')

  React.useEffect(() => {
    _ebsdStore.set(windowId, {
      tab, cif, spaceGroup, voltage, pcx, pcy, pcz, step, minD, background,
      bgSigma, nBands, zoneAxes, keep, refine, refineSteps, dictReady,
    })
  }, [windowId, tab, cif, spaceGroup, voltage, pcx, pcy, pcz, step, minD,
      background, bgSigma, nBands, zoneAxes, keep, refine, refineSteps, dictReady])

  // The backend knows the PC the data was recorded with; adopt it so the very
  // first overlay is drawn with the right geometry instead of a guessed centre.
  useWizardEvent('spyde:ebsd_dictionary_ready', windowId, (d) => {
    const n = Number(d.n_orientations ?? 0)
    const pc = d.pc as number[] | undefined
    if (pc && pc.length === 3) { setPcx(pc[0]); setPcy(pc[1]); setPcz(pc[2]) }
    setDictReady(true)
    setStatus(`Dictionary ready (${n.toLocaleString()} orientations) — move the crosshair.`)
  })
  useWizardEvent('spyde:ebsd_match', windowId, (d) => {
    if (!d.ok) return
    setMatch({
      phi1: Number(d.phi1), Phi: Number(d.Phi), phi2: Number(d.phi2),
      score: Number(d.score),
    })
  })

  const sendRefine = useDebouncedAction(sendAction, 'ebsd_refine', windowId)
  const { recents, remember } = useCifRecents()
  const base = (p: string) => p.split(/[/\\]/).pop() || p

  const addCif = (path: string) => {
    setCif(path); remember(path)
    setStatus('Crystal loaded — build the dictionary.')
  }
  const pickCif = async () => {
    const path = await window.electron.pickFile({ name: 'Crystal (.cif)', extensions: ['cif'] })
    if (path) addCif(path)
  }
  const build = () => {
    setStatus('Building dictionary…')
    sendAction('ebsd_build_dictionary', {
      cif_path: cif, space_group: spaceGroup, accelerating_voltage: voltage,
      pc_x: pcx, pc_y: pcy, pc_z: pcz, step_deg: step, min_dspacing: minD,
      background, background_sigma: bgSigma,
      n_bands: nBands, show_zone_axes: zoneAxes,
    }, windowId)
    setDictReady(true)          // backend acks with ebsd_dictionary_ready
    setTab('Refine')
  }
  // Debounced so dragging a PC slider doesn't fire a match per pixel.
  const tune = (next: Partial<{ nBands: number; zoneAxes: boolean; pcx: number; pcy: number; pcz: number }>) => {
    const p = {
      n_bands: next.nBands ?? nBands,
      show_zone_axes: next.zoneAxes ?? zoneAxes,
      pc_x: next.pcx ?? pcx, pc_y: next.pcy ?? pcy, pc_z: next.pcz ?? pcz,
    }
    sendRefine(() => p)
  }
  const run = () => {
    setStatus('Indexing the scan…')
    sendAction('ebsd_run', { keep, refine, refine_steps: refineSteps }, windowId)
  }

  return (
    <WizardShell testid="ebsd-wizard" title="EBSD Indexing" posStyle={caretPos}
      onClose={onClose} closeTestid="ebsd-close" status={status} statusTestid="ebsd-status">
      <TabRow tabs={TABS} active={tab} onSelect={setTab} testid={(t) => `ebsd-tab-${t}`}
        locked={(t) => (t === 'Refine' || t === 'Run') && !dictReady} />

      {tab === 'Load' && (
        <div style={S.page}>
          <label style={S.lbl}>Crystal (.cif)</label>
          <div style={{ display: 'flex', gap: 6, alignItems: 'stretch' }}>
            <button data-testid="ebsd-pick-cif" style={{ ...S.fileBtn, flex: 1, alignSelf: 'auto' }}
              onClick={pickCif}>＋ From file</button>
            <CodPicker windowId={windowId} sendAction={sendAction} onCif={addCif} />
          </div>
          <RecentCifs recents={recents} exclude={cif ? [cif] : []} onPick={addCif} />
          <div data-testid="ebsd-cif" style={S.cifList}>
            {cif
              ? (
                <div style={S.cifRow} title={cif}>
                  <span style={S.cifName}>{base(cif)}</span>
                  <button data-testid="ebsd-cif-remove" style={S.close}
                    onClick={() => setCif('')}>✕</button>
                </div>
              )
              : <span style={S.hint}>No .cif — a generic cubic band set is used.</span>}
          </div>
          {!cif && (
            <Field label="Space group">
              <NumInput value={spaceGroup} onChange={setSpaceGroup} step="1" width={60} testid="ebsd-spacegroup" />
            </Field>
          )}
          <Field label="Voltage (kV)">
            <NumInput value={voltage} onChange={setVoltage} step="1" width={60} testid="ebsd-voltage" />
          </Field>
          <div style={S.hint}>Projection centre (fraction of the detector):</div>
          <Field label="PC x"><NumInput value={pcx} onChange={setPcx} step="0.005" width={70} testid="ebsd-pcx" /></Field>
          <Field label="PC y"><NumInput value={pcy} onChange={setPcy} step="0.005" width={70} testid="ebsd-pcy" /></Field>
          <Field label="Distance"><NumInput value={pcz} onChange={setPcz} step="0.005" width={70} testid="ebsd-pcz" /></Field>
        </div>
      )}

      {tab === 'Library' && (
        <div style={S.page}>
          <Field label="Angle step (°)">
            <NumInput value={step} onChange={setStep} step="0.5" width={60} testid="ebsd-step" />
          </Field>
          <Field label="Min d (Å)">
            <NumInput value={minD} onChange={setMinD} step="0.05" width={60} testid="ebsd-mind" />
          </Field>
          <Field label="Background">
            <Select value={background} options={BG_METHODS}
              onChange={setBackground} testid="ebsd-background" />
          </Field>
          <Field label="σ (px)">
            <NumInput value={bgSigma} onChange={setBgSigma} step="0.5" width={60} testid="ebsd-bgsigma" />
          </Field>
          <div style={S.hint}>
            A finer step indexes closer but costs simulation time and slows the
            live preview.
          </div>
          <button data-testid="ebsd-build" style={S.primary} onClick={build}>Build Dictionary</button>
        </div>
      )}

      {tab === 'Refine' && (
        <div style={S.page}>
          <div style={S.hint}>
            Move the crosshair on the navigator: the matched orientation&apos;s
            Kikuchi bands are drawn on the pattern. Nudge the PC until the lines
            sit on the bands.
          </div>
          <div data-testid="ebsd-match" style={S.hint}>
            {match
              ? `NCC ${match.score.toFixed(3)} — φ1 ${match.phi1.toFixed(1)}° Φ ${match.Phi.toFixed(1)}° φ2 ${match.phi2.toFixed(1)}°`
              : 'No match yet — move the crosshair.'}
          </div>
          <Field label="Bands">
            <Slider testid="ebsd-nbands" value={nBands} min={1} max={40} step={1}
              onChange={(n) => { setNBands(n); tune({ nBands: n }) }} />
          </Field>
          <Check testid="ebsd-zone-axes" checked={zoneAxes} label="Mark zone axes"
            onChange={(b) => { setZoneAxes(b); tune({ zoneAxes: b }) }} />
          <Field label="PC x">
            <Slider testid="ebsd-refine-pcx" value={pcx} min={0.2} max={0.8} step={0.005}
              onChange={(n) => { setPcx(n); tune({ pcx: n }) }} />
          </Field>
          <Field label="PC y">
            <Slider testid="ebsd-refine-pcy" value={pcy} min={0.2} max={0.8} step={0.005}
              onChange={(n) => { setPcy(n); tune({ pcy: n }) }} />
          </Field>
          <Field label="Distance">
            <Slider testid="ebsd-refine-pcz" value={pcz} min={0.2} max={1.2} step={0.005}
              onChange={(n) => { setPcz(n); tune({ pcz: n }) }} />
          </Field>
        </div>
      )}

      {tab === 'Run' && (
        <div style={S.page}>
          <Field label="N best">
            <NumInput value={keep} onChange={setKeep} step="1" width={56} testid="ebsd-keep" />
          </Field>
          <div style={S.hint}>
            More than one match per pattern is what the orientation-similarity
            quality map needs.
          </div>
          <Check testid="ebsd-do-refine" checked={refine} label="Refine orientations"
            onChange={setRefine} />
          {refine && (
            <Field label="Steps">
              <NumInput value={refineSteps} onChange={setRefineSteps} step="10" width={64} testid="ebsd-refine-steps" />
            </Field>
          )}
          <button data-testid="ebsd-run" style={S.primary} onClick={run}>Index Map</button>
        </div>
      )}
    </WizardShell>
  )
}
