/**
 * FindVectorsWizard.tsx — the Find-Diffraction-Vectors caret (Qt parity).
 *
 * Opening the caret starts a LIVE preview of the found peaks on the diffraction
 * pattern (`fv_open`); the Method dropdown + parameter sliders re-run it
 * (debounced `fv_tune`) and moving the navigator previews other patterns.
 * "Compute" (`fv_run`) runs the full-dataset batch → a new vectors window;
 * closing the caret tears the preview down (`fv_close`).
 *
 * Three detection methods:
 *   • Neural — the SpotUNet disk detector (parameter-free disk size; threshold
 *     is the model's confidence, ~0.3; Model dropdown selects a registry model,
 *     populated by the `fv_models` payload). The default.
 *   • NXCORR — window-normalised cross-correlation against a flat disk
 *     (Disk Radius slider; threshold is a [-1,1] correlation score).
 *   • DoG — Difference-of-Gaussians band-pass, best for small (2-3 px) spots
 *     and beam-stopped patterns (σ₁/σ₂ sliders; threshold is a band-pass SNR).
 */
import React from 'react'
import { WizardShell, Field, Slider, Select, Check, S } from './WizardShell'
import { useWizardLifecycle, useDebouncedAction, useWizardEvent } from './wizardHooks'

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

type Method = 'neural' | 'nxcorr' | 'dog'
const METHODS: readonly { value: Method; label: string }[] = [
  { value: 'neural', label: 'Neural (SpotUNet)' },
  { value: 'nxcorr', label: 'NXCORR (disk)' },
  { value: 'dog', label: 'DoG (small spots)' },
]
// Sensible threshold default per method (neural confidence vs NXCORR score vs DoG SNR).
const THR_DEFAULT: Record<Method, number> = { neural: 0.3, nxcorr: 0.5, dog: 10 }

// Per-window tuned state kept OUTSIDE the component (like the OM/VOM wizards)
// so the caret closing on Compute doesn't lose the tuning — reopening restores
// the last-used parameters.
interface FvSaved {
  method: Method; modelId: string; sigma: number; radius: number
  sigma1: number; sigma2: number; bgSigma: number
  threshold: number; minDist: number; subpixel: boolean; beamstop: boolean
  beamstopDilate: number; showTransform: boolean; persistence: boolean
}
const _fvStore = new Map<number, FvSaved>()

// Neural local-norm high-pass scale (px). 12 matches the backend default; the
// backend's one-shot auto-calibration (`fv_calibration`) overwrites it unless
// the user already moved the slider.
const BG_SIGMA_DEFAULT = 12

export function FindVectorsWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const saved = _fvStore.get(windowId)
  const [method, setMethod] = React.useState<Method>(saved?.method ?? 'neural')
  const [modelId, setModelId] = React.useState(saved?.modelId ?? '')
  const [models, setModels] = React.useState<readonly { value: string; label: string }[]>([])
  const [sigma, setSigma] = React.useState(saved?.sigma ?? 1.0)
  const [radius, setRadius] = React.useState(saved?.radius ?? 5)
  const [sigma1, setSigma1] = React.useState(saved?.sigma1 ?? 0.8)
  const [sigma2, setSigma2] = React.useState(saved?.sigma2 ?? 2.0)
  const [bgSigma, setBgSigma] = React.useState(saved?.bgSigma ?? BG_SIGMA_DEFAULT)
  const [threshold, setThreshold] = React.useState(saved?.threshold ?? 0.3)
  const [minDist, setMinDist] = React.useState(saved?.minDist ?? 5)
  const [subpixel, setSubpixel] = React.useState(saved?.subpixel ?? true)
  const [beamstop, setBeamstop] = React.useState(saved?.beamstop ?? false)
  const [beamstopDilate, setBeamstopDilate] = React.useState(saved?.beamstopDilate ?? 5)
  const [showTransform, setShowTransform] = React.useState(saved?.showTransform ?? false)
  const [persistence, setPersistence] = React.useState(saved?.persistence ?? false)
  const [status, setStatus] = React.useState('Tune the parameters — peaks preview under the crosshair.')

  React.useEffect(() => {
    _fvStore.set(windowId, {
      method, modelId, sigma, radius, sigma1, sigma2, bgSigma, threshold, minDist, subpixel,
      beamstop, beamstopDilate, showTransform, persistence,
    })
  }, [windowId, method, modelId, sigma, radius, sigma1, sigma2, bgSigma, threshold, minDist,
      subpixel, beamstop, beamstopDilate, showTransform, persistence])

  // Live refs so the debounced tune always sends the latest of EVERY control.
  const vals = React.useRef({ method, modelId, sigma, radius, sigma1, sigma2, bgSigma, threshold, minDist, subpixel, beamstop, beamstopDilate, showTransform, persistence })
  vals.current = { method, modelId, sigma, radius, sigma1, sigma2, bgSigma, threshold, minDist, subpixel, beamstop, beamstopDilate, showTransform, persistence }
  const params = () => ({
    method: vals.current.method, model_id: vals.current.modelId,
    sigma: vals.current.sigma, kernel_radius: vals.current.radius,
    dog_sigma1: vals.current.sigma1, dog_sigma2: vals.current.sigma2,
    bg_sigma: vals.current.bgSigma,
    threshold: vals.current.threshold, min_distance: vals.current.minDist,
    subpixel: vals.current.subpixel, beamstop_auto: vals.current.beamstop,
    beamstop_dilate: vals.current.beamstopDilate,
    show_transform: vals.current.showTransform,
    persistence: vals.current.persistence,
  })

  // Start the live preview when the caret opens; tear it down when it closes
  // (StrictMode-safe: exactly one fv_open reaches the backend).
  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'fv_open', openPayload: params, closeAction: 'fv_close',
  })

  // Debounced live tune — dispatch on control settle so matches don't flood;
  // a pending tune is cancelled on unmount so it can't hit a torn-down preview.
  const sendTune = useDebouncedAction(sendAction, 'fv_tune', windowId)
  const tune = () => sendTune(params)
  const live = <T,>(set: (v: T) => void) => (v: T) => { set(v); tune() }

  // Populate the Model dropdown from the backend registry. Requested ONCE on
  // mount via a ref (sendAction must never be an effect dep — it's recreated
  // every render and would re-fire the request in a loop).
  const sendRef = React.useRef(sendAction)
  sendRef.current = sendAction
  React.useEffect(() => {
    sendRef.current('fv_models', {}, windowId)
  }, [windowId])
  useWizardEvent('spyde:fv_models', windowId, (d) => {
    const list = Array.isArray(d.models) ? (d.models as { id: string; label?: string }[]) : []
    setModels(list.map((m) => ({ value: m.id, label: m.label ?? m.id })))
    // '' = registry default; surface it as the concrete id once known.
    if (!vals.current.modelId && typeof d.default === 'string') setModelId(d.default)
    if (d.refreshed) setStatus(`Model list refreshed — ${list.length} available.`)
  })

  // "Check for new models": pull the latest registry from Hugging Face; the
  // backend re-emits fv_models (refreshed: true) with the merged list.
  const refreshModels = () => {
    setStatus('Checking for new models…')
    sendRef.current('fv_refresh_models', {}, windowId)
  }

  // Backend one-shot auto-calibration (neural): adopt bg σ / threshold unless
  // the user already moved them off their defaults (override wins), then
  // re-run the preview with the calibrated values.
  useWizardEvent('spyde:fv_calibration', windowId, (d) => {
    if (vals.current.method !== 'neural') return
    let adopted = false
    if (typeof d.bg_sigma === 'number' && vals.current.bgSigma === BG_SIGMA_DEFAULT
        && d.bg_sigma !== BG_SIGMA_DEFAULT) {
      setBgSigma(d.bg_sigma)
      vals.current = { ...vals.current, bgSigma: d.bg_sigma }
      adopted = true
    }
    if (typeof d.thresh === 'number' && vals.current.threshold === THR_DEFAULT.neural
        && d.thresh !== THR_DEFAULT.neural) {
      setThreshold(d.thresh)
      vals.current = { ...vals.current, threshold: d.thresh }
      adopted = true
    }
    if (adopted) {
      const conf = typeof d.confidence === 'number' ? ` (conf ${d.confidence.toFixed(2)})` : ''
      setStatus(`Auto-calibrated: high-pass σ=${Number(d.bg_sigma).toFixed(1)}, `
        + `threshold ${Number(d.thresh).toFixed(2)}${conf}.`)
      tune()
    }
  })

  // Adopt the backend's auto-estimated disk radius (Qt parity) the first time it
  // arrives, then re-run the preview with it. The user can still override.
  const autoApplied = React.useRef(false)
  useWizardEvent('spyde:fv_auto_params', windowId, (d) => {
    if (autoApplied.current) return
    autoApplied.current = true
    if (typeof d.kernel_radius === 'number') setRadius(d.kernel_radius)
    if (typeof d.min_distance === 'number') setMinDist(d.min_distance)
    setStatus(`Disk radius auto-set to ${d.kernel_radius} px from the pattern.`)
    tune()
  })

  // Switching method resets the threshold to that method's natural scale and
  // re-runs the preview (NXCORR score and DoG SNR are not comparable numbers).
  const onMethod = (m: Method) => {
    setMethod(m)
    setThreshold(THR_DEFAULT[m])
    vals.current = { ...vals.current, method: m, threshold: THR_DEFAULT[m] }
    tune()
  }

  const compute = () => {
    sendAction('fv_run', params(), windowId)
    // Compute collapses the caret back into the toolbar button (its tuned
    // state is kept in _fvStore). The unmount fires fv_close, which only
    // drops the live preview — the batch keeps running.
    onClose()
  }

  const isDog = method === 'dog'
  const isNeural = method === 'neural'

  return (
    <WizardShell testid="find-vectors-wizard" title="Find Diffraction Vectors" posStyle={caretPos}
      onClose={onClose} closeTestid="fv-close" status={status} statusTestid="fv-status"
      width={320}>
      <div style={S.hint}>Move the crosshair on the navigator to preview the found peaks.</div>
      <Field label="Method">
        <Select testid="fv-method" value={method} options={METHODS} onChange={onMethod} />
      </Field>
      {isNeural && models.length > 0 && (
        <Field label="Model">
          <div style={modelRowStyle}>
            <Select testid="fv-model" value={modelId} options={models}
              onChange={live(setModelId)} />
            <button data-testid="fv-refresh-models" style={refreshBtnStyle}
              title="Check Hugging Face for new models" onClick={refreshModels}>↻</button>
          </div>
        </Field>
      )}
      <div style={gridStyle}>
        <Cell label="Nav Blur σ">
          <Slider testid="fv-sigma" value={sigma} min={0} max={5} step={0.1}
            onChange={live(setSigma)} fmt={(n) => n.toFixed(1)} />
        </Cell>
        {!isDog && !isNeural && (
          <Cell label="Disk Radius">
            <Slider testid="fv-radius" value={radius} min={1} max={30} step={1} onChange={live(setRadius)} />
          </Cell>
        )}
        {isNeural && (
          <Cell label="High-pass σ">
            <Slider testid="fv-bg-sigma" value={bgSigma} min={2} max={24} step={0.5}
              onChange={live(setBgSigma)} fmt={(n) => n.toFixed(1)} />
          </Cell>
        )}
        {isDog && (
          <>
            <Cell label="DoG σ₁ (spot)">
              <Slider testid="fv-dog-sigma1" value={sigma1} min={0.4} max={3} step={0.1}
                onChange={live(setSigma1)} fmt={(n) => n.toFixed(1)} />
            </Cell>
            <Cell label="DoG σ₂ (bg)">
              <Slider testid="fv-dog-sigma2" value={sigma2} min={1} max={6} step={0.1}
                onChange={live(setSigma2)} fmt={(n) => n.toFixed(1)} />
            </Cell>
          </>
        )}
        <Cell label={isDog ? 'Threshold (SNR)' : isNeural ? 'Threshold (conf.)' : 'Threshold'}>
          <Slider testid="fv-threshold" value={threshold}
            min={0} max={isDog ? 30 : 1} step={isDog ? 0.5 : 0.05}
            onChange={live(setThreshold)} fmt={(n) => (isDog ? n.toFixed(1) : n.toFixed(2))} />
        </Cell>
        <Cell label="Min Distance">
          <Slider testid="fv-mindist" value={minDist} min={1} max={30} step={1} onChange={live(setMinDist)} />
        </Cell>
        {beamstop && (
          <Cell label="Beam-stop dilate">
            <Slider testid="fv-beamstop-dilate" value={beamstopDilate} min={0} max={20} step={1}
              onChange={live(setBeamstopDilate)} />
          </Cell>
        )}
      </div>
      <div style={gridStyle}>
        <Check testid="fv-beamstop" checked={beamstop} onChange={live(setBeamstop)}
          label="Mask beam stop" />
        <Check testid="fv-show-transform" checked={showTransform} onChange={live(setShowTransform)}
          label={isDog ? 'Show DoG' : isNeural ? 'Show heatmap' : 'Show corr.'} />
        <Check testid="fv-subpixel" checked={subpixel} onChange={live(setSubpixel)}
          label="Subpixel" />
        {isNeural && (
          // Batch-only stage-2 refine (scan-neighbour persistence + Friedel);
          // the live preview has no neighbours so it only affects Compute.
          <Check testid="fv-persistence" checked={persistence} onChange={live(setPersistence)}
            label="Neighbor refine" />
        )}
      </div>
      <button data-testid="fv-compute" style={S.primary} onClick={compute}>Compute</button>
    </WizardShell>
  )
}

const gridStyle: React.CSSProperties = {
  display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 10, rowGap: 6,
}
const modelRowStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 4, minWidth: 0,
}
const refreshBtnStyle: React.CSSProperties = {
  flex: '0 0 auto', width: 22, height: 22, padding: 0, lineHeight: '20px',
  background: 'transparent', color: 'inherit', border: '1px solid #444',
  borderRadius: 4, cursor: 'pointer',
}
const cellStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0,
}

// A labelled control as a compact stacked cell (label above control) for the
// two-column grid. MUST be module-scope, NOT defined inside the component —
// a component defined inline is a new type every render, so React remounts the
// sliders on each keystroke and they lose their drag ("sliders don't work").
function Cell({ label, children }: { label: string; children: React.ReactNode }) {
  return <div style={cellStyle}><label style={S.lbl}>{label}</label>{children}</div>
}
