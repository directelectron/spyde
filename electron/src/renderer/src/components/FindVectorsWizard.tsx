/**
 * FindVectorsWizard.tsx — the Find-Diffraction-Vectors caret (Qt parity).
 *
 * Opening the caret starts a LIVE preview of the found peaks on the diffraction
 * pattern (`fv_preview`); the Method dropdown + parameter sliders re-run it
 * (debounced `fv_tune`) and moving the navigator previews other patterns.
 * "Compute" (`fv_run`) runs the full-dataset batch → a new vectors window;
 * closing the caret tears the preview down (`fv_stop`).
 *
 * Two detection methods:
 *   • NXCORR — window-normalised cross-correlation against a flat disk
 *     (Disk Radius slider; threshold is a [-1,1] correlation score).
 *   • DoG — Difference-of-Gaussians band-pass, best for small (2-3 px) spots
 *     and beam-stopped patterns (σ₁/σ₂ sliders; threshold is a band-pass SNR).
 */
import React from 'react'
import { WizardShell, Field, Slider, Select, Check, S } from './WizardShell'

interface Props {
  openUp: boolean
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

type Method = 'nxcorr' | 'dog'
const METHODS: readonly { value: Method; label: string }[] = [
  { value: 'nxcorr', label: 'NXCORR (disk)' },
  { value: 'dog', label: 'DoG (small spots)' },
]
// Sensible threshold default per method (NXCORR score vs DoG SNR).
const THR_DEFAULT: Record<Method, number> = { nxcorr: 0.5, dog: 10 }

export function FindVectorsWizard({ openUp, windowId, sendAction, onClose }: Props) {
  const [method, setMethod] = React.useState<Method>('nxcorr')
  const [sigma, setSigma] = React.useState(1.0)
  const [radius, setRadius] = React.useState(5)
  const [sigma1, setSigma1] = React.useState(0.8)
  const [sigma2, setSigma2] = React.useState(2.0)
  const [threshold, setThreshold] = React.useState(0.5)
  const [minDist, setMinDist] = React.useState(5)
  const [subpixel, setSubpixel] = React.useState(true)
  const [beamstop, setBeamstop] = React.useState(false)
  const [status, setStatus] = React.useState('Tune the parameters — peaks preview under the crosshair.')

  const tuneTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  // Live refs so the debounced tune always sends the latest of EVERY control.
  const vals = React.useRef({ method, sigma, radius, sigma1, sigma2, threshold, minDist, subpixel, beamstop })
  vals.current = { method, sigma, radius, sigma1, sigma2, threshold, minDist, subpixel, beamstop }
  const params = () => ({
    method: vals.current.method,
    sigma: vals.current.sigma, kernel_radius: vals.current.radius,
    dog_sigma1: vals.current.sigma1, dog_sigma2: vals.current.sigma2,
    threshold: vals.current.threshold, min_distance: vals.current.minDist,
    subpixel: vals.current.subpixel, beamstop_auto: vals.current.beamstop,
  })

  // Start the live preview when the caret opens; tear it down when it closes.
  React.useEffect(() => {
    sendAction('fv_preview', params(), windowId)
    return () => { sendAction('fv_stop', {}, windowId) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Adopt the backend's auto-estimated disk radius (Qt parity) the first time it
  // arrives, then re-run the preview with it. The user can still override.
  const autoApplied = React.useRef(false)
  React.useEffect(() => {
    const onAuto = (e: Event) => {
      const d = (e as CustomEvent).detail as Record<string, unknown>
      if (d.window_id != null && d.window_id !== windowId) return
      if (autoApplied.current) return
      autoApplied.current = true
      if (typeof d.kernel_radius === 'number') setRadius(d.kernel_radius)
      if (typeof d.min_distance === 'number') setMinDist(d.min_distance)
      setStatus(`Disk radius auto-set to ${d.kernel_radius} px from the pattern.`)
      tune()
    }
    window.addEventListener('spyde:fv_auto_params', onAuto)
    return () => window.removeEventListener('spyde:fv_auto_params', onAuto)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [windowId])

  // Debounced live tune — dispatch on control settle so matches don't flood.
  const tune = () => {
    if (tuneTimer.current) clearTimeout(tuneTimer.current)
    tuneTimer.current = setTimeout(() => sendAction('fv_tune', params(), windowId), 120)
  }
  const live = <T,>(set: (v: T) => void) => (v: T) => { set(v); tune() }

  // Switching method resets the threshold to that method's natural scale and
  // re-runs the preview (NXCORR score and DoG SNR are not comparable numbers).
  const onMethod = (m: Method) => {
    setMethod(m)
    setThreshold(THR_DEFAULT[m])
    vals.current = { ...vals.current, method: m, threshold: THR_DEFAULT[m] }
    tune()
  }

  const compute = () => {
    setStatus('Finding diffraction vectors…')
    sendAction('fv_run', params(), windowId)
  }

  const isDog = method === 'dog'

  return (
    <WizardShell testid="find-vectors-wizard" title="Find Diffraction Vectors" openUp={openUp}
      onClose={onClose} closeTestid="fv-close" status={status} statusTestid="fv-status">
      <div style={S.hint}>Move the crosshair on the navigator to preview the found peaks.</div>
      <Field label="Method">
        <Select testid="fv-method" value={method} options={METHODS} onChange={onMethod} />
      </Field>
      <Field label="Nav Blur σ">
        <Slider testid="fv-sigma" value={sigma} min={0} max={5} step={0.1}
          onChange={live(setSigma)} fmt={(n) => n.toFixed(1)} />
      </Field>
      {!isDog && (
        <Field label="Disk Radius">
          <Slider testid="fv-radius" value={radius} min={1} max={30} step={1} onChange={live(setRadius)} />
        </Field>
      )}
      {isDog && (
        <>
          <Field label="DoG σ₁ (spot)">
            <Slider testid="fv-dog-sigma1" value={sigma1} min={0.4} max={3} step={0.1}
              onChange={live(setSigma1)} fmt={(n) => n.toFixed(1)} />
          </Field>
          <Field label="DoG σ₂ (bg)">
            <Slider testid="fv-dog-sigma2" value={sigma2} min={1} max={6} step={0.1}
              onChange={live(setSigma2)} fmt={(n) => n.toFixed(1)} />
          </Field>
        </>
      )}
      <Field label={isDog ? 'Threshold (SNR)' : 'Threshold'}>
        <Slider testid="fv-threshold" value={threshold}
          min={0} max={isDog ? 30 : 1} step={isDog ? 0.5 : 0.05}
          onChange={live(setThreshold)} fmt={(n) => (isDog ? n.toFixed(1) : n.toFixed(2))} />
      </Field>
      <Field label="Min Distance">
        <Slider testid="fv-mindist" value={minDist} min={1} max={30} step={1} onChange={live(setMinDist)} />
      </Field>
      <Check testid="fv-beamstop" checked={beamstop} onChange={live(setBeamstop)}
        label="Mask beam stop (auto)" />
      <Check testid="fv-subpixel" checked={subpixel} onChange={live(setSubpixel)}
        label="Subpixel refinement" />
      <button data-testid="fv-compute" style={S.primary} onClick={compute}>Compute</button>
    </WizardShell>
  )
}
