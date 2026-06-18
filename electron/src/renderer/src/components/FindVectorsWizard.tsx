/**
 * FindVectorsWizard.tsx — the Find-Diffraction-Vectors caret (Qt parity).
 *
 * Opening the caret starts a LIVE preview of the found peaks on the diffraction
 * pattern (`fv_preview`); the σ / disk-radius / threshold / min-distance /
 * subpixel sliders re-run it (debounced `fv_tune`) and moving the navigator
 * previews other patterns. "Compute" (`fv_run`) runs the full-dataset batch →
 * a new vectors window; closing the caret tears the preview down (`fv_stop`).
 */
import React from 'react'
import { WizardShell, Field, Slider, Check, S } from './WizardShell'

interface Props {
  openUp: boolean
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

export function FindVectorsWizard({ openUp, windowId, sendAction, onClose }: Props) {
  const [sigma, setSigma] = React.useState(1.0)
  const [radius, setRadius] = React.useState(5)
  const [threshold, setThreshold] = React.useState(0.5)
  const [minDist, setMinDist] = React.useState(5)
  const [subpixel, setSubpixel] = React.useState(true)
  const [status, setStatus] = React.useState('Tune the parameters — peaks preview under the crosshair.')

  const tuneTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  // Live refs so the debounced tune always sends the latest of EVERY slider.
  const vals = React.useRef({ sigma, radius, threshold, minDist, subpixel })
  vals.current = { sigma, radius, threshold, minDist, subpixel }
  const params = () => ({
    sigma: vals.current.sigma, kernel_radius: vals.current.radius,
    threshold: vals.current.threshold, min_distance: vals.current.minDist,
    subpixel: vals.current.subpixel,
  })

  // Start the live preview when the caret opens; tear it down when it closes.
  React.useEffect(() => {
    sendAction('fv_preview', params(), windowId)
    return () => { sendAction('fv_stop', {}, windowId) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Debounced live tune — dispatch on slider settle so matches don't flood.
  const tune = () => {
    if (tuneTimer.current) clearTimeout(tuneTimer.current)
    tuneTimer.current = setTimeout(() => sendAction('fv_tune', params(), windowId), 120)
  }
  const live = <T,>(set: (v: T) => void) => (v: T) => { set(v); tune() }

  const compute = () => {
    setStatus('Finding diffraction vectors…')
    sendAction('fv_run', params(), windowId)
  }

  return (
    <WizardShell testid="find-vectors-wizard" title="Find Diffraction Vectors" openUp={openUp}
      onClose={onClose} closeTestid="fv-close" status={status} statusTestid="fv-status">
      <div style={S.hint}>Move the crosshair on the navigator to preview the found peaks.</div>
      <Field label="Nav Blur σ">
        <Slider testid="fv-sigma" value={sigma} min={0} max={5} step={0.1}
          onChange={live(setSigma)} fmt={(n) => n.toFixed(1)} />
      </Field>
      <Field label="Disk Radius">
        <Slider testid="fv-radius" value={radius} min={1} max={30} step={1} onChange={live(setRadius)} />
      </Field>
      <Field label="Threshold">
        <Slider testid="fv-threshold" value={threshold} min={0} max={1} step={0.05}
          onChange={live(setThreshold)} fmt={(n) => n.toFixed(2)} />
      </Field>
      <Field label="Min Distance">
        <Slider testid="fv-mindist" value={minDist} min={1} max={30} step={1} onChange={live(setMinDist)} />
      </Field>
      <Check testid="fv-subpixel" checked={subpixel} onChange={live(setSubpixel)}
        label="Subpixel refinement" />
      <button data-testid="fv-compute" style={S.primary} onClick={compute}>Compute</button>
    </WizardShell>
  )
}
