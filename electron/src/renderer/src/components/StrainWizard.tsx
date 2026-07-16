/**
 * StrainWizard.tsx — the Strain Mapping caret.
 *
 * Opening the caret runs the live strain field (`strain_open`) and shows an
 * interactive overlay on the source diffraction pattern: the reference spots as
 * GREEN (selected) / grey (excluded) circles — double-click one to toggle whether it
 * drives the fit. Moving the navigator picks a new reference pixel; off the
 * reference pixel the overlay draws displacement arrows (reference spot → matched
 * peak within the match radius).
 *
 *   Method        — Region (relative, the navigator pixel) or CIF (absolute,
 *                   from a crystal's ideal spacings → prompts for the .cif).
 *   Match radius  — how far (px) a frame peak can be from a reference spot to
 *                   count as matched (drives the arrows).
 *   Min matches   — a pixel's affine fit needs REDUNDANCY (≥4 matched spots);
 *                   under-determined pixels render masked gray, not noise.
 *   Ref region    — pool the reference over a (2r+1)² scan neighbourhood
 *                   (consensus median per reflection — noise/FP robust).
 *   Colour range  — symmetric ± clim for the map (0 = robust auto).
 *   Weight by     — fade unreliable pixels: alpha from coverage or fit error.
 *   Commit        — freeze the current strain field as a new signal tree
 *                   (`strain_commit` → the standard Commit affordance).
 */
import React from 'react'
import { WizardShell, Field, NumInput, Select, S } from './WizardShell'
import { useWizardLifecycle, CommitButton } from './wizardHooks'

const METHODS = [
  { value: 'region' as const, label: 'Region (relative)' },
  { value: 'cif' as const, label: 'CIF (absolute)…' },
]
type Method = typeof METHODS[number]['value']

const WEIGHTS = [
  { value: 'none' as const, label: 'None' },
  { value: 'coverage' as const, label: 'Coverage' },
  { value: 'error' as const, label: 'Fit error' },
]
type Weight = typeof WEIGHTS[number]['value']

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

export function StrainWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const [method, setMethod] = React.useState<Method>('region')
  const [matchRadius, setMatchRadius] = React.useState(6)
  const [minMatches, setMinMatches] = React.useState(4)
  const [refRadius, setRefRadius] = React.useState(2)
  const [vmax, setVmax] = React.useState(0)          // 0 = robust auto range
  const [weight, setWeight] = React.useState<Weight>('none')
  const [status, setStatus] = React.useState('Double-click reference spots to use/ignore; move the navigator to displace.')

  // Open → run the live strain field (opens the strain map + selection overlay).
  // Close (caret deselected / toggled off) → tear it ALL down: strain map window,
  // overlay, nav hooks. The source DP/navigator are left untouched.
  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'strain_open',
    openPayload: () => ({
      match_radius_px: matchRadius, min_matches: minMatches,
      ref_radius: refRadius, vmax, weight,
    }),
    closeAction: 'strain_close',
  })

  const onMethod = async (m: Method) => {
    setMethod(m)
    if (m === 'cif') {
      const p = await window.electron.pickFile({ name: 'Crystal (.cif)', extensions: ['cif'] })
      if (p) { sendAction('strain_set_method', { method: 'cif', cif_path: p }, windowId); setStatus('Absolute (CIF) reference.') }
      else setMethod('region')   // cancelled the picker → stay on Region
    } else {
      sendAction('strain_set_method', { method: 'region' }, windowId)
      setStatus('Relative reference = the navigator pixel.')
    }
  }

  const onRadius = (r: number) => {
    setMatchRadius(r)
    sendAction('strain_set_match_radius', { match_radius_px: r }, windowId)
  }

  const onMinMatches = (n: number) => {
    setMinMatches(n)
    sendAction('strain_set_fit', { min_matches: n }, windowId)
  }
  const onRefRadius = (r: number) => {
    setRefRadius(r)
    sendAction('strain_set_fit', { ref_radius: r }, windowId)
  }
  const onVmax = (v: number) => {
    setVmax(v)
    sendAction('strain_set_display', { vmax: v }, windowId)
  }
  const onWeight = (w: Weight) => {
    setWeight(w)
    sendAction('strain_set_display', { weight: w }, windowId)
  }

  return (
    <WizardShell testid="strain-wizard" title="Strain Mapping" posStyle={caretPos}
      onClose={onClose} closeTestid="strain-close" status={status} statusTestid="strain-status">
      <div style={S.page}>
        <div style={S.hint}>
          Green circles = reference spots used in the fit. Double-click a spot to drop/restore it.
          Move the navigator to set the reference pixel; off it, arrows show each spot's displacement.
        </div>
        <Field label="Method">
          <Select testid="strain-method" value={method} options={METHODS} onChange={onMethod} />
        </Field>
        <Field label="Match radius (px)">
          <NumInput testid="strain-match-radius" value={matchRadius} onChange={onRadius} step="1" />
        </Field>
        <Field label="Min. matched spots">
          <NumInput testid="strain-min-matches" value={minMatches} onChange={onMinMatches} step="1" />
        </Field>
        <Field label="Ref. region radius (px)">
          <NumInput testid="strain-ref-radius" value={refRadius} onChange={onRefRadius} step="1" />
        </Field>
        <Field label="Colour range ± (0 = auto)">
          <NumInput testid="strain-vmax" value={vmax} onChange={onVmax} step="0.001" />
        </Field>
        <Field label="Weight display by">
          <Select testid="strain-weight" value={weight} options={WEIGHTS} onChange={onWeight} />
        </Field>
        <CommitButton wizardKey="strain" windowId={windowId} sendAction={sendAction}
          onCommit={() => setStatus('Committed — new strain signal tree created.')} />
      </div>
    </WizardShell>
  )
}
