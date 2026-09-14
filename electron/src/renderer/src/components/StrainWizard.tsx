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
 * Deliberately minimal: contrast (min/max) and colormap belong to the PLOT
 * WIDGET's histogram dock (the strain window emits the standard histogram and
 * responds to its handles), and the fit-robustness knobs (min matched spots,
 * reference pooling radius) live on good defaults in the backend, adjustable
 * via the `strain_set_fit` action for scripted use — not wizard clutter.
 *
 *   Method        — Region (relative, the navigator pixel) or CIF (absolute,
 *                   from a crystal's ideal spacings → prompts for the .cif).
 *   Match radius  — how far (px) a frame peak can be from a reference spot to
 *                   count as matched (drives the arrows).
 *   Rotation / Flip — the detector's frame turned into the scan's x/y, so
 *                   "εxx" means strain along the scan's x and not the
 *                   detector's (`strain_set_rotation`). The same angle is the
 *                   x/y arrow pair drawn on the reference pattern — drag an
 *                   arrow head and the slider follows. "From DPC" lists the
 *                   DPC results open in the session and takes the angle and
 *                   handedness one of them found for this scan.
 *   Commit        — freeze the current strain field as a new signal tree
 *                   (`strain_commit` → the standard Commit affordance).
 */
import React from 'react'
import { WizardShell, Field, Select, Slider, Check, S } from './WizardShell'
import { useWizardLifecycle, useWizardEvent, CommitButton } from './wizardHooks'

const METHODS = [
  { value: 'region' as const, label: 'Region (relative)' },
  { value: 'cif' as const, label: 'CIF (absolute)…' },
]
type Method = typeof METHODS[number]['value']

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

export function StrainWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const [method, setMethod] = React.useState<Method>('region')
  const [matchRadius, setMatchRadius] = React.useState(6)
  const [rotation, setRotation] = React.useState(0)
  const [flip, setFlip] = React.useState(false)
  const [dpcSources, setDpcSources] = React.useState<{ index: number; label: string }[]>([])
  const [status, setStatus] = React.useState('Double-click reference spots to use/ignore; move the navigator to displace.')

  // Open → run the live strain field (opens the strain map + selection overlay).
  // Close (caret deselected / toggled off) → tear it ALL down: strain map window,
  // overlay, nav hooks. The source DP/navigator are left untouched.
  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'strain_open',
    openPayload: () => ({ match_radius_px: matchRadius }),
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

  const onRotation = (deg: number) => {
    setRotation(deg)
    sendAction('strain_set_rotation', { rotation: deg, flip }, windowId)
  }
  const onFlip = (b: boolean) => {
    setFlip(b)
    sendAction('strain_set_rotation', { rotation, flip: b }, windowId)
  }
  const fromDpc = (v: string) => {
    if (v === '') return
    sendAction('strain_set_rotation', { from_dpc: Number(v) }, windowId)
  }
  const listDpc = () => sendAction('strain_set_rotation', { list_dpc: true }, windowId)
  // The backend reports the basis the map is drawn with — after a drag of the
  // arrows on the reference pattern, or "From DPC" — and, when asked, the DPC
  // results it could take one from.
  useWizardEvent('spyde:strain_rotation', windowId, (d) => {
    if (typeof d.rotation === 'number') setRotation(Math.round(Number(d.rotation) * 10) / 10)
    if (typeof d.flip === 'boolean') setFlip(d.flip)
    if (Array.isArray(d.dpc_sources)) {
      setDpcSources(d.dpc_sources as { index: number; label: string }[])
    } else {
      setStatus(`Basis: ${Number(d.rotation).toFixed(1)}°${d.flip ? ', x/y flipped' : ''}.`)
    }
  })
  const dpcOptions = [
    { value: '', label: dpcSources.length ? 'From DPC…' : 'From DPC: none open' },
    ...dpcSources.map((s) => ({ value: String(s.index), label: s.label })),
  ]

  return (
    <WizardShell testid="strain-wizard" title="Strain Mapping" posStyle={caretPos}
      onClose={onClose} closeTestid="strain-close" status={status} statusTestid="strain-status">
      <div style={S.page}>
        <div style={S.hint}>
          Green circles = reference spots used in the fit. Double-click a spot to drop/restore it.
          Move the navigator to set the reference pixel. Contrast: drag the histogram handles
          in the plot sidebar.
        </div>
        <Field label="Method">
          <Select testid="strain-method" value={method} options={METHODS} onChange={onMethod} />
        </Field>
        <Field label="Match radius (px)">
          <Slider testid="strain-match-radius" value={matchRadius} min={1} max={15} step={1}
            onChange={onRadius} fmt={(v: number) => `${v} px`} />
        </Field>
        <Field label="Rotation (°)">
          <Slider testid="strain-rotation" value={rotation} min={-180} max={180} step={0.5}
            onChange={onRotation} fmt={(v: number) => `${v}°`} />
        </Field>
        <div style={S.hint}>
          The pink x / gold y arrows on the reference pattern are the scan's axes — drag an
          arrow head to turn them.
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Check testid="strain-flip" checked={flip} onChange={onFlip} label="Flip x/y" />
          <Select testid="strain-rotation-from-dpc" value={''} options={dpcOptions}
            onChange={fromDpc} />
          <button data-testid="strain-dpc-refresh" style={S.primary} onClick={listDpc}
            title="Look again for DPC results open in this session">↻</button>
        </div>
        <CommitButton wizardKey="strain" windowId={windowId} sendAction={sendAction}
          onCommit={() => setStatus('Committed — new strain signal tree created.')} />
      </div>
    </WizardShell>
  )
}
