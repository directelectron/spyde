/**
 * CenterZeroBeamWizard.tsx — the Center-Zero-Beam caret (Qt two-tab parity).
 *
 *   Automatic — pick a method (+ optional centred half-width window + linear
 *               flat-field) → "Center" dispatches `czb_run`.
 *   Manual    — "Place crosshair" drops a draggable crosshair on the DP
 *               (`czb_open`); drag it onto the zero beam; "Apply"
 *               dispatches `czb_pick`. The crosshair is removed on Apply or
 *               when the caret / Manual tab is left (`czb_close`).
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Check, S } from './WizardShell'
import { useWizardLifecycle } from './wizardHooks'

const TABS = ['Automatic', 'Manual'] as const
type Tab = typeof TABS[number]

interface Props {
  openUp: boolean
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

export function CenterZeroBeamWizard({ openUp, windowId, sendAction, onClose }: Props) {
  const [tab, setTab] = React.useState<Tab>('Automatic')
  const [method, setMethod] = React.useState('center_of_mass')
  const [halfWidth, setHalfWidth] = React.useState(0)
  const [flat, setFlat] = React.useState(false)
  const [status, setStatus] = React.useState('Center the direct beam automatically or by hand.')

  // Manual crosshair lifecycle: add when the Manual tab is active, remove
  // otherwise (and always on unmount). Re-fires on tab switch.
  useWizardLifecycle({
    windowId, sendAction,
    openAction: tab === 'Manual' ? 'czb_open' : 'czb_close',
    closeAction: 'czb_close',
    deps: [tab],
  })

  const center = () => {
    setStatus('Centering…')
    sendAction('czb_run', { method, half_square_width: halfWidth, make_flat_field: flat }, windowId)
  }
  const apply = () => {
    setStatus('Applying manual center…')
    sendAction('czb_pick', {}, windowId)
  }

  return (
    <WizardShell testid="center-zero-beam-wizard" title="Center Zero Beam" openUp={openUp}
      onClose={onClose} closeTestid="czb-close" status={status} statusTestid="czb-status">
      <TabRow tabs={TABS} active={tab} onSelect={setTab} testid={(t) => `czb-tab-${t}`} />

      {tab === 'Automatic' && (
        <div style={S.page}>
          <Field label="Method">
            <select data-testid="czb-method" style={S.sel} value={method}
              onChange={(e) => setMethod(e.target.value)}>
              <option value="center_of_mass">center of mass</option>
            </select>
          </Field>
          <Field label="Half-width (px)">
            <NumInput testid="czb-halfwidth" value={halfWidth} onChange={setHalfWidth} step="1" />
          </Field>
          <Check testid="czb-flatfield" checked={flat} onChange={setFlat} label="Linear flat field" />
          <button data-testid="czb-center" style={S.primary} onClick={center}>Center</button>
        </div>
      )}

      {tab === 'Manual' && (
        <div style={S.page}>
          <div style={S.hint}>Drag the yellow crosshair on the pattern onto the zero beam, then Apply.</div>
          <button data-testid="czb-apply" style={S.primary} onClick={apply}>Apply</button>
        </div>
      )}
    </WizardShell>
  )
}
