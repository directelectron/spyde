/**
 * CenterZeroBeamWizard.tsx — the Center-Zero-Beam caret (Qt two-tab parity).
 *
 *   Automatic — pick a method (+ optional centred half-width window + linear
 *               flat-field) → "Center" dispatches `czb_run`. The half-width
 *               window is shown as a DRAGGABLE/resizable rectangle widget on
 *               the DP (czb_set_region); dragging it is the primary input —
 *               its live size is read back into the half-width field via the
 *               same `spyde:figure_event` re-broadcast the Crop caret uses
 *               (see CropWizard.tsx), and czb_run itself re-reads the widget
 *               so a stale field value can't win.
 *   Manual    — "Place crosshair" drops a draggable crosshair on the DP
 *               (`czb_open`); drag it onto the zero beam; "Apply"
 *               dispatches `czb_pick`. The crosshair is removed on Apply or
 *               when the caret / Manual tab is left (`czb_close`).
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Check, S } from './WizardShell'
import { useWizardLifecycle, useWizardEvent } from './wizardHooks'
import { useSpyDE } from '../kernel/SpyDEContext'

const TABS = ['Automatic', 'Manual'] as const
type Tab = typeof TABS[number]

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

export function CenterZeroBeamWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const { state } = useSpyDE()
  const [tab, setTab] = React.useState<Tab>('Automatic')
  const [method, setMethod] = React.useState('center_of_mass')
  const [halfWidth, setHalfWidth] = React.useState(0)
  const [flat, setFlat] = React.useState(false)
  const [status, setStatus] = React.useState('Center the direct beam automatically or by hand.')

  // The live figId for this window's DP — spyde:figure_event carries {figId,
  // event}, not windowId, so this resolves which widget events are ours.
  const figId = state.windows.get(windowId)?.figures.find(f => !f.isNavigator)?.figId
    ?? state.windows.get(windowId)?.figures[0]?.figId

  // Drag → field: the search-window rectangle's own pointer_move/pointer_up
  // events carry its live size; half-width = half the (square) side.
  useWizardEvent('figure_event', windowId, (detail) => {
    if (tab !== 'Automatic') return
    const d = detail as { figId?: string; event?: Record<string, unknown> }
    if (!figId || d.figId !== figId) return
    const ev = d.event
    if (!ev || ev.type !== 'rectangle') return
    const w = Number(ev.w), h = Number(ev.h)
    if (!Number.isFinite(w) || !Number.isFinite(h)) return
    setHalfWidth(Math.round(Math.min(w, h) / 2))
  })

  // Manual crosshair lifecycle: add when the Manual tab is active, remove
  // otherwise (and always on unmount). Re-fires on tab switch.
  useWizardLifecycle({
    windowId, sendAction,
    openAction: tab === 'Manual' ? 'czb_open' : 'czb_close',
    closeAction: 'czb_close',
    deps: [tab],
  })

  // Automatic tab: outline the centering search window on the DP live as the
  // half-width changes. Deferred one tick so it lands AFTER the lifecycle's
  // (also deferred) czb_close when both fire in the same commit — otherwise
  // the close would wipe the box we just drew. czb_close removes it.
  React.useEffect(() => {
    if (tab !== 'Automatic') return
    const t = setTimeout(() => {
      sendAction('czb_set_region', { half_square_width: halfWidth }, windowId)
    }, 0)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, halfWidth])

  const center = () => {
    setStatus('Centering…')
    sendAction('czb_run', { method, half_square_width: halfWidth, make_flat_field: flat }, windowId)
  }
  const apply = () => {
    setStatus('Applying manual center…')
    sendAction('czb_pick', {}, windowId)
  }

  return (
    <WizardShell testid="center-zero-beam-wizard" title="Center Zero Beam" posStyle={caretPos}
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
