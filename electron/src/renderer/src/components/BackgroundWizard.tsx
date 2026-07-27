/**
 * BackgroundWizard.tsx — the Remove Background caret.
 *
 * This existed only as a backend half. `background_action.py` has the whole
 * staged set — `bg_open` / `bg_close` / `bg_set_model` / `bg_set_region` /
 * `bg_apply`, a `PARAMETERS` schema registered under the `bg` key, and a
 * controller that adds a green span widget to the plot and removes it in
 * `remove()`. What was missing was the caret, so the action was reached
 * through the plain YAML toolbar path: clicking it fired `bg_open`, which
 * added the span, and nothing ever unmounted to fire `bg_close`. The span
 * stayed on the plot for the life of the window.
 *
 * The lifecycle is not something to reimplement here — `useWizardLifecycle`
 * fires the open on mount and the close on unmount, and FloatingToolbar only
 * renders one caret at a time, so selecting another action unmounts this one
 * and the teardown that already existed finally runs.
 */
import React from 'react'
import { WizardShell, Field, NumInput, Select, S } from './WizardShell'
import { useWizardLifecycle, useWizardEvent, useDebouncedAction } from './wizardHooks'

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

const MODELS = ['PowerLaw', 'Offset', 'Polynomial', 'Exponential']

export function BackgroundWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const [model, setModel] = React.useState('PowerLaw')
  const [region, setRegion] = React.useState({ x0: 0, x1: 0 })
  const [status, setStatus] = React.useState(
    'Drag the green band over a region that is background only.')

  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'bg_open',
    closeAction: 'bg_close',
    deps: [windowId],
  })

  // The band is the PRIMARY input, so the caret follows it rather than owning
  // it — dragging on the plot updates these fields, not the other way round.
  useWizardEvent('spyde:bg_state', windowId, (detail) => {
    const d = detail as { model?: string; x0?: number; x1?: number; status?: string }
    if (d.model) setModel(d.model)
    if (Number.isFinite(d.x0) && Number.isFinite(d.x1)) {
      setRegion({ x0: Number(d.x0), x1: Number(d.x1) })
    }
    if (d.status) setStatus(d.status)
  })

  // Typing in the fields re-places the band. Debounced: each one re-fits the
  // preview, and a keystroke rate of those is work arriving faster than it is
  // served — the same trap the Fit caret's navigator coalescer exists for.
  const pushRegion = useDebouncedAction(sendAction, 'bg_set_region', windowId)

  const setField = (key: 'x0' | 'x1', v: number) => {
    const next = { ...region, [key]: v }
    setRegion(next)
    pushRegion(() => next)
  }

  return (
    <WizardShell testid="bg-wizard" title="Remove Background" posStyle={caretPos}
      width={380} onClose={onClose} closeTestid="bg-close"
      status={status} statusTestid="bg-status">
      {/* Keep mousedown inside the caret — letting it bubble re-renders the
          subwindow subtree between mousedown and mouseup, so the click never
          lands. See the same note in FitWizard. */}
      <div onMouseDown={(e) => e.stopPropagation()}>
        <div style={S.page}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
            <Field label="Background">
              <Select testid="bg-model" value={model}
                options={MODELS.map((m) => ({ value: m, label: m }))}
                onChange={(v) => { setModel(v); sendAction('bg_set_model', { model: v }, windowId) }} />
            </Field>
            <Field label="From">
              <NumInput testid="bg-x0" value={region.x0} step="any" width={70}
                onChange={(v) => setField('x0', v)} />
            </Field>
            <Field label="To">
              <NumInput testid="bg-x1" value={region.x1} step="any" width={70}
                onChange={(v) => setField('x1', v)} />
            </Field>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button data-testid="bg-apply" style={S.primary}
              title="Fit the background at every position and subtract it into a new node"
              onClick={() => {
                setStatus('Removing the background…')
                sendAction('bg_apply', {}, windowId)
              }}>
              Remove Background
            </button>
          </div>
        </div>
      </div>
    </WizardShell>
  )
}
