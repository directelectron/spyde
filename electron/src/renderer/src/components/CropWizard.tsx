/**
 * CropWizard.tsx — the Crop caret (laundry item #4: interactive rectangle ROI
 * instead of typed-in-only values).
 *
 * Activating Crop drops a draggable/resizable rectangle widget on the source
 * plot (backend: crop_open) covering the full frame; the user drags it to the
 * desired spatial box. The widget is the PRIMARY input (mirrors the Center
 * Zero Beam search-window precedent) — Crop reads its LIVE geometry at Run
 * time, not the (possibly stale) typed fields. The numeric fields stay
 * editable and sync BOTH ways:
 *   - drag → fields: read the widget's own pointer_move/pointer_up geometry
 *     off the existing `spyde:figure_event` CustomEvent (already re-broadcast
 *     by SpyDEContext for every widget interaction) — no new IPC needed.
 *   - field edit → widget: dispatches `crop_set_region`, which repositions
 *     the live widget to match.
 * Closing the caret (✕, toggling Crop off, or the window closing) sends
 * `crop_close`, which hides the widget (crop_open/close mirror the Center
 * Zero Beam manual-crosshair teardown in center_zero_beam.py).
 */
import React from 'react'
import { WizardShell, Field, NumInput, S } from './WizardShell'
import { useWizardLifecycle, useWizardEvent } from './wizardHooks'
import { useSpyDE } from '../kernel/SpyDEContext'

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

export function CropWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const { state } = useSpyDE()
  const [box, setBox] = React.useState({ x0: 0, x1: 0, y0: 0, y1: 0 })
  const [status, setStatus] = React.useState(
    'Drag the box to the region to keep, then Crop.',
  )

  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'crop_open',
    closeAction: 'crop_close',
    deps: [windowId],
  })

  // The current window's live figId — spyde:figure_event carries {figId,
  // event}, not windowId, so this resolves which events belong to OUR plot.
  const figId = state.windows.get(windowId)?.figures.find(f => !f.isNavigator)?.figId
    ?? state.windows.get(windowId)?.figures[0]?.figId

  // Drag → fields: the rectangle widget's pointer_move/pointer_up events carry
  // its full geometry (x, y, w, h). Cheap — no backend round trip.
  useWizardEvent('figure_event', windowId, (detail) => {
    const d = detail as { figId?: string; event?: Record<string, unknown> }
    if (!figId || d.figId !== figId) return
    const ev = d.event
    if (!ev || ev.type !== 'rectangle') return
    const x = Number(ev.x), y = Number(ev.y), w = Number(ev.w), h = Number(ev.h)
    if (![x, y, w, h].every(Number.isFinite)) return
    setBox({ x0: Math.round(x), x1: Math.round(x + w), y0: Math.round(y), y1: Math.round(y + h) })
  })

  // Field edit → widget: push the new box back so the on-plot rectangle stays
  // in sync (crop_set_region repositions the live widget; cheap, no recompute).
  const setField = (key: keyof typeof box, v: number) => {
    const next = { ...box, [key]: v }
    setBox(next)
    sendAction('crop_set_region', next, windowId)
  }

  const doCrop = () => {
    setStatus('Cropping…')
    // The backend reads the WIDGET's live geometry as the primary input (see
    // CropAction.build_kwargs) — the typed fields ride along as a fallback for
    // hosts with no widget (notebook/script use of CropAction directly).
    sendAction('toolbar_action', { name: 'Crop', params: box }, windowId)
    onClose()
  }

  return (
    <WizardShell testid="crop-wizard" title="Crop" posStyle={caretPos}
      onClose={onClose} closeTestid="crop-close" status={status} statusTestid="crop-status">
      <div style={S.page}>
        <Field label="X start"><NumInput testid="crop-x0" value={box.x0} step="1" onChange={(v) => setField('x0', v)} /></Field>
        <Field label="X end"><NumInput testid="crop-x1" value={box.x1} step="1" onChange={(v) => setField('x1', v)} /></Field>
        <Field label="Y start"><NumInput testid="crop-y0" value={box.y0} step="1" onChange={(v) => setField('y0', v)} /></Field>
        <Field label="Y end"><NumInput testid="crop-y1" value={box.y1} step="1" onChange={(v) => setField('y1', v)} /></Field>
        <button data-testid="crop-run" style={S.primary} onClick={doCrop}>Crop</button>
      </div>
    </WizardShell>
  )
}
