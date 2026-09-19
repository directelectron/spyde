/**
 * MultiAngleGate.tsx — bridges the SpyDE context's `multiAngleLoaderOpen` flag
 * (set from File → Load Multi-Angle 4D STEM…) to the MultiAngleLoader, and
 * hands it the context's `sendAction` so the dialog can drive the whole
 * `maped_*` conversation itself.
 *
 * Unlike StackGate, which owns the one confirm action, this dialog sends many
 * actions across its three tabs and renders from the `maped_state` the backend
 * answers each one with — so the bridge is the sender, not a callback per step.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { MultiAngleLoader } from './MultiAngleLoader'

export function MultiAngleGate() {
  const { multiAngleLoaderOpen, closeMultiAngleLoader, sendAction } = useSpyDE()
  if (!multiAngleLoaderOpen) return null
  return <MultiAngleLoader sendAction={sendAction} onClose={closeMultiAngleLoader} />
}
