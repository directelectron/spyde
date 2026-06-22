/**
 * NavShapeGate.tsx — bridges the SpyDE context's pending `navShapePrompt` to the
 * NavShapeDialog, and dispatches the user's choice back to the backend as
 * `confirm_nav_shape`. Lives inside the provider so it sees both real IPC and
 * test-injected `nav_shape_prompt` messages.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { NavShapeDialog } from './NavShapeDialog'

export function NavShapeGate() {
  const { state, sendAction, clearNavShapePrompt } = useSpyDE()
  const prompt = state.navShapePrompt
  if (!prompt) return null
  return (
    <NavShapeDialog
      prompt={prompt}
      onConfirm={(navShape, step, units) => {
        sendAction('confirm_nav_shape', { nav_shape: navShape, step_size: step, units })
        clearNavShapePrompt()
      }}
      onCancel={() => {
        // Cancel still opens the dataset as-loaded (no reshape / recalibration).
        sendAction('confirm_nav_shape', {})
        clearNavShapePrompt()
      }}
    />
  )
}
