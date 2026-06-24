/**
 * StackGate.tsx — bridges the SpyDE context's `stackDialogOpen` flag (set when
 * the File → Load Stack… menu fires) to the StackDialog, and dispatches the
 * ordered paths back to the backend as `open_stack`.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { StackDialog } from './StackDialog'

export function StackGate() {
  const { stackDialogOpen, closeStackDialog, sendAction } = useSpyDE()
  if (!stackDialogOpen) return null
  return (
    <StackDialog
      onConfirm={(paths) => {
        sendAction('open_stack', { paths })
        closeStackDialog()
      }}
      onCancel={closeStackDialog}
    />
  )
}
