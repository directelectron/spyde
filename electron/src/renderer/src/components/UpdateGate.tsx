/**
 * UpdateGate.tsx — bridges the SpyDE context's `updateDialogOpen` flag (set
 * from Help -> Check for Updates…) to UpdateDialog.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { UpdateDialog } from './UpdateDialog'

export function UpdateGate() {
  const { updateDialogOpen, closeUpdateDialog } = useSpyDE()
  if (!updateDialogOpen) return null
  return <UpdateDialog onClose={closeUpdateDialog} />
}
