/**
 * GpuHelpGate.tsx — bridges the SpyDE context's `gpuHelpDialogOpen` flag
 * (set from Help -> GPU & CUDA) to GpuHelpDialog.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { GpuHelpDialog } from './GpuHelpDialog'

export function GpuHelpGate() {
  const { gpuHelpDialogOpen, closeGpuHelpDialog } = useSpyDE()
  if (!gpuHelpDialogOpen) return null
  return <GpuHelpDialog onClose={closeGpuHelpDialog} />
}
