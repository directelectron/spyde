/**
 * GpuStatusGate.tsx — bridges the SpyDE context's `gpuStatusDialogOpen` flag
 * (set from Help -> GPU Status…) to GpuStatusDialog.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { GpuStatusDialog } from './GpuStatusDialog'

export function GpuStatusGate() {
  const { gpuStatusDialogOpen, closeGpuStatusDialog } = useSpyDE()
  if (!gpuStatusDialogOpen) return null
  return <GpuStatusDialog onClose={closeGpuStatusDialog} />
}
