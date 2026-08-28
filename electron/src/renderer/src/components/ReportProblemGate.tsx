/**
 * ReportProblemGate.tsx — bridges the SpyDE context's `reportDialogOpen` flag
 * (set from Help -> Report a Problem…) to ReportProblemDialog.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { ReportProblemDialog } from './ReportProblemDialog'

export function ReportProblemGate() {
  const { reportDialogOpen, closeReportDialog } = useSpyDE()
  if (!reportDialogOpen) return null
  return <ReportProblemDialog onClose={closeReportDialog} />
}
