/**
 * UpdateBanner.tsx — small dismissible "SpyDE X.Y.Z is available" banner shown
 * when electron-updater finds a newer release and the Check-for-Updates
 * dialog isn't already open. Clicking it opens the dialog; the × dismisses it
 * for this session only (electron-updater will still report the same status
 * next time the dialog opens or a fresh check runs).
 *
 * There's no toast/notification primitive in this codebase (only the single-
 * line StatusBar text + the non-dismissible BackendExitedOverlay) — this is
 * the first, so it's deliberately small and self-contained rather than a new
 * generic "toast" system.
 */
import React, { useEffect, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'

type UpdateStatus =
  | { state: 'available'; version: string }
  | { state: 'downloaded'; version: string }
  | { state: string }

export function UpdateBanner() {
  const { updateDialogOpen, openUpdateDialog } = useSpyDE()
  const [status, setStatus] = useState<UpdateStatus>({ state: 'idle' })
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    const dispose = window.electron.onUpdateStatus((s) => setStatus(s as UpdateStatus))
    return () => dispose?.()
  }, [])

  const available = status.state === 'available' || status.state === 'downloaded'
  if (!available || dismissed || updateDialogOpen) return null

  const version = (status as { version: string }).version
  const label = status.state === 'downloaded'
    ? `SpyDE ${version} downloaded — restart to install`
    : `SpyDE ${version} is available`

  return (
    <div style={styles.banner} data-testid="update-banner">
      <span style={styles.text} onClick={openUpdateDialog}>{label} — Update</span>
      <button
        data-testid="update-banner-dismiss"
        style={styles.dismiss}
        onClick={() => setDismissed(true)}
        aria-label="Dismiss"
      >
        ×
      </button>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  banner: {
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '4px 12px',
    background: 'rgba(137,180,250,0.12)',
    borderBottom: '1px solid rgba(137,180,250,0.3)',
    flexShrink: 0,
  },
  text: {
    fontSize: 12, color: '#89b4fa', cursor: 'pointer', flex: 1,
  },
  dismiss: {
    background: 'none', border: 'none', color: '#89b4fa',
    fontSize: 15, lineHeight: 1, cursor: 'pointer', padding: '0 2px',
  },
}
