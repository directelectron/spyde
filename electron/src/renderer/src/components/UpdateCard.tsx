/**
 * UpdateCard.tsx — Claude-Code-style "new version available" card, pinned to the
 * BOTTOM-LEFT corner (the DownloadToasts corner/z-surface idiom, flipped to the
 * left). Replaces the old full-width top UpdateBanner.
 *
 * It reacts to the WHOLE electron-updater lifecycle (updater.ts), not just
 * 'available'/'downloaded' the way the banner did — crucially it now surfaces
 * ERRORS with a Retry, so a flaky/timed-out check is visible and recoverable
 * instead of silently dropped.
 *
 *   available    → "SpyDE X.Y.Z is available" + Download (release notes on expand)
 *   downloading  → progress bar + percent (DownloadToasts' track/fill idiom)
 *   downloaded   → "Restart to update" → quitAndInstallUpdate()
 *   error        → friendly message + Retry → checkForUpdates()
 *   checking     → unobtrusive "Checking for updates…"
 *   idle / not-available → nothing
 *
 * Seeds its initial state from getUpdateInfo().status on mount, so a card that
 * mounts mid-flight (e.g. the startup check already fired) shows the current
 * state rather than nothing. Then subscribes to onUpdateStatus for live updates.
 *
 * Dismissable per session (× like the banner). Dismiss is cleared whenever a NEW
 * status arrives, so a fresh version / a later error re-shows the card — and an
 * error is never dismissed away from its Retry for good. Hidden while the full
 * Check-for-Updates dialog is open (that's the full-control surface).
 */
import React, { useEffect, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'

type UpdateStatus =
  | { state: 'idle' }
  | { state: 'checking' }
  | { state: 'available'; version: string; releaseNotes?: string }
  | { state: 'not-available' }
  | { state: 'downloading'; percent: number }
  | { state: 'downloaded'; version: string }
  | { state: 'error'; message: string }
  | { state: string }

export function UpdateCard() {
  const { updateDialogOpen, openUpdateDialog } = useSpyDE()
  const [status, setStatus] = useState<UpdateStatus>({ state: 'idle' })
  const [dismissed, setDismissed] = useState(false)
  const [notesOpen, setNotesOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    // Seed from the last-known status so a card mounting AFTER the startup check
    // fired still reflects reality (the "mounted mid-flight shows nothing" gap).
    window.electron.getUpdateInfo().then((info) => {
      if (!cancelled) setStatus(info.status as UpdateStatus)
    }).catch(() => { /* dev/e2e without the IPC handler — stay idle */ })

    const dispose = window.electron.onUpdateStatus((s) => {
      // Any NEW status un-dismisses the card: a fresh version or a later error
      // must be able to re-appear even after the user × 'd a prior one.
      setDismissed(false)
      setNotesOpen(false)
      setStatus(s as UpdateStatus)
    })
    return () => { cancelled = true; dispose?.() }
  }, [])

  // idle / not-available render nothing; the dialog and full-control surface own
  // "you're up to date". checking is shown but unobtrusive.
  const st = status.state
  const shown = st === 'available' || st === 'downloading' || st === 'downloaded' ||
    st === 'error' || st === 'checking'
  if (!shown || dismissed || updateDialogOpen) return null

  const notes = st === 'available' ? (status as { releaseNotes?: string }).releaseNotes : undefined

  return (
    <div style={S.stack} data-testid="update-card">
      <div style={S.card}>
        <div style={S.row}>
          <span style={S.title}>{titleFor(status)}</span>
          <button
            data-testid="update-card-dismiss"
            style={S.dismiss}
            onClick={() => setDismissed(true)}
            aria-label="Dismiss"
            title="Dismiss"
          >
            ×
          </button>
        </div>

        {st === 'checking' && (
          <div style={S.subtle} data-testid="update-card-checking">Checking for updates…</div>
        )}

        {st === 'error' && (
          <div style={S.errText} data-testid="update-card-error">
            {(status as { message: string }).message}
          </div>
        )}

        {st === 'downloading' && (
          <>
            <div style={S.track} data-testid="update-card-bar">
              <div style={{ ...S.fill, width: `${(status as { percent: number }).percent}%` }} />
            </div>
            <div style={S.bytes}>{(status as { percent: number }).percent}%</div>
          </>
        )}

        {notes && notesOpen && (
          <div style={S.notes} data-testid="update-card-notes">{notes}</div>
        )}

        {(st === 'available' || st === 'downloaded' || st === 'error') && (
          <div style={S.actions}>
            {st === 'available' && (
              <button
                data-testid="update-card-download"
                style={S.primary}
                onClick={() => window.electron.downloadUpdate()}
              >
                Download
              </button>
            )}
            {st === 'downloaded' && (
              <button
                data-testid="update-card-restart"
                style={S.primary}
                onClick={() => window.electron.quitAndInstallUpdate()}
              >
                Restart to update
              </button>
            )}
            {st === 'error' && (
              <button
                data-testid="update-card-retry"
                style={S.primary}
                onClick={() => window.electron.checkForUpdates()}
              >
                Retry
              </button>
            )}
            {notes && (
              <button
                data-testid="update-card-notes-toggle"
                style={S.ghost}
                onClick={() => setNotesOpen((v) => !v)}
              >
                {notesOpen ? 'Hide notes' : 'Release notes'}
              </button>
            )}
            {(st === 'available' || st === 'downloaded') && (
              <button
                data-testid="update-card-details"
                style={S.ghost}
                onClick={openUpdateDialog}
                title="Open the full update dialog"
              >
                Details
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function titleFor(status: UpdateStatus): string {
  switch (status.state) {
    case 'available':
      return `SpyDE ${(status as { version: string }).version} is available`
    case 'downloading':
      return 'Downloading update…'
    case 'downloaded':
      return `SpyDE ${(status as { version: string }).version} downloaded`
    case 'error':
      return 'Update failed'
    case 'checking':
      return 'SpyDE updates'
    default:
      return 'SpyDE updates'
  }
}

// Bottom-LEFT mirror of DownloadToasts (which lives bottom-RIGHT), same dark
// card surface + z-index so the two stacks read as one system on opposite
// corners, above the StatusBar.
const S: Record<string, React.CSSProperties> = {
  stack: {
    position: 'fixed', left: 12, bottom: 44, zIndex: 9300,
    display: 'flex', flexDirection: 'column', gap: 8,
    width: 280,
  },
  card: {
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 8,
    padding: '10px 12px', boxShadow: '0 10px 28px rgba(0,0,0,0.5)',
    display: 'flex', flexDirection: 'column', gap: 8,
  },
  row: { display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 },
  title: {
    fontSize: 12.5, fontWeight: 600, color: '#cdd6f4', lineHeight: 1.3,
  },
  dismiss: {
    flex: '0 0 auto', background: 'none', border: 'none', color: '#7f849c',
    fontSize: 16, lineHeight: 1, cursor: 'pointer', padding: '0 2px', marginTop: -2,
  },
  subtle: { fontSize: 11.5, color: '#a6adc8' },
  errText: { fontSize: 11.5, color: '#f38ba8', lineHeight: 1.4 },
  notes: {
    fontSize: 11, color: '#a6adc8', lineHeight: 1.45, whiteSpace: 'pre-wrap',
    maxHeight: 140, overflowY: 'auto',
    background: '#11111b', border: '1px solid #313244', borderRadius: 6, padding: '6px 8px',
  },
  track: {
    height: 5, borderRadius: 3, background: '#313244', overflow: 'hidden',
  },
  fill: {
    height: '100%', borderRadius: 3, background: '#89b4fa',
    transition: 'width 200ms linear',
  },
  bytes: { fontSize: 10, color: '#a6adc8' },
  actions: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' },
  primary: {
    background: '#89b4fa', border: 'none', color: '#11111b', fontWeight: 600,
    borderRadius: 6, padding: '5px 14px', fontSize: 11.5, cursor: 'pointer',
  },
  ghost: {
    background: 'transparent', border: '1px solid #45475a', color: '#cdd6f4',
    borderRadius: 6, padding: '5px 10px', fontSize: 11, cursor: 'pointer',
  },
}
