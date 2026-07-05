/**
 * UpdateDialog.tsx — Help -> Check for Updates…
 *
 * Shows the running version, a stable/beta channel toggle, and drives the
 * check -> download -> restart flow via electron-updater (wired in
 * electron/src/main/updater.ts). autoDownload is off, so nothing happens
 * until the user explicitly clicks through each step.
 */
import React, { useEffect, useState } from 'react'

const ACCENT = '#89b4fa'

type UpdateStatus =
  | { state: 'idle' }
  | { state: 'checking' }
  | { state: 'available'; version: string; releaseNotes?: string }
  | { state: 'not-available' }
  | { state: 'downloading'; percent: number }
  | { state: 'downloaded'; version: string }
  | { state: 'error'; message: string }

export function UpdateDialog({ onClose }: { onClose: () => void }) {
  const [channel, setChannel] = useState<'stable' | 'beta'>('stable')
  const [supported, setSupported] = useState(true)
  const [appVersion, setAppVersion] = useState('')
  const [status, setStatus] = useState<UpdateStatus>({ state: 'idle' })

  useEffect(() => {
    let cancelled = false
    window.electron.getUpdateInfo().then((info) => {
      if (cancelled) return
      setChannel(info.channel)
      setSupported(info.supported)
      setAppVersion(info.appVersion)
      setStatus(info.status as UpdateStatus)
    })
    const dispose = window.electron.onUpdateStatus((s) => setStatus(s as UpdateStatus))
    return () => {
      cancelled = true
      dispose?.()
    }
  }, [])

  const setChannelAndPersist = (next: 'stable' | 'beta') => {
    setChannel(next)
    window.electron.setUpdateChannel(next)
  }

  return (
    <div style={styles.overlay} data-testid="update-dialog">
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>Check for Updates</h3>
        <p style={styles.sub}>
          Current version <strong>{appVersion || '…'}</strong>
        </p>

        {!supported && (
          <p style={styles.notice}>
            This build doesn't support auto-update (a dev or unpackaged run).
            Download new releases from GitHub instead.
          </p>
        )}

        <div style={styles.channelRow}>
          <span style={styles.channelLabel}>Update channel</span>
          <div style={styles.channelToggle}>
            {(['stable', 'beta'] as const).map((c) => (
              <button
                key={c}
                data-testid={`update-channel-${c}`}
                onClick={() => setChannelAndPersist(c)}
                style={{
                  ...styles.channelBtn,
                  background: channel === c ? ACCENT : 'transparent',
                  color: channel === c ? '#11111b' : '#cdd6f4',
                }}
              >
                {c === 'stable' ? 'Stable' : 'Beta'}
              </button>
            ))}
          </div>
        </div>

        <StatusPanel status={status} supported={supported} />

        <div style={styles.footer}>
          <button data-testid="update-close" style={styles.cancel} onClick={onClose}>
            Close
          </button>
          <ActionButton status={status} supported={supported} />
        </div>
      </div>
    </div>
  )
}

function StatusPanel({ status, supported }: { status: UpdateStatus; supported: boolean }) {
  if (!supported) return null
  switch (status.state) {
    case 'checking':
      return <p style={styles.status}>Checking for updates…</p>
    case 'available':
      return (
        <p style={styles.status} data-testid="update-available-text">
          SpyDE {status.version} is available.
        </p>
      )
    case 'not-available':
      return <p style={styles.status}>You're up to date.</p>
    case 'downloading':
      return (
        <div style={styles.status} data-testid="update-download-progress">
          <div>Downloading… {status.percent}%</div>
          <div style={styles.progressTrack}>
            <div style={{ ...styles.progressFill, width: `${status.percent}%` }} />
          </div>
        </div>
      )
    case 'downloaded':
      return (
        <p style={styles.status} data-testid="update-downloaded-text">
          SpyDE {status.version} downloaded — restart to install.
        </p>
      )
    case 'error':
      return <p style={styles.error}>Update check failed: {status.message}</p>
    default:
      return null
  }
}

function ActionButton({ status, supported }: { status: UpdateStatus; supported: boolean }) {
  if (!supported) return null
  if (status.state === 'downloaded') {
    return (
      <button
        data-testid="update-restart"
        style={styles.confirm}
        onClick={() => window.electron.quitAndInstallUpdate()}
      >
        Restart to Install
      </button>
    )
  }
  if (status.state === 'available') {
    return (
      <button
        data-testid="update-download"
        style={styles.confirm}
        onClick={() => window.electron.downloadUpdate()}
      >
        Download
      </button>
    )
  }
  const checking = status.state === 'checking' || status.state === 'downloading'
  return (
    <button
      data-testid="update-check-now"
      style={{ ...styles.confirm, opacity: checking ? 0.6 : 1 }}
      disabled={checking}
      onClick={() => window.electron.checkForUpdates()}
    >
      Check Now
    </button>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 9500,
    background: 'rgba(17,17,27,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  dialog: {
    width: 380, display: 'flex', flexDirection: 'column',
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 10,
    padding: 18, color: '#cdd6f4', boxShadow: '0 16px 40px rgba(0,0,0,0.55)',
    fontSize: 13,
  },
  title: { margin: '0 0 4px', fontSize: 16, fontWeight: 600 },
  sub: { margin: '0 0 14px', fontSize: 12, color: '#a6adc8' },
  notice: {
    margin: '0 0 14px', fontSize: 11.5, color: '#f9e2af',
    background: 'rgba(249,226,175,0.08)', border: '1px solid rgba(249,226,175,0.25)',
    borderRadius: 6, padding: '8px 10px', lineHeight: 1.4,
  },
  channelRow: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    marginBottom: 14,
  },
  channelLabel: { fontSize: 12, color: '#a6adc8' },
  channelToggle: {
    display: 'flex', background: '#11111b', borderRadius: 7,
    border: '1px solid #313244', padding: 2, gap: 2,
  },
  channelBtn: {
    border: 'none', borderRadius: 5, padding: '4px 12px',
    fontSize: 12, fontWeight: 600, cursor: 'pointer',
    transition: 'background 120ms ease, color 120ms ease',
  },
  status: {
    fontSize: 12.5, color: '#cdd6f4', margin: '0 0 14px',
    background: '#11111b', border: '1px solid #313244', borderRadius: 6,
    padding: '10px 12px',
  },
  error: {
    fontSize: 12.5, color: '#f38ba8', margin: '0 0 14px',
    background: 'rgba(243,139,168,0.08)', border: '1px solid rgba(243,139,168,0.3)',
    borderRadius: 6, padding: '10px 12px',
  },
  progressTrack: {
    marginTop: 6, height: 5, borderRadius: 3, background: '#313244', overflow: 'hidden',
  },
  progressFill: { height: '100%', background: ACCENT, borderRadius: 3 },
  footer: { display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 'auto' },
  cancel: {
    background: 'transparent', border: '1px solid #313244', color: '#cdd6f4',
    borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 12,
  },
  confirm: {
    background: ACCENT, border: 'none', color: '#11111b', fontWeight: 600,
    borderRadius: 6, padding: '6px 18px', cursor: 'pointer', fontSize: 12,
  },
}
