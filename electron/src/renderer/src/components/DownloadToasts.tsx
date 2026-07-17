/**
 * DownloadToasts.tsx — bottom-right notification cards for Examples-menu
 * downloads (backend pooch fetches).
 *
 * Driven by the `spyde:download_progress` / `spyde:download_done` CustomEvents
 * (re-broadcast in SpyDEContext). Each card: dataset name, a determinate
 * progress bar (indeterminate shimmer when the size is unknown), byte readout,
 * and a Cancel button that sends the `download_cancel` action with the
 * download's token — the backend aborts the pooch stream and deletes the
 * partial temp file.
 *
 * A toast only ever appears while bytes actually flow (a cache hit downloads
 * nothing), and `download_done` removes it — success, failure and cancel all
 * end the card; errors additionally surface through the normal status/error
 * line.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'

interface Download {
  label: string
  done: number
  total: number
}

const fmtMB = (bytes: number) => {
  const mb = bytes / 1e6
  return mb >= 100 ? mb.toFixed(0) : mb >= 10 ? mb.toFixed(1) : mb.toFixed(2)
}

export function DownloadToasts() {
  const { sendAction } = useSpyDE()
  const [downloads, setDownloads] = React.useState<Record<string, Download>>({})
  // Cancel is fire-once: grey the button immediately so a slow abort (the flag
  // is only checked per received chunk) doesn't invite repeated clicks.
  const [cancelling, setCancelling] = React.useState<Record<string, boolean>>({})

  React.useEffect(() => {
    const onProgress = (e: Event) => {
      const d = (e as CustomEvent).detail as {
        token: string; label: string; done: number; total: number
      }
      setDownloads((prev) => ({
        ...prev,
        [d.token]: { label: String(d.label), done: Number(d.done), total: Number(d.total) },
      }))
    }
    const onDone = (e: Event) => {
      const token = String((e as CustomEvent).detail?.token ?? '')
      setDownloads((prev) => {
        if (!(token in prev)) return prev
        const next = { ...prev }
        delete next[token]
        return next
      })
      setCancelling((prev) => {
        if (!(token in prev)) return prev
        const next = { ...prev }
        delete next[token]
        return next
      })
    }
    window.addEventListener('spyde:download_progress', onProgress)
    window.addEventListener('spyde:download_done', onDone)
    return () => {
      window.removeEventListener('spyde:download_progress', onProgress)
      window.removeEventListener('spyde:download_done', onDone)
    }
  }, [])

  const entries = Object.entries(downloads)
  if (entries.length === 0) return null

  return (
    <div style={S.stack} data-testid="download-toasts">
      {entries.map(([token, d]) => {
        const pct = d.total > 0 ? Math.min(100, (100 * d.done) / d.total) : null
        const isCancelling = !!cancelling[token]
        return (
          <div key={token} style={S.card} data-testid={`download-toast-${d.label}`}>
            <div style={S.row}>
              <span style={S.title}>Downloading {d.label}</span>
              <button
                data-testid={`download-cancel-${d.label}`}
                style={{ ...S.cancel, opacity: isCancelling ? 0.5 : 1 }}
                disabled={isCancelling}
                onClick={() => {
                  setCancelling((prev) => ({ ...prev, [token]: true }))
                  sendAction('download_cancel', { token })
                }}
              >
                {isCancelling ? 'Cancelling…' : 'Cancel'}
              </button>
            </div>
            <div style={S.track} data-testid={`download-bar-${d.label}`}>
              <div
                style={{
                  ...S.fill,
                  // Unknown size → indeterminate: a partial bar sliding via the
                  // shared shimmer keyframes (injected below).
                  ...(pct == null
                    ? { width: '30%', animation: 'spyde-dl-slide 1.2s ease-in-out infinite' }
                    : { width: `${pct}%` }),
                }}
              />
            </div>
            <div style={S.bytes}>
              {pct == null
                ? `${fmtMB(d.done)} MB`
                : `${fmtMB(d.done)} / ${fmtMB(d.total)} MB (${pct.toFixed(0)}%)`}
            </div>
          </div>
        )
      })}
      {/* One-time keyframes for the indeterminate bar (ConsoleBar/StatusBar idiom). */}
      <style>{`@keyframes spyde-dl-slide {
        0% { margin-left: 0% } 50% { margin-left: 70% } 100% { margin-left: 0% }
      }`}</style>
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  stack: {
    position: 'fixed', right: 12, bottom: 44, zIndex: 9300,
    display: 'flex', flexDirection: 'column', gap: 8,
    width: 280,
  },
  card: {
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 8,
    padding: '8px 10px', boxShadow: '0 10px 28px rgba(0,0,0,0.5)',
    display: 'flex', flexDirection: 'column', gap: 6,
  },
  row: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 },
  title: {
    fontSize: 11.5, fontWeight: 600, color: '#cdd6f4',
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  cancel: {
    flex: '0 0 auto', background: 'transparent', color: '#f38ba8',
    border: '1px solid #45475a', borderRadius: 5, padding: '2px 8px',
    fontSize: 10.5, cursor: 'pointer',
  },
  track: {
    height: 5, borderRadius: 3, background: '#313244', overflow: 'hidden',
  },
  fill: {
    height: '100%', borderRadius: 3, background: '#89b4fa',
    transition: 'width 200ms linear',
  },
  bytes: { fontSize: 10, color: '#a6adc8' },
}
