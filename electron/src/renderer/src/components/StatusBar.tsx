import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { DaskMonitor } from './DaskMonitor'

// One-time keyframes for the loading spinner (renderer has no global CSS file).
if (typeof document !== 'undefined' && !document.getElementById('spyde-spin-kf')) {
  const el = document.createElement('style')
  el.id = 'spyde-spin-kf'
  el.textContent = '@keyframes spyde-spin { to { transform: rotate(360deg) } }'
  document.head.appendChild(el)
}

export function StatusBar({ logOpen, onToggleLog }: {
  logOpen?: boolean
  onToggleLog?: () => void
}) {
  const { state, openStackDialog, tileWindowsRef } = useSpyDE()
  const hasWindows = Array.from(state.windows.values()).some(w => w.visible)
  // Badge unseen warnings/errors so problems are noticeable while the log is hidden.
  const problems = state.logEntries.filter(
    (e) => e.level === 'WARNING' || e.level === 'ERROR' || e.level === 'CRITICAL',
  ).length

  const busy = state.loading.busy

  return (
    <div style={styles.bar}>
      {/* Spinner during a long file read so the cold-cache load of a big file
          doesn't look hung. */}
      {busy && <span data-testid="loading-spinner" style={styles.spinner} aria-label="Loading" />}
      <span style={styles.text} data-testid="status-text">
        {busy && state.loading.text ? state.loading.text : state.status}
      </span>
      {/* Live compute readout (CPU / GPU / tasks) — click for per-worker detail
          + the full-dashboard link (the old standalone "Dask dashboard ↗"
          button folded into the popover). */}
      <DaskMonitor />
      <button
        data-testid="toggle-log"
        style={{ ...styles.btn, ...(logOpen ? styles.btnActive : null) }}
        onClick={onToggleLog}
        title={logOpen ? 'Hide application log' : 'Show application log'}
      >
        Log
        {problems > 0 && <span style={styles.badge} data-testid="log-badge">{problems}</span>}
      </button>
      <button
        data-testid="tile-windows"
        style={{ ...styles.btn, ...(hasWindows ? null : styles.btnDisabled) }}
        disabled={!hasWindows}
        onClick={() => tileWindowsRef.current?.()}
        title="Arrange all windows into a grid"
      >
        Tile
      </button>
      <button
        style={styles.btn}
        onClick={() => window.electron.openFile()}
      >
        Open…
      </button>
      <button
        data-testid="open-stack"
        style={styles.btn}
        onClick={openStackDialog}
        title="Combine several datasets into one stack"
      >
        Stack…
      </button>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  bar: {
    display: 'flex', alignItems: 'center', gap: 12,
    padding: '0 12px',
    height: 28,
    background: '#181825',
    borderTop: '1px solid #313244',
    flexShrink: 0,
    userSelect: 'none',
  },
  text: { fontSize: 12, color: '#a6adc8', flex: 1 },
  spinner: {
    width: 12, height: 12, borderRadius: '50%', flexShrink: 0,
    border: '2px solid #45475a', borderTopColor: '#89b4fa',
    animation: 'spyde-spin 0.8s linear infinite',
  },
  btn: {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    background: '#313244', border: 'none', color: '#cdd6f4',
    fontSize: 12, cursor: 'pointer', padding: '2px 10px',
    borderRadius: 4,
  },
  btnActive: { background: '#45475a', color: '#89b4fa' },
  btnDisabled: { color: '#585b70', cursor: 'default' },
  badge: {
    fontSize: 10, lineHeight: '14px', minWidth: 14, textAlign: 'center',
    color: '#11111b', background: '#f9e2af', borderRadius: 8, padding: '0 4px',
    fontWeight: 700,
  },
}
