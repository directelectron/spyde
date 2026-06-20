import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'

export function StatusBar({ logOpen, onToggleLog }: {
  logOpen?: boolean
  onToggleLog?: () => void
}) {
  const { state } = useSpyDE()
  // Badge unseen warnings/errors so problems are noticeable while the log is hidden.
  const problems = state.logEntries.filter(
    (e) => e.level === 'WARNING' || e.level === 'ERROR' || e.level === 'CRITICAL',
  ).length

  return (
    <div style={styles.bar}>
      <span style={styles.text} data-testid="status-text">
        {state.status}
      </span>
      {state.dashboardUrl && (
        <button
          style={styles.link}
          onClick={() => window.electron.openExternal(state.dashboardUrl!)}
        >
          Dask dashboard ↗
        </button>
      )}
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
        style={styles.btn}
        onClick={() => window.electron.openFile()}
      >
        Open…
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
  link: {
    background: 'none', border: 'none', color: '#89b4fa',
    fontSize: 12, cursor: 'pointer', padding: 0,
  },
  btn: {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    background: '#313244', border: 'none', color: '#cdd6f4',
    fontSize: 12, cursor: 'pointer', padding: '2px 10px',
    borderRadius: 4,
  },
  btnActive: { background: '#45475a', color: '#89b4fa' },
  badge: {
    fontSize: 10, lineHeight: '14px', minWidth: 14, textAlign: 'center',
    color: '#11111b', background: '#f9e2af', borderRadius: 8, padding: '0 4px',
    fontWeight: 700,
  },
}
