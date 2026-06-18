import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'

export function StatusBar() {
  const { state, sendAction } = useSpyDE()

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
    background: '#313244', border: 'none', color: '#cdd6f4',
    fontSize: 12, cursor: 'pointer', padding: '2px 10px',
    borderRadius: 4,
  },
}
