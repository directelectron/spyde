/**
 * LogPanel.tsx — the application log: a bottom drawer that streams Python
 * logging records from the backend, with an on/off toggle (owned by App via the
 * status-bar button) and a verbosity switcher (DEBUG…CRITICAL) styled like the
 * dock's navigator switcher. Switching the level tells the backend to change
 * verbosity and backfills recent history.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useSpyDE, LogEntry } from '../kernel/SpyDEContext'

const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] as const

// Catppuccin-ish per-level colour so severity reads at a glance.
const LEVEL_COLOR: Record<string, string> = {
  DEBUG: '#6c7086',
  INFO: '#a6adc8',
  WARNING: '#f9e2af',
  ERROR: '#f38ba8',
  CRITICAL: '#eba0ac',
}

function clock(time: number): string {
  const d = new Date(time * 1000)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

// Strip the leading "spyde." so the logger column stays short and readable.
function shortName(name: string): string {
  return name.startsWith('spyde.') ? name.slice('spyde.'.length) : name
}

export function LogPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { state, sendAction } = useSpyDE()
  const [clearAt, setClearAt] = useState(0)        // hide entries older than this
  const bodyRef = useRef<HTMLDivElement>(null)
  const followRef = useRef(true)                   // auto-scroll unless user scrolled up

  // On open, ask the backend for the current level's history (backfill).
  useEffect(() => {
    if (open) sendAction('set_log_level', { level: state.logLevel })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const rows = useMemo(
    () => state.logEntries.filter((e) => e.time >= clearAt),
    [state.logEntries, clearAt],
  )

  // Auto-scroll to the newest line while the user is parked at the bottom.
  useEffect(() => {
    const el = bodyRef.current
    if (el && followRef.current) el.scrollTop = el.scrollHeight
  }, [rows.length, open])

  const onScroll = () => {
    const el = bodyRef.current
    if (!el) return
    followRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
  }

  const onLevel = (e: React.ChangeEvent<HTMLSelectElement>) =>
    sendAction('set_log_level', { level: e.target.value })

  if (!open) return null

  return (
    <div style={styles.root} data-testid="log-panel">
      <div style={styles.header}>
        <span style={styles.title}>Application Log</span>
        <span style={styles.count} data-testid="log-count">{rows.length}</span>
        <span style={{ flex: 1 }} />
        <label style={styles.levelLabel}>Level</label>
        <select
          data-testid="log-level-select"
          style={styles.select}
          value={state.logLevel}
          onChange={onLevel}
        >
          {LEVELS.map((l) => <option key={l} value={l}>{l}</option>)}
        </select>
        <button
          data-testid="log-clear"
          style={styles.btn}
          onClick={() => { setClearAt(Date.now() / 1000); followRef.current = true }}
          title="Clear the visible log"
        >
          Clear
        </button>
        <button
          data-testid="log-close"
          style={styles.iconBtn}
          onClick={onClose}
          title="Hide the log panel"
          aria-label="Hide the log panel"
        >
          ×
        </button>
      </div>

      <div style={styles.body} ref={bodyRef} onScroll={onScroll} data-testid="log-body">
        {rows.length === 0 ? (
          <div style={styles.empty} data-testid="log-empty">No log records at this level yet.</div>
        ) : (
          rows.map((e, i) => <LogRow key={i} entry={e} />)
        )}
      </div>
    </div>
  )
}

function LogRow({ entry }: { entry: LogEntry }) {
  const color = LEVEL_COLOR[entry.level] ?? '#cdd6f4'
  return (
    <div style={styles.row} data-testid="log-row" data-level={entry.level}>
      <span style={styles.time}>{clock(entry.time)}</span>
      <span style={{ ...styles.level, color }}>{entry.level.padEnd(8)}</span>
      <span style={styles.name}>{shortName(entry.name)}</span>
      <span style={{ ...styles.msg, color: entry.level === 'DEBUG' ? '#9399b2' : '#cdd6f4' }}>
        {entry.msg}
      </span>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    height: 220,
    flexShrink: 0,
    display: 'flex',
    flexDirection: 'column',
    background: '#11111b',
    borderTop: '1px solid #313244',
  },
  header: {
    display: 'flex', alignItems: 'center', gap: 8,
    height: 30, flexShrink: 0,
    padding: '0 10px',
    background: '#181825',
    borderBottom: '1px solid #313244',
    userSelect: 'none',
  },
  title: { fontSize: 12, fontWeight: 600, color: '#cdd6f4', letterSpacing: 0.3 },
  count: {
    fontSize: 10.5, color: '#a6adc8',
    background: '#313244', borderRadius: 9, padding: '1px 7px',
  },
  levelLabel: { fontSize: 11, color: '#a6adc8' },
  select: {
    background: '#1e1e2e', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 4, padding: '3px 6px',
    fontSize: 12,
  },
  btn: {
    background: '#313244', border: 'none', color: '#cdd6f4',
    fontSize: 12, cursor: 'pointer', padding: '3px 10px', borderRadius: 4,
  },
  iconBtn: {
    background: 'transparent', border: 'none', color: '#a6adc8',
    fontSize: 18, lineHeight: '18px', cursor: 'pointer', padding: '0 4px',
  },
  body: {
    flex: 1, minHeight: 0, overflowY: 'auto',
    padding: '6px 10px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 11.5, lineHeight: 1.5,
  },
  empty: { color: '#6c7086', fontStyle: 'italic', padding: '8px 0' },
  row: { display: 'flex', gap: 8, whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
  time: { color: '#6c7086', flexShrink: 0 },
  level: { flexShrink: 0, whiteSpace: 'pre', fontWeight: 600 },
  name: { color: '#7f849c', flexShrink: 0 },
  msg: { flex: 1, whiteSpace: 'pre-wrap' },
}
