/**
 * BottomDock.tsx — the tabbed table dock at the bottom of the app shell.
 *
 * Slots into the App's bottom stack exactly like LogPanel (a `flexShrink: 0`
 * sibling of the body, no z-index games), and hosts the generic `DataTable`:
 *
 *   Table  — one row per particle/track (`particles_table.rows`, columns come
 *            from the backend so this file stays data-agnostic).
 *   Events — the birth/death/merge/split stream (`particles_table.events`),
 *            whose columns ARE fixed here because that record shape is fixed
 *            by `spyde/particles/track.py::ParticleEvent.to_dict`.
 *
 * Visibility lives in SpyDEContext (`tableDockOpen`), not in App state, because
 * MenuBar's View menu has to read and toggle it and only sees the context.
 *
 * Height: resizable by its TOP edge with the SubWindow / ReportSidebar
 * Pointer-Capture gesture (NOT react-rnd, which the app declares but
 * deliberately never uses). Capped at 50% of the window — the whole bottom
 * stack is `flexShrink: 0`, so LogPanel (220) + this + ConsoleBar + StatusBar
 * would otherwise squeeze MDIArea (`flex: 1, minHeight: 0`) to nothing.
 *
 * Backend contract (the Python side is a separate workstream and may not exist
 * yet — the dock degrades to a clear empty state until it does):
 *   → sendAction('particles_query', { window_id })
 *   ← spyde:particles_table   (see ParticlesTableMessage in kernel/protocol.ts)
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import type { ParticlesTableMessage } from '../kernel/protocol'
import { DataTable, toCsv, type DataColumn, type DataRow } from './DataTable'

const MIN_H = 120
const DEFAULT_H = 260
/** Hard ceiling as a fraction of the window — mirrors the `maxHeight: '50%'`
 *  style guard so a drag can't do what the stylesheet forbids. */
const MAX_FRACTION = 0.5

type TabKey = 'table' | 'events'

/** Plan C2's lane colours, reused so the table and the navigator event lane
 *  agree: green birth, red death, mauve merge, yellow split. */
const EVENT_COLORS: Record<string, string> = {
  birth: '#a6e3a1',
  death: '#f38ba8',
  merge: '#cba6f7',
  split: '#f9e2af',
}

/** The Events tab's columns. Fixed here (not backend-supplied) because
 *  `ParticleEvent.to_dict()` is a fixed record: {frame, kind, tracks, particles}. */
const EVENT_COLUMNS: DataColumn[] = [
  { key: 'frame', label: 'frame', width: 80, numeric: true },
  {
    key: 'kind', label: 'event', width: 120, kind: 'swatch',
    color: (v) => EVENT_COLORS[String(v)] ?? '#89b4fa',
  },
  { key: 'tracks', label: 'track ids', width: 150, sortable: false },
  { key: 'particles', label: 'particle rows', sortable: false },
]

export function BottomDock() {
  const { state, sendAction, tableDockOpen, closeTableDock } = useSpyDE()
  const [tab, setTab] = React.useState<TabKey>('table')
  const [height, setHeight] = React.useState(DEFAULT_H)
  const [query, setQuery] = React.useState('')
  const [copied, setCopied] = React.useState(false)
  const [table, setTable] = React.useState<ParticlesTableMessage | null>(null)

  const activeId = state.activeWindowId

  // `sendAction` is recreated on EVERY provider render, so listing it in a dep
  // array re-runs the effect on every unrelated state update — and an effect
  // that re-requests data whose reply IS state becomes an infinite loop (the
  // "flashing preview" bug, documented at ConsoleBar.tsx:225). Route sends
  // through a ref instead.
  const sendRef = React.useRef(sendAction)
  sendRef.current = sendAction

  // Ask for this window's table when the dock opens and whenever the active
  // window changes. Guarded on `activeId` so an empty session sends nothing
  // (the backend logs "Unknown action" for anything it doesn't handle yet).
  React.useEffect(() => {
    if (!tableDockOpen || activeId == null) return
    sendRef.current('particles_query', { window_id: activeId }, activeId)
  }, [tableDockOpen, activeId])

  // The backend's reply, re-broadcast as a DOM CustomEvent by SpyDEContext (the
  // LayersSection idiom) — no reducer state for a panel-local payload.
  React.useEffect(() => {
    const on = (e: Event) => {
      const msg = (e as CustomEvent).detail as ParticlesTableMessage
      // A table for a DIFFERENT window is not ours; `window_id: null` means the
      // backend sent an unscoped table, which always applies.
      if (msg.window_id != null && activeId != null && msg.window_id !== activeId) return
      setTable(msg)
    }
    window.addEventListener('spyde:particles_table', on)
    return () => window.removeEventListener('spyde:particles_table', on)
  }, [activeId])

  // ── Top-edge resize (Pointer-Capture, per SubWindow / ReportSidebar) ───────
  const resizeGesture = React.useRef<{ py: number; h: number } | null>(null)
  const onResizeDown = (e: React.PointerEvent) => {
    try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId) } catch { /* */ }
    resizeGesture.current = { py: e.clientY, h: height }
  }
  const onResizeMove = (e: React.PointerEvent) => {
    const g = resizeGesture.current
    if (!g) return
    // Dragging the TOP edge upwards grows the dock (its bottom edge is pinned).
    const max = Math.max(MIN_H, Math.round(window.innerHeight * MAX_FRACTION))
    setHeight(Math.min(max, Math.max(MIN_H, g.h + (g.py - e.clientY))))
  }
  const onResizeUp = (e: React.PointerEvent) => {
    if (!resizeGesture.current) return
    try { (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId) } catch { /* */ }
    resizeGesture.current = null
  }

  // ── Rows for the active tab ───────────────────────────────────────────────
  const columns: DataColumn[] = React.useMemo(() => {
    if (tab === 'events') return EVENT_COLUMNS
    return (table?.columns ?? []).map((c) => ({ ...c }))
  }, [tab, table])

  const allRows: DataRow[] = React.useMemo(() => {
    if (tab === 'events') return (table?.events ?? []) as unknown as DataRow[]
    return table?.rows ?? []
  }, [tab, table])

  // Free-text filter across every column value. Client-side and deliberately
  // simple — the dock owns filtering so DataTable stays a pure view.
  const rows = React.useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return allRows
    return allRows.filter((r) =>
      columns.some((c) => String(r[c.key] ?? '').toLowerCase().includes(q)),
    )
  }, [allRows, columns, query])

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(toCsv(columns, rows))
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch { /* clipboard unavailable (rare in Electron) — no-op */ }
  }

  // Selection stays renderer-side and is re-broadcast as a CustomEvent so a
  // future particle overlay can highlight the picked rows on the frame without
  // this dock guessing a backend action name that doesn't exist yet.
  const onSelect = (keys: (string | number)[], picked: DataRow[]) => {
    window.dispatchEvent(new CustomEvent('spyde:particles_selection', {
      detail: { tab, window_id: table?.window_id ?? activeId ?? null, keys, rows: picked },
    }))
  }

  if (!tableDockOpen) return null

  const emptyMessage = table
    ? (tab === 'events'
        ? 'No events in this result.\nEvents appear once the linker has run.'
        : 'No rows in this result.')
    : (activeId == null
        ? 'No table data.\nOpen a dataset and run Segment Particles to populate this dock.'
        : 'No table data for this window yet.\nRun Segment Particles, then press Refresh.')

  const title = table?.title ?? 'Particles'

  return (
    <div
      style={{ ...styles.root, height }}
      data-testid="bottom-dock"
      data-tab={tab}
      data-height={height}
    >
      <ResizeHandle onDown={onResizeDown} onMove={onResizeMove} onUp={onResizeUp} />

      <div style={styles.header}>
        <span style={styles.title}>{title}</span>
        <div style={styles.tabs} role="tablist">
          <TabButton id="table" label="Table" active={tab === 'table'} onPick={setTab} />
          <TabButton id="events" label="Events" active={tab === 'events'} onPick={setTab} />
        </div>
        <span style={styles.count} data-testid="bottom-dock-count">
          {rows.length}{rows.length !== allRows.length ? ` / ${allRows.length}` : ''}
        </span>
        {table?.partial && (
          <span style={styles.streaming} data-testid="bottom-dock-streaming">streaming…</span>
        )}
        <input
          data-testid="bottom-dock-search"
          style={styles.search}
          type="text"
          placeholder="Filter rows…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          title="Filter visible rows (matches any column)"
        />
        <span style={{ flex: 1 }} />
        {/* NB: any <Dropdown> belongs HERE, in the header — its menu is
            absolutely positioned at zIndex 9500 and an `overflow: auto` table
            body clips it (see PlotControlDock's Layers note). */}
        <button
          data-testid="bottom-dock-refresh"
          style={{ ...styles.btn, ...(activeId == null ? styles.btnDisabled : null) }}
          disabled={activeId == null}
          onClick={() => {
            if (activeId == null) return
            sendRef.current('particles_query', { window_id: activeId }, activeId)
          }}
          title="Ask the backend for this window's table again"
        >
          Refresh
        </button>
        <button
          data-testid="bottom-dock-copy"
          style={styles.btn}
          onClick={onCopy}
          title="Copy the visible rows as CSV"
        >
          {copied ? 'Copied' : 'Copy CSV'}
        </button>
        <button
          data-testid="bottom-dock-close"
          style={styles.iconBtn}
          onClick={closeTableDock}
          title="Hide the table dock"
          aria-label="Hide the table dock"
        >
          ×
        </button>
      </div>

      {/* Keyed by tab so switching tabs remounts with a clean sort + selection
          (the two tabs share no columns, so carrying either across is wrong). */}
      <DataTable
        key={tab}
        testid="particle-table"
        columns={columns}
        rows={rows}
        rowKey={(row, index) => (typeof row.id === 'number' ? row.id : index)}
        selectionMode="multi"
        onSelect={onSelect}
        emptyMessage={emptyMessage}
      />
    </div>
  )
}

function TabButton({ id, label, active, onPick }: {
  id: TabKey; label: string; active: boolean; onPick: (t: TabKey) => void
}) {
  const [hover, setHover] = React.useState(false)
  return (
    <button
      role="tab"
      aria-selected={active}
      data-testid={`bottom-dock-tab-${id}`}
      data-active={active ? 'true' : undefined}
      onClick={() => onPick(id)}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        ...styles.tab,
        background: active ? '#313244' : hover ? '#242438' : 'transparent',
        color: active ? '#89b4fa' : '#a6adc8',
        fontWeight: active ? 600 : 400,
      }}
    >
      {label}
    </button>
  )
}

function ResizeHandle({ onDown, onMove, onUp }: {
  onDown: (e: React.PointerEvent) => void
  onMove: (e: React.PointerEvent) => void
  onUp: (e: React.PointerEvent) => void
}) {
  const [hover, setHover] = React.useState(false)
  return (
    <div
      data-testid="bottom-dock-resize-handle"
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{ ...styles.resizeHandle, background: hover ? '#89b4fa' : 'transparent' }}
    />
  )
}

/** Exported so a particle overlay can paint events in the same colours. */
export { EVENT_COLORS }

const styles: Record<string, React.CSSProperties> = {
  root: {
    position: 'relative',
    flexShrink: 0,
    // Belt-and-braces with the drag clamp: the bottom stack never shrinks, so an
    // unbounded dock would starve the MDI area.
    maxHeight: '50%',
    minHeight: MIN_H,
    display: 'flex',
    flexDirection: 'column',
    background: '#11111b',
    borderTop: '1px solid #313244',
  },
  resizeHandle: {
    position: 'absolute', top: -3, left: 0, right: 0, height: 6,
    cursor: 'ns-resize', zIndex: 5,
    transition: 'background 120ms ease',
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
  tabs: { display: 'flex', alignItems: 'center', gap: 2, marginLeft: 4 },
  tab: {
    border: 'none', borderRadius: 5, cursor: 'pointer',
    padding: '3px 11px', fontSize: 12,
    transition: 'background 100ms ease, color 100ms ease',
  },
  count: {
    fontSize: 10.5, color: '#a6adc8',
    background: '#313244', borderRadius: 9, padding: '1px 7px',
    fontVariantNumeric: 'tabular-nums',
  },
  streaming: { fontSize: 10.5, color: '#f9e2af' },
  search: {
    background: '#1e1e2e', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 4, padding: '3px 8px',
    fontSize: 12, width: 180,
  },
  btn: {
    background: '#313244', border: 'none', color: '#cdd6f4',
    fontSize: 12, cursor: 'pointer', padding: '3px 10px', borderRadius: 4,
  },
  btnDisabled: { color: '#585b70', cursor: 'default' },
  iconBtn: {
    background: 'transparent', border: 'none', color: '#a6adc8',
    fontSize: 18, lineHeight: '18px', cursor: 'pointer', padding: '0 4px',
  },
}
