/**
 * LogPanel.tsx — the application log: a bottom drawer that streams Python
 * logging records from the backend, with an on/off toggle (owned by App via the
 * status-bar button) and a verbosity switcher (DEBUG…CRITICAL) styled like the
 * dock's navigator switcher. Switching the level tells the backend to change
 * verbosity and backfills recent history.
 *
 * PERFORMANCE — the row list is VIRTUALISED; do not "simplify" it back to
 * `rows.map(...)`. The panel used to render every buffered record, so each
 * incoming line re-ran ~1000 row components and reconciled ~5000 DOM nodes:
 * measured 1.4 ms/line at 300 buffered records and 4–37 ms/line once the buffer
 * hit its 1000 cap, all of it blocking the renderer's main thread. That is why
 * "the app gets slow around 1000 log lines" — the cost is linear in the BUFFER,
 * not in what's on screen, and it plateaus at its worst exactly at the cap.
 * Only the ~13 rows the 220 px-tall body can show (plus overscan) are mounted
 * now, so the cost is bounded by the VIEWPORT and independent of buffer size.
 * Appends are additionally coalesced per animation frame in SpyDEContext
 * (`queueLog`), so a burst of records costs one render rather than N.
 */
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useSpyDE, type LogEntry } from '../kernel/SpyDEContext'

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

// Derive a short area tag from an entry, falling back to the logger name's
// leading segment if the backend didn't tag it (older records / third party).
function areaOf(e: LogEntry): string {
  if (e.area) return e.area
  const n = e.name.startsWith('spyde.') ? e.name.slice(6) : e.name
  return n.split('.')[0] || 'other'
}

// Stable per-area colour so a given subsystem reads the same across the log.
const AREA_COLORS = ['#89b4fa', '#a6e3a1', '#f9e2af', '#fab387', '#f5c2e7',
  '#94e2d5', '#cba6f7', '#eba0ac', '#74c7ec', '#b4befe']
function areaColor(area: string): string {
  let h = 0
  for (let i = 0; i < area.length; i++) h = (h * 31 + area.charCodeAt(i)) | 0
  return AREA_COLORS[Math.abs(h) % AREA_COLORS.length]
}

// ── Virtualisation ──────────────────────────────────────────────────────────
// Rows WRAP (long messages, embedded tracebacks), so heights are NOT uniform and
// cannot be assumed. Unmeasured rows are assumed one line tall; a row is measured
// exactly ONCE, the first time it mounts, and the cache is what the scroll
// offsets are built from. So estimates only ever affect rows the user has not
// looked at yet, and they converge as the user scrolls.
const ROW_EST = 18            // px — one line at fontSize 11.5 × lineHeight 1.5
const ROW_MIN = 14            // px — floor used to size the follow window
const OVERSCAN = 8            // rows rendered beyond each edge of the viewport
const HEIGHT_CACHE_MAX = 4000 // measured heights retained before pruning

/** Largest i in [0, n) with offsets[i] <= y (offsets is ascending, length n+1). */
function indexAt(offsets: Float64Array, n: number, y: number): number {
  let lo = 0, hi = n - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (offsets[mid] <= y) lo = mid
    else hi = mid - 1
  }
  return lo
}

interface Viewport { top: number; height: number }

/**
 * Windowed list state for `items`: which slice to mount, how to anchor it, and
 * how tall the full list is. `keyOf` must be STABLE per item across renders (the
 * log buffer is a ring, so an array index identifies a different record after
 * every shift) — it keys both the React element and the measured-height cache.
 *
 * TWO anchoring modes, and the split is load-bearing rather than cosmetic:
 *
 *  - FOLLOWING (parked at the newest line, the normal case): the slice is the
 *    last `cover` items and it is anchored to the BOTTOM of the scroller. It does
 *    NOT depend on the height estimates of everything above it, so a first-time
 *    measurement can never move it. Deriving the follow window from measured
 *    offsets instead is an infinite loop: pinning scrollTop to the bottom makes
 *    the window depend on `total`, measuring the newly exposed rows corrects
 *    `total`, which moves the window onto more unmeasured rows, … — React kills
 *    it with "Maximum update depth exceeded" (error #185).
 *  - SCROLLED BACK: the slice comes from the measured offsets and is anchored to
 *    the top at `offsets[start]`. Scroll is not pinned here, so measuring settles
 *    in one extra pass.
 */
function useWindowed<T>(
  items: T[],
  keyOf: (item: T, index: number) => number,
  view: Viewport,
  follow: boolean,
) {
  const heights = useRef(new Map<number, number>())
  const [measureTick, setMeasureTick] = useState(0)

  const { offsets, total } = useMemo(() => {
    const H = heights.current
    const offs = new Float64Array(items.length + 1)
    let acc = 0
    for (let i = 0; i < items.length; i++) {
      offs[i] = acc
      acc += H.get(keyOf(items[i], i)) ?? ROW_EST
    }
    offs[items.length] = acc
    return { offsets: offs, total: acc }
    // keyOf is stable; measureTick invalidates when a measurement changed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, measureTick])

  const n = items.length
  let start = 0, end = 0
  if (n > 0) {
    if (follow) {
      // ROW_MIN (not ROW_EST) so the slice always over-covers the viewport.
      const cover = Math.ceil(Math.max(view.height, 1) / ROW_MIN) + OVERSCAN
      start = Math.max(0, n - cover)
      end = n
    } else {
      start = Math.max(0, indexAt(offsets, n, view.top) - OVERSCAN)
      end = Math.min(n, indexAt(offsets, n, view.top + Math.max(view.height, 1)) + 1 + OVERSCAN)
    }
  }

  /**
   * Measure the mounted rows. ONCE per key — heights only change when the panel
   * width changes, which `resetHeights` handles. Re-measuring every pass is what
   * lets a measurement cascade run away.
   */
  const measure = useCallback((container: HTMLElement | null) => {
    if (!container) return
    const H = heights.current
    const kids = container.children
    let changed = false
    for (let i = 0; i < kids.length; i++) {
      const el = kids[i] as HTMLElement
      const k = Number(el.dataset.vkey)
      if (Number.isNaN(k) || H.has(k)) continue
      const h = el.offsetHeight
      if (h > 0) { H.set(k, h); changed = true }
    }
    // Record keys (>= 0) are monotonic and the buffer is a ring, so old ones are
    // dead forever; drop the oldest half rather than clearing (a clear would
    // re-estimate — and so re-lay-out — every row still on screen). Raw-output
    // keys are negative and already bounded by that buffer, so they are exempt.
    if (H.size > HEIGHT_CACHE_MAX) {
      let maxK = -Infinity
      for (const k of H.keys()) if (k > maxK) maxK = k
      const cut = maxK - HEIGHT_CACHE_MAX / 2
      for (const k of Array.from(H.keys())) if (k >= 0 && k < cut) H.delete(k)
      changed = true
    }
    if (changed) setMeasureTick((t) => t + 1)
  }, [])

  /** Forget all measurements (panel resized → every wrapped row re-wraps). */
  const resetHeights = useCallback(() => {
    if (heights.current.size === 0) return
    heights.current.clear()
    setMeasureTick((t) => t + 1)
  }, [])

  return {
    start, end, total, follow,
    offsetTop: n === 0 ? 0 : offsets[start],
    measure, resetHeights,
  }
}

export function LogPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { state, sendAction } = useSpyDE()
  const [clearAt, setClearAt] = useState(0)        // hide entries older than this
  const [query, setQuery] = useState('')           // free-text search filter
  const [areaFilter, setAreaFilter] = useState('') // '' = all areas
  // Two views: structured backend log records (default) OR the RAW process
  // stdout/stderr the backend/uv emitted. The raw stream is captured even when
  // the backend dies before any structured record arrives (a startup crash), so
  // it's the only place that early failure output is reachable outside the
  // backend-exited overlay. Plain lines + the same search box; no area chips.
  const [raw, setRaw] = useState(false)
  const bodyRef = useRef<HTMLDivElement>(null)
  const sliceRef = useRef<HTMLDivElement>(null)    // the mounted-rows container
  const followRef = useRef(true)                   // auto-scroll unless user scrolled up
  // Mirrored into state because the window ANCHOR depends on it (see useWindowed).
  const [following, setFollowing] = useState(true)
  const [view, setView] = useState<Viewport>({ top: 0, height: 220 })

  // On open, ask the backend for the current level's history (backfill).
  useEffect(() => {
    if (open) sendAction('set_log_level', { level: state.logLevel })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // All areas present in the current buffer, for the area dropdown.
  const areas = useMemo(() => {
    const s = new Set<string>()
    for (const e of state.logEntries) s.add(areaOf(e))
    return Array.from(s).sort()
  }, [state.logEntries])

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    return state.logEntries.filter((e) => {
      if (e.time < clearAt) return false
      if (areaFilter && areaOf(e) !== areaFilter) return false
      if (q) {
        const hay = `${e.level} ${e.name} ${areaOf(e)} ${e.msg}`.toLowerCase()
        if (!hay.includes(q)) return false
      }
      return true
    })
  }, [state.logEntries, clearAt, query, areaFilter])

  // Raw stdout/stderr lines (the "Raw output" view). Filtered by the same search
  // box; no area filter (there are no areas). Kept separate so the structured
  // and raw views never interleave.
  const rawRows = useMemo(() => {
    const q = query.trim().toLowerCase()
    return state.streamLines.filter((l) => !q || l.text.toLowerCase().includes(q))
  }, [state.streamLines, query])

  // Count shown in the header + whether the current view has anything.
  const shownCount = raw ? rawRows.length : rows.length

  // The virtual window over whichever list is showing. Structured records key on
  // the stable `seq` stamped at ingest; raw stdout lines have no id, so they key
  // on NEGATIVE index — distinct from any seq, so the two views can never collide
  // in the shared height cache when the user flips between them.
  const items: unknown[] = raw ? rawRows : rows
  const keyOf = useCallback(
    (item: unknown, index: number) =>
      typeof (item as LogEntry)?.seq === 'number' ? (item as LogEntry).seq! : -(index + 1),
    [],
  )
  const win = useWindowed(items, keyOf, view, following)

  // Track the scroll position + viewport height that drive the window.
  //
  // The unchanged case must NOT reach setState at all — it is checked against a
  // ref, not by returning the previous value from the updater. React's
  // same-value bailout only applies when the fiber has no other pending update,
  // and under a burst of log records there always is one: the "no-op" setState
  // then schedules a render, whose layout effect calls this again, … until React
  // gives up with "Maximum update depth exceeded" (#185). This is the whole
  // reason this reads as a redundant double-guard.
  const viewRef = useRef(view)
  const syncView = useCallback(() => {
    const el = bodyRef.current
    if (!el) return
    const top = el.scrollTop, height = el.clientHeight
    if (viewRef.current.top === top && viewRef.current.height === height) return
    viewRef.current = { top, height }
    setView(viewRef.current)
  }, [])

  const followStateRef = useRef(true)
  const setFollow = useCallback((next: boolean) => {
    followRef.current = next
    if (followStateRef.current === next) return   // see syncView — no no-op setState
    followStateRef.current = next
    setFollowing(next)
  }, [])

  const onScroll = () => {
    const el = bodyRef.current
    if (!el) return
    setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 24)
    syncView()
  }

  // ONE layout effect owns: measure the mounted rows → pin to the bottom while
  // following → resync the window. It must run before paint (useLayoutEffect) so
  // a batch of new records never paints against the previous scroll offset.
  //
  // Auto-scroll is suppressed while a text selection is active in the log —
  // scrolling would collapse/yank the user's selection as records stream in.
  useLayoutEffect(() => {
    const el = bodyRef.current
    if (!el || !open) return
    win.measure(sliceRef.current)
    if (followRef.current) {
      const sel = window.getSelection?.()
      const selectingHere = sel && !sel.isCollapsed && el.contains(sel.anchorNode)
      if (!selectingHere) el.scrollTop = el.scrollHeight
    }
    syncView()
  })

  // A width change re-wraps every long row, invalidating every measured height.
  useEffect(() => {
    const el = bodyRef.current
    if (!el || !open || typeof ResizeObserver === 'undefined') return
    let last = el.clientWidth
    const ro = new ResizeObserver(() => {
      if (el.clientWidth !== last) { last = el.clientWidth; win.resetHeights() }
      syncView()
    })
    ro.observe(el)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, raw])

  const onLevel = (e: React.ChangeEvent<HTMLSelectElement>) =>
    sendAction('set_log_level', { level: e.target.value })

  // Per-frame update profiling: toggle the backend flag live (no restart / env
  // var). When on, each navigator move logs [NAV-PROFILE] + [PAINT-PROFILE] lines
  // (read/dtype/prefetch/lod/levels/transport ms) into this panel — filter on
  // "PROFILE" or paste them to report where a slow update spends its time.
  const [profiling, setProfiling] = useState(false)
  const onToggleProfile = () => {
    const next = !profiling
    setProfiling(next)
    sendAction('set_debug_flag', { name: 'nav_profile', value: next })
    if (next) setQuery('PROFILE')          // auto-filter to the profile lines
    else if (query === 'PROFILE') setQuery('')
  }

  // Copy the visible log as plain text (tab-separated, one record per line).
  const [copied, setCopied] = useState(false)
  const onCopy = async () => {
    const text = raw
      ? rawRows.map((l) => l.text.replace(/\n$/, '')).join('\n')
      : rows
          .map((e) => `${clock(e.time)}\t${e.level}\t[${areaOf(e)}]\t${shortName(e.name)}\t${e.msg}`)
          .join('\n')
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      // clipboard API unavailable (rare in Electron) — no-op
    }
  }

  if (!open) return null

  return (
    <div style={styles.root} data-testid="log-panel">
      <div style={styles.header}>
        <span style={styles.title}>{raw ? 'Raw Output' : 'Application Log'}</span>
        <span style={styles.count} data-testid="log-count">{shownCount}</span>
        <button
          data-testid="log-raw-toggle"
          style={{ ...styles.btn, ...(raw ? styles.btnActive : null) }}
          onClick={() => { setRaw(v => !v); setFollow(true) }}
          title={raw
            ? 'Show the structured backend log records'
            : 'Show the raw process stdout/stderr (captures startup crashes the structured log misses)'}
        >
          {raw ? 'Structured log' : 'Raw output'}
        </button>
        <input
          data-testid="log-search"
          style={styles.search}
          type="text"
          placeholder={raw ? 'Search raw output…' : 'Search logs…'}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          title={raw
            ? 'Filter visible raw output lines'
            : 'Filter visible log lines (matches level, logger, area, and message)'}
        />
        {!raw && (
          <select
            data-testid="log-area-select"
            style={styles.select}
            value={areaFilter}
            onChange={(e) => setAreaFilter(e.target.value)}
            title="Show only one subsystem's logs"
          >
            <option value="">All areas</option>
            {areas.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        )}
        <span style={{ flex: 1 }} />
        {!raw && <>
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
            data-testid="log-profile"
            style={{ ...styles.btn, ...(profiling ? styles.btnActive : null) }}
            onClick={onToggleProfile}
            title="Toggle per-frame navigator update timing (read / levels / transport ms per move)"
          >
            {profiling ? 'Profiling ●' : 'Profile'}
          </button>
        </>}
        <button
          data-testid="log-copy"
          style={styles.btn}
          onClick={onCopy}
          title="Copy the visible log to the clipboard"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
        {!raw && (
          <button
            data-testid="log-clear"
            style={styles.btn}
            onClick={() => { setClearAt(Date.now() / 1000); setFollow(true) }}
            title="Clear the visible log"
          >
            Clear
          </button>
        )}
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
        {shownCount === 0 ? (
          <div style={styles.empty} data-testid="log-empty">
            {raw ? 'No raw output captured yet.' : 'No log records at this level yet.'}
          </div>
        ) : (
          // Full-height spacer keeps the scrollbar honest; only the visible slice
          // is mounted, translated into place.
          <div style={{ height: win.total, position: 'relative' }} data-testid="log-scroller">
            <div
              ref={sliceRef}
              style={win.follow
                // Anchored to the BOTTOM while following: exact regardless of how
                // good the estimates for everything above it are.
                ? { position: 'absolute', bottom: 0, left: 0, right: 0 }
                : { position: 'absolute', top: win.offsetTop, left: 0, right: 0 }}
            >
              {raw
                ? rawRows.slice(win.start, win.end).map((l, i) => (
                    // vkey MUST match keyOf's negative-index scheme — it is what
                    // the height cache is keyed by.
                    <RawRow key={win.start + i} vkey={-(win.start + i + 1)} line={l} />
                  ))
                : rows.slice(win.start, win.end).map((e, i) => (
                    <LogRow
                      key={keyOf(e, win.start + i)}
                      vkey={keyOf(e, win.start + i)}
                      entry={e}
                      onPickArea={setAreaFilter}
                    />
                  ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Memoised: `entry` objects are immutable once buffered and `onPickArea` is a
// stable setState, so a mounted row never re-renders for an unrelated record.
const LogRow = React.memo(function LogRow({ entry, vkey, onPickArea }: {
  entry: LogEntry
  vkey: number
  onPickArea: (a: string) => void
}) {
  const color = LEVEL_COLOR[entry.level] ?? '#cdd6f4'
  const area = areaOf(entry)
  return (
    <div
      style={styles.row}
      data-testid="log-row"
      data-vkey={vkey}
      data-level={entry.level}
      data-area={area}
    >
      <span style={styles.time}>{clock(entry.time)}</span>
      <span style={{ ...styles.level, color }}>{entry.level.padEnd(8)}</span>
      <span
        style={{ ...styles.area, color: areaColor(area) }}
        data-testid="log-area-chip"
        title={`Filter to “${area}”  (logger: ${entry.name})`}
        onClick={() => onPickArea(area)}
      >
        {area}
      </span>
      <span style={{ ...styles.msg, color: entry.level === 'DEBUG' ? '#9399b2' : '#cdd6f4' }}>
        {entry.msg}
      </span>
    </div>
  )
})

const RawRow = React.memo(function RawRow({ line, vkey }: {
  line: { text: string; kind: 'stdout' | 'stderr' }
  vkey: number
}) {
  return (
    <div
      data-testid="log-raw-row"
      data-vkey={vkey}
      style={{ ...styles.rawRow, color: line.kind === 'stderr' ? '#f9c0c9' : '#a6adc8' }}
    >
      {line.text.replace(/\n$/, '')}
    </div>
  )
})

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
  search: {
    background: '#1e1e2e', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 4, padding: '3px 8px',
    fontSize: 12, width: 200,
  },
  btn: {
    background: '#313244', border: 'none', color: '#cdd6f4',
    fontSize: 12, cursor: 'pointer', padding: '3px 10px', borderRadius: 4,
  },
  btnActive: {
    background: '#89b4fa', color: '#11111b', fontWeight: 600,
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
    // Explicitly selectable: the app shell may set user-select:none globally
    // (drag regions / chrome), which would otherwise block selecting log text.
    userSelect: 'text', WebkitUserSelect: 'text', cursor: 'text',
  },
  empty: { color: '#6c7086', fontStyle: 'italic', padding: '8px 0' },
  row: { display: 'flex', gap: 8, whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
  rawRow: { whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
  time: { color: '#6c7086', flexShrink: 0 },
  level: { flexShrink: 0, whiteSpace: 'pre', fontWeight: 600 },
  area: {
    flexShrink: 0, fontWeight: 600, cursor: 'pointer',
    minWidth: 72, whiteSpace: 'pre',
  },
  msg: { flex: 1, whiteSpace: 'pre-wrap' },
}
