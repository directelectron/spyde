/**
 * DataTable.tsx — the app's generic, data-agnostic table.
 *
 * Columns, sorting, row selection and row virtualisation are all independent of
 * what the rows MEAN; the particle dock (BottomDock) is its first consumer, and
 * the vector list / per-phase OM statistics / fit component list are the next
 * ones. Nothing in here knows about particles.
 *
 * Deliberate implementation choices:
 *
 * - **No virtualisation library, no `react-rnd`.** The renderer hand-rolls this
 *   kind of thing (see SubWindow.tsx / ReportSidebar.tsx, which both explicitly
 *   reject react-rnd for a Pointer-Capture gesture). Windowing here is a plain
 *   `scrollTop`-driven slice with two spacer divs — ~30 lines, no dependency.
 *   `scrollTop` is quantised to the row grid, so scrolling only re-renders when
 *   it crosses a row boundary rather than on every wheel tick.
 *
 * - **Flex divs, not `<table>`.** A virtualised `<table>` needs spacer `<tr>`s
 *   whose height browsers treat as a suggestion; a flex row grid also matches
 *   the app's existing columnar layout (DaskMonitor's worker rows) — fixed-width
 *   `flexShrink: 0` cells, `tabular-nums` on the numbers. ARIA roles carry the
 *   table semantics that the markup no longer does.
 *
 * - **The body sets `userSelect: 'text'` explicitly.** `index.html` sets
 *   `:root { user-select: none }` app-wide (desktop feel: dragging a plot must
 *   not blue-highlight it), so without this a user cannot select a cell's text
 *   to copy it — the same fix LogPanel's body carries.
 *
 * - **Never put a `Dropdown` in a row.** Its menu is `position: absolute;
 *   zIndex: 9500`, which an `overflow: auto` scroll container CLIPS — see the
 *   note at PlotControlDock.tsx's Layers rows. Selects belong in the host
 *   panel's header.
 *
 * Every control carries a `data-testid` (project rule, electron/tests/README.md).
 */
import React from 'react'

// ── Public types ─────────────────────────────────────────────────────────────

export type ColumnAlign = 'left' | 'center' | 'right'
export type SortDir = 'asc' | 'desc'

/** A row is an opaque bag of values; `columns[].key` indexes into it. */
export type DataRow = Record<string, unknown>

export interface DataColumn {
  /** Property read from each row. */
  key: string
  /** Header text. */
  label: string
  /** Fixed pixel width. Omitted → the column flexes to fill leftover space. */
  width?: number
  /** Defaults to 'right' for `numeric` columns, 'left' otherwise. */
  align?: ColumnAlign
  /** Right-align + tabular figures, so digits line up down the column. */
  numeric?: boolean
  /** Header cycles asc → desc → unsorted. Default: true for every column. */
  sortable?: boolean
  /** 'swatch' prefixes the cell with a colour chip (particles by track id). */
  kind?: 'text' | 'swatch'
  /** Decimal places for a numeric cell (default: integers plain, floats 3 dp). */
  precision?: number
  /** Appended to the formatted value ("nm", "nm²"). */
  units?: string
  /** Full control of the displayed text (sorting still uses the raw value). */
  format?: (value: unknown, row: DataRow, index: number) => string
  /** Swatch colour; defaults to `swatchColor(value)`. */
  color?: (value: unknown, row: DataRow, index: number) => string
  /** Header tooltip. */
  title?: string
}

export type RowKey = string | number

export interface DataTableProps {
  columns: DataColumn[]
  rows: DataRow[]
  /** Stable identity for selection. `index` is the row's position in `rows`
   *  (BEFORE sorting), so a key derived from it survives a re-sort. */
  rowKey?: (row: DataRow, index: number) => RowKey
  rowHeight?: number
  headerHeight?: number
  /** Extra rows rendered above/below the viewport (default 8). */
  overscan?: number
  selectionMode?: 'none' | 'single' | 'multi'
  /** Fires on every selection change with the selected keys AND rows. */
  onSelect?: (keys: RowKey[], rows: DataRow[]) => void
  /** Double-click / Enter on a row. */
  onRowActivate?: (row: DataRow, index: number) => void
  initialSort?: { key: string; dir: SortDir } | null
  emptyMessage?: React.ReactNode
  /** Prefix for every `data-testid` this table emits. */
  testid?: string
}

// ── Palette ──────────────────────────────────────────────────────────────────

/** SpyDE's six accents, cycled for swatch cells (particle track colours). */
export const SWATCH_COLORS = [
  '#89b4fa', '#f38ba8', '#a6e3a1', '#f9e2af', '#cba6f7', '#94e2d5',
] as const

/** Stable colour for a swatch value: a non-negative integer indexes the palette
 *  directly (track 0 is always blue); anything else is hashed into it. */
export function swatchColor(value: unknown): string {
  if (typeof value === 'number' && Number.isFinite(value)) {
    const n = Math.trunc(Math.abs(value))
    return SWATCH_COLORS[n % SWATCH_COLORS.length]
  }
  const s = String(value ?? '')
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
  return SWATCH_COLORS[Math.abs(h) % SWATCH_COLORS.length]
}

// ── Value formatting / comparison ────────────────────────────────────────────

const isBlank = (v: unknown) =>
  v == null || v === '' || (typeof v === 'number' && !Number.isFinite(v))

/** Display text for a cell when the column supplies no `format`. */
export function formatValue(value: unknown, col?: DataColumn): string {
  if (isBlank(value)) return '—'
  if (typeof value === 'number') {
    let s: string
    if (col?.precision != null) s = value.toFixed(col.precision)
    else if (Number.isInteger(value)) s = String(value)
    else {
      const a = Math.abs(value)
      s = a < 1e-3 || a >= 1e6 ? value.toExponential(2) : value.toFixed(3)
    }
    return col?.units ? `${s} ${col.units}` : s
  }
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (Array.isArray(value)) return value.length ? value.join(', ') : '—'
  return String(value)
}

function compareValues(a: unknown, b: unknown): number {
  if (typeof a === 'number' && typeof b === 'number') return a - b
  if (typeof a === 'boolean' || typeof b === 'boolean') {
    return (a ? 1 : 0) - (b ? 1 : 0)
  }
  return String(a).localeCompare(String(b), undefined, { numeric: true })
}

/** The visible table as CSV (header row + one line per row), for a Copy button.
 *  Values go through the column's own formatting, so what you paste is what you
 *  saw — quoted only when a comma/quote/newline forces it. */
export function toCsv(columns: DataColumn[], rows: DataRow[]): string {
  const cell = (s: string) =>
    /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  const head = columns.map((c) => cell(c.label)).join(',')
  const body = rows.map((r, i) =>
    columns
      .map((c) => cell(c.format ? c.format(r[c.key], r, i) : formatValue(r[c.key], c)))
      .join(','),
  )
  return [head, ...body].join('\n')
}

// ── Component ────────────────────────────────────────────────────────────────

export function DataTable({
  columns,
  rows,
  rowKey,
  rowHeight = 24,
  headerHeight = 26,
  overscan = 8,
  selectionMode = 'single',
  onSelect,
  onRowActivate,
  initialSort = null,
  emptyMessage = 'No rows.',
  testid = 'data-table',
}: DataTableProps) {
  const bodyRef = React.useRef<HTMLDivElement>(null)
  // Quantised scroll position: the index of the first row at/above the viewport
  // top. Only changes when scrolling crosses a row boundary, so a wheel gesture
  // re-renders ~once per row instead of once per event.
  const [firstRow, setFirstRow] = React.useState(0)
  const [viewH, setViewH] = React.useState(0)
  const [sort, setSort] = React.useState<{ key: string; dir: SortDir } | null>(initialSort)
  const [selected, setSelected] = React.useState<Set<RowKey>>(() => new Set())
  // Anchor for shift-click ranges — an index into the SORTED view.
  const anchorRef = React.useRef<number | null>(null)

  const keyOf = React.useCallback(
    (row: DataRow, index: number): RowKey => (rowKey ? rowKey(row, index) : index),
    [rowKey],
  )

  // Viewport height drives how many rows exist at all. A ResizeObserver keeps it
  // right through dock resizes and window resizes alike.
  React.useEffect(() => {
    const el = bodyRef.current
    if (!el) return
    setViewH(el.clientHeight)
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => setViewH(el.clientHeight))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const onScroll = () => {
    const el = bodyRef.current
    if (!el) return
    const next = Math.max(0, Math.floor(el.scrollTop / rowHeight))
    setFirstRow((cur) => (cur === next ? cur : next))
  }

  // Sort a decorated copy so the ORIGINAL index survives (it is the default row
  // key and the tiebreak that keeps the sort stable).
  const view = React.useMemo(() => {
    const decorated = rows.map((row, index) => ({ row, index }))
    if (!sort) return decorated
    const sign = sort.dir === 'asc' ? 1 : -1
    const k = sort.key
    decorated.sort((a, b) => {
      const va = a.row[k]
      const vb = b.row[k]
      const ba = isBlank(va)
      const bb = isBlank(vb)
      // Blanks sink to the bottom in BOTH directions — "no value" is not an
      // extreme value, and floating them to the top of a descending sort hides
      // the rows you asked to see.
      if (ba !== bb) return ba ? 1 : -1
      if (ba) return a.index - b.index
      const c = compareValues(va, vb) * sign
      return c !== 0 ? c : a.index - b.index
    })
    return decorated
  }, [rows, sort])

  const total = view.length
  // `viewH || 320` keeps the FIRST paint non-empty: the ResizeObserver has not
  // reported yet, and rendering zero rows then would flash a blank table.
  const perView = Math.max(1, Math.ceil((viewH || 320) / rowHeight) + 1)
  // Clamp against a stale `firstRow` (rows can shrink under us when the host
  // filters), then widen by the overscan.
  const clamped = Math.min(firstRow, Math.max(0, total - perView))
  const start = Math.max(0, clamped - overscan)
  const end = Math.min(total, start + perView + overscan * 2)
  const padTop = start * rowHeight
  const padBottom = Math.max(0, (total - end) * rowHeight)

  // Minimum content width so fixed columns never squash; anything wider gets a
  // horizontal scrollbar and the sticky header scrolls with it (correct — a
  // sticky header only pins vertically).
  const minRowWidth = React.useMemo(
    () => columns.reduce((w, c) => w + (c.width ?? MIN_FLEX_WIDTH), 0),
    [columns],
  )

  const emit = (next: Set<RowKey>) => {
    setSelected(next)
    if (!onSelect) return
    const picked: DataRow[] = []
    const keys: RowKey[] = []
    for (const { row, index } of view) {
      const k = keyOf(row, index)
      if (next.has(k)) { keys.push(k); picked.push(row) }
    }
    onSelect(keys, picked)
  }

  const onRowClick = (e: React.MouseEvent, viewIndex: number) => {
    if (selectionMode === 'none') return
    const { row, index } = view[viewIndex]
    const k = keyOf(row, index)
    const multi = selectionMode === 'multi'
    if (multi && (e.ctrlKey || e.metaKey)) {
      const next = new Set(selected)
      if (next.has(k)) next.delete(k)
      else next.add(k)
      anchorRef.current = viewIndex
      emit(next)
      return
    }
    if (multi && e.shiftKey && anchorRef.current != null) {
      const lo = Math.min(anchorRef.current, viewIndex)
      const hi = Math.max(anchorRef.current, viewIndex)
      const next = new Set<RowKey>()
      for (let i = lo; i <= hi; i++) next.add(keyOf(view[i].row, view[i].index))
      emit(next)
      return
    }
    anchorRef.current = viewIndex
    emit(new Set<RowKey>([k]))
  }

  // Identity-stable row callbacks. Without these every Row gets a fresh closure
  // on each render and `React.memo` below can never skip anything — the memo
  // would be decorative rather than the reason scrolling stays cheap at 100k
  // rows. The refs carry the LATEST handler without changing identity.
  const clickRef = React.useRef(onRowClick)
  clickRef.current = onRowClick
  const stableClick = React.useCallback(
    (e: React.MouseEvent, viewIndex: number) => clickRef.current(e, viewIndex), [],
  )
  const activateRef = React.useRef(onRowActivate)
  activateRef.current = onRowActivate
  const stableActivate = React.useCallback(
    (row: DataRow, index: number) => activateRef.current?.(row, index), [],
  )

  const onHeaderClick = (col: DataColumn) => {
    if (col.sortable === false) return
    setSort((cur) => {
      if (!cur || cur.key !== col.key) return { key: col.key, dir: 'asc' }
      if (cur.dir === 'asc') return { key: col.key, dir: 'desc' }
      return null                       // third click clears the sort
    })
    // A re-sort moves rows under a range anchor that no longer means anything.
    anchorRef.current = null
    const el = bodyRef.current
    if (el) { el.scrollTop = 0; setFirstRow(0) }
  }

  return (
    <div style={S.root} data-testid={testid} data-rows={total}>
      <div
        style={{ ...S.body, ...(total === 0 ? { overflow: 'hidden' } : null) }}
        ref={bodyRef}
        onScroll={onScroll}
        data-testid={`${testid}-body`}
        role="table"
        aria-rowcount={total}
      >
        <div style={{ minWidth: minRowWidth }}>
          <div
            style={{ ...S.head, height: headerHeight }}
            role="row"
            data-testid={`${testid}-head`}
          >
            {columns.map((col, ci) => {
              const sorted = sort?.key === col.key ? sort.dir : null
              const canSort = col.sortable !== false
              return (
                <div
                  // Position-based: a caller MAY repeat a key (the same measured
                  // property shown twice under different formatting), and a
                  // duplicate React key silently drops cells.
                  key={`${col.key}-${ci}`}
                  role="columnheader"
                  aria-sort={sorted === 'asc' ? 'ascending'
                    : sorted === 'desc' ? 'descending' : 'none'}
                  data-testid={`${testid}-th-${col.key}`}
                  data-sort={sorted ?? 'none'}
                  title={col.title ?? (canSort ? `Sort by ${col.label}` : col.label)}
                  onClick={() => onHeaderClick(col)}
                  style={{
                    ...S.th,
                    ...cellBox(col),
                    cursor: canSort ? 'pointer' : 'default',
                    color: sorted ? '#89b4fa' : '#a6adc8',
                  }}
                >
                  <span style={S.thLabel}>{col.label}</span>
                  {canSort && (
                    <span style={{ ...S.sortMark, opacity: sorted ? 1 : 0.25 }}>
                      {sorted === 'desc' ? '▾' : '▴'}
                    </span>
                  )}
                </div>
              )
            })}
          </div>

          {total === 0 ? (
            <div style={S.empty} data-testid={`${testid}-empty`}>{emptyMessage}</div>
          ) : (
            <>
              <div style={{ height: padTop }} aria-hidden />
              {view.slice(start, end).map(({ row, index }, i) => {
                const k = keyOf(row, index)
                return (
                  <Row
                    key={k}
                    columns={columns}
                    row={row}
                    index={index}
                    // Position in the SORTED view — stable while scrolling (only
                    // a re-sort or a filter moves it), which is what lets the
                    // memo below skip unchanged rows.
                    viewIndex={start + i}
                    height={rowHeight}
                    selected={selected.has(k)}
                    testid={testid}
                    onPick={stableClick}
                    onActivate={stableActivate}
                  />
                )
              })}
              <div style={{ height: padBottom }} aria-hidden />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// One rendered row. Split out and memoised so a scroll re-renders only the rows
// that ENTERED the window: every prop is identity-stable for a row that stayed
// (see the stableClick/stableActivate refs above — without them the memo would
// never hit).
const Row = React.memo(function Row({
  columns, row, index, viewIndex, height, selected, testid, onPick, onActivate,
}: {
  columns: DataColumn[]
  row: DataRow
  index: number
  viewIndex: number
  height: number
  selected: boolean
  testid: string
  onPick: (e: React.MouseEvent, viewIndex: number) => void
  onActivate: (row: DataRow, index: number) => void
}) {
  const [hover, setHover] = React.useState(false)
  return (
    <div
      role="row"
      data-testid={`${testid}-row`}
      data-row-index={index}
      data-selected={selected ? 'true' : undefined}
      aria-selected={selected}
      onClick={(e) => onPick(e, viewIndex)}
      onDoubleClick={() => onActivate(row, index)}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        ...S.row,
        height,
        background: selected ? 'rgba(137,180,250,0.18)'
          : hover ? 'rgba(137,180,250,0.07)'
            : index % 2 ? 'rgba(255,255,255,0.014)' : 'transparent',
        color: selected ? '#cdd6f4' : '#bac2de',
      }}
    >
      {columns.map((col, ci) => {
        const value = row[col.key]
        const text = col.format ? col.format(value, row, index) : formatValue(value, col)
        return (
          <div
            key={`${col.key}-${ci}`}
            role="cell"
            data-testid={`${testid}-cell`}
            data-col={col.key}
            title={text}
            style={{ ...S.td, ...cellBox(col) }}
          >
            {col.kind === 'swatch' && (
              <span
                data-testid={`${testid}-swatch`}
                style={{
                  ...S.swatch,
                  background: col.color ? col.color(value, row, index) : swatchColor(value),
                }}
              />
            )}
            <span style={S.cellText}>{text}</span>
          </div>
        )
      })}
    </div>
  )
})

/** Fixed-width columns never shrink (DaskMonitor's numeric-column recipe);
 *  width-less columns share the leftover space. */
const MIN_FLEX_WIDTH = 90
function cellBox(col: DataColumn): React.CSSProperties {
  const align = col.align ?? (col.numeric ? 'right' : 'left')
  return {
    ...(col.width != null
      ? { flex: `0 0 ${col.width}px`, width: col.width }
      : { flex: '1 1 auto', minWidth: MIN_FLEX_WIDTH }),
    justifyContent:
      align === 'right' ? 'flex-end' : align === 'center' ? 'center' : 'flex-start',
    textAlign: align,
    fontVariantNumeric: col.numeric ? 'tabular-nums' : undefined,
  }
}

const S: Record<string, React.CSSProperties> = {
  root: { flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' },
  body: {
    flex: 1, minHeight: 0, overflow: 'auto',
    // `index.html` sets `user-select: none` on :root for the desktop feel, so a
    // table has to opt its own text back IN or cells cannot be copied.
    userSelect: 'text', WebkitUserSelect: 'text',
  },
  head: {
    position: 'sticky', top: 0, zIndex: 2,
    display: 'flex', alignItems: 'center',
    background: '#181825',
    borderBottom: '1px solid #313244',
    fontSize: 10.5, fontWeight: 600, letterSpacing: 0.3,
    userSelect: 'none',
  },
  th: {
    display: 'flex', alignItems: 'center', gap: 3,
    padding: '0 8px', height: '100%', overflow: 'hidden',
    whiteSpace: 'nowrap',
  },
  thLabel: { overflow: 'hidden', textOverflow: 'ellipsis' },
  sortMark: { fontSize: 9, flex: '0 0 auto', color: 'inherit' },
  row: {
    display: 'flex', alignItems: 'center',
    fontSize: 11.5, cursor: 'default',
    borderBottom: '1px solid rgba(49,50,68,0.4)',
  },
  td: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '0 8px', height: '100%', overflow: 'hidden', whiteSpace: 'nowrap',
  },
  cellText: { overflow: 'hidden', textOverflow: 'ellipsis' },
  swatch: {
    width: 9, height: 9, borderRadius: 2, flex: '0 0 auto',
    border: '1px solid rgba(0,0,0,0.45)',
  },
  empty: {
    color: '#6c7086', fontStyle: 'italic', fontSize: 11.5,
    padding: '14px 10px', whiteSpace: 'pre-line',
  },
}
