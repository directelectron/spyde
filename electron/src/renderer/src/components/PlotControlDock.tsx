/**
 * PlotControlDock.tsx — right-hand dock with controls for the active plot.
 *
 * Colormap, contrast (display range), and basic metadata. Sends set_colormap /
 * set_clim actions to Python targeting the active window.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import type { TreeNode, AxisRow } from '../kernel/SpyDEContext'
import { CompositionPanel } from './CompositionPanel'

// Compact workflow tree: each step is a row with a depth guide-rail, a node dot,
// and the step name. The active (displayed) node is highlighted. Hovering tints
// the row so it reads as clickable.
function TreeNodes({ nodes, depth, activeId, onPick }:
  { nodes: TreeNode[]; depth: number; activeId: number | null; onPick: (id: number) => void }) {
  const [hover, setHover] = React.useState<number | null>(null)
  return (
    <>
      {nodes.map((n) => {
        const active = n.signal_id === activeId
        const hot = n.signal_id === hover
        return (
          <div key={n.signal_id}>
            <button
              data-testid={`tree-node-${n.name}`}
              data-active={active ? 'true' : undefined}
              onMouseEnter={() => setHover(n.signal_id)}
              onMouseLeave={() => setHover(h => (h === n.signal_id ? null : h))}
              onClick={() => onPick(n.signal_id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6, width: '100%',
                textAlign: 'left', border: 'none', cursor: 'pointer',
                fontSize: 11, padding: '3px 6px', borderRadius: 5,
                paddingLeft: 6 + depth * 12,
                color: active ? '#cdd6f4' : '#a6adc8',
                fontWeight: active ? 600 : 400,
                background: active ? 'rgba(137,180,250,0.16)'
                  : hot ? 'rgba(137,180,250,0.07)' : 'none',
              }}
            >
              {/* depth rail + node dot */}
              {depth > 0 && <span style={{ color: '#45475a', fontSize: 10 }}>└</span>}
              <span style={{
                width: 6, height: 6, borderRadius: '50%', flex: '0 0 auto',
                background: active ? '#89b4fa' : '#585b70',
              }} />
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {n.name}
              </span>
            </button>
            {n.children?.length > 0 && (
              <TreeNodes nodes={n.children} depth={depth + 1} activeId={activeId} onPick={onPick} />
            )}
          </div>
        )
      })}
    </>
  )
}

const COLORMAPS = [
  'gray', 'viridis', 'inferno', 'magma', 'plasma',
  'cividis', 'hot', 'jet', 'turbo', 'twilight',
]

// Click-to-edit cell (Qt-like): shows the value as text; click turns it into an
// input that commits on blur/Enter and reverts on Escape. Avoids the "wall of
// always-on input boxes" look.
function EditableCell({ value, editable, onCommit, testid }:
  { value: string; editable: boolean; onCommit: (v: string) => void; testid: string }) {
  const [editing, setEditing] = React.useState(false)
  const [draft, setDraft] = React.useState(value)
  React.useEffect(() => { if (!editing) setDraft(value) }, [value, editing])

  if (!editable) return <span style={styles.axCellRO} data-testid={testid}>—</span>
  if (!editing) {
    return (
      <span data-testid={testid} style={styles.axText} title="click to edit"
        onClick={() => { setDraft(value); setEditing(true) }}>
        {value === '' ? <span style={styles.axPlaceholder}>—</span> : value}
      </span>
    )
  }
  return (
    <input
      data-testid={`${testid}-input`} autoFocus style={styles.axInput}
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => { setEditing(false); if (draft !== value) onCommit(draft) }}
      onKeyDown={(e) => {
        if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
        else if (e.key === 'Escape') { setDraft(value); setEditing(false) }
      }}
    />
  )
}

// Editable axes calibration table. Name / scale / offset / units commit straight
// to the dataset's axes_manager (which re-pushes every plot → the change shows in
// the plot immediately). The dataset SHAPE lives in the Metadata panel now, so
// there's no size column here.
function AxesTable({ axes, onEdit }:
  { axes: AxisRow[]; onEdit: (index: number, field: string, value: string) => void }) {
  const txt = (ax: AxisRow, field: keyof AxisRow) => {
    const v = ax[field]
    return v == null ? '' : String(v)
  }
  return (
    <table data-testid="axes-table" style={styles.axTable}>
      <thead>
        <tr style={styles.axHeadRow}>
          <th style={styles.axTh}></th>
          <th style={styles.axTh}>name</th>
          <th style={styles.axTh}>scale</th>
          <th style={styles.axTh}>offset</th>
          <th style={styles.axTh}>units</th>
        </tr>
      </thead>
      <tbody>
        {axes.map((ax) => (
          <tr key={ax.index} data-testid={`axis-row-${ax.index}`}>
            <td style={styles.axRole} title={ax.navigate ? 'navigation' : 'signal'}>
              {ax.navigate ? 'nav' : 'sig'}
            </td>
            {(['name', 'scale', 'offset', 'units'] as const).map((field) => (
              <td key={field} style={styles.axTd}>
                <EditableCell
                  testid={`axis-${ax.index}-${field}`}
                  value={txt(ax, field)}
                  editable={field === 'name' || field === 'units' || ax[field] != null}
                  onCommit={(v) => onEdit(ax.index, field, v)}
                />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Histogram({ counts, edges, vmin, vmax, onClim }:
  { counts: number[]; edges: number[]; vmin: number; vmax: number
    onClim: (mn: number, mx: number) => void }) {
  if (!counts.length) return null
  const max = Math.max(...counts) || 1
  const lo = edges[0], hi = edges[edges.length - 1]
  const span = hi - lo || 1
  const W = 216, H = 84
  const bw = W / counts.length
  const xOf = (v: number) => ((v - lo) / span) * W
  const vOf = (x: number) => lo + (Math.max(0, Math.min(W, x)) / W) * span
  const fmt = (v: number) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01))
    ? v.toExponential(1) : v.toFixed(2)

  const svgRef = React.useRef<SVGSVGElement>(null)
  const [drag, setDrag] = React.useState<null | 'min' | 'max'>(null)

  React.useEffect(() => {
    if (!drag) return
    const move = (e: PointerEvent) => {
      const rect = svgRef.current?.getBoundingClientRect()
      if (!rect) return
      const v = vOf(e.clientX - rect.left)
      if (drag === 'min') onClim(Math.min(v, vmax), vmax)
      else onClim(vmin, Math.max(v, vmin))
    }
    const up = () => setDrag(null)
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up, { once: true })
    return () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up) }
  }, [drag, vmin, vmax, lo, span])

  const handle = (which: 'min' | 'max', v: number) => {
    const x = xOf(v)
    return (
      <g key={which}>
        <line x1={x} y1={0} x2={x} y2={H} stroke="#f38ba8" strokeWidth={3} />
        {/* grip caps so the thick lines read as draggable handles */}
        <rect x={x - 3} y={0} width={6} height={5} rx={1.5} fill="#f38ba8" />
        <rect x={x - 3} y={H - 5} width={6} height={5} rx={1.5} fill="#f38ba8" />
        {/* fat invisible grab target */}
        <rect
          data-testid={`hist-${which}-handle`}
          x={x - 6} y={0} width={12} height={H}
          fill="transparent" style={{ cursor: 'ew-resize' }}
          onPointerDown={(e) => { e.preventDefault(); setDrag(which) }}
        />
      </g>
    )
  }

  return (
    <div>
      <svg ref={svgRef} width={W} height={H} data-testid="histogram"
        style={{ background: '#1e1e2e', borderRadius: 4, touchAction: 'none', display: 'block' }}>
        {/* selected range tint */}
        <rect x={xOf(vmin)} y={0} width={Math.max(0, xOf(vmax) - xOf(vmin))} height={H}
          fill="#89b4fa" opacity={0.12} />
        {counts.map((c, i) => {
          const h = (c / max) * (H - 4)
          return <rect key={i} x={i * bw} y={H - h} width={Math.max(1, bw - 0.5)} height={h} fill="#89b4fa" />
        })}
        {handle('min', vmin)}
        {handle('max', vmax)}
      </svg>
      {/* min / max display-range labels (replaces the old Scale section) */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
        <span data-testid="clim-min" style={{ fontSize: 10, color: '#a6adc8' }}>{fmt(vmin)}</span>
        <span data-testid="clim-max" style={{ fontSize: 10, color: '#a6adc8' }}>{fmt(vmax)}</span>
      </div>
    </div>
  )
}

export function PlotControlDock() {
  const { state, sendAction } = useSpyDE()
  const activeId = state.activeWindowId
  const win = activeId != null ? state.windows.get(activeId) : undefined
  const hist = activeId != null ? state.histograms.get(activeId) : undefined
  const meta = activeId != null ? state.metadata.get(activeId) : undefined
  const navSelectors = Array.from(state.selectors.values())
  const tree = activeId != null ? state.signalTrees.get(activeId) : undefined
  const axes = activeId != null ? state.axes.get(activeId) : undefined

  const onAxisEdit = (index: number, field: string, value: string) => {
    if (activeId == null) return
    sendAction('set_axis', { index, field, value }, activeId)
  }

  const onColormap = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (activeId == null) return
    sendAction('set_colormap', { name: e.target.value }, activeId)
  }

  // Display range (clim) is driven by dragging the histogram handles. A manual
  // override holds while the user drags; it resets when fresh data (a new
  // histogram) arrives so the handles follow the new auto-levels.
  const [clim, setClim] = React.useState<{ min: number; max: number } | null>(null)
  React.useEffect(() => { setClim(null) }, [hist])
  const vmin = clim?.min ?? hist?.vmin ?? 0
  const vmax = clim?.max ?? hist?.vmax ?? 1
  const onClim = (mn: number, mx: number) => {
    setClim({ min: mn, max: mx })
    if (activeId != null) sendAction('set_clim', { vmin: mn, vmax: mx }, activeId)
  }

  // Section order (per spec): Histogram, Colormap, Signal type, Metadata,
  // Axes (editable calibration), Scale (display range), Navigator Selector.
  return (
    <div data-testid="plot-control-dock" style={styles.dock}>
      <div style={styles.header}>Plot Control{win ? ` — ${win.title}` : ''}</div>

      {win == null && navSelectors.length === 0 && (
        <div style={styles.empty} data-testid="dock-empty">No active plot</div>
      )}

      {/* 1. Histogram */}
      {win && (
        <div style={styles.section} data-testid="histogram-section">
          <div style={styles.label}>Histogram</div>
          {hist
            ? <Histogram counts={hist.counts} edges={hist.edges} vmin={vmin} vmax={vmax} onClim={onClim} />
            : <div style={styles.empty} data-testid="histogram-empty">—</div>}
          <div style={styles.hint}>drag the handles to set contrast</div>
        </div>
      )}

      {/* 2. Colormap */}
      {win && (
        <div style={styles.section}>
          <label style={styles.label} htmlFor="cmap">Colormap</label>
          <select
            id="cmap"
            data-testid="colormap-select"
            style={styles.select}
            defaultValue="gray"
            onChange={onColormap}
          >
            {COLORMAPS.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      )}

      {/* 3. Workflow (signal-tree node switcher — the steps taken) */}
      {win && tree && (
        <div style={styles.section} data-testid="signal-tree">
          <div style={styles.label}>Workflow</div>
          <TreeNodes
            nodes={[tree]}
            depth={0}
            activeId={activeId != null ? (state.signalTreeActive.get(activeId) ?? null) : null}
            onPick={(id) => activeId != null && sendAction('select_signal_node', { signal_id: id }, activeId)}
          />
        </div>
      )}

      {/* 3.5 Composition (sample elements + atomic % → HyperSpy metadata) */}
      {win && (
        <CompositionPanel
          activeId={activeId}
          composition={activeId != null ? state.composition.get(activeId) : undefined}
          sendAction={sendAction}
        />
      )}

      {/* 4. Metadata — two columns to save vertical space */}
      {win && meta && (
        <div style={{ ...styles.section, overflowY: 'auto' }} data-testid="metadata-panel">
          {Object.entries(meta).map(([group, fields]) => (
            <div key={group} style={{ marginBottom: 8 }}>
              <div style={{ ...styles.label, fontWeight: 600, marginBottom: 4 }}>{group}</div>
              <div style={styles.metaGrid}>
                {Object.entries(fields).map(([k, v]) => (
                  <div key={k} style={styles.metaCell}>
                    <span style={styles.metaKey}>{k}</span>
                    <span style={styles.metaVal}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 5. Axes (editable calibration table — written back to the dataset) */}
      {win && axes && axes.length > 0 && (
        <div style={styles.section} data-testid="axes-section">
          <div style={styles.label}>Axes</div>
          <AxesTable axes={axes} onEdit={onAxisEdit} />
        </div>
      )}

      {/* Navigator Selector (bottom) */}
      {navSelectors.length > 0 && (
        <div style={styles.section} data-testid="selector-control">
          <div style={styles.label}>Navigator Selector</div>
          {navSelectors.map((s) => (
            <div key={s.windowId} style={styles.toggleRow}>
              <button
                data-testid="selector-crosshair"
                style={s.mode === 'crosshair' ? styles.toggleActive : styles.toggle}
                onClick={() => sendAction('set_selector_mode', { integrate: false }, s.windowId)}
              >
                ✛ Point
              </button>
              <button
                data-testid="selector-integrate"
                style={s.mode === 'integrate' ? styles.toggleActive : styles.toggle}
                onClick={() => sendAction('set_selector_mode', { integrate: true }, s.windowId)}
              >
                ▭ Integrate
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  dock: {
    width: 240, flexShrink: 0,
    height: '100%',
    background: '#181825',
    borderLeft: '1px solid #313244',
    display: 'flex', flexDirection: 'column',
    color: '#cdd6f4',
  },
  header: {
    padding: '10px 12px', fontSize: 13, fontWeight: 600,
    borderBottom: '1px solid #313244', color: '#cdd6f4',
  },
  empty: { padding: 16, fontSize: 12, color: '#6c7086' },
  section: {
    padding: '10px 12px',
    borderBottom: '1px solid #1e1e2e',
    display: 'flex', flexDirection: 'column', gap: 6,
  },
  label: { fontSize: 11, color: '#a6adc8' },
  select: {
    background: '#1e1e2e', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 4, padding: '4px 6px',
    fontSize: 12,
  },
  row: { display: 'flex', gap: 6 },
  input: {
    flex: 1, minWidth: 0,
    background: '#1e1e2e', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 4, padding: '4px 6px',
    fontSize: 12,
  },
  btn: {
    background: '#313244', color: '#cdd6f4', border: 'none',
    borderRadius: 4, padding: '4px 10px', fontSize: 12, cursor: 'pointer',
    alignSelf: 'flex-start',
  },
  metaRow: {
    display: 'flex', justifyContent: 'space-between', gap: 8,
    fontSize: 11, padding: '1px 0',
  },
  // Two-column metadata grid: each cell is a compact key (tiny, top) + value.
  metaGrid: {
    display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 10, rowGap: 3,
  },
  metaCell: { display: 'flex', flexDirection: 'column', minWidth: 0 },
  metaKey: { color: '#6c7086', fontSize: 9, whiteSpace: 'nowrap',
             overflow: 'hidden', textOverflow: 'ellipsis' },
  metaVal: { color: '#cdd6f4', fontSize: 11, whiteSpace: 'nowrap',
             overflow: 'hidden', textOverflow: 'ellipsis' },
  hint: { fontSize: 10, color: '#6c7086', marginTop: 4 },
  toggleRow: { display: 'flex', gap: 6 },
  toggle: {
    flex: 1, background: '#1e1e2e', color: '#a6adc8',
    border: '1px solid #313244', borderRadius: 4, padding: '4px 6px',
    fontSize: 11, cursor: 'pointer',
  },
  toggleActive: {
    flex: 1, background: '#89b4fa', color: '#11111b',
    border: '1px solid #89b4fa', borderRadius: 4, padding: '4px 6px',
    fontSize: 11, cursor: 'pointer', fontWeight: 600,
  },
  axTable: { width: '100%', borderCollapse: 'collapse', fontSize: 10 },
  axHeadRow: { color: '#6c7086' },
  axTh: { textAlign: 'left', fontWeight: 500, padding: '0 2px 2px', fontSize: 10 },
  axTd: { padding: '1px 1px' },
  axTdRO: { padding: '1px 3px', color: '#a6adc8', textAlign: 'center' },
  axRole: { padding: '1px 3px', color: '#6c7086', fontSize: 9 },
  axInput: {
    width: '100%', minWidth: 0, boxSizing: 'border-box',
    background: '#1e1e2e', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 3, padding: '2px 3px',
    fontSize: 10,
  },
  axCellRO: { color: '#6c7086' },
  axText: {
    display: 'block', minWidth: 0, padding: '2px 4px', borderRadius: 3,
    color: '#cdd6f4', fontSize: 10, cursor: 'text',
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  axPlaceholder: { color: '#45475a' },
}
