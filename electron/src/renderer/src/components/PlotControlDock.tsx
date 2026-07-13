/**
 * PlotControlDock.tsx — right-hand dock with controls for the active plot.
 *
 * Colormap, contrast (display range), and basic metadata. Sends set_colormap /
 * set_clim actions to Python targeting the active window.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import type { TreeNode, AxisRow } from '../kernel/SpyDEContext'
import type { LayerState, LayersStateMessage } from '../kernel/protocol'
import { WORKFLOW_NODE_DRAG_MIME } from '../kernel/dnd'
import { COLORMAPS } from '../kernel/colormaps'
import { useKeyedDebounce } from './wizardHooks'
import { CompositionPanel } from './CompositionPanel'

// Compact workflow tree: each step is a row with a depth guide-rail, a node dot,
// and the step name. The active (displayed) node is highlighted. Hovering tints
// the row so it reads as clickable.
function TreeNodes({ nodes, depth, activeId, windowId, onPick }:
  { nodes: TreeNode[]; depth: number; activeId: number | null
    windowId: number | null; onPick: (id: number) => void }) {
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
              // Drag a workflow node into the console to bind it (backend
              // console_bind_node picks a var name for this exact tree node).
              draggable={windowId != null}
              onDragStart={(e) => {
                if (windowId == null) return
                e.dataTransfer.setData(WORKFLOW_NODE_DRAG_MIME, JSON.stringify({
                  windowId, signalId: n.signal_id, name: n.name,
                }))
                e.dataTransfer.effectAllowed = 'copy'
              }}
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
              <TreeNodes nodes={n.children} depth={depth + 1} activeId={activeId}
                windowId={windowId} onPick={onPick} />
            )}
          </div>
        )
      })}
    </>
  )
}

// Click-to-edit cell (Qt-like): shows the value as text; click turns it into an
// input that commits on blur/Enter and reverts on Escape. Avoids the "wall of
// always-on input boxes" look. ``display`` is the (possibly rounded) text shown
// when not editing; editing always exposes the full-precision ``value``.
function EditableCell({ value, display, editable, onCommit, testid }:
  { value: string; display?: string; editable: boolean
    onCommit: (v: string) => void; testid: string }) {
  const [editing, setEditing] = React.useState(false)
  const [draft, setDraft] = React.useState(value)
  React.useEffect(() => { if (!editing) setDraft(value) }, [value, editing])
  const shown = display ?? value

  if (!editable) return <span style={styles.axCellRO} data-testid={testid}>—</span>
  if (!editing) {
    return (
      <span data-testid={testid} style={styles.axText} title="click to edit"
        onClick={() => { setDraft(value); setEditing(true) }}>
        {shown === '' ? <span style={styles.axPlaceholder}>—</span> : shown}
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
function AxesTable({ axes, onEdit, offsetPick, onToggleOffsetPick }:
  { axes: AxisRow[]; onEdit: (index: number, field: string, value: string) => void
    offsetPick: boolean; onToggleOffsetPick: () => void }) {
  const txt = (ax: AxisRow, field: keyof AxisRow) => {
    const v = ax[field]
    return v == null ? '' : String(v)
  }
  // Display scale/offset rounded to 2 dp (full precision shows on click-to-edit).
  // Very small / large magnitudes fall back to 2-sig-fig exponential so a tiny
  // calibration (e.g. 0.0042 Å⁻¹/px) doesn't render as "0.00".
  const disp = (ax: AxisRow, field: keyof AxisRow) => {
    if (field !== 'scale' && field !== 'offset') return undefined
    const v = ax[field]
    if (v == null) return undefined
    const n = Number(v)
    if (!Number.isFinite(n)) return undefined
    if (n !== 0 && Math.abs(n) < 0.01) return n.toExponential(1)
    return n.toFixed(2)
  }
  const hasSignal = axes.some((ax) => !ax.navigate)
  return (
    <>
      <table data-testid="axes-table" style={styles.axTable}>
        <thead>
          <tr style={styles.axHeadRow}>
            <th style={styles.axTh}></th>
            <th style={styles.axTh}>name</th>
            <th style={styles.axTh}>scale</th>
            <th style={styles.axTh}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                offset
                {hasSignal && (
                  <button
                    data-testid="offset-pick-toggle"
                    title="Set origin: drag a crosshair on the image to mark (0,0)"
                    onClick={onToggleOffsetPick}
                    style={offsetPick ? styles.offPickOn : styles.offPick}
                  >+</button>
                )}
              </span>
            </th>
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
                    display={disp(ax, field)}
                    editable={field === 'name' || field === 'units' || ax[field] != null}
                    onCommit={(v) => onEdit(ax.index, field, v)}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {offsetPick && (
        <div style={styles.hint} data-testid="offset-pick-hint">
          Drag the orange crosshair onto the (0,0) point — offsets update live.
        </div>
      )}
    </>
  )
}

function Histogram({ counts, edges, vmin, vmax, threshold, onClim }:
  { counts: number[]; edges: number[]; vmin: number; vmax: number
    threshold?: number | null
    onClim: (mn: number, mx: number) => void }) {
  if (!counts.length) return null
  const max = Math.max(...counts) || 1
  const lo = edges[0], hi = edges[edges.length - 1]
  const dataSpan = hi - lo || 1
  // Draw an axis that extends PAST the data max so the upper-contrast handle can
  // be pulled above 100% (darkens the image). Headroom = 2× the data span beyond
  // the data max; also stretch to cover a held vmax (e.g. a brighter frame in a
  // 5-D series whose contrast was carried over). The bars only span [lo, hi].
  const axLo = lo
  const axHi = Math.max(hi + 2 * dataSpan, vmax + 0.05 * dataSpan)
  const span = axHi - axLo || 1
  const W = 276, H = 84
  const bw = (dataSpan / span) * W / counts.length
  // No right-edge clamp: dragging past the drawn area maps to values > data max.
  const xOf = (v: number) => ((v - axLo) / span) * W
  const vOf = (x: number) => axLo + (Math.max(0, x) / W) * span
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
        {/* data-max marker: dragging the max handle to the RIGHT of this line
            pushes the upper clim above the data range (image gets darker). */}
        {xOf(hi) < W && (
          <line data-testid="hist-datamax" x1={xOf(hi)} y1={0} x2={xOf(hi)} y2={H}
            stroke="#585b70" strokeWidth={1} strokeDasharray="2 2" />
        )}
        {/* Find-Vectors detector threshold: dotted orange line (in image units) */}
        {threshold != null && threshold >= lo && threshold <= hi && (
          <line data-testid="hist-threshold" x1={xOf(threshold)} y1={0}
            x2={xOf(threshold)} y2={H} stroke="#ffae57" strokeWidth={1.5}
            strokeDasharray="3 3" />
        )}
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

// One layer row: colour-coded title, colormap select, alpha slider (debounced
// live overlay_set), a visibility toggle, and remove. `sendSet` is a per-row
// debounced sender (mirrors wizardHooks' useDebouncedAction pattern) so a
// dragged alpha slider doesn't flood overlay_set.
function LayerRow({ layer, dotColor, onCmap, onAlpha, onVisible, onRemove }: {
  layer: LayerState
  dotColor: string
  onCmap: (cmap: string) => void
  onAlpha: (alpha: number) => void
  onVisible: (visible: boolean) => void
  onRemove: () => void
}) {
  // Local alpha draft so the slider tracks the pointer smoothly between
  // debounced sends (mirrors the clim-drag pattern above).
  const [draftAlpha, setDraftAlpha] = React.useState(layer.alpha)
  React.useEffect(() => { setDraftAlpha(layer.alpha) }, [layer.alpha])

  return (
    <div data-testid={`layer-row-${layer.id}`} style={styles.layerRow}>
      <div style={styles.toggleRow}>
        <span style={{ ...styles.selectorDot, background: dotColor }} title={layer.title} />
        <span style={styles.layerTitle} title={layer.title}>{layer.title || 'Layer'}</span>
        <button
          data-testid={`layer-visible-${layer.id}`}
          title={layer.visible ? 'Hide layer' : 'Show layer'}
          onClick={() => onVisible(!layer.visible)}
          style={layer.visible ? styles.eyeOn : styles.eyeOff}
        >
          {layer.visible ? '◉' : '○'}
        </button>
        <button
          data-testid={`layer-remove-${layer.id}`}
          title="Remove layer"
          onClick={onRemove}
          style={styles.removeBtn}
        >
          {'×'}
        </button>
      </div>
      <div style={styles.row}>
        <select
          data-testid={`layer-cmap-${layer.id}`}
          style={{ ...styles.select, flex: 1 }}
          value={layer.cmap}
          onChange={(e) => onCmap(e.target.value)}
        >
          {COLORMAPS.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
      <div style={styles.toggleRow}>
        <span style={styles.hint}>alpha</span>
        <input
          data-testid={`layer-alpha-${layer.id}`}
          type="range" min={0} max={1} step={0.05}
          value={draftAlpha}
          onChange={(e) => {
            const v = Number(e.target.value)
            setDraftAlpha(v)
            onAlpha(v)
          }}
          style={{ flex: 1 }}
        />
        <span style={{ ...styles.hint, minWidth: 28, textAlign: 'right' }}>
          {draftAlpha.toFixed(2)}
        </span>
      </div>
    </div>
  )
}

// A small palette cycled per-row so each layer's title dot reads as visually
// distinct (mirrors the backend's own _LAYER_CMAP_CYCLE intent, but this is
// just a UI accent — the authoritative appearance is layer.cmap).
const LAYER_DOT_COLORS = ['#f38ba8', '#a6e3a1', '#f9e2af', '#89b4fa', '#cba6f7', '#94e2d5']

// "Layers" section: live overlay stack for the ACTIVE window (MDI image
// layering — spyde/actions/overlay.py). Listens for `spyde:layers_state`
// CustomEvents (re-broadcast by SpyDEContext) filtered to the active window,
// and re-queries on active-window change. Renders nothing when there are no
// layers (including no active window).
function LayersSection({ activeId, sendAction }: {
  activeId: number | null
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}) {
  const [layers, setLayers] = React.useState<LayerState[]>([])

  React.useEffect(() => {
    setLayers([])
    if (activeId == null) return
    sendAction('overlay_query', { window_id: activeId }, activeId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId])

  React.useEffect(() => {
    const on = (e: Event) => {
      const msg = (e as CustomEvent).detail as LayersStateMessage
      if (activeId == null || msg.window_id !== activeId) return
      setLayers(msg.layers ?? [])
    }
    window.addEventListener('spyde:layers_state', on)
    return () => window.removeEventListener('spyde:layers_state', on)
  }, [activeId])

  // Debounced per-layer overlay_set sender — keyed by layer id so dragging one
  // layer's alpha doesn't cancel another's pending send.
  const debounce = useKeyedDebounce(150)
  const sendSet = (layerId: string, payload: Record<string, unknown>) => {
    if (activeId == null) return
    debounce(layerId, () => {
      sendAction('overlay_set', { window_id: activeId, layer_id: layerId, ...payload }, activeId)
    })
  }

  if (activeId == null || layers.length === 0) return null

  return (
    <div style={styles.section} data-testid="layers-section">
      <div style={styles.label}>Layers</div>
      {layers.map((layer, i) => (
        <LayerRow
          key={layer.id}
          layer={layer}
          dotColor={LAYER_DOT_COLORS[i % LAYER_DOT_COLORS.length]}
          onCmap={(cmap) => sendAction('overlay_set', { window_id: activeId, layer_id: layer.id, cmap }, activeId)}
          onAlpha={(alpha) => sendSet(layer.id, { alpha })}
          onVisible={(visible) => sendAction('overlay_set', { window_id: activeId, layer_id: layer.id, visible }, activeId)}
          onRemove={() => sendAction('overlay_remove', { window_id: activeId, layer_id: layer.id }, activeId)}
        />
      ))}
    </div>
  )
}

export function PlotControlDock() {
  const { state, sendAction } = useSpyDE()
  const activeId = state.activeWindowId
  const win = activeId != null ? state.windows.get(activeId) : undefined
  const hist = activeId != null ? state.histograms.get(activeId) : undefined
  const meta = activeId != null ? state.metadata.get(activeId) : undefined
  const tree = activeId != null ? state.signalTrees.get(activeId) : undefined
  // Only the ACTIVE signal tree's selectors are listed — every window of a
  // tree receives the same signal_tree payload, so two windows belong to the
  // same tree iff their trees share a root signal_id. With no tree context
  // (e.g. a bare result window is focused) fall back to showing all.
  const activeTreeRoot = tree?.signal_id
  const navSelectors = Array.from(state.selectors.values()).filter(s => {
    if (activeTreeRoot == null) return true
    return state.signalTrees.get(s.windowId)?.signal_id === activeTreeRoot
  })
  const axes = activeId != null ? state.axes.get(activeId) : undefined
  const sigType = activeId != null ? state.signalTypes.get(activeId) : undefined

  const onAxisEdit = (index: number, field: string, value: string) => {
    if (activeId == null) return
    sendAction('set_axis', { index, field, value }, activeId)
  }

  // "Set origin" crosshair tool: toggles a draggable crosshair on the signal
  // plot whose position the backend turns into the signal-axis offsets live.
  const [offsetPick, setOffsetPick] = React.useState(false)
  // Drop the tool when the active window changes (the crosshair lives on that
  // window's plot).
  React.useEffect(() => { setOffsetPick(false) }, [activeId])
  const onToggleOffsetPick = () => {
    if (activeId == null) return
    const next = !offsetPick
    setOffsetPick(next)
    sendAction('set_offset_crosshair', { on: next }, activeId)
  }

  const onColormap = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (activeId == null) return
    sendAction('set_colormap', { name: e.target.value }, activeId)
  }

  const onSignalType = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (activeId == null) return
    sendAction('set_signal_type', { signal_type: e.target.value }, activeId)
  }
  // Human label for a HyperSpy signal_type (the empty type = a generic signal).
  const sigTypeLabel = (t: string) => t === '' ? 'Generic (none)' : t

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
            ? <Histogram counts={hist.counts} edges={hist.edges} vmin={vmin} vmax={vmax}
                threshold={hist.threshold} onClim={onClim} />
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

      {/* 3. Signal type (HyperSpy signal_type — re-casts the signal class) */}
      {win && sigType && (
        <div style={styles.section} data-testid="signal-type-section">
          <label style={styles.label} htmlFor="sigtype">Signal type</label>
          <select
            id="sigtype"
            data-testid="signal-type-select"
            style={styles.select}
            value={sigType.current}
            onChange={onSignalType}
          >
            {sigType.options.map(t => (
              <option key={t} value={t}>{sigTypeLabel(t)}</option>
            ))}
          </select>
        </div>
      )}

      {/* 4. Workflow (signal-tree node switcher — the steps taken) */}
      {win && tree && (
        <div style={styles.section} data-testid="signal-tree">
          <div style={styles.label}>Workflow</div>
          <TreeNodes
            nodes={[tree]}
            depth={0}
            activeId={activeId != null ? (state.signalTreeActive.get(activeId) ?? null) : null}
            windowId={activeId}
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
          <AxesTable axes={axes} onEdit={onAxisEdit}
            offsetPick={offsetPick} onToggleOffsetPick={onToggleOffsetPick} />
        </div>
      )}

      {/* 6. Layers — live MDI image overlay stack (spyde/actions/overlay.py) */}
      {win && <LayersSection activeId={activeId} sendAction={sendAction} />}

      {/* Navigator Selector (bottom) — one row per selector, with its colour dot */}
      {navSelectors.length > 0 && (
        <div style={styles.section} data-testid="selector-control">
          <div style={styles.label}>Navigator Selector</div>
          {navSelectors.map((s) => (
            <div key={s.selectorId ?? s.windowId} style={styles.toggleRow}>
              <span
                data-testid="selector-dot"
                style={{ ...styles.selectorDot, background: s.color ?? '#00e676' }}
                title={s.title ?? 'Navigator'}
              />
              <button
                data-testid="selector-crosshair"
                style={s.mode === 'crosshair' ? styles.toggleActive : styles.toggle}
                onClick={() => sendAction('set_selector_mode',
                  { integrate: false, selector_id: s.selectorId }, s.windowId)}
              >
                ✛ Point
              </button>
              <button
                data-testid="selector-integrate"
                style={s.mode === 'integrate' ? styles.toggleActive : styles.toggle}
                onClick={() => sendAction('set_selector_mode',
                  { integrate: true, selector_id: s.selectorId }, s.windowId)}
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
    width: 300, flexShrink: 0,
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
  toggleRow: { display: 'flex', gap: 6, alignItems: 'center' },
  selectorDot: {
    width: 10, height: 10, borderRadius: '50%', flexShrink: 0,
    border: '1px solid rgba(0,0,0,0.4)',
  },
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
  // "+" origin-pick toggle in the offset header — matches the orange on-plot
  // crosshair when active so the two read as the same tool.
  offPick: {
    border: '1px solid #45475a', background: '#1e1e2e', color: '#a6adc8',
    borderRadius: 3, width: 14, height: 14, lineHeight: '12px', fontSize: 11,
    padding: 0, cursor: 'pointer', fontWeight: 700,
  },
  offPickOn: {
    border: '1px solid #ffae57', background: '#ffae57', color: '#11111b',
    borderRadius: 3, width: 14, height: 14, lineHeight: '12px', fontSize: 11,
    padding: 0, cursor: 'pointer', fontWeight: 700,
  },
  layerRow: {
    display: 'flex', flexDirection: 'column', gap: 4,
    padding: '6px 0', borderBottom: '1px solid #1e1e2e',
  },
  layerTitle: {
    flex: 1, fontSize: 11, color: '#cdd6f4', minWidth: 0,
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  eyeOn: {
    background: 'none', border: 'none', color: '#89b4fa', cursor: 'pointer',
    fontSize: 12, padding: '0 4px', lineHeight: 1,
  },
  eyeOff: {
    background: 'none', border: 'none', color: '#585b70', cursor: 'pointer',
    fontSize: 12, padding: '0 4px', lineHeight: 1,
  },
  removeBtn: {
    background: 'none', border: 'none', color: '#f38ba8', cursor: 'pointer',
    fontSize: 13, padding: '0 4px', lineHeight: 1, fontWeight: 700,
  },
}
