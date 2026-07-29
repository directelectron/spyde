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
import { UnitText } from '../kernel/units'
import { useKeyedDebounce } from './wizardHooks'
import { CompositionPanel } from './CompositionPanel'
import { MetadataPanel } from './MetadataPanel'
import { Dropdown } from './Dropdown'

// Dropdown options are {value,label}; the colormap list is shared app-wide.
const CMAP_OPTS = COLORMAPS.map((c) => ({ value: c, label: c }))

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
                fontSize: 10, padding: '2px 6px', borderRadius: 4,
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
// always-on input boxes" look. ``display`` is the (possibly rounded, possibly
// LaTeX-rendered) content shown when not editing; editing always exposes the
// raw, full-precision ``value``. A non-editable cell falls back to showing
// ``value`` itself (real read-only content, e.g. a derived metadata field)
// UNLESS the caller has no value at all (the axes table's null scale/offset),
// signalled by passing "" — that still renders as "—" via the placeholder
// branch below.
function EditableCell({ value, display, editable, onCommit, testid }:
  { value: string; display?: React.ReactNode; editable: boolean
    onCommit: (v: string) => void; testid: string }) {
  const [editing, setEditing] = React.useState(false)
  const [draft, setDraft] = React.useState(value)
  React.useEffect(() => { if (!editing) setDraft(value) }, [value, editing])
  const shown = display ?? value

  if (!editable) {
    return (
      <span style={styles.axCellRO} data-testid={testid}>
        {shown === '' ? '—' : shown}
      </span>
    )
  }
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
  //
  // `units` is the same idea with a different renderer: HyperSpy stores the
  // units as raw LaTeX ("$\AA^{-1}$"), so the cell DISPLAYS it rendered (Å⁻¹)
  // and click-to-edit still exposes — and commits — the raw string unchanged.
  const disp = (ax: AxisRow, field: keyof AxisRow): React.ReactNode => {
    if (field === 'units') {
      const u = txt(ax, field)
      return u === '' ? undefined : <UnitText raw={u} />
    }
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
                    // Mirrors the backend's live crosshair state, so a test (and
                    // an accessibility reader) can see on/off without decoding
                    // the inline style.
                    data-on={offsetPick ? 'true' : 'false'}
                    aria-pressed={offsetPick}
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

/** Small dock button that tints on hover and again while held — inline styles
 *  can't express :hover/:active, and the dock has no stylesheet, so the state is
 *  local (the same idiom Pill.tsx uses). */
function TintButton({ children, onClick, testid, title }: {
  children: React.ReactNode
  onClick: () => void
  testid?: string
  title?: string
}) {
  const [hover, setHover] = React.useState(false)
  const [held, setHeld] = React.useState(false)
  return (
    <button
      data-testid={testid}
      title={title}
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => { setHover(false); setHeld(false) }}
      onPointerDown={() => setHeld(true)}
      onPointerUp={() => setHeld(false)}
      style={{
        ...styles.autoBtn,
        background: held ? '#585b70' : hover ? '#313244' : '#1e1e2e',
        borderColor: held || hover ? '#585b70' : '#313244',
        color: held || hover ? '#cdd6f4' : '#a6adc8',
      }}
    >{children}</button>
  )
}

function Histogram({ counts, edges, vmin, vmax, threshold, clipped, onClim, onAuto, onReset }:
  { counts: number[]; edges: number[]; vmin: number; vmax: number
    threshold?: number | null
    clipped?: boolean
    onClim: (mn: number, mx: number) => void
    onAuto: () => void
    onReset: () => void }) {
  if (!counts.length) return null
  const max = Math.max(...counts) || 1
  const lo = edges[0], hi = edges[edges.length - 1]
  const dataSpan = hi - lo || 1
  // Draw a little axis PAST the binned range so the upper handle can be pulled
  // above 100% (darkens the image). The bars only span [lo, hi].
  //
  // The headroom used to be 2× the data span, which cost two thirds of the
  // widget. That was survivable when the bins covered min–max; now that the
  // backend bins over robust quantiles (see Plot._hist_range) the drawn range is
  // the range worth resolving, so the handles get essentially all of the width.
  // Dragging is not limited by it either way — neither xOf nor vOf clamps.
  //
  // The axis deliberately does NOT stretch to cover an out-of-range vmax (after
  // Reset, or a contrast held from a brighter frame): that stretch re-created
  // the squish, since a vmax out at a hot-pixel value compresses every bar into
  // the left edge. A handle beyond the axis pins at the border with the
  // off-range arrow instead — visible, grabbable, and the bars keep the width.
  const axLo = lo
  const axHi = hi + 0.25 * dataSpan
  const span = axHi - axLo || 1
  const W = 276, H = 62
  const bw = (dataSpan / span) * W / counts.length
  // No clamping either side: dragging past the drawn area maps to values outside
  // the binned range, which is the only way to reach a clipped tail.
  const xOf = (v: number) => ((v - axLo) / span) * W
  const vOf = (x: number) => axLo + (x / W) * span
  const fmt = (v: number) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01))
    ? v.toExponential(1) : v.toFixed(2)

  const svgRef = React.useRef<SVGSVGElement>(null)
  const [drag, setDrag] = React.useState<null | 'min' | 'max'>(null)
  // The handles are a HOVER affordance: at rest the clim edges are just a hairline
  // (the tinted range already shows where they are), and the grabbable pink bars
  // fade in when the pointer is over the widget. Two fat pink bars parked over the
  // data all the time read as part of the plot and hide the left-most bins.
  const [hover, setHover] = React.useState(false)
  const armed = hover || drag != null

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
    // PIN the handle inside the widget when its value falls outside the drawn
    // range. Dragging is unclamped (that is how you reach a clipped tail), so
    // without this a handle dragged past an edge is drawn off-canvas — invisible
    // AND unreachable, with no way to drag it back. Pinned, it stays grabbable
    // and the first inward drag brings the value back into range; the arrow says
    // the real value is further out that way.
    const raw = xOf(v)
    const x = Math.min(Math.max(raw, 3), W - 3)
    const off = raw < 0 ? -1 : raw > W ? 1 : 0
    const mid = H / 2
    return (
      <g key={which}>
        <line x1={x} y1={0} x2={x} y2={H} stroke="#f38ba8"
          strokeWidth={armed ? 3 : 1} opacity={armed ? 1 : 0.5}
          style={{ transition: 'stroke-width 90ms, opacity 90ms' }} />
        {/* grip caps so the thick lines read as draggable handles — only once
            hovered, since that is the only time they can be grabbed */}
        {armed && <>
          <rect x={x - 3} y={0} width={6} height={5} rx={1.5} fill="#f38ba8" />
          <rect x={x - 3} y={H - 5} width={6} height={5} rx={1.5} fill="#f38ba8" />
        </>}
        {/* the off-range arrow is NOT part of the hover affordance: it says the
            real value is outside the drawn range, which is worth showing at rest */}
        {off !== 0 && (
          <polygon data-testid={`hist-${which}-offscreen`}
            points={off < 0
              ? `${x - 3},${mid} ${x + 6},${mid - 5} ${x + 6},${mid + 5}`
              : `${x + 3},${mid} ${x - 6},${mid - 5} ${x - 6},${mid + 5}`}
            fill="#f38ba8" />
        )}
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
        onPointerEnter={() => setHover(true)}
        onPointerLeave={() => setHover(false)}
        style={{ background: '#1e1e2e', borderRadius: 4, touchAction: 'none', display: 'block' }}>
        <title>Drag the handles to set contrast</title>
        {/* selected range tint */}
        <rect x={xOf(vmin)} y={0} width={Math.max(0, xOf(vmax) - xOf(vmin))} height={H}
          fill="#89b4fa" opacity={0.12} />
        {counts.map((c, i) => {
          // LOG-scaled bars. Counting electrons is Poisson, so a frame's
          // occupancy falls off by orders of magnitude away from the background
          // peak: on a linear axis the peak is one full-height bar and the entire
          // useful tail — the diffraction spots you are setting contrast for —
          // rounds to zero pixels. log1p keeps 0 at 0 and still separates a count
          // of 1 from a count of 10.
          const h = (Math.log1p(c) / Math.log1p(max)) * (H - 4)
          // The end bins are overflow when the range is clipped: everything
          // beyond the quantiles piled in. Shaded differently so a tall edge bar
          // is not misread as a real population at that value.
          const overflow = clipped && (i === 0 || i === counts.length - 1) && c > 0
          return <rect key={i} x={i * bw} y={H - h} width={Math.max(1, bw - 0.5)}
            height={h} fill={overflow ? '#585b70' : '#89b4fa'} />
        })}
        {/* end-of-bins marker: dragging the max handle to the RIGHT of this line
            pushes the upper clim past the binned range (image gets darker). */}
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
      <div style={{ display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginTop: 2 }}>
        <span data-testid="clim-min" style={{ fontSize: 9, color: '#a6adc8' }}>{fmt(vmin)}</span>
        <span data-testid="clim-max" style={{ fontSize: 9, color: '#a6adc8' }}>{fmt(vmax)}</span>
      </div>
      {/* …and the two range buttons on their own row, sized/styled like the
          Point / Integrate selector-mode pair so the dock reads as one system. */}
      <div style={{ ...styles.toggleRow, marginTop: 3 }}>
        <TintButton testid="clim-auto" onClick={onAuto}
          title="Contrast from the robust levels for this frame (2–99%)">◐ Auto</TintButton>
        <TintButton testid="clim-reset" onClick={onReset}
          title="Show the full data range, tail included">↺ Reset</TintButton>
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
        <div style={{ flex: 1, minWidth: 0 }}>
          <Dropdown testid={`layer-cmap-${layer.id}`} value={layer.cmap}
            options={CMAP_OPTS} onChange={onCmap} />
        </div>
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
  const metaEditable = activeId != null ? state.metadataEditable.get(activeId) : undefined
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

  // Instrument-metadata cell edit — same click-to-edit idiom as the axes
  // table (EditableCell). The backend resolves (group, prop) back to the
  // writable hyperspy metadata key + type (float/int/string) via the same
  // METADATA_WIDGET_CONFIG the panel's values come from, so no key/type needs
  // to travel over the wire — see spyde/backend/_session_axes.py:_set_metadata.
  const onMetadataEdit = (group: string, prop: string, value: string) => {
    if (activeId == null) return
    sendAction('set_metadata', { group, prop, value }, activeId)
  }

  // "Set origin" crosshair tool: toggles a draggable crosshair on the signal
  // plot whose position the backend turns into the signal-axis offsets live.
  //
  // The state is BACKEND-OWNED, per window (`offset_pick` messages →
  // state.offsetPick): the backend holds the widget on the plot, so it is the
  // only thing that knows whether a crosshair is alive. A renderer-local
  // boolean drifted out of sync — focusing another window reset the flag (the
  // "+" went dark) while the crosshair stayed on the first window's plot, and
  // the next click then targeted the NEW active plot, so the stale crosshair
  // could never be dismissed from the UI. Reading per-window state also means
  // clicking back onto the window that owns the crosshair shows "+" lit again.
  const offsetPick = activeId != null && (state.offsetPick.get(activeId) ?? false)
  const onToggleOffsetPick = () => {
    if (activeId == null) return
    // Fire-and-follow: the backend answers with the state it actually reached
    // (a toggle it refuses — e.g. a 1-D signal with < 2 display axes — must not
    // leave a lit button with no crosshair under it).
    sendAction('set_offset_crosshair', { on: !offsetPick }, activeId)
  }

  // The colormap dropdown mirrors the old uncontrolled <select>: local state,
  // 'gray' initial, persists across window switches (the backend is the source
  // of truth for what each window actually shows).
  const [cmapSel, setCmapSel] = React.useState('gray')
  const onColormap = (name: string) => {
    setCmapSel(name)
    if (activeId == null) return
    sendAction('set_colormap', { name }, activeId)
  }

  const onSignalType = (t: string) => {
    if (activeId == null) return
    sendAction('set_signal_type', { signal_type: t }, activeId)
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
  // Auto: hand the range back to the backend, which re-derives it from the frame
  // currently on screen and re-emits the histogram. Drop the local override up
  // front so the handles follow that answer rather than sticking where the user
  // last dragged them (the histogram effect above would clear it anyway; doing it
  // here means the handles don't sit stale for the round trip).
  const onAuto = () => {
    setClim(null)
    if (activeId != null) sendAction('auto_clim', { mode: 'robust' }, activeId)
  }
  // Reset: the OTHER thing you want when the contrast is unhelpful — show
  // everything, tail included. Same round trip as Auto, so the histogram comes
  // back binned over the full extent (nothing clipped) rather than leaving the
  // handles pinned outside a robust view of it.
  const onReset = () => {
    setClim(null)
    if (activeId != null) sendAction('auto_clim', { mode: 'full' }, activeId)
  }

  // Section order (per spec): Histogram, Colormap, Signal type, Metadata,
  // Axes (editable calibration), Scale (display range), Navigator Selector.
  return (
    <div data-testid="plot-control-dock" style={styles.dock}>
      <div style={styles.header} title={win?.title}>
        Plot Control{win ? ` — ${win.title}` : ''}
      </div>

      {/* Single scroll region for every section — see `styles.body`. */}
      <div style={styles.body} data-testid="dock-body">

      {win == null && navSelectors.length === 0 && (
        <div style={styles.empty} data-testid="dock-empty">No active plot</div>
      )}

      {/* 1. Histogram */}
      {win && (
        <div style={styles.section} data-testid="histogram-section">
          <div style={styles.label}>Histogram</div>
          {hist
            ? <Histogram counts={hist.counts} edges={hist.edges} vmin={vmin} vmax={vmax}
                threshold={hist.threshold} clipped={hist.clipped} onClim={onClim}
                onAuto={onAuto} onReset={onReset} />
            : <div style={styles.empty} data-testid="histogram-empty">—</div>}
        </div>
      )}

      {/* 2 + 3. Colormap and signal type (HyperSpy signal_type — re-casts the
          signal class) share ONE row: two labelled dropdowns, each set once in a
          while, that cost two full sections stacked. `signal-type-section` stays
          a wrapper so the section-order test and anything targeting it hold. */}
      {win && (
        <div style={styles.section}>
          <div style={styles.row}>
            <div style={styles.half}>
              <div style={styles.label}>Colormap</div>
              <Dropdown testid="colormap-select" value={cmapSel}
                options={CMAP_OPTS} onChange={onColormap} />
            </div>
            {sigType && (
              <div style={styles.half} data-testid="signal-type-section">
                <div style={styles.label}>Signal type</div>
                <Dropdown testid="signal-type-select" value={sigType.current}
                  options={sigType.options.map((t) => ({ value: t, label: sigTypeLabel(t) }))}
                  onChange={onSignalType} />
              </div>
            )}
          </div>
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

      {/* 4. Metadata — a curated summary + a per-field detail popover, and the
          ONLY part of the dock that scrolls (see MetadataPanel's design note). */}
      {win && meta && (
        <MetadataPanel meta={meta} editable={metaEditable ?? {}}
          info={state.metadataInfo} onEdit={onMetadataEdit}
          chunking={activeId != null ? state.chunking.get(activeId) : undefined} />
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
              {/* Point is a SPLIT control: the button picks the mode, and a
                  caret at its right edge opens the frame-width menu. It has to
                  be a styled wrapper around two siblings rather than a caret
                  nested in the button, because buttons cannot contain buttons
                  — so the wrapper carries the active/inactive look and both
                  children are transparent inside it. */}
              <div style={{
                ...(s.mode === 'crosshair' ? styles.toggleActive : styles.toggle),
                ...styles.splitWrap,
              }}>
                <button
                  data-testid="selector-crosshair"
                  style={{
                    ...styles.splitMain,
                    // Explicit, not inherited: a <button> does not take the
                    // parent's colour from the UA stylesheet, and the `font`
                    // shorthand in an inline style wiped the label entirely.
                    color: s.mode === 'crosshair' ? '#11111b' : '#a6adc8',
                    fontWeight: s.mode === 'crosshair' ? 600 : 400,
                  }}
                  onClick={() => sendAction('set_selector_mode',
                    { integrate: false, selector_id: s.selectorId }, s.windowId)}
                >
                  ✛ Point
                  {/* The width, subtly, so you can see what the pointer is
                      reading without opening the menu. */}
                  {s.sumFrames != null && s.sumFrames > 1 && (
                    <span style={styles.sumBadge}>{s.sumFrames}f</span>
                  )}
                </button>
                {s.sumFrames != null && s.mode === 'crosshair' && (
                  <Dropdown
                    testid="selector-sum-frames"
                    value={String(s.sumFrames ?? 1)}
                    triggerText=""
                    bare
                    caretColor={s.mode === 'crosshair' ? '#11111b' : '#6c7086'}
                    options={sumFrameOptions(s.navSize ?? 0, s.navScale ?? 0)}
                    onChange={(v) => sendAction('set_selector_sum',
                      { frames: Number(v), selector_id: s.selectorId }, s.windowId)}
                  />
                )}
              </div>
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
    </div>
  )
}

/** The Sum-frames ladder: powers of two up to the navigation length.
 *
 *  Powers of two rather than round frame rates because a rate cannot generally
 *  divide the acquisition cadence — asking for "60 fps" on a 2564 fps camera
 *  silently rounds to 43 frames, so the number you picked is not the number
 *  you got. A frame count is always exact, and the rate it produces is shown
 *  beside it when the axis is time. */
function sumFrameOptions(navSize: number, navScale: number) {
  const opts: { value: string; label: string }[] = []
  for (let n = 1; n <= Math.max(1, navSize); n *= 2) {
    const rate = navScale > 0 ? 1 / (n * navScale) : 0
    const hz = rate >= 1000 ? `${(rate / 1000).toFixed(1)} kfps`
      : rate >= 1 ? `${rate.toFixed(0)} fps`
        : rate > 0 ? `${rate.toFixed(2)} fps` : ''
    const frames = n === 1 ? '1 frame' : `${n} frames`
    opts.push({ value: String(n), label: hz ? `${frames} — ${hz}` : frames })
    if (n >= 64) break
  }
  return opts
}

const styles: Record<string, React.CSSProperties> = {
  // The wrapper owns the button's look; the children sit transparently in it.
  // NO overflow:hidden — the caret's menu is absolutely positioned inside this
  // wrapper, so clipping it renders the options in the DOM but never visible.
  splitWrap: {
    display: 'flex', alignItems: 'center', padding: 0,
  },
  splitMain: {
    flex: 1, background: 'transparent', border: 'none', fontSize: 11,
    padding: '4px 2px 4px 6px', cursor: 'pointer', textAlign: 'center',
  },
  sumBadge: {
    marginLeft: 6, fontSize: 10, fontWeight: 600, opacity: 0.62,
    fontVariantNumeric: 'tabular-nums',
  },
  dock: {
    width: 300, flexShrink: 0,
    height: '100%',
    background: '#181825',
    borderLeft: '1px solid #313244',
    display: 'flex', flexDirection: 'column',
    color: '#cdd6f4',
  },
  header: {
    padding: '6px 10px', fontSize: 12, fontWeight: 600,
    borderBottom: '1px solid #313244', color: '#cdd6f4', flexShrink: 0,
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  // The controls below the header are PINNED (every `section` is flexShrink:0):
  // the metadata panel is the one elastic child, so it absorbs the height and is
  // the only thing that scrolls. Without that the dock was a fixed-height column
  // whose bottom sections (Layers, Navigator Selector) were cut off unreachably
  // on a laptop screen. `overflowY` here is the last resort for a window too
  // short even for the pinned controls plus the panel's minimum.
  body: { flex: 1, minHeight: 0, overflowY: 'auto', display: 'flex',
          flexDirection: 'column' },
  empty: { padding: 12, fontSize: 11, color: '#6c7086' },
  section: {
    padding: '6px 10px', flexShrink: 0,
    borderBottom: '1px solid #1e1e2e',
    display: 'flex', flexDirection: 'column', gap: 4,
  },
  label: { fontSize: 10, color: '#a6adc8' },
  select: {
    background: '#1e1e2e', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 4, padding: '2px 6px',
    fontSize: 11,
  },
  row: { display: 'flex', gap: 4 },
  // One of two equal columns in a shared section row. minWidth:0 so a long
  // dropdown value (signal types are wordy) truncates instead of pushing its
  // neighbour out of the dock.
  half: { flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 4 },
  input: {
    flex: 1, minWidth: 0,
    background: '#1e1e2e', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 4, padding: '2px 6px',
    fontSize: 11,
  },
  btn: {
    background: '#313244', color: '#cdd6f4', border: 'none',
    borderRadius: 4, padding: '4px 10px', fontSize: 12, cursor: 'pointer',
    alignSelf: 'flex-start',
  },
  hint: { fontSize: 10, color: '#6c7086', marginTop: 2 },
  // ONE metric for every paired button in the dock (Auto/Reset, Point/Integrate):
  // same height, same radius, same type size, so the rows line up and none of
  // them costs more vertical space than it has to on a laptop screen.
  autoBtn: {
    flex: 1, background: '#1e1e2e', color: '#a6adc8',
    border: '1px solid #313244', borderRadius: 4, padding: '2px 6px',
    fontSize: 10, lineHeight: '15px', cursor: 'pointer',
  },
  toggleRow: { display: 'flex', gap: 4, alignItems: 'center' },
  selectorDot: {
    width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
    border: '1px solid rgba(0,0,0,0.4)',
  },
  toggle: {
    flex: 1, background: '#1e1e2e', color: '#a6adc8',
    border: '1px solid #313244', borderRadius: 4, padding: '2px 6px',
    fontSize: 10, lineHeight: '15px', cursor: 'pointer',
  },
  toggleActive: {
    flex: 1, background: '#89b4fa', color: '#11111b',
    border: '1px solid #89b4fa', borderRadius: 4, padding: '2px 6px',
    fontSize: 10, lineHeight: '15px', cursor: 'pointer', fontWeight: 600,
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
  // Read-only twin of `axText`. It carried no fontSize, so every non-editable
  // cell (Dtype, Shape, the axes units) inherited the 16px document default and
  // rendered LARGER than its own key — the single biggest space waster in the
  // dock, and enough to wrap "nav 6 × 6 · sig 128 × 128" onto a second line.
  axCellRO: { color: '#6c7086', fontSize: 10 },
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
    display: 'flex', flexDirection: 'column', gap: 3,
    padding: '4px 0', borderBottom: '1px solid #1e1e2e',
  },
  layerTitle: {
    flex: 1, fontSize: 10, color: '#cdd6f4', minWidth: 0,
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
