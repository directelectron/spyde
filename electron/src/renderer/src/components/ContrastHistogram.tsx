/**
 * ContrastHistogram.tsx — the log-scaled histogram with draggable black/white
 * level handles.
 *
 * Lifted verbatim out of PlotControlDock so the report figure editor drives
 * contrast with the SAME widget the live plot dock does: two surfaces that let
 * a user set a display range must agree about what the bars mean, where the
 * headroom is, and what Auto and Reset do — a second implementation drifts,
 * and the two are read side by side.
 *
 * Purely presentational: it owns no clim state and sends no actions. The host
 * supplies the binned data and the three callbacks.
 */
import React from 'react'

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

export function ContrastHistogram({ counts, edges, vmin, vmax, threshold, clipped, onClim, onAuto, onReset, testidPrefix = "" }:
  { counts: number[]; edges: number[]; vmin: number; vmax: number
    threshold?: number | null
    clipped?: boolean
    onClim: (mn: number, mx: number) => void
    onAuto: () => void
    onReset: () => void
    /** Scopes every testid to one host. Two of these can be on screen at once
     *  (the plot dock and a report cell), and an unscoped `clim-auto` then
     *  matches both — a spec targeting one silently drives the other. */
    testidPrefix?: string }) {
  const tid = (name: string) => (testidPrefix ? `${testidPrefix}-${name}` : name)
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
          <polygon data-testid={tid(`hist-${which}-offscreen`)}
            points={off < 0
              ? `${x - 3},${mid} ${x + 6},${mid - 5} ${x + 6},${mid + 5}`
              : `${x + 3},${mid} ${x - 6},${mid - 5} ${x - 6},${mid + 5}`}
            fill="#f38ba8" />
        )}
        {/* fat invisible grab target */}
        <rect
          data-testid={tid(`hist-${which}-handle`)}
          x={x - 6} y={0} width={12} height={H}
          fill="transparent" style={{ cursor: 'ew-resize' }}
          onPointerDown={(e) => { e.preventDefault(); setDrag(which) }}
        />
      </g>
    )
  }

  return (
    <div>
      <svg ref={svgRef} width={W} height={H} data-testid={tid("histogram")}
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
          <line data-testid={tid("hist-datamax")} x1={xOf(hi)} y1={0} x2={xOf(hi)} y2={H}
            stroke="#585b70" strokeWidth={1} strokeDasharray="2 2" />
        )}
        {/* Find-Vectors detector threshold: dotted orange line (in image units) */}
        {threshold != null && threshold >= lo && threshold <= hi && (
          <line data-testid={tid("hist-threshold")} x1={xOf(threshold)} y1={0}
            x2={xOf(threshold)} y2={H} stroke="#ffae57" strokeWidth={1.5}
            strokeDasharray="3 3" />
        )}
        {handle('min', vmin)}
        {handle('max', vmax)}
      </svg>
      {/* min / max display-range labels (replaces the old Scale section) */}
      <div style={{ display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginTop: 2 }}>
        <span data-testid={tid("clim-min")} style={{ fontSize: 9, color: '#a6adc8' }}>{fmt(vmin)}</span>
        <span data-testid={tid("clim-max")} style={{ fontSize: 9, color: '#a6adc8' }}>{fmt(vmax)}</span>
      </div>
      {/* …and the two range buttons on their own row, sized/styled like the
          Point / Integrate selector-mode pair so the dock reads as one system. */}
      <div style={{ ...styles.toggleRow, marginTop: 3 }}>
        <TintButton testid={tid("clim-auto")} onClick={onAuto}
          title="Contrast from the robust levels for this frame (2–99%)">◐ Auto</TintButton>
        <TintButton testid={tid("clim-reset")} onClick={onReset}
          title="Show the full data range, tail included">↺ Reset</TintButton>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  autoBtn: {
    flex: 1, background: '#1e1e2e', color: '#a6adc8',
    border: '1px solid #313244', borderRadius: 4, padding: '2px 6px',
    fontSize: 10, lineHeight: '15px', cursor: 'pointer',
  },
  toggleRow: { display: 'flex', gap: 4, alignItems: 'center' },
}
