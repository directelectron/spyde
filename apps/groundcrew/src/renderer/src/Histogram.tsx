/**
 * Histogram.tsx — SpyDE's contrast histogram, ported verbatim.
 *
 * A deliberate COPY of `electron/src/renderer/src/components/PlotControlDock.tsx`,
 * down to the palette: an engineer moving between SpyDE and Ground Crew should
 * find the same widget behaving the same way, not a lookalike with different
 * drag semantics. The comments below are the originals — they record why each
 * piece is the way it is, and every one of those reasons still applies here.
 *
 * It keeps SpyDE's catppuccin colours rather than Ground Crew's tokens, which
 * is the whole point of "same style". They are named in `SPYDE` so it is
 * obvious they are borrowed and must be changed together if SpyDE's change.
 *
 * The one difference: SpyDE is handed explicit bin `edges`; the DE Server
 * reports a uniform histogram as counts plus a min/max, so edges are derived.
 * Uniform bins make that identical.
 */
import React from 'react'

/** SpyDE's palette. Borrowed on purpose — see the module docstring. */
const SPYDE = {
  well: '#1e1e2e',
  bar: '#89b4fa',
  overflow: '#585b70',
  handle: '#f38ba8',
  line: '#313244',
  text: '#a6adc8',
  textBright: '#cdd6f4',
} as const

const W = 276, H = 62

export function Histogram({ counts, lo, hi, vmin, vmax, clipped, onClim, onAuto, onReset }: {
  counts: number[]
  /** First and last bin edge — the range the counts span. */
  lo: number
  hi: number
  vmin: number
  vmax: number
  clipped?: boolean
  onClim: (mn: number, mx: number) => void
  onAuto: () => void
  onReset: () => void
}) {
  const svgRef = React.useRef<SVGSVGElement>(null)
  const [drag, setDrag] = React.useState<null | 'min' | 'max'>(null)
  // The handles are a HOVER affordance: at rest the clim edges are just a hairline
  // (the tinted range already shows where they are), and the grabbable pink bars
  // fade in when the pointer is over the widget. Two fat pink bars parked over the
  // data all the time read as part of the plot and hide the left-most bins.
  const [hover, setHover] = React.useState(false)

  const max = Math.max(...counts, 0) || 1
  const dataSpan = hi - lo || 1
  // Draw a little axis PAST the binned range so the upper handle can be pulled
  // above 100% (darkens the image). The bars only span [lo, hi].
  //
  // The axis deliberately does NOT stretch to cover an out-of-range vmax (after
  // Reset, or a contrast held from a brighter frame): that stretch re-created
  // the squish, since a vmax out at a hot-pixel value compresses every bar into
  // the left edge. A handle beyond the axis pins at the border with the
  // off-range arrow instead — visible, grabbable, and the bars keep the width.
  const axLo = lo
  const axHi = hi + 0.25 * dataSpan
  const span = axHi - axLo || 1
  const bw = counts.length ? (dataSpan / span) * W / counts.length : 0
  // No clamping either side: dragging past the drawn area maps to values outside
  // the binned range, which is the only way to reach a clipped tail.
  const xOf = (v: number) => ((v - axLo) / span) * W
  const vOf = (x: number) => axLo + (x / W) * span
  const fmt = (v: number) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01))
    ? v.toExponential(1) : v.toFixed(2)

  const armed = hover || drag != null

  React.useEffect(() => {
    if (!drag) return
    const move = (e: PointerEvent) => {
      const rect = svgRef.current?.getBoundingClientRect()
      if (!rect || !rect.width) return
      // Rescale CSS pixels into viewBox units. SpyDE's copy renders at a fixed
      // `width={W}`, so the two coincide and it maps the offset directly; this
      // one is `width: 100%` in a narrower panel, and dropping the scale made
      // every drag land at the wrong value — far enough left that the handle
      // appeared not to move at all.
      const v = vOf((e.clientX - rect.left) * (W / rect.width))
      if (drag === 'min') onClim(Math.min(v, vmax), vmax)
      else onClim(vmin, Math.max(v, vmin))
    }
    const up = () => setDrag(null)
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up, { once: true })
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [drag, vmin, vmax, lo, span])

  if (!counts.length) return null

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
        <line x1={x} y1={0} x2={x} y2={H} stroke={SPYDE.handle}
          strokeWidth={armed ? 3 : 1} opacity={armed ? 1 : 0.5}
          style={{ transition: 'stroke-width 90ms, opacity 90ms' }} />
        {/* grip caps so the thick lines read as draggable handles — only once
            hovered, since that is the only time they can be grabbed */}
        {armed && <>
          <rect x={x - 3} y={0} width={6} height={5} rx={1.5} fill={SPYDE.handle} />
          <rect x={x - 3} y={H - 5} width={6} height={5} rx={1.5} fill={SPYDE.handle} />
        </>}
        {/* the off-range arrow is NOT part of the hover affordance: it says the
            real value is outside the drawn range, which is worth showing at rest */}
        {off !== 0 && (
          <polygon data-testid={`hist-${which}-offscreen`}
            points={off < 0
              ? `${x - 3},${mid} ${x + 6},${mid - 5} ${x + 6},${mid + 5}`
              : `${x + 3},${mid} ${x - 6},${mid - 5} ${x - 6},${mid + 5}`}
            fill={SPYDE.handle} />
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
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} height={H} data-testid="histogram"
        onPointerEnter={() => setHover(true)}
        onPointerLeave={() => setHover(false)}
        style={{
          width: '100%', background: SPYDE.well, borderRadius: 4,
          touchAction: 'none', display: 'block',
        }}>
        <title>Drag the handles to set contrast</title>
        {/* selected range tint */}
        <rect x={xOf(vmin)} y={0} width={Math.max(0, xOf(vmax) - xOf(vmin))} height={H}
          fill={SPYDE.bar} opacity={0.12} />
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
            height={h} fill={overflow ? SPYDE.overflow : SPYDE.bar} />
        })}
        {/* end-of-bins marker: dragging the max handle to the RIGHT of this line
            pushes the upper clim past the binned range (image gets darker). */}
        {xOf(hi) < W && (
          <line data-testid="hist-datamax" x1={xOf(hi)} y1={0} x2={xOf(hi)} y2={H}
            stroke={SPYDE.overflow} strokeWidth={1} strokeDasharray="2 2" />
        )}
        {handle('min', vmin)}
        {handle('max', vmax)}
      </svg>
      {/* min / max display-range labels */}
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginTop: 2,
      }}>
        <span data-testid="clim-min" style={{ fontSize: 9, color: SPYDE.text }}>{fmt(vmin)}</span>
        <span data-testid="clim-max" style={{ fontSize: 9, color: SPYDE.text }}>{fmt(vmax)}</span>
      </div>
      {/* …and the two range buttons on their own row. */}
      <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginTop: 3 }}>
        <TintButton testid="clim-auto" onClick={onAuto}
          title="Contrast from the robust levels for this frame (2–98%)">◐ Auto</TintButton>
        <TintButton testid="clim-reset" onClick={onReset}
          title="Show the full data range, tail included">↺ Reset</TintButton>
      </div>
    </div>
  )
}

function TintButton({ children, onClick, testid, title }: {
  children: React.ReactNode; onClick: () => void; testid?: string; title?: string
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
        flex: 1, borderRadius: 4, padding: '2px 6px',
        fontSize: 10, lineHeight: '15px', cursor: 'pointer',
        background: held ? SPYDE.overflow : hover ? SPYDE.line : SPYDE.well,
        border: `1px solid ${held || hover ? SPYDE.overflow : SPYDE.line}`,
        color: held || hover ? SPYDE.textBright : SPYDE.text,
      }}
    >{children}</button>
  )
}
