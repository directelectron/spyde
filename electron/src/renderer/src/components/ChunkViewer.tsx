/**
 * ChunkViewer.tsx — what "lazy" means for THIS dataset, drawn.
 *
 * dask ships a block diagram in `_repr_html_` (which is what HyperSpy shows in a
 * notebook) and it is a good idea; this is that idea in the app's palette, in
 * the dock's own caret box, with the one thing dask cannot know added. dask sees
 * an N-d array of blocks. SpyDE knows which of those axes are NAVIGATION and
 * which are SIGNAL, and that distinction decides whether the chunking is good or
 * ruinous here: the navigator displays one frame as `data[iy, ix]`, so a chunk
 * spanning whole frames costs one block read, while a chunk that SPLITS the
 * signal axes makes every frame a multi-block read (and seams the navigator
 * sum). See the storage-alignment note in CLAUDE.md.
 *
 * So the diagram is the two spaces the user actually thinks in — a navigation
 * grid and a signal frame, each carved by its real chunk boundaries — rather
 * than one abstract N-d block stack, and the verdict line says which case this
 * is.
 */
import React from 'react'
import type { ChunkInfo } from '../kernel/SpyDEContext'
import { CaretBox } from './CaretBox'

/** Largest drawn subdivision per axis. Past this the boundaries are closer than
 *  a pixel and the drawing says nothing the count doesn't; the label carries the
 *  real number. */
const MAX_LINES = 40

const fmtBytes = (n: number) => {
  if (!isFinite(n) || n <= 0) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.min(u.length - 1, Math.floor(Math.log10(n) / 3))
  return `${(n / 1000 ** i).toFixed(i === 0 ? 0 : 1)} ${u[i]}`
}

/** Cumulative fractional offsets of the chunk boundaries along one axis. */
function boundaries(sizes: number[], total: number, count: number): number[] {
  // Truncated payload: fall back to an even split so the drawing still conveys
  // "this many blocks" rather than crowding them all at the left edge.
  const use = count > sizes.length
    ? Array.from({ length: Math.min(count, MAX_LINES) }, () => total / Math.min(count, MAX_LINES))
    : sizes
  const out: number[] = []
  let acc = 0
  for (const s of use.slice(0, MAX_LINES)) {
    acc += s
    if (acc < total) out.push(acc / total)
  }
  return out
}

/** One space (navigation or signal) as a rectangle carved into its blocks.
 *  A 1-D space is drawn as a single row of columns. */
function SpaceBox({ dims, w, h, tint, label }: {
  dims: Array<{ size: number; sizes: number[]; count: number; name: string }>
  w: number; h: number; tint: string; label: string
}) {
  // Array order is (slowest … fastest), i.e. (y, x) — so the LAST dimension is
  // horizontal, matching how the image itself is displayed.
  const xd = dims[dims.length - 1]
  const yd = dims.length > 1 ? dims[dims.length - 2] : null
  const vx = boundaries(xd.sizes, xd.size, xd.count)
  const vy = yd ? boundaries(yd.sizes, yd.size, yd.count) : []
  // A 3rd+ navigation dimension (a 5-D in-situ series) can't go in the plane;
  // it multiplies the grid instead, so it is stated rather than drawn.
  const extra = dims.slice(0, Math.max(0, dims.length - 2))

  return (
    <div style={S.space}>
      <div style={S.spaceLabel}>{label}</div>
      <svg width={w} height={h} style={{ display: 'block' }}>
        <rect x={0.5} y={0.5} width={w - 1} height={h - 1} rx={2}
          fill={tint} stroke="#585b70" strokeWidth={1} />
        {vx.map((f, i) => (
          <line key={`x${i}`} x1={f * w} y1={0} x2={f * w} y2={h}
            stroke="#89b4fa" strokeWidth={1} opacity={0.75} />
        ))}
        {vy.map((f, i) => (
          <line key={`y${i}`} x1={0} y1={f * h} x2={w} y2={f * h}
            stroke="#89b4fa" strokeWidth={1} opacity={0.75} />
        ))}
      </svg>
      {/* Per axis: how long it is, and how many blocks it is cut into. Terse
          because the box is 250px wide — the tooltip spells it out. */}
      <div style={S.axisRow} title="axis length / number of blocks along it">
        {dims.map((d, i) => (
          <span key={i} style={S.axisTag}>
            {d.size}<span style={S.blocks}>/{d.count}</span>
          </span>
        ))}
      </div>
      {extra.length > 0 && <div style={S.extra}>+{extra.length}d stacked</div>}
    </div>
  )
}

export function ChunkViewer({ info, anchor, el, onClose }: {
  info: ChunkInfo
  anchor: DOMRect
  el: HTMLElement
  onClose: () => void
}) {
  const dims = info.shape.map((size, i) => ({
    size,
    sizes: info.chunks[i] ?? [size],
    count: info.counts[i] ?? 1,
    name: info.names[i] ?? '',
  }))
  const nav = dims.slice(0, info.nav_ndim)
  const sig = dims.slice(info.nav_ndim)
  // Chunk shape as the user would pass it to hs.load(chunks=…).
  const chunkShape = dims.map((d) => Math.max(...(d.sizes.length ? d.sizes : [d.size])))

  return (
    <CaretBox anchor={anchor} el={el} width={250} testid="chunk-viewer" onClose={onClose}>
      <div style={S.title}>Chunking</div>

      <div style={S.diagram}>
        {nav.length > 0 && (
          <SpaceBox dims={nav} w={116} h={74} tint="rgba(137,180,250,0.10)"
            label="Navigation" />
        )}
        {sig.length > 0 && (
          <SpaceBox dims={sig} w={74} h={74}
            // The signal box is the verdict: whole frames read blue, a split one
            // reads red, because that is the case that costs you.
            tint={info.signal_split ? 'rgba(243,139,168,0.12)' : 'rgba(137,180,250,0.10)'}
            label="Signal" />
        )}
      </div>

      <div style={S.verdict} data-testid="chunk-verdict">
        {info.signal_split ? (
          <span style={S.bad}
            title={`Reload with chunks spanning the full signal: chunks=(${nav.map(() => 32).join(', ')}, ${sig.map(() => -1).join(', ')})`}>
            ⚠ Chunks split the signal axes — one frame spans several blocks, so
            every navigator move reads more than it needs.
          </span>
        ) : (
          <span style={S.good}>✓ Whole frames per block — a navigator move reads one.</span>
        )}
      </div>

      <div style={S.stats}>
        <Stat k="Shape" v={info.shape.join(' × ')} />
        <Stat k="Chunk" v={chunkShape.join(' × ')} />
        <Stat k="Blocks" v={info.n_chunks.toLocaleString()} />
        <Stat k="Per block" v={fmtBytes(info.chunk_bytes)} />
        <Stat k="Total" v={fmtBytes(info.nbytes)} />
        <Stat k="Dtype" v={info.dtype} mono />
      </div>
    </CaretBox>
  )
}

function Stat({ k, v, mono }: { k: string; v: string; mono?: boolean }) {
  return (
    <div style={S.stat} title={`${k}: ${v}`}>
      <span style={S.statKey}>{k}</span>
      <span style={{ ...S.statVal, fontFamily: mono ? 'ui-monospace, monospace' : undefined }}>{v}</span>
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  title: { fontSize: 11, fontWeight: 600, color: '#cdd6f4' },
  diagram: { display: 'flex', gap: 10, alignItems: 'flex-start' },
  space: { display: 'flex', flexDirection: 'column', gap: 2 },
  spaceLabel: { fontSize: 9, color: '#a6adc8', fontWeight: 600 },
  axisRow: { display: 'flex', gap: 6, flexWrap: 'wrap' },
  axisTag: { fontSize: 9, color: '#6c7086', whiteSpace: 'nowrap' },
  blocks: { color: '#89b4fa' },
  extra: { fontSize: 9, color: '#6c7086' },
  verdict: { fontSize: 9, lineHeight: 1.4 },
  good: { color: '#a6e3a1' },
  bad: { color: '#f38ba8' },
  stats: {
    // TWO columns, not three: "24 × 24 × 32 × 32" is the whole point of this
    // box and it does not fit in a third of 250px without ellipsing.
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3px 8px',
    borderTop: '1px solid #313244', paddingTop: 5,
  },
  stat: { display: 'flex', flexDirection: 'column', minWidth: 0 },
  statKey: { fontSize: 8, color: '#6c7086' },
  statVal: {
    fontSize: 10, color: '#cdd6f4', overflow: 'hidden',
    textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
}
