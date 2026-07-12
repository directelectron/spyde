/**
 * ConsolePreviewPanel.tsx — the live-preview pop-out that floats ABOVE the
 * console bar while the eye toggle is on (or a one-shot Ctrl+Enter preview is
 * active). Anchored to the console root like the completions popup, but
 * right-aligned; it floats, so its size never reflows the app.
 *
 *   ┌──────────────────┐
 *   │   ▒▓█ preview    │  ← image ≤208px (small frames UPSCALED, pixelated),
 *   │   ▓█▒            │    sparkline 208×72, or a text line
 *   │ 128×128 uint16 · 12 ms │
 *   └──────────────────┘
 *   >>> ⟨s1⟩ > 100        👁 …
 *
 * The backend (`console_preview_result`) replies with one of four kinds:
 *   • image      — raw uint8 GRAYSCALE bytes (w*h, row-major) → aspect-fit canvas
 *   • sparkline  — ≤512 points (null = a gap in the stroke) → 1-px polyline
 *   • scalar     — a short text repr
 *   • unavailable— a quiet reason (or empty → keep the previous content dimmed)
 *
 * DESIGN RULE — the panel NEVER goes red and NEVER flashes. A failing /
 * expensive expression must read as "nothing to show", not as an error. The
 * LAST content stays up at FULL opacity while a new preview computes and is
 * swapped in place when the reply lands — no dim/blank between keystrokes.
 */
import React, { useEffect, useRef } from 'react'
import type { ConsolePreviewResult } from '../kernel/SpyDEContext'

const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace'

// Max edge for the image canvas — small frames are UPSCALED to fill it
// (nearest-neighbour via imageRendering:pixelated), big ones letterboxed down.
const PANEL_EDGE = 208
const SPARK_W = 208
const SPARK_H = 72

interface Props {
  preview: ConsolePreviewResult | null
}

// ── uint8 grayscale bytes → RGBA ImageData (an offscreen canvas of the SOURCE
//    size), so we can aspect-fit `drawImage` it at any scale. atob → binary
//    string → Uint8Array; each gray byte becomes an opaque (r=g=b) pixel.
function grayToImageData(dataB64: string, w: number, h: number): ImageData | null {
  if (!dataB64 || w <= 0 || h <= 0) return null
  let bin: string
  try {
    bin = atob(dataB64)
  } catch {
    return null
  }
  const n = w * h
  if (bin.length < n) return null
  const rgba = new Uint8ClampedArray(n * 4)
  for (let i = 0; i < n; i++) {
    const g = bin.charCodeAt(i) & 0xff
    const o = i * 4
    rgba[o] = g
    rgba[o + 1] = g
    rgba[o + 2] = g
    rgba[o + 3] = 255
  }
  return new ImageData(rgba, w, h)
}

// Draw an ImageData filling a destination 2D context of (destW × destH) — the
// canvas element itself is already sized to the preview's aspect, so this is a
// straight nearest-neighbour scale via a temporary source-sized canvas.
function drawScaled(
  ctx: CanvasRenderingContext2D,
  img: ImageData,
  destW: number,
  destH: number,
): void {
  ctx.clearRect(0, 0, destW, destH)
  const src = document.createElement('canvas')
  src.width = img.width
  src.height = img.height
  const sctx = src.getContext('2d')
  if (!sctx) return
  sctx.putImageData(img, 0, 0)
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(src, 0, 0, destW, destH)
}

// Draw a sparkline: normalise finite points into [0,1], map to the canvas,
// break the stroke on `null` gaps.
function drawSparkline(
  ctx: CanvasRenderingContext2D,
  points: (number | null)[],
  destW: number,
  destH: number,
): void {
  ctx.clearRect(0, 0, destW, destH)
  const finite = points.filter((p): p is number => p != null && Number.isFinite(p))
  if (finite.length === 0) return
  let min = finite[0]
  let max = finite[0]
  for (const v of finite) {
    if (v < min) min = v
    if (v > max) max = v
  }
  const span = max - min || 1
  const pad = 3
  const innerW = Math.max(1, destW - pad * 2)
  const innerH = Math.max(1, destH - pad * 2)
  const n = points.length
  ctx.strokeStyle = '#89b4fa'
  ctx.lineWidth = 1
  ctx.beginPath()
  let penDown = false
  for (let i = 0; i < n; i++) {
    const p = points[i]
    if (p == null || !Number.isFinite(p)) {
      penDown = false
      continue
    }
    const x = pad + (n <= 1 ? 0 : (i / (n - 1)) * innerW)
    // Higher value → higher on screen (smaller y).
    const y = pad + innerH - ((p - min) / span) * innerH
    if (!penDown) {
      ctx.moveTo(x, y)
      penDown = true
    } else {
      ctx.lineTo(x, y)
    }
  }
  ctx.stroke()
}

function shapeDtypeLine(p: ConsolePreviewResult): string {
  const shape = p.shape && p.shape.length ? p.shape.join('×') : ''
  const dtype = p.dtype ?? ''
  return [shape, dtype].filter(Boolean).join(' ')
}

export function ConsolePreviewPanel({ preview }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Image canvas size: fit the preview's aspect into PANEL_EDGE, UPSCALING
  // small frames (the whole point of the pop-out — the inline slot was too
  // small to read).
  let imgW = 0
  let imgH = 0
  if (preview && preview.kind === 'image' && preview.w > 0 && preview.h > 0) {
    const s = Math.min(PANEL_EDGE / preview.w, PANEL_EDGE / preview.h)
    imgW = Math.max(1, Math.round(preview.w * s))
    imgH = Math.max(1, Math.round(preview.h * s))
  }

  // Paint the canvas whenever the preview changes.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (!preview) return
    if (preview.kind === 'image') {
      const img = grayToImageData(preview.dataB64, preview.w, preview.h)
      if (img) drawScaled(ctx, img, canvas.width, canvas.height)
      else ctx.clearRect(0, 0, canvas.width, canvas.height)
    } else if (preview.kind === 'sparkline') {
      drawSparkline(ctx, preview.points ?? [], canvas.width, canvas.height)
    }
  }, [preview, imgW, imgH])

  // Choose the body by kind.
  let body: React.ReactNode
  if (!preview) {
    // Nothing yet (or empty expression) → a quiet placeholder.
    body = <span style={styles.placeholder}>no preview yet</span>
  } else if (preview.kind === 'image') {
    body = (
      <canvas
        ref={canvasRef}
        data-testid="console-preview-canvas"
        width={imgW}
        height={imgH}
        style={styles.canvas}
      />
    )
  } else if (preview.kind === 'sparkline') {
    body = (
      <canvas
        ref={canvasRef}
        data-testid="console-preview-canvas"
        width={SPARK_W}
        height={SPARK_H}
        style={styles.canvas}
      />
    )
  } else if (preview.kind === 'scalar') {
    body = <span style={styles.scalar}>{preview.text}</span>
  } else {
    // unavailable — a quiet reason. EMPTY reason means "keep prior content
    // dimmed" (never a jump / never red); render a muted placeholder line.
    body = preview.reason
      ? <span style={styles.reason}>{preview.reason}</span>
      : <span style={styles.placeholder}>·</span>
  }

  // Badge line: shape×dtype (when known) + the backend's elapsed time.
  const badge = preview && preview.kind !== 'unavailable' ? shapeDtypeLine(preview) : ''
  const elapsed = preview?.elapsedMs != null ? `${Math.round(preview.elapsedMs)} ms` : ''

  return (
    <div data-testid="console-preview-panel" style={styles.panel}>
      <div style={styles.body}>{body}</div>
      {(badge || elapsed) && (
        <div style={styles.badgeRow}>
          <span style={styles.badge}>{badge}</span>
          <span style={styles.elapsed}>{elapsed}</span>
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  // Anchored to the EYE button's position:relative wrapper in ConsoleBar, so
  // the pop-out opens directly above the eye (right edges aligned) — borrows
  // the completions-popup look. Floats over the MDI area, so size changes
  // never reflow the app.
  panel: {
    position: 'absolute',
    right: 0,
    bottom: '100%',
    marginBottom: 10,
    background: '#1e1e2e',
    border: '1px solid #313244',
    borderRadius: 6,
    boxShadow: '0 -6px 20px rgba(0,0,0,0.45)',
    padding: 8,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 6,
    minWidth: 120,
    maxWidth: PANEL_EDGE + 16,
    zIndex: 9200,
  },
  body: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 24,
  },
  canvas: {
    display: 'block',
    imageRendering: 'pixelated',
    background: '#11111b',
    borderRadius: 3,
  },
  scalar: {
    fontFamily: MONO,
    fontSize: 12,
    color: '#cdd6f4',
    maxWidth: PANEL_EDGE,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  reason: {
    fontFamily: MONO,
    fontSize: 10,
    color: '#6c7086',
    maxWidth: PANEL_EDGE,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  placeholder: {
    fontFamily: MONO,
    fontSize: 10,
    color: '#45475a',
  },
  badgeRow: {
    display: 'flex',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: 10,
    alignSelf: 'stretch',
  },
  badge: {
    fontFamily: MONO,
    fontSize: 9,
    color: '#a6adc8',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  elapsed: {
    fontFamily: MONO,
    fontSize: 9,
    color: '#6c7086',
    flexShrink: 0,
  },
}
