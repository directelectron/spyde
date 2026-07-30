/**
 * ClassStrip.tsx — the floating in-canvas brush strip for Segment Particles
 * (plan B0: "Controls live on a floating strip next to the plot, not in the
 * caret").
 *
 * While painting you are looking at the IMAGE, so the three things switched
 * most often — active class, brush size, eraser — sit under the cursor. A
 * ~300 px round trip to the caret per class switch is friction you feel a
 * thousand times over a labelling session.
 *
 * Deliberately swatch-ONLY: class NAMES and per-class labelled-pixel counts
 * stay in the caret's class list, which is the authoritative view (plan B7).
 * Duplicating them here would make the strip wide enough to cover the data it
 * exists to sit next to, and there would then be two places showing counts
 * that can disagree.
 *
 * Positioning is supplied by the caller (`posStyle`), because only
 * FloatingToolbar knows the owning window's live rect — the strip is a DOM
 * child of the floating toolbar (which is parented to the window root and so
 * tracks move/resize for free) placed back up over the top-left of the figure.
 */
import React from 'react'
import type { SegClassInfo } from '../kernel/protocol'

interface Props {
  /** Authoritative class list from `seg_state`. */
  classes: SegClassInfo[]
  /** Currently painting class id. */
  activeId: number
  onSelect: (id: number) => void
  /** Brush diameter in image pixels (the backend's `brush` param). */
  brush: number
  onBrush: (b: number) => void
  eraser: boolean
  onEraser: (b: boolean) => void
  /** Absolute placement over the figure, computed by FloatingToolbar. */
  posStyle: React.CSSProperties
}

export function ClassStrip({
  classes, activeId, onSelect, brush, onBrush, eraser, onEraser, posStyle,
}: Props) {
  return (
    <div data-testid="seg-class-strip" style={{ ...posStyle, ...S.strip }}>
      {classes.map(c => {
        const active = !eraser && c.id === activeId
        return (
          <button
            key={c.id}
            data-testid={`seg-strip-class-${c.id}`}
            data-active={active ? 'true' : 'false'}
            title={`${c.name} — ${c.pixels.toLocaleString()} px labelled`}
            onClick={() => { onEraser(false); onSelect(c.id) }}
            style={{
              ...S.swatch,
              background: c.colour,
              // A selected swatch is ringed rather than resized: the strip must
              // not reflow when you switch class, or the next swatch moves out
              // from under the cursor mid-stroke.
              boxShadow: active ? '0 0 0 2px #cdd6f4' : '0 0 0 1px #45475a',
            }}
          />
        )
      })}

      <span style={S.sep} />

      <button
        data-testid="seg-strip-eraser"
        data-active={eraser ? 'true' : 'false'}
        title="Eraser — removes labels under the brush (not a background class)"
        onClick={() => onEraser(!eraser)}
        style={{ ...S.iconBtn, ...(eraser ? S.iconBtnOn : {}) }}
      >⌫</button>

      <span style={S.sep} />

      {/* Brush size lives here rather than in the caret's Scribble tab for the
          same reason as the swatches: it is adjusted between strokes. */}
      <input
        data-testid="seg-strip-brush"
        type="range" min={1} max={32} step={1} value={brush}
        title="Brush size (px)"
        onChange={(e) => onBrush(Number(e.target.value))}
        style={S.range}
      />
      <span data-testid="seg-strip-brush-val" style={S.brushVal}>{brush}</span>
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  strip: {
    display: 'flex', alignItems: 'center', gap: 5,
    background: 'rgba(24,24,37,0.94)', border: '1px solid #313244',
    borderRadius: 8, padding: '4px 7px', zIndex: 13,
    boxShadow: '0 6px 20px rgba(0,0,0,0.5)',
    width: 'max-content',
  },
  swatch: {
    width: 16, height: 16, borderRadius: 4, border: 'none', padding: 0,
    cursor: 'pointer', flex: '0 0 auto',
  },
  sep: { width: 1, height: 16, background: '#313244', flex: '0 0 auto' },
  iconBtn: {
    width: 20, height: 18, padding: 0, fontSize: 11, lineHeight: '16px',
    background: 'transparent', color: '#cdd6f4', border: '1px solid #45475a',
    borderRadius: 4, cursor: 'pointer', flex: '0 0 auto',
  },
  // Full `border` shorthand, not a `borderColor` longhand over iconBtn's
  // shorthand — see the note on SegmentWizard's classRowActiveStyle.
  iconBtnOn: {
    background: '#89b4fa', color: '#11111b', border: '1px solid #89b4fa',
  },
  range: { width: 70, flex: '0 0 auto' },
  brushVal: {
    fontSize: 10, color: '#cdd6f4', minWidth: 14, textAlign: 'right',
    fontVariantNumeric: 'tabular-nums',
  },
}
