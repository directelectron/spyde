/**
 * SlideOverview.tsx — a THUMBNAIL GRID of all slides for Present mode.
 *
 * THE PRESENTER'S "JUMP AROUND" + REORDER TOOL. A full-screen grid overlay (on
 * top of the Present-mode stage) that shows ONE thumbnail per slide:
 *   • CLICK a thumbnail → jump to that slide (setIndex) and close the grid.
 *   • DRAG a thumbnail onto another position → reorder the WHOLE slide via the
 *     `report_move_slide {from, to}` backend verb. After the move the report
 *     re-emits `report_state`, the deck regroups, and the grid re-renders.
 *
 * THUMBNAILS ARE CHEAP + STATIC — NOT live iframes. Each thumbnail reuses
 * PresentMode's `SlidePreview`, which renders markdown + a figure cell's BAKED
 * PNG (`cell.png` data URL, the offline snapshot) + image cells' photos, scaled
 * DOWN. It deliberately does NOT mount the live `SeamlessFigureFrame` (those live
 * only in the hidden audience stack behind this overlay), so a grid of N slides
 * is N static previews, not N heavy live embeds. The full slide is live; the
 * thumbnail is a picture of it.
 *
 * Opened from Present mode with `O` (or the grid button in the present header).
 * ESC / clicking the backdrop closes the grid WITHOUT exiting Present mode (the
 * present-mode ESC=exit still works once the grid is closed). The current slide's
 * thumbnail is highlighted.
 */
import React from 'react'
import type { ReportCell } from '../kernel/protocol'
import { SlidePreview, slideMeta, slideNotes } from './PresentMode'

interface Props {
  /** All slides (cell-groups), same grouping as Present mode. */
  slides: ReportCell[][]
  /** The currently-active slide index (highlighted in the grid). */
  index: number
  /** Jump to a slide (Present mode's setIndex) — also closes the grid. */
  onJump: (index: number) => void
  /** Close the overview without exiting Present mode. */
  onClose: () => void
  /** Reorder a WHOLE slide: fires `report_move_slide {from, to}`. */
  onMoveSlide: (from: number, to: number) => void
}

/** A short human LABEL for a slide: its first markdown heading / first non-empty
 *  line, else its first cell's caption, else "Slide N". Strips leading markdown
 *  heading markers and trims. Kept tiny — it's a grid caption, not a title. */
function slideLabel(cells: ReportCell[], slideNumber: number): string {
  for (const c of cells) {
    if (c.cell_type === 'markdown' && (c.source ?? '').trim()) {
      const firstLine = (c.source ?? '')
        .split('\n')
        .map(l => l.trim())
        .find(l => l.length > 0)
      if (firstLine) {
        // Strip a leading markdown heading marker (#, ##, …) + surrounding
        // emphasis so a title reads clean in the grid.
        const cleaned = firstLine
          .replace(/^#{1,6}\s+/, '')
          .replace(/[*_`]/g, '')
          .trim()
        if (cleaned) return cleaned
      }
    }
  }
  // No markdown title — fall back to a caption (figure/image) then the number.
  for (const c of cells) {
    if ((c.caption ?? '').trim()) return (c.caption ?? '').trim()
  }
  return `Slide ${slideNumber}`
}

export function SlideOverview({ slides, index, onJump, onClose, onMoveSlide }: Props) {
  // The slide currently being dragged (its index) + the current drop target.
  // `dragFromRef` mirrors `dragFrom` synchronously — React state hasn't committed
  // yet when a rapid synthetic dragstart→drop fires (test / clicker), so the drop
  // handler reads the ref (the sidebar's `dragCellId.current` idiom).
  const [dragFrom, setDragFrom] = React.useState<number | null>(null)
  const dragFromRef = React.useRef<number | null>(null)
  const [dropOn, setDropOn] = React.useState<number | null>(null)

  // ESC closes the grid (but not Present mode). Capture-phase so it beats the
  // present-mode window handler (which would otherwise exit the whole deck).
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        e.stopPropagation()
        onClose()
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  const endDrag = () => { dragFromRef.current = null; setDragFrom(null); setDropOn(null) }

  return (
    <div
      style={styles.overlay}
      data-testid="slide-overview"
      // Click on the backdrop (not a thumbnail) closes the grid.
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div style={styles.header}>
        <span style={styles.title} data-testid="slide-overview-title">
          Slide overview — {slides.length} slide{slides.length === 1 ? '' : 's'}
        </span>
        <span style={styles.hint}>Click a slide to jump · drag to reorder · Esc to close</span>
        <div style={{ flex: 1 }} />
        <button
          data-testid="slide-overview-close"
          style={styles.closeBtn}
          title="Close overview (Esc)"
          onClick={onClose}
        >✕</button>
      </div>

      <div style={styles.grid} data-testid="slide-overview-grid">
        {slides.map((cells, si) => {
          const meta = slideMeta(cells)
          const hasNotes = slideNotes(cells).trim().length > 0
          const isTitle = meta.kind === 'title'
          const label = slideLabel(cells, si + 1)
          const isCurrent = si === index
          const isDropTarget = dropOn === si && dragFrom !== null && dragFrom !== si
          return (
            <div
              key={si}
              data-testid={`slide-thumb-${si}`}
              data-slide-index={si}
              data-current={isCurrent ? '1' : '0'}
              draggable
              style={{
                ...styles.thumb,
                ...(isCurrent ? styles.thumbCurrent : {}),
                ...(dragFrom === si ? styles.thumbDragging : {}),
                ...(isDropTarget ? styles.thumbDropTarget : {}),
              }}
              title={`${label} — click to jump, drag to reorder`}
              onClick={() => onJump(si)}
              onDragStart={(e) => {
                dragFromRef.current = si
                setDragFrom(si)
                e.dataTransfer.effectAllowed = 'move'
                // A private marker (some browsers require SOME data to start a drag).
                e.dataTransfer.setData('application/x-spyde-slide', String(si))
              }}
              onDragOver={(e) => {
                if (dragFromRef.current === null) return
                e.preventDefault()
                e.dataTransfer.dropEffect = 'move'
                if (dropOn !== si) setDropOn(si)
              }}
              onDrop={(e) => {
                e.preventDefault()
                const from = dragFromRef.current
                endDrag()
                if (from !== null && from !== si) onMoveSlide(from, si)
              }}
              onDragEnd={endDrag}
            >
              {/* Drop indicator bar (left edge) when this is the drop target. */}
              {isDropTarget && <div style={styles.dropBar} data-testid={`slide-drop-${si}`} />}

              <div style={styles.thumbStage}>
                {/* The scaled-down STATIC preview (baked PNGs, not live iframes). */}
                <SlidePreview cells={cells} />
              </div>

              <div style={styles.thumbBar}>
                <span style={styles.thumbNum} data-testid={`slide-thumb-num-${si}`}>
                  {si + 1}
                </span>
                <span style={styles.thumbLabel} data-testid={`slide-thumb-label-${si}`}>
                  {label}
                </span>
                <div style={{ flex: 1 }} />
                {isTitle && (
                  <span style={styles.badge} title="Title / section slide"
                    data-testid={`slide-thumb-title-${si}`}>T</span>
                )}
                {hasNotes && (
                  <span style={styles.badge} title="Has speaker notes"
                    data-testid={`slide-thumb-notes-${si}`}>📝</span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 9600,
    background: 'rgba(10,10,16,0.96)', color: '#e8e8f0',
    display: 'flex', flexDirection: 'column',
    padding: '18px 24px 24px',
  },
  header: {
    display: 'flex', alignItems: 'baseline', gap: 16,
    paddingBottom: 12, borderBottom: '1px solid #313244', marginBottom: 12,
  },
  title: { fontSize: 20, fontWeight: 700, color: '#cdd6f4' },
  hint: { fontSize: 13, color: '#7f849c' },
  closeBtn: {
    background: 'rgba(30,30,46,0.8)', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 8,
    width: 36, height: 36, fontSize: 16, cursor: 'pointer',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
    gap: 18, overflowY: 'auto', alignContent: 'start',
    paddingRight: 4,
  },
  thumb: {
    position: 'relative',
    display: 'flex', flexDirection: 'column',
    border: '2px solid #313244', borderRadius: 10, overflow: 'hidden',
    background: '#14141f', cursor: 'pointer',
    transition: 'border-color 0.12s, transform 0.12s',
  },
  thumbCurrent: { borderColor: '#89b4fa', boxShadow: '0 0 0 2px rgba(137,180,250,0.35)' },
  thumbDragging: { opacity: 0.4 },
  thumbDropTarget: { borderColor: '#a6e3a1' },
  dropBar: {
    position: 'absolute', top: 0, bottom: 0, left: 0, width: 4,
    background: '#a6e3a1', zIndex: 2, borderRadius: '2px 0 0 2px',
  },
  thumbStage: {
    // A fixed 16:9-ish stage so every thumbnail is the same size; SlidePreview
    // fills it and scales its content down.
    width: '100%', aspectRatio: '16 / 10',
    background: '#0e0e16', overflow: 'hidden',
    pointerEvents: 'none',   // the whole card handles click/drag, not the preview
  },
  thumbBar: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '7px 10px', borderTop: '1px solid #313244',
    background: '#181825',
  },
  thumbNum: {
    fontSize: 13, fontWeight: 700, color: '#89b4fa',
    fontVariantNumeric: 'tabular-nums', minWidth: 18,
  },
  thumbLabel: {
    fontSize: 13, color: '#cdd6f4',
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
    maxWidth: 150,
  },
  badge: {
    fontSize: 12, color: '#a6adc8',
    background: 'rgba(137,180,250,0.14)', borderRadius: 5,
    padding: '1px 6px', fontWeight: 600,
  },
}
