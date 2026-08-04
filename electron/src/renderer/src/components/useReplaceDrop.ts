/**
 * useReplaceDrop.ts — "drop something here to REPLACE what's already there".
 *
 * A picture in a report can come from two places, and once one is in the cell
 * there was no way to swap it: an image cell had only its reorder wiring, and a
 * split cell's drop zone existed solely while the figure side was EMPTY. So
 * changing a picture meant deleting the cell and re-adding it — which loses the
 * caption, the display width, and (on a slide's first cell) the slide break and
 * speaker notes.
 *
 * Both sources land here:
 *   • an image FILE from the OS  → report_set_cell_image (bytes swapped in place)
 *   • a figure/window PILL       → report_add_figure {at_cell} (a LIVE figure)
 *
 * Both verbs target the EXISTING cell id, so the cell keeps its identity and
 * everything hanging off it. The two transports are genuinely different
 * (`Files` vs `application/x-spyde-*`), so a target that wants both has to test
 * for both — see imageDrop.ts.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { FIGURE_DRAG_MIME, WINDOW_DRAG_MIME, peekWindowDrag } from '../kernel/dnd'
import { hasImageFiles, imageExtOf, imageFilesFrom, readFileAsDataURL } from './imageDrop'

const PILL_MIMES = [FIGURE_DRAG_MIME, WINDOW_DRAG_MIME]
const isPillDrag = (dt: DataTransfer) => PILL_MIMES.some(m => dt.types.includes(m))

/** The source window (+ optional view / figure id) behind a pill drop. */
function pillPayload(dt: DataTransfer): {
  windowId: number; view?: string; figId?: string
} | null {
  const raw = dt.getData(FIGURE_DRAG_MIME)
  if (raw) {
    try {
      const parsed = JSON.parse(raw)
      if (typeof parsed?.windowId === 'number') return parsed
    } catch { /* fall through to the window mime / stash */ }
  }
  const win = dt.getData(WINDOW_DRAG_MIME)
  if (win) {
    const n = parseInt(win, 10)
    if (Number.isFinite(n)) return { windowId: n }
  }
  // The drag stash — set at dragstart, read when the DataTransfer is empty
  // (some platforms withhold getData outside the drop handler).
  return peekWindowDrag()
}

export interface ReplaceDrop {
  active: boolean
  handlers: {
    onDragOver: (e: React.DragEvent) => void
    onDragLeave: (e: React.DragEvent) => void
    onDrop: (e: React.DragEvent) => void
  }
}

/**
 * Drop handlers that REPLACE the content of `cellId` in place.
 *
 * `active` is true while a droppable drag is over the target, for the caller's
 * highlight. Handlers stopPropagation so the drop never also reaches the
 * sidebar body, which would append a NEW cell underneath — the exact bug this
 * exists to avoid.
 */
export function useReplaceDrop(cellId: string): ReplaceDrop {
  const { sendAction } = useSpyDE()
  const [active, setActive] = React.useState(false)

  const onDragOver = (e: React.DragEvent) => {
    if (!isPillDrag(e.dataTransfer) && !hasImageFiles(e.dataTransfer)) return
    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'copy'
    setActive(true)
  }

  const onDragLeave = (e: React.DragEvent) => {
    // Only clear when the pointer actually leaves the box, not when it crosses
    // a child (the caption, the resize grip, the overlay itself).
    if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) {
      setActive(false)
    }
  }

  const onDrop = (e: React.DragEvent) => {
    // FILE first, so it can never be mistaken for a pill.
    if (hasImageFiles(e.dataTransfer)) {
      e.preventDefault()
      e.stopPropagation()
      setActive(false)
      const file = imageFilesFrom(e.dataTransfer)[0]
      if (!file) return
      void (async () => {
        try {
          const dataUrl = await readFileAsDataURL(file)
          if (!dataUrl) return
          sendAction('report_set_cell_image', {
            cell_id: cellId, image_b64: dataUrl, image_ext: imageExtOf(file),
          })
        } catch { /* unreadable file — leave the picture as it was */ }
      })()
      return
    }
    if (!isPillDrag(e.dataTransfer)) return
    e.preventDefault()
    e.stopPropagation()
    setActive(false)
    const src = pillPayload(e.dataTransfer)
    if (src == null) return
    // at_cell targets THIS cell, so the backend converts it in place rather
    // than appending a figure below the picture it was meant to replace.
    sendAction('report_add_figure', {
      source_window_id: src.windowId, at_cell: cellId,
      ...(src.view !== undefined ? { view: src.view } : {}),
      ...(src.figId !== undefined ? { fig_id: src.figId } : {}),
    })
  }

  return { active, handlers: { onDragOver, onDragLeave, onDrop } }
}
