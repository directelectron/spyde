/**
 * reportClipboard.ts — the Report Builder's INTERNAL cell clipboard.
 *
 * A tiny module-scope store holding at most one serialized cell (the shape the
 * backend's `report_paste_cell` consumes). Copy writes it; the sidebar's Paste
 * button reads it and dispatches `report_paste_cell {cell}`. A subscribe hook
 * lets the Paste button enable only while the clipboard holds a cell.
 *
 * This is deliberately app-internal (not the OS clipboard): it carries the full
 * FigureSpec recipe + baked PNG so a paste rebuilds a LIVE figure when the source
 * still resolves. (A figure Copy ALSO best-effort mirrors the PNG to the OS
 * clipboard via `clipboardWritePng` so pasting into Word/Slack works.)
 */

/** A serialized markdown cell (source + its rendered html fragment). */
export interface SerializedMarkdownCell {
  cell_type: 'markdown'
  source: string
  html: string
}

/** A serialized figure cell (caption + the pixel-free FigureSpec + a baked PNG
 *  data URL for the offline fallback). */
export interface SerializedFigureCell {
  cell_type: 'figure'
  caption: string
  figure: unknown
  png?: string | null
}

/** A serialized image (photo) cell — the caption + the raw image inlined as a
 *  data URL (the backend re-holds the bytes on paste). */
export interface SerializedImageCell {
  cell_type: 'image'
  caption: string
  image_ext: string
  image: string
}

export type SerializedCell =
  | SerializedMarkdownCell | SerializedFigureCell | SerializedImageCell

let current: SerializedCell | null = null
const listeners = new Set<() => void>()

export const reportClipboard = {
  /** The currently held cell, or null. */
  get(): SerializedCell | null {
    return current
  },
  /** Replace the held cell and notify subscribers (enables the Paste button). */
  set(cell: SerializedCell): void {
    current = cell
    listeners.forEach((l) => l())
  },
  /** Subscribe to changes; returns an unsubscribe. Used by the Paste button's
   *  `useSyncExternalStore` to reactively enable/disable. */
  subscribe(listener: () => void): () => void {
    listeners.add(listener)
    return () => {
      listeners.delete(listener)
    }
  },
}
