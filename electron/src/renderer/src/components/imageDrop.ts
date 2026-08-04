/**
 * imageDrop.ts — the shared "an image FILE is being dragged in from the OS"
 * helpers, used by every report drop target.
 *
 * These lived privately in ReportSidebar, which is why only the SIDEBAR BODY
 * ever recognised a dropped PNG. The split cell's figure side and the empty
 * figure placeholder both gate their drop handlers on the figure/window PILL
 * mimes alone, so a dragged file failed their test, bubbled to the body, and was
 * appended as a NEW image cell BELOW the split instead of filling the slot the
 * user aimed at. Sharing them is what lets a drop zone accept both kinds.
 *
 * A pill drag and a file drag are genuinely different transports — a pill
 * carries `application/x-spyde-*` data, a file carries `Files` — so a drop
 * target that wants both has to test for both.
 */

/** The image file extensions a PHOTO cell may carry (mirrors the backend's
 *  IMAGE_EXTS). Anything else is normalised to png, as the backend does. */
export const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'gif', 'webp'] as const

/** Map an image file's MIME / name to one of {@link IMAGE_EXTS}. */
export function imageExtOf(file: File): string {
  const fromType = (file.type.split('/')[1] || '').toLowerCase()
  if ((IMAGE_EXTS as readonly string[]).includes(fromType)) return fromType
  const fromName = (file.name.split('.').pop() || '').toLowerCase()
  if ((IMAGE_EXTS as readonly string[]).includes(fromName)) return fromName
  return 'png'
}

/** Read a File/Blob as a data URL. Rejects on read error. */
export function readFileAsDataURL(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader()
    fr.onload = () => resolve(String(fr.result || ''))
    fr.onerror = () => reject(fr.error)
    fr.readAsDataURL(file)
  })
}

/**
 * True when a DataTransfer carries at least one image FILE (the drop-a-photo
 * path, distinct from a figure/window pill drop).
 *
 * The `items` fallback is load-bearing: during `dragover` the browser does not
 * expose `files` yet (only on `drop`), so a target that tested `files` alone
 * would never call preventDefault and would therefore never RECEIVE the drop.
 */
export function hasImageFiles(dt: DataTransfer): boolean {
  if (dt.files && dt.files.length) {
    for (const f of Array.from(dt.files)) {
      if (f.type.startsWith('image/')) return true
    }
  }
  if (dt.items && dt.items.length) {
    for (const it of Array.from(dt.items)) {
      if (it.kind === 'file' && it.type.startsWith('image/')) return true
    }
  }
  return false
}

/** The image files on a drop, in order (empty when there are none). */
export function imageFilesFrom(dt: DataTransfer): File[] {
  if (!dt.files || !dt.files.length) return []
  return Array.from(dt.files).filter(f => f.type.startsWith('image/'))
}
