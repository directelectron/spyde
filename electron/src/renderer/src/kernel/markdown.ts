/**
 * markdown.ts — the single markdown render pipeline for the Report Builder.
 *
 * `marked` (configured synchronously — no async extensions) → `DOMPurify.sanitize`.
 * Used for BOTH the in-app rendered view (ReportCell) and the sanitized `html`
 * fragment shipped on every markdown commit (`report_add_cell`/`report_update_cell`)
 * so static HTML export embeds real rendered HTML rather than an ugly `<pre>`
 * fallback. Factoring it here keeps the display render and the export-cache render
 * byte-identical.
 */
import { marked } from 'marked'
import DOMPurify from 'dompurify'

/** Render markdown source → sanitized HTML fragment (safe for the DOM AND for
 *  the exported page). */
export function renderMarkdown(src: string): string {
  const html = marked.parse(src ?? '', { async: false }) as string
  return DOMPurify.sanitize(html)
}
