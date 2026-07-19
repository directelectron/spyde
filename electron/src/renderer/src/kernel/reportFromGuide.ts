/**
 * reportFromGuide.ts — seed a starter PRESENTATION (Report) from a guide.
 *
 * Phase 6 (Present mode): a guide (guides/, id + ordered steps with title/body/
 * optional image) becomes a fast starting deck the user then edits in the Report
 * Builder. Each step → ONE slide: a markdown cell whose source is the step title
 * as a heading + the step body, with `slide_break` set on every step after the
 * first (so `ReportDoc.slides()` groups one step per slide). The FIRST slide also
 * carries a `live_action` pointing at the guide (and, when the guide id maps to a
 * curated tutorial dataset, the matching `tutorial`) so Present mode's
 * "Launch live ▶" button can hand off into the real app.
 *
 * Generation is renderer-side (via the existing report actions: `report_new`
 * then `report_add_cell` per slide, then `report_toggle_slide_break` /
 * `report_set_live_action`) — no new backend generation path. Keeping it simple
 * (steps → markdown slides) is deliberate; the user edits from here.
 */
import type { Guide } from '@guides/index'

type SendAction = (action: string, payload?: Record<string, unknown>) => void

// Guide id → curated tutorial dataset name (spyde/backend/tutorial_data.py
// TUTORIAL_LOADERS keys). A guide with no obvious dataset maps to undefined — the
// live_action then only starts the guide tour (no dataset load). Best-effort: if
// Phase 1's `tutorial_load` isn't wired, the Launch-live button degrades to just
// exiting Present mode (see PresentMode.tsx).
const GUIDE_TUTORIAL: Record<string, string | undefined> = {
  'find-vectors': 'find_vectors',
  'virtual-imaging': 'find_vectors',
  orientation: 'orientation',
}

/** The markdown source for one guide step's slide: the title as an H1 heading
 *  followed by the step body (already markdown). */
function slideSource(title: string, body: string): string {
  const heading = title?.trim() ? `# ${title.trim()}\n\n` : ''
  return `${heading}${(body ?? '').trim()}`
}

/**
 * Build a new presentation from a guide by dispatching report actions. Returns
 * nothing — the backend re-emits `report_state`, which drives the sidebar. The
 * caller should ensure the report sidebar is open so the deck is visible.
 *
 * The sequence:
 *   1. report_new (fresh empty doc — replaces any open report),
 *   2. one report_add_cell per step (markdown, appended in order),
 *   3. report_toggle_slide_break {value:true} on every step after the first,
 *   4. report_set_live_action on the first cell (guide + tutorial handoff),
 *   5. report_set_title to the guide title.
 *
 * Because report_add_cell mints its OWN cell id (returned only via report_state,
 * not synchronously), steps 3–4 can't target ids we don't have yet. Instead they
 * ride ON the add: report_add_cell accepts slide_break / live_action fields
 * directly (see the backend handler), so each cell is created already-marked.
 */
export function reportFromGuide(guide: Guide, sendAction: SendAction): void {
  if (!guide || !Array.isArray(guide.steps) || guide.steps.length === 0) return
  sendAction('report_new', {})
  sendAction('report_set_title', { title: guide.title })
  guide.steps.forEach((step, i) => {
    const source = slideSource(step.title, step.body)
    const payload: Record<string, unknown> = {
      cell_type: 'markdown',
      source,
      // A break on every step after the first → one step per slide.
      slide_break: i > 0,
    }
    // The first slide carries the go-live handle for the whole deck.
    if (i === 0) {
      const tutorial = GUIDE_TUTORIAL[guide.id]
      payload.live_action = {
        guide: guide.id,
        ...(tutorial ? { tutorial } : {}),
      }
    }
    sendAction('report_add_cell', payload)
  })
}
