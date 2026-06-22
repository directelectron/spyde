/**
 * guides/types.ts — the single-source guide format.
 *
 * A Guide is authored ONCE here and rendered in two places:
 *   • in-app   — as an interactive coachmark tour (Tour.tsx) that spotlights the
 *                real UI element named by each step's `anchor` and floats a
 *                callout bubble next to it,
 *   • on the web — as a scrollable walkthrough (the docs site) that shows the
 *                same step text alongside a screenshot.
 *
 * Keeping one source means the in-app tour and the website never drift. The
 * format is intentionally tiny: ordered steps, each anchored to a UI element by
 * its stable `data-testid`, with a short markdown body.
 */

/** Where the callout bubble sits relative to its anchored element. */
export type Placement = 'top' | 'bottom' | 'left' | 'right' | 'center'

export interface GuideStep {
  /**
   * Stable selector for the UI element this step is about. Prefer a
   * `data-testid` value (we resolve `[data-testid="<anchor>"]`); a raw CSS
   * selector starting with `.`/`#`/`[` is also accepted. Use `null` for a
   * step that isn't tied to an element (rendered centered in-app).
   */
  anchor: string | null
  /** Short imperative title, e.g. "Open a diffraction pattern". */
  title: string
  /** Markdown body — a sentence or two. Callout boxes via `> 💡` blockquotes. */
  body: string
  /** Bubble placement in-app (default 'bottom'). */
  placement?: Placement
  /**
   * Screenshot for the WEB rendering (path relative to the guide's media dir).
   * Optional — the in-app tour spotlights the live UI and ignores this.
   */
  image?: string
}

export interface Guide {
  /** URL/registry slug, e.g. "find-vectors". */
  id: string
  /** Display title, e.g. "Finding Diffraction Vectors". */
  title: string
  /** One-line summary for the guide index / web card. */
  summary: string
  /** Ordered steps. */
  steps: GuideStep[]
}
