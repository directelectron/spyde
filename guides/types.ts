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

/**
 * How a guide step is REACHED when generating screenshots automatically (the
 * `guide_screenshots.spec.ts` Playwright run walks these). A guide with `drive`
 * blocks is both documentation AND an executable screenplay, so the docs media
 * never drifts from the real UI. Purely optional: steps without `drive` are just
 * screenshotted in the current state. Waits are SIGNAL-based (a backend message,
 * a subwindow count, a non-black canvas) so slow operations are handled without
 * brittle fixed sleeps.
 */
export interface GuideDrive {
  /** What to do to reach this step. Defaults to 'none' (screenshot as-is). */
  action?: 'none' | 'click' | 'hover' | 'backend'
  /** Element to act on for click/hover. Defaults to the step's `anchor`. */
  testid?: string
  /** Backend test-only action name for action:'backend' (e.g. run_test_orientation). */
  backend?: string
  /**
   * Payload for an `action:'backend'` drive. E.g. a data-loading step uses
   * `{action:'backend', backend:'tutorial_load', payload:{name:'find_vectors'}}`.
   * Both the in-app driver (`guideDriver.ts`) and the screenshot spec pass this
   * through to the backend action.
   */
  payload?: Record<string, unknown>
  /** What to wait for AFTER the action, before screenshotting. */
  waitFor?: {
    /** At least N subwindows exist. */
    subwindows?: number
    /** A `data-testid` element is visible. */
    visible?: string
    /** Any canvas shows the colour (markers landed): 'bright' | 'red' | 'green'. */
    pixels?: 'bright' | 'red' | 'green'
  }
  /** Per-step wait budget (ms) for the `waitFor` signal. Default 60000. */
  timeoutMs?: number
  /** Small last-resort settle (ms) after the wait, for paint to flush. Default 0. */
  settleMs?: number
  /**
   * What to screenshot: 'page' (whole window, default) or a `data-testid` whose
   * subwindow is cropped (e.g. the signal window for an overlay close-up).
   */
  shotTarget?: 'page' | string
}

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
   * Optional — the in-app tour spotlights the live UI and ignores this. The
   * `guide_screenshots.spec.ts` run WRITES this file (docs-site media dir).
   */
  image?: string
  /**
   * Optional: how to reach this step when auto-generating screenshots. Omit for
   * a step that is just captured in the current state. See {@link GuideDrive}.
   */
  drive?: GuideDrive
  /**
   * In-app opt-in: when true, the coachmark Tour renders a "Show me ▶" button
   * that runs this step's `drive` live (via `guideDriver.ts`) and advances to
   * the next step once its `waitFor` signal is met. DEFAULT false — a step must
   * be EXPLICITLY marked safe to auto-run. Steps left unmarked stay manual (the
   * spotlight guides the user to do it themselves).
   *
   * CRITICAL: never mark a long/full-scan COMPUTE step `autoDrive` (e.g. the
   * find-vectors "Compute across the whole scan" step) — it can run for minutes
   * and would appear to hang the tour. Leave those manual.
   */
  autoDrive?: boolean
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
  /**
   * In-app opt-in: a drive the coachmark Tour runs automatically ON OPEN, before
   * showing step 1 — typically `{action:'backend', backend:'tutorial_load',
   * payload:{name:'…'}}` to load the walkthrough's small instant tutorial
   * dataset so the tour starts with data already on screen. The Tour shows a
   * "Loading tutorial data…" state while its `waitFor` resolves. Omit to start
   * with whatever is already open. Ignored by the docs website + screenshot spec
   * (they walk each step's own `drive`).
   */
  autoload?: GuideDrive
}
