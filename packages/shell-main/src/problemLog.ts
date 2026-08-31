/**
 * problemLog.ts — a bounded in-memory record of things that went wrong.
 *
 * Its own module, importing nothing, so that anything can write to it: the
 * updater records its own failures here, and errorReport.ts reads the ring
 * while also reading the updater's state. Merging the two would make that a
 * cycle.
 *
 * The ring is memory only and never leaves the machine on its own — it exists
 * so that a report the user writes minutes after a failure still knows what
 * failed. See errorReport.ts for what happens to it.
 */

/** One thing that went wrong, as recorded when it happened. */
export interface Problem {
  /** When, as an ISO timestamp. */
  at: string
  /** Coarse source, e.g. 'main', 'backend', 'updater'. */
  kind: string
  /** The message/stack, truncated. */
  detail: string
}

/** Keep it small: it is read in full into every report. */
const MAX_PROBLEMS = 25
const MAX_DETAIL_CHARS = 4000

const problems: Problem[] = []

/**
 * Note that something went wrong.
 *
 * Deliberately cheap and total: it never throws and never sends, so it is safe
 * to call from a crash handler, a stderr pump, or an updater error path.
 */
export function recordProblem(kind: string, detail: unknown): void {
  try {
    const text = detail instanceof Error
      ? (detail.stack ?? detail.message)
      : String(detail)
    problems.push({
      at: new Date().toISOString(),
      kind,
      detail: text.slice(0, MAX_DETAIL_CHARS),
    })
    if (problems.length > MAX_PROBLEMS) problems.splice(0, problems.length - MAX_PROBLEMS)
  } catch { /* a reporting failure must never become the failure */ }
}

/** The problems recorded so far, oldest first. */
export function recordedProblems(): Problem[] {
  return [...problems]
}
