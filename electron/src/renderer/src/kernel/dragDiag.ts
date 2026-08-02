/**
 * dragDiag.ts — a always-on trace of the window-pill → report-cell drag, so a
 * drop that "does nothing" reports WHICH STAGE it died at instead of being
 * guessed about.
 *
 * The compose drag crosses six places, and a failure at any one of them looks
 * identical from the outside (nothing happens):
 *
 *   1. dragstart on the Pill            — did the drag start at all? MIMEs stamped?
 *   2. the global classifier            — did dragKind become 'window'?
 *   3. the shield mounting              — is there a drop target over the iframe?
 *   4. dragenter/dragover on the shield — is the target receiving events? which zone?
 *   5. drop on the shield               — did it fire? was the payload readable?
 *   6. the action send                  — was repfig_compose actually dispatched?
 *
 * A synthesized Playwright drag exercises 1–6 in one process and passes; a real
 * OS drag can fail at 2, 4 or 5 and looks the same. Hence: record, don't reason.
 *
 * Cost is a push onto a bounded array per event — negligible next to the drag
 * itself, so this stays enabled rather than hiding behind a flag nobody knows to
 * set when they hit the bug.
 *
 * READING IT: do the drag, then in DevTools (Cmd+Alt+I) run
 *     __spydeDragDump()
 * or copy the `[drag-diag]` block auto-printed on every drop / dragend.
 */

const MAX = 400

export interface DragDiagEntry {
  t: number
  stage: string
  detail: Record<string, unknown>
}

let entries: DragDiagEntry[] = []
let t0 = 0

/** Record one stage of the drag. */
export function dlog(stage: string, detail: Record<string, unknown> = {}): void {
  const now = typeof performance !== 'undefined' ? performance.now() : 0
  if (stage.startsWith('1.') || t0 === 0) { t0 = now; entries = []; onceKeys.clear() }
  entries.push({ t: Math.round(now - t0), stage, detail })
  if (entries.length > MAX) entries.shift()
}

/**
 * Record a stage only the FIRST time it occurs in this drag. dragover fires at
 * ~60 Hz; without this the trace is 300 identical lines and the one interesting
 * event scrolls off the end of the ring buffer.
 */
const onceKeys = new Set<string>()
export function dlogOnce(stage: string, detail: Record<string, unknown> = {}): void {
  const key = `${stage}|${JSON.stringify(detail)}`
  if (onceKeys.has(key)) return
  onceKeys.add(key)
  dlog(stage, detail)
}

/** Human-readable dump of the last drag. */
export function dragDump(): string {
  if (!entries.length) return '[drag-diag] no drag recorded yet'
  const lines = entries.map(e => {
    const d = Object.entries(e.detail)
      .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
      .join(' ')
    return `  +${String(e.t).padStart(5)}ms  ${e.stage.padEnd(26)} ${d}`
  })
  return `[drag-diag] ${entries.length} events\n${lines.join('\n')}`
}

/** The raw entries (tests assert on these). */
export function dragEntries(): DragDiagEntry[] {
  return entries.slice()
}

/** Did a given stage occur in the last drag? */
export function sawStage(stage: string): boolean {
  return entries.some(e => e.stage === stage)
}

/**
 * Auto-printing is OPT-IN; recording is always on.
 *
 * Keeping the ring buffer live costs a few array pushes per drag and means the
 * next time a drag misbehaves the evidence is ALREADY there — `__spydeDragDump()`
 * answers it on the spot instead of needing a rebuild, a flag, and a repro. But
 * printing a block on every drag turns the console into noise, so that half is
 * behind a switch:
 *
 *     __spydeDragDebug(true)      // persists; survives reload
 */
const DEBUG_KEY = 'spyde.dragDebug'
function autoPrintEnabled(): boolean {
  try { return window.localStorage.getItem(DEBUG_KEY) === '1' } catch { return false }
}

/** Print the trace — called at the end of a drag (drop / dragend). */
export function dragDumpToConsole(reason: string): void {
  if (!autoPrintEnabled()) return
  // console.info so it survives a filtered console but isn't styled as an error.
  // eslint-disable-next-line no-console
  console.info(`[drag-diag] === end of drag (${reason}) ===\n${dragDump()}`)
}

if (typeof window !== 'undefined') {
  const w = window as unknown as Record<string, unknown>
  w.__spydeDragDump = () => {
    // eslint-disable-next-line no-console
    console.info(dragDump())
    return dragDump()
  }
  w.__spydeDragEntries = dragEntries
  w.__spydeDragDebug = (on = true) => {
    try { window.localStorage.setItem(DEBUG_KEY, on ? '1' : '0') } catch { /* ignore */ }
    // eslint-disable-next-line no-console
    console.info(`[drag-diag] auto-print ${on ? 'ON' : 'OFF'}`)
    return on
  }
}
