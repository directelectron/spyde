/**
 * guideDriver.ts — the RENDERER-SIDE guide "drive" engine.
 *
 * A `GuideDrive` (guides/types.ts) is a tiny screenplay for reaching a guide
 * step: do a backend action / click / hover, then wait on a SIGNAL (a subwindow
 * count, an element becoming visible, marker pixels of a colour appearing). The
 * `guide_screenshots.spec.ts` Playwright run executes these against `page.*` to
 * capture docs screenshots. THIS module executes the SAME semantics live in the
 * running app (DOM events + `sendAction`) so the in-app coachmark Tour can
 * auto-load its tutorial dataset and offer a "Show me ▶" button per step.
 *
 * Intentional duplication: the spec runs in the Playwright/`page` world and this
 * runs in-DOM; a full code share across those two worlds is not worth the
 * indirection. What IS shared is the `GuideDrive` TYPE (guides/types.ts) and the
 * pixel-colour thresholds (COLOR_THRESHOLDS below), kept identical to the
 * harness's `countColorPixels` so "red markers appeared" means the same thing in
 * both. If you change the harness thresholds, change these to match.
 */
import type { GuideDrive, GuideStep } from '@guides/index'

/** How `sendAction` is threaded in from the SpyDE context. */
export interface DriveEnv {
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}

/**
 * Per-colour RGB predicates, byte-for-byte the same as the harness's
 * `countColorPixels` (electron/tests/_harness.cjs). Keep in sync.
 *  - bright: any non-black pixel (something painted).
 *  - red:    #ff3030 find-vectors markers.
 *  - green:  #30ff60 matched-template overlay (blue band rejects the navigator's
 *            pure-green crosshair, whose blue≈0).
 */
const COLOR_THRESHOLDS: Record<'bright' | 'red' | 'green', (r: number, g: number, b: number) => boolean> = {
  bright: (r, g, b) => r > 30 || g > 30 || b > 30,
  red: (r, g, b) => r > 120 && g < 90 && b < 90,
  green: (r, g, b) => g > 150 && r < 130 && b > 50 && b < 170,
}

const testidSel = (tid: string) => `[data-testid="${tid}"]`

/** The signal (non-navigator) subwindow, chosen by its S- breadcrumb — mirrors
 *  the spec's `page.getByTestId('subwindow').filter({ has: … /^S-/ })`. */
function signalSubwindow(): HTMLElement | null {
  const wins = Array.from(document.querySelectorAll<HTMLElement>('[data-testid="subwindow"]'))
  for (const w of wins) {
    const bc = w.querySelector('[data-testid="window-breadcrumb"]')
    if (bc && /^S-/.test((bc.textContent || '').trim())) return w
  }
  return null
}

/** Is an element in the DOM AND actually laid-out/visible (non-zero box)? */
function isVisible(el: HTMLElement | null): boolean {
  if (!el) return false
  const r = el.getBoundingClientRect()
  if (r.width === 0 && r.height === 0) return false
  const cs = window.getComputedStyle(el)
  return cs.visibility !== 'hidden' && cs.display !== 'none' && cs.opacity !== '0'
}

/** Fire a real bubbling mouse event so React's synthetic handlers run — the
 *  in-DOM equivalent of Playwright's `.click()` / `.hover()`. */
function fireMouse(el: Element, type: string): void {
  el.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, view: window }))
}

/** Reveal the floating toolbar the way the spec does: the toolbar action / sub-
 *  action buttons are opacity:0 until the signal window is hovered, so hover its
 *  titlebar (which bubbles to the subwindow's onMouseEnter → showToolbar) first. */
function revealToolbar(): void {
  const sig = signalSubwindow()
  if (!sig) return
  const bar = sig.querySelector('[data-testid="subwindow-titlebar"]') || sig
  // mouseover bubbles (mouseenter does not) — React binds onMouseEnter to the
  // enter/over pair; the bubbling mouseover reaches the subwindow's handler.
  fireMouse(bar, 'mouseover')
  fireMouse(bar, 'mouseenter')
}

/**
 * Resolve the element a click/hover drive targets. Scoped to the SIGNAL window
 * for per-window controls (toolbar/titlebar/action buttons), falling back to a
 * document-wide lookup — mirrors the spec's `scope = sig || page`.
 */
function resolveTarget(tid: string): HTMLElement | null {
  const sel = testidSel(tid)
  const sig = signalSubwindow()
  if (sig) {
    const scoped = sig.querySelector<HTMLElement>(sel)
    if (scoped) return scoped
  }
  return document.querySelector<HTMLElement>(sel)
}

/** Count canvas pixels of `kind` across every same-origin frame's canvases —
 *  the in-DOM port of the harness's `countColorPixels`. Cross-origin figure
 *  iframes throw on getImageData and are skipped (best-effort, like the spec). */
function countColorPixels(kind: 'bright' | 'red' | 'green'): number {
  const pred = COLOR_THRESHOLDS[kind]
  const docs: Document[] = [document]
  for (const f of Array.from(document.querySelectorAll('iframe'))) {
    try {
      const d = (f as HTMLIFrameElement).contentDocument
      if (d) docs.push(d)
    } catch { /* cross-origin — skip */ }
  }
  let n = 0
  for (const doc of docs) {
    for (const c of Array.from(doc.querySelectorAll('canvas'))) {
      const ctx = c.getContext('2d', { willReadFrequently: true })
      if (!ctx || !c.width || !c.height) continue
      let data: Uint8ClampedArray
      try {
        data = ctx.getImageData(0, 0, c.width, c.height).data
      } catch { continue }
      for (let p = 0; p < data.length; p += 4) {
        if (pred(data[p], data[p + 1], data[p + 2])) n++
      }
    }
  }
  return n
}

/**
 * Poll `check` on animation frames (falling back to a timer) until it returns a
 * truthy value or the deadline passes. Resolves true on success, false on
 * timeout — the caller decides what a timeout means.
 */
function pollUntil(check: () => boolean, timeoutMs: number): Promise<boolean> {
  return new Promise((resolve) => {
    const deadline = performance.now() + timeoutMs
    let raf = 0
    let timer = 0
    const done = (ok: boolean) => {
      if (raf) cancelAnimationFrame(raf)
      if (timer) window.clearTimeout(timer)
      resolve(ok)
    }
    const tick = () => {
      let ok = false
      try { ok = check() } catch { ok = false }
      if (ok) return done(true)
      if (performance.now() >= deadline) return done(false)
      raf = requestAnimationFrame(tick)
    }
    // Safety net in case rAF is throttled (backgrounded window): also poll on a
    // coarse timer so a wait can't stall past its budget.
    const poll = () => {
      let ok = false
      try { ok = check() } catch { ok = false }
      if (ok) return done(true)
      if (performance.now() >= deadline) return done(false)
      timer = window.setTimeout(poll, 150)
    }
    tick()
    poll()
  })
}

/** Await a `GuideDrive.waitFor` signal. Resolves when met; rejects on timeout. */
async function awaitWaitFor(w: NonNullable<GuideDrive['waitFor']>, timeoutMs: number): Promise<void> {
  const conditions: Array<() => boolean> = []
  if (typeof w.subwindows === 'number') {
    const need = w.subwindows
    conditions.push(() => document.querySelectorAll('[data-testid="subwindow"]').length >= need)
  }
  if (w.visible) {
    const tid = w.visible
    conditions.push(() => isVisible(document.querySelector<HTMLElement>(testidSel(tid))))
  }
  if (w.pixels) {
    const kind = w.pixels
    conditions.push(() => countColorPixels(kind) > 0)
  }
  if (conditions.length === 0) return
  const ok = await pollUntil(() => conditions.every((c) => c()), timeoutMs)
  if (!ok) {
    const labels = [
      typeof w.subwindows === 'number' ? `${w.subwindows} subwindows` : null,
      w.visible ? `visible:${w.visible}` : null,
      w.pixels ? `pixels:${w.pixels}` : null,
    ].filter(Boolean).join(', ')
    throw new Error(`guide drive waitFor timed out (${labels})`)
  }
}

const sleep = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms))

/**
 * Execute a single `GuideDrive` in the LIVE app.
 *
 * - action 'backend'  → `sendAction(drive.backend, drive.payload ?? {})`
 * - action 'click'    → resolve `[data-testid]` (reveal the toolbar first for
 *                       action-btn-/subaction- targets), then dispatch a real
 *                       bubbling `click`.
 * - action 'hover'    → same resolution, dispatch `mouseover` + `mouseenter`.
 * Then awaits `drive.waitFor` (subwindows / visible / pixels) up to
 * `drive.timeoutMs` (default 60 s), then a `drive.settleMs` paint settle.
 *
 * Rejects on a missing target element or a `waitFor` timeout — the caller (the
 * Tour) catches and shows a "couldn't run automatically" note WITHOUT wedging.
 */
export async function runDrive(drive: GuideDrive, step: GuideStep, env: DriveEnv): Promise<void> {
  const action = drive.action ?? 'none'
  const timeout = drive.timeoutMs ?? 60_000

  if (action === 'backend') {
    if (!drive.backend) throw new Error('guide drive action:backend has no `backend` name')
    env.sendAction(drive.backend, drive.payload ?? {})
  } else if (action === 'click' || action === 'hover') {
    const tid = drive.testid || step.anchor
    if (!tid) throw new Error('guide drive click/hover has no testid or anchor')
    // Toolbar action buttons + sub-actions are hidden until the window is
    // hovered — reveal the floating toolbar first so the target exists/lands.
    if (/^(action-btn-|subaction-)/.test(tid)) revealToolbar()
    // The toolbar reveal + React re-render is async; wait briefly for the
    // target to actually appear before giving up.
    let el = resolveTarget(tid)
    if (!el) {
      await pollUntil(() => (el = resolveTarget(tid)) != null, 2_000)
    }
    if (!el) throw new Error(`guide drive target not found: [data-testid="${tid}"]`)
    if (action === 'hover') {
      fireMouse(el, 'mouseover')
      fireMouse(el, 'mouseenter')
    } else {
      // Match a real click: pointer/mouse down+up then click, all bubbling.
      fireMouse(el, 'mousedown')
      fireMouse(el, 'mouseup')
      fireMouse(el, 'click')
    }
  }

  if (drive.waitFor) await awaitWaitFor(drive.waitFor, timeout)
  if (drive.settleMs) await sleep(drive.settleMs)
}

// Exported for tests / reuse — the pixel semantics are shared with the harness.
export { countColorPixels, COLOR_THRESHOLDS }
