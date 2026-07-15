/**
 * mdi_overlay.spec.ts — MDI live image LAYERING end-to-end (Report Builder
 * Phase 2, Part 2 — spyde/actions/overlay.py).
 *
 * Real Dask + bundled-synthetic si_grains. si_grains is loaded TWICE → two
 * independent trees, each with a navigator + signal window; both signal windows
 * have IDENTICAL 128×128 diffraction-pattern shapes (a same-tree nav↔DP pair
 * would DIFFER in shape and the backend refuses the overlay — see part f). We
 * drop signal-A's pill onto signal-B (the TARGET) → overlay_add layers A's image
 * over B's, defaulting to a non-gray cmap (magma) at alpha 0.5.
 *
 * Verified in the real app, screenshots each stage (a blank/black target frame is
 * a failure, not a pass):
 *   a. drop signal-A → signal-B → confirm popover → accept → overlay_add succeeds
 *   b. dock Layers section shows one layer-row for the target; the target figure's
 *      COLORED pixels rise (the magma layer paints where the base was grayscale)
 *   c. alpha→0 drops the colored-pixel count back near the pre-overlay level
 *   d. drive the TARGET's navigator crosshair (test_nav_drag on signal_trees[-1]
 *      = tree B, the target's tree) → the layered figure still updates, no errors
 *   e. remove the layer via the dock × → Layers section disappears, figure gray
 *   f. an INCOMPATIBLE drop (a navigator onto a signal — differing shapes) surfaces
 *      the backend's status refusal WITHOUT a crash
 *
 * Backend emit/emit_error never reach Playwright stdout (PLOTAPP line protocol);
 * SPYDE_LOG_LEVEL=WARNING tees logging to stderr → ctx.backend.logBuffer, scanned
 * for the refusal string in (f) and for tracebacks in the final audit.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_phase2_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

test.beforeAll(async () => {
  // INFO (not WARNING) so the test_nav_drag "[REDRAW] … moves changed" verdict
  // (logged at INFO) reaches the harness's stderr buffer in part (d). The
  // backendErrorLines filter still only matches ERROR/Traceback, so INFO chatter
  // doesn't pollute the traceback audit.
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  // Load si_grains TWICE → two trees, two 128×128 signal windows (same shape).
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2000)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 4, 120_000)
  await page.waitForTimeout(3000)   // let both DPs paint
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

// The two signal windows (S- breadcrumb prefix), in creation order.
function sigWindows(page: any) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
}
function navWindows(page: any) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
}

// The numeric windowId a subwindow's iframe belongs to (figure-<figId> → the
// window is addressed by windowId in overlay_add; but the shield/dock use the
// windowId from the state). We resolve it from the subwindow's data-testid path
// via the overlay-drop-shield / layers testids which embed the window id. Easier:
// read the window's active-select via clicking it, then activeWindowId. Instead we
// stamp a marker attr on the pill and resolve windowId through the DOM: the
// subwindow root carries the id in nested testids. Use the reliable path: click
// the window to focus it, then read the plot-control-dock header + layers section
// keyed by activeWindowId. For the drag we only need the pill element + the
// target content box, which we tag directly.

/**
 * Native HTML5 window-pill drag onto a specific TARGET window's overlay shield,
 * split so the shield mounts. Fires dragstart + a dragover over the MDI area to
 * promote dragKind='window' (which mounts one overlay-drop-shield over EVERY
 * window's out-of-process iframe), yields to React, then dragover+drop onto the
 * shield INSIDE the target window (there is one shield per window — dropping on
 * the wrong one layers onto the wrong window). The shared DataTransfer is stashed
 * on window across the two evaluates. ``srcSel`` marks the drag source pill;
 * ``targetWinSel`` marks the target subwindow root.
 */
async function dragPillToShield(page: any, srcSel: string, targetWinSel: string) {
  await page.evaluate(({ srcSel }: any) => {
    const src = document.querySelector(srcSel) as HTMLElement
    const mdi = document.querySelector('[data-testid="mdi-area"]') as HTMLElement
    if (!src) throw new Error('drag src not found: ' + srcSel)
    const dt = new DataTransfer()
    ;(window as any).__mdidt = dt
    const fire = (target: HTMLElement, type: string) => {
      const r = target.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
    }
    fire(src, 'dragstart')
    fire(mdi, 'dragover')   // promote dragKind='window' → shields mount
  }, { srcSel })
  await page.waitForTimeout(300)  // let the shields mount

  await page.evaluate(({ srcSel, targetWinSel }: any) => {
    const dt = (window as any).__mdidt as DataTransfer
    const win = document.querySelector(targetWinSel) as HTMLElement
    // The overlay shield INSIDE the target window (one per window).
    const shield = win?.querySelector('[data-testid^="overlay-drop-shield-"]') as HTMLElement
    const src = document.querySelector(srcSel) as HTMLElement
    if (!shield) throw new Error('overlay shield not mounted inside ' + targetWinSel)
    const fire = (target: HTMLElement, type: string) => {
      const r = target.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
    }
    fire(shield, 'dragenter'); fire(shield, 'dragover'); fire(shield, 'drop')
    if (src) fire(src, 'dragend')
  }, { srcSel, targetWinSel })
}

// Count COLORED (non-gray) canvas pixels inside a specific window's iframe. A
// grayscale base image has r≈g≈b; a magma/cividis overlay tints pixels so the
// channels diverge. We match the target window's iframe by resolving its figId.
async function coloredPixelsInWindow(page: any, targetWinTestId: string): Promise<number> {
  const src: string | null = await page.evaluate((sel: string) => {
    const win = document.querySelector(sel)
    if (!win) return null
    // The visible figure iframe (display:block) in this window's figure box.
    const ifrs = Array.from(win.querySelectorAll('iframe[data-testid^="figure-"]')) as HTMLIFrameElement[]
    const vis = ifrs.find(f => getComputedStyle(f).display !== 'none') ?? ifrs[0]
    return vis?.src || null
  }, targetWinTestId)
  if (!src) return -1
  const frame = page.frames().find((f: any) => f.url() === src)
  if (!frame) return -1
  try {
    return await frame.evaluate(() => {
      let n = 0
      for (const c of Array.from(document.querySelectorAll('canvas'))) {
        const cv = c as HTMLCanvasElement
        const cctx = cv.getContext('2d')
        if (!cctx || !cv.width || !cv.height) continue
        const d = cctx.getImageData(0, 0, cv.width, cv.height).data
        for (let p = 0; p < d.length; p += 4) {
          const r = d[p], g = d[p + 1], b = d[p + 2]
          // Non-black and channels diverge → a colored (non-gray) pixel.
          if ((r > 24 || g > 24 || b > 24) &&
              (Math.max(r, g, b) - Math.min(r, g, b) > 28)) n++
        }
      }
      return n
    })
  } catch { return -1 }
}

// Give the two signal windows a stable test attr on their root + content shield
// path. Returns the DOM testids for [targetWin, targetShield].
async function tagWindows(page: any) {
  const sigs = sigWindows(page)
  await expect(sigs).toHaveCount(2, { timeout: 30_000 })
  // Tag the SECOND signal window (tree B) as the TARGET, the FIRST as SOURCE.
  await sigs.nth(0).evaluate((el: HTMLElement) => el.setAttribute('data-mdi-win', 'source'))
  await sigs.nth(1).evaluate((el: HTMLElement) => el.setAttribute('data-mdi-win', 'target'))
  // Tag the source window's breadcrumb pill as the drag source.
  await sigs.nth(0).getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-mdi-src', '1'))
}

test('a) drop signal-A onto signal-B → confirm → overlay_add succeeds', async () => {
  const { page } = ctx
  await tagWindows(page)

  // Focus the target window so the Plot-Control dock tracks it (needed for the
  // Layers section in step b). Click its titlebar strip (not the pill).
  const target = page.locator('[data-mdi-win="target"]')
  await target.getByTestId('subwindow-titlebar').click({ position: { x: 40, y: 8 } })
  await page.waitForTimeout(300)

  // Baseline colored-pixel count of the TARGET before any overlay (grayscale DP →
  // ~0 colored pixels).
  const before = await coloredPixelsInWindow(page, '[data-mdi-win="target"]')
  console.log('[mdi] target colored pixels BEFORE overlay =', before)
  ;(ctx as any).__beforeColored = before

  await page.screenshot({ path: join(SHOTS, '10-two-signals.png') })

  // Drag source pill → the TARGET window's overlay shield.
  await dragPillToShield(page, '[data-mdi-src="1"]', '[data-mdi-win="target"]')

  // The confirm popover appears on the target window.
  const confirm = page.locator('[data-testid^="overlay-confirm-"]')
  await expect(confirm.first()).toBeVisible({ timeout: 10_000 })
  await page.screenshot({ path: join(SHOTS, '11-overlay-confirm.png') })
  // Accept.
  await page.locator('[data-testid^="overlay-confirm-ok-"]').first().click()

  // overlay_add lands → the dock Layers section appears for the target window.
  await expect(page.getByTestId('layers-section')).toBeVisible({ timeout: 15_000 })
  ctx.assertNoJsErrors()
})

test('b) dock reflects one layer-row + target figure gains colored pixels', async () => {
  const { page } = ctx
  const section = page.getByTestId('layers-section')
  await expect(section).toBeVisible()
  const rows = section.locator('[data-testid^="layer-row-"]')
  await expect(rows).toHaveCount(1, { timeout: 10_000 })
  // Record the layer id for later steps (alpha / remove).
  const layerId = await rows.first().evaluate((el) =>
    (el.getAttribute('data-testid') || '').replace('layer-row-', ''))
  ;(ctx as any).__layerId = layerId
  console.log('[mdi] layer id =', layerId)

  // The magma layer (alpha 0.5) tints the grayscale base → colored pixels appear.
  const before = (ctx as any).__beforeColored as number
  await expect.poll(async () =>
    coloredPixelsInWindow(page, '[data-mdi-win="target"]'), {
    timeout: 20_000,
    message: 'target figure gained no colored pixels after overlay (layer not composited)',
  }).toBeGreaterThan(Math.max(500, before + 500))
  const after = await coloredPixelsInWindow(page, '[data-mdi-win="target"]')
  console.log('[mdi] target colored pixels AFTER overlay =', after)

  await page.waitForTimeout(500)
  await page.screenshot({ path: join(SHOTS, '12-overlay-composited.png') })
  ctx.assertNoJsErrors()
})

test('c) alpha → 0 drops the colored pixels back near baseline', async () => {
  const { page } = ctx
  const layerId = (ctx as any).__layerId as string
  const before = (ctx as any).__beforeColored as number
  const slider = page.getByTestId(`layer-alpha-${layerId}`)
  await expect(slider).toBeVisible()
  // Drive the range input to 0 and fire input/change (the row sends overlay_set
  // {alpha:0} debounced).
  await slider.evaluate((el: HTMLInputElement) => {
    const set = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value')!.set!
    set.call(el, '0')
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  })

  // With alpha 0 the layer contributes nothing → colored pixels fall back toward
  // the pre-overlay baseline.
  await expect.poll(async () =>
    coloredPixelsInWindow(page, '[data-mdi-win="target"]'), {
    timeout: 20_000,
    message: 'alpha→0 did not remove the layer contribution',
  }).toBeLessThan(before + 800)

  await page.waitForTimeout(400)
  await page.screenshot({ path: join(SHOTS, '13-alpha-zero.png') })
  ctx.assertNoJsErrors()

  // Restore alpha to 0.5 so the live-nav step (d) still shows a composited layer.
  await slider.evaluate((el: HTMLInputElement) => {
    const set = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value')!.set!
    set.call(el, '0.5')
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  })
  await expect.poll(async () =>
    coloredPixelsInWindow(page, '[data-mdi-win="target"]'), {
    timeout: 20_000, message: 'restoring alpha=0.5 did not re-composite the layer',
  }).toBeGreaterThan(before + 500)
})

test('d) driving the target navigator keeps the layered figure updating', async () => {
  const { page } = ctx
  // The TARGET is tree B's signal window; test_nav_drag drives signal_trees[-1]
  // (= tree B). So its verdict reflects moves on the target's OWN navigator, which
  // repaints the base AND (via refresh_plot_layers) the layer. si_grains is 6×6
  // nav; scrub the CORNERS (different grains → the DP genuinely changes). Count
  // this run's verdict lines so a stale line from any earlier drag can't satisfy
  // the poll.
  const verdictBefore = ctx.backend.logBuffer.filter((l: string) =>
    /test_nav_drag:\s*\d+\/\d+\s+moves changed/.test(l)).length
  const targets = [[0, 0], [5, 0], [5, 5], [0, 5], [2, 3]]
  await backendAction(page, 'test_nav_drag', { targets })

  // The verdict line lands in the backend log (INFO-tee'd stderr). Wait for a NEW
  // one past the pre-drag count.
  await expect.poll(() =>
    ctx.backend.logBuffer.filter((l: string) =>
      /test_nav_drag:\s*\d+\/\d+\s+moves changed/.test(l)).length, {
    timeout: 60_000, message: 'nav_drag verdict never appeared in backend log',
  }).toBeGreaterThan(verdictBefore)

  const verdicts = ctx.backend.logBuffer.filter((l: string) =>
    /test_nav_drag:\s*\d+\/\d+\s+moves changed/.test(l))
  const verdict = verdicts[verdicts.length - 1]
  const m = verdict.match(/test_nav_drag:\s*(\d+)\/(\d+)\s+moves changed/)!
  const changed = Number(m[1]); const total = Number(m[2])
  console.log('[mdi] nav_drag verdict:', verdict.slice(verdict.indexOf('test_nav_drag')))
  // The nav path must not be WEDGED — corner moves land in different grains, so the
  // base DP repaints on most of them (adjacent-within-a-grain frames can repeat, so
  // we require a solid majority rather than every single move).
  expect(changed, `only ${changed}/${total} target-nav moves repainted (nav wedged?)`)
    .toBeGreaterThanOrEqual(Math.ceil(total / 2))

  // The layer is still present + composited after scrubbing (no colored-pixel
  // collapse, no dropped layer).
  await expect(page.getByTestId('layers-section')).toBeVisible()
  const before = (ctx as any).__beforeColored as number
  await expect.poll(async () =>
    coloredPixelsInWindow(page, '[data-mdi-win="target"]'), {
    timeout: 15_000, message: 'layer lost its composite after nav scrub',
  }).toBeGreaterThan(before + 300)

  await page.waitForTimeout(400)
  await page.screenshot({ path: join(SHOTS, '14-after-nav-scrub.png') })
  ctx.assertNoJsErrors()
})

test('e) removing the layer via the dock × returns the figure to gray', async () => {
  const { page } = ctx
  const layerId = (ctx as any).__layerId as string
  const before = (ctx as any).__beforeColored as number
  await page.getByTestId(`layer-remove-${layerId}`).click()

  // Layers section disappears (no layers left → renders nothing).
  await expect(page.getByTestId('layers-section')).toHaveCount(0, { timeout: 10_000 })

  // The composite is gone → colored pixels fall back near baseline.
  await expect.poll(async () =>
    coloredPixelsInWindow(page, '[data-mdi-win="target"]'), {
    timeout: 20_000, message: 'removing the layer did not restore the gray base',
  }).toBeLessThan(before + 800)

  await page.waitForTimeout(400)
  await page.screenshot({ path: join(SHOTS, '15-layer-removed.png') })
  ctx.assertNoJsErrors()
})

test('f) an incompatible drop (navigator onto signal) is refused, no crash', async () => {
  const { page } = ctx
  // Tag a NAVIGATOR window's pill as the source. A same-tree nav (6×6) has a
  // different frame shape from the 128×128 DP, so overlay_add must REFUSE with a
  // status message (not crash). Drop it onto the target signal window's shield.
  // First clear the previous (sig-A) source tag so data-mdi-src is unique.
  await page.evaluate(() => {
    document.querySelectorAll('[data-mdi-src="1"]')
      .forEach(el => el.removeAttribute('data-mdi-src'))
  })
  const navs = navWindows(page)
  await expect(navs.first()).toBeVisible({ timeout: 15_000 })
  await navs.nth(0).getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => { el.setAttribute('data-mdi-src', '1') })

  const errsBefore = backendErrorLines(ctx.backend).length
  await dragPillToShield(page, '[data-mdi-src="1"]', '[data-mdi-win="target"]')
  // Confirm popover appears (the renderer doesn't pre-validate shape) → accept →
  // the BACKEND refuses.
  const confirm = page.locator('[data-testid^="overlay-confirm-"]')
  await expect(confirm.first()).toBeVisible({ timeout: 10_000 })
  await page.locator('[data-testid^="overlay-confirm-ok-"]').first().click()

  // The refusal is an emit_status → a PLOTAPP status message that the main process
  // relays to the renderer over IPC (it does NOT reach Playwright stdout / the log
  // buffer — the "PLOTAPP echo allowlist" trap). It lands in state.status, shown in
  // the status bar's status-text element. Poll the DOM for the refusal string
  // ("Overlay refused: frame shapes differ (...)" from overlay.py).
  await expect.poll(async () =>
    (await page.getByTestId('status-text').textContent()) ?? '', {
    timeout: 15_000, message: 'incompatible drop produced no status refusal in the status bar',
  }).toMatch(/Overlay refused|frame shapes differ/i)

  // No new Python traceback fired (a refusal is a status, not an error).
  const errsAfter = backendErrorLines(ctx.backend)
  expect(errsAfter.length,
    `incompatible drop raised a traceback:\n${errsAfter.slice(errsBefore).join('\n')}`)
    .toBe(errsBefore)
  // The target still has NO layer (the refused overlay wasn't added).
  await expect(page.getByTestId('layers-section')).toHaveCount(0)

  await page.waitForTimeout(300)
  await page.screenshot({ path: join(SHOTS, '16-incompatible-refused.png') })
  ctx.assertNoJsErrors()
})

test('g) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[mdi] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
