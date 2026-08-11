/**
 * window_recovery.spec.ts — plot windows must never be stranded off-screen.
 *
 * The bug this pins: MDI subwindows live at ABSOLUTE pixel coordinates that are
 * assigned once and never re-derived, inside an area that is `overflow: hidden`.
 * Shrink the area and every window outside the new bounds becomes invisible —
 * and unreachable, because the top bar lists MINIMIZED windows only, so a
 * stranded window is listed nowhere. It is still open, still streaming, just not
 * anywhere you can see it.
 *
 * That is the "I closed and reopened my laptop and all my plots are gone"
 * report: a Mac lid-close changes the display configuration and resizes the app
 * window. Measured on the real failure, the Python backend, the Dask cluster and
 * the renderer PROCESS all survive it — so the windows were never lost.
 *
 * Driven here by resizing the BrowserWindow, which is the same event the display
 * change produces and needs no sleeping laptop. Tile first, so windows actually
 * occupy the full area and the shrink has something to strand (default placement
 * packs from the top-left and would survive by luck, testing nothing).
 *
 * Run:
 *   npx playwright test tests/window_recovery.spec.ts --project=electron \
 *     --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'window_recovery_shots')

const BIG = { w: 1400, h: 900 }
const SMALL = { w: 640, h: 560 }

async function setWindowSize(app: any, size: { w: number; h: number }) {
  await app.evaluate(async ({ BrowserWindow }, s) => {
    BrowserWindow.getAllWindows()[0].setSize(s.w, s.h, false)
  }, size)
}

/** Every subwindow's box, plus the area box they must stay inside. */
async function layout(page: any) {
  const area = await page.getByTestId('mdi-area').boundingBox()
  const wins = page.getByTestId('subwindow')
  const n = await wins.count()
  const boxes: Array<{ x: number; y: number; w: number; h: number }> = []
  for (let i = 0; i < n; i++) {
    const b = await wins.nth(i).boundingBox()
    if (b) boxes.push({ x: b.x, y: b.y, w: b.width, h: b.height })
  }
  return { area, boxes }
}

/**
 * Windows that spill outside the area — i.e. are clipped away by its
 * `overflow: hidden`. Containment (not merely "some sliver is reachable") is the
 * contract, because the area here is always big enough to hold both windows at
 * the minimum size: a window that spills is one the user has lost part of, and
 * one that spills entirely is the reported bug. Tolerance covers sub-pixel
 * layout rounding only.
 */
function outsideArea(area: any, boxes: Array<{ x: number; y: number; w: number; h: number }>) {
  const T = 2
  return boxes.filter(b =>
    b.x < area.x - T ||
    b.y < area.y - T ||
    b.x + b.w > area.x + area.width + T ||
    b.y + b.h > area.y + area.height + T)
}

test.setTimeout(300_000)

test('shrinking the window never strands a plot off-screen', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page, app } = ctx

  try {
    await setWindowSize(app, BIG)
    await backendAction(page, 'load_test_data_si_grains', {})
    await waitForSubwindowCount(page, 2)
    await page.waitForTimeout(2500)

    // Spread them over the whole area, so some sit near the right/bottom edges.
    await page.getByTestId('tile-windows').click()
    await page.waitForTimeout(1200)
    await page.screenshot({ path: join(SHOTS, '01-tiled-large.png') })

    const before = await layout(page)
    console.log('[recovery] before area =', JSON.stringify(before.area),
                'boxes =', JSON.stringify(before.boxes))
    expect(before.boxes.length).toBe(2)
    expect(outsideArea(before.area, before.boxes)).toEqual([])
    // The shrink must actually be able to strand something: at least one window
    // has to start beyond where the narrowed area's right edge will land.
    const reachRight = Math.max(...before.boxes.map(b => b.x + b.w))
    expect(reachRight).toBeGreaterThan(SMALL.w)

    // ── The display change ────────────────────────────────────────────────
    await setWindowSize(app, SMALL)
    await expect
      .poll(async () => (await page.getByTestId('mdi-area').boundingBox())?.width ?? 0,
            { timeout: 20_000 })
      .toBeLessThan(BIG.w - 200)
    await page.waitForTimeout(1500)
    await page.screenshot({ path: join(SHOTS, '02-after-shrink.png') })

    const after = await layout(page)
    console.log('[recovery] after shrink area =', JSON.stringify(after.area),
                'boxes =', JSON.stringify(after.boxes))
    // Nothing closed — the windows were never lost, only moved.
    expect(after.boxes.length).toBe(2)
    const lost = outsideArea(after.area, after.boxes)
    expect(lost, `clipped off-screen after shrink: ${JSON.stringify(lost)} ` +
                 `area=${JSON.stringify(after.area)}`).toEqual([])

    // ── And back again ────────────────────────────────────────────────────
    // Growing the area must not disturb what recovery just placed.
    await setWindowSize(app, BIG)
    await page.waitForTimeout(2000)
    await page.screenshot({ path: join(SHOTS, '03-restored-large.png') })

    const restored = await layout(page)
    expect(restored.boxes.length).toBe(2)
    expect(outsideArea(restored.area, restored.boxes)).toEqual([])

    ctx.assertNoJsErrors()
  } finally {
    await ctx.app.close().catch(() => {})
  }
})
