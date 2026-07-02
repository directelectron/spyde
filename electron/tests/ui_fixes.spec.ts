/**
 * ui_fixes.spec.ts — E2E verification of the 2026-07 UI-degradation fix batch:
 *
 *  1  toolbar floats BELOW the window (same z-level, no hover-raise)
 *  2  MDI minimize → top-bar chip → restore
 *  3  Tile reserves room for the below-window toolbar
 *  4  wizard caret opens below; flips beside the window near the area bottom
 *  5  Workflow panel is always present in the dock (no toolbar toggle)
 *  6  Rebin repaints the DP immediately + adds the "Binned" workflow node
 *  7  Virtual Imaging: + auto-opens the new ROI caret (with Commit); commit
 *     creates a new tree; deselecting the action closes the live VI artifacts
 *  8  Line Profile: solid+dashed line ROI (not a box) + 1-D output window
 *  9  Center Zero Beam: half-width box overlay on the DP
 *
 * Eager data (SPYDE_NO_DASK) — geometry/lifecycle only, no cluster needed.
 * Screenshots land in electron/ui_fixes_shots/ (read them — they ARE the test).
 */
import { test, expect, Page } from '@playwright/test'
import { join } from 'path'
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'ui_fixes_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>
let page: Page

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  page = ctx.page
  await page.waitForTimeout(1500)              // let the stdin pump settle
  await backendAction(page, 'load_test_data')  // eager 4D stack → nav + DP
  await waitForSubwindowCount(page, 2, 60_000)
  await page.waitForTimeout(1000)
})

test.afterAll(async () => { await ctx?.app?.close() })

const shot = (name: string) => page.screenshot({ path: join(SHOTS, name) })

function dpWindow() {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Rebin') }).first()
}

test('1: toolbar floats below the window at the window z-level', async () => {
  const win = dpWindow()
  await win.getByTestId('subwindow-titlebar').hover()
  const tb = win.getByTestId('floating-toolbar')
  await expect(tb).toBeVisible()
  const wb = (await win.boundingBox())!
  const tbb = (await tb.boundingBox())!
  // The bar hangs BELOW the window's bottom edge (not over the figure).
  expect(tbb.y).toBeGreaterThanOrEqual(wb.y + wb.height - 2)
  // No hover-raise: the window's z-index stays in the managed focus range.
  const z = await win.evaluate((el) => Number(getComputedStyle(el).zIndex))
  expect(z).toBeLessThan(1000)
  await shot('01-toolbar-below.png')
})

test('2: minimize lists the window in the top bar; the chip restores it', async () => {
  const win = dpWindow()
  await win.getByTestId('minimize-btn').click()
  await expect(win).toBeHidden()
  const bar = page.getByTestId('minimized-bar')
  await expect(bar).toBeVisible()
  await shot('02-minimized.png')
  await bar.locator('button').first().click()
  await expect(dpWindow()).toBeVisible()
  await expect(page.getByTestId('minimized-bar')).toHaveCount(0)
})

test('3: Tile leaves the toolbar strip below each window', async () => {
  await page.getByTestId('tile-windows').click()
  await page.waitForTimeout(400)
  const area = (await page.getByTestId('mdi-area').boundingBox())!
  for (const w of await page.getByTestId('subwindow').all()) {
    const b = (await w.boundingBox())!
    // Every tiled window leaves ≥ the toolbar reserve below itself (to the
    // area bottom or the next row).
    expect(b.y + b.height + 40).toBeLessThanOrEqual(area.y + area.height + 2)
  }
  await shot('03-tiled.png')
})

test('4: wizard caret prefers below; flips beside the window near the bottom', async () => {
  const win = dpWindow()
  const area = (await page.getByTestId('mdi-area').boundingBox())!
  const tb = win.getByTestId('subwindow-titlebar')

  // Test 3 tiled the windows to FULL height — move this one to the TOP and
  // SHRINK it so there is genuinely room below for the caret.
  const tbb0 = (await tb.boundingBox())!
  await page.mouse.move(tbb0.x + tbb0.width / 2, tbb0.y + tbb0.height / 2)
  await page.mouse.down()
  await page.mouse.move(area.x + area.width / 2, area.y + 40, { steps: 12 })
  await page.mouse.up()
  await page.waitForTimeout(200)
  const wb0 = (await win.boundingBox())!
  const rh = (await win.getByTestId('resize-handle').boundingBox())!
  await page.mouse.move(rh.x + rh.width / 2, rh.y + rh.height / 2)
  await page.mouse.down()
  await page.mouse.move(wb0.x + 420, wb0.y + 300, { steps: 10 })
  await page.mouse.up()
  await page.waitForTimeout(300)

  await win.getByTestId('subwindow-titlebar').hover()
  await win.getByTestId('action-btn-Center Zero Beam').click()
  const caret = page.getByTestId('center-zero-beam-wizard')
  await expect(caret).toBeVisible()
  const wb1 = (await win.boundingBox())!
  const cb1 = (await caret.boundingBox())!
  expect(cb1.y, 'caret should open below the window').toBeGreaterThanOrEqual(
    wb1.y + wb1.height - 2)
  await shot('04a-caret-below.png')

  // Drag the window to the bottom of the area → the caret must move BESIDE it.
  const tbb = (await tb.boundingBox())!
  await page.mouse.move(tbb.x + tbb.width / 2, tbb.y + tbb.height / 2)
  await page.mouse.down()
  await page.mouse.move(area.x + area.width / 3,
    area.y + area.height - 60, { steps: 12 })
  await page.mouse.up()
  await page.waitForTimeout(300)
  const wb2 = (await win.boundingBox())!
  const cb2 = (await caret.boundingBox())!
  const beside = cb2.x >= wb2.x + wb2.width - 4 || cb2.x + cb2.width <= wb2.x + 4
  expect(beside, 'caret should float beside the window when below is off-area')
    .toBe(true)
  await shot('04b-caret-beside.png')

  // Drag back up → the caret snaps back below.
  const tbb2 = (await tb.boundingBox())!
  await page.mouse.move(tbb2.x + tbb2.width / 2, tbb2.y + tbb2.height / 2)
  await page.mouse.down()
  await page.mouse.move(area.x + 80, area.y + 40, { steps: 12 })
  await page.mouse.up()
  await page.waitForTimeout(300)
  const wb3 = (await win.boundingBox())!
  const cb3 = (await caret.boundingBox())!
  expect(cb3.y, 'caret should snap back below once there is room')
    .toBeGreaterThanOrEqual(wb3.y + wb3.height - 2)
  await page.getByTestId('czb-close').click()
})

test('5: the Workflow panel is present in the dock without any toolbar toggle', async () => {
  await dpWindow().getByTestId('subwindow-titlebar').click()
  await expect(page.getByTestId('signal-tree')).toBeVisible()
  await expect(page.getByTestId('tree-node-root')).toBeVisible()
})

test('6: Rebin repaints the DP immediately and adds the Binned node', async () => {
  const win = dpWindow()
  // The DP figure id (the shown iframe of this window).
  const figId = await win.locator('iframe:visible').first()
    .getAttribute('data-testid').then(t => t!.replace('figure-', ''))
  const sigBefore = await page.evaluate((id) => window._spyde_test_image_sig?.(id), figId)

  await win.getByTestId('subwindow-titlebar').hover()
  await win.getByTestId('action-btn-Rebin').click()
  await page.getByTestId('action-run').click()

  // The DP must repaint WITHOUT any crosshair nudge…
  await expect.poll(async () =>
    page.evaluate((id) => window._spyde_test_image_sig?.(id), figId), {
    timeout: 30_000, message: 'DP did not repaint after Rebin',
  }).not.toBe(sigBefore)
  // …and the workflow gains the Binned node.
  await expect(page.getByTestId('tree-node-Binned')).toBeVisible({ timeout: 15_000 })
  await shot('06-rebin.png')
})

test('7: VI + auto-opens the ROI caret; Commit makes a tree; deselect closes live VI', async () => {
  const win = dpWindow()
  await win.getByTestId('subwindow-titlebar').hover()
  await win.getByTestId('action-btn-Virtual Imaging').click()
  await expect(page.getByTestId('sub-toolbar')).toBeVisible()
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('subaction-add_virtual_image').click()
  await waitForSubwindowCount(page, before + 1, 30_000)
  // The new item's caret opened AUTOMATICALLY, with the Commit affordance.
  const caret = page.locator('[data-testid^="vi-caret-"]')
  await expect(caret).toBeVisible({ timeout: 10_000 })
  const commit = page.locator('[data-testid^="vi-commit-"]')
  await expect(commit).toBeVisible()
  await shot('07a-vi-caret.png')

  // Commit → a NEW signal tree (its own window) appears.
  const beforeCommit = await page.getByTestId('subwindow').count()
  await commit.click()
  await waitForSubwindowCount(page, beforeCommit + 1, 30_000)
  await shot('07b-vi-committed.png')

  // Deselect the (open, lit) Virtual Imaging action → the LIVE VI output
  // closes; the committed tree survives. The commit focused the NEW window on
  // top of the toolbar (same-z design), so raise the source window first.
  const beforeOff = await page.getByTestId('subwindow').count()
  await win.getByTestId('subwindow-titlebar').click()   // focus-raise
  await win.getByTestId('subwindow-titlebar').hover()
  await win.getByTestId('action-btn-Virtual Imaging').click()   // deselect-all
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 15_000, message: 'live VI window did not close on deselect',
  }).toBe(beforeOff - 1)
  await shot('07c-vi-deselected.png')
})

test('8: Line Profile draws the two-line ROI and a 1-D output', async () => {
  const win = dpWindow()
  const before = await page.getByTestId('subwindow').count()
  await win.getByTestId('subwindow-titlebar').hover()
  await win.getByTestId('action-btn-Line Profile').click()
  await waitForSubwindowCount(page, before + 1, 30_000)
  await page.waitForTimeout(800)
  // The ROI colour (#ffd166) appears on the DP canvas: solid line + handles.
  const yellow = await countYellowPixels()
  expect(yellow, 'no line-profile ROI pixels on the DP').toBeGreaterThan(20)
  await shot('08-line-profile.png')
})

test('9: Center Zero Beam shows the half-width box overlay', async () => {
  const win = dpWindow()
  await win.getByTestId('subwindow-titlebar').hover()
  await win.getByTestId('action-btn-Center Zero Beam').click()
  await expect(page.getByTestId('center-zero-beam-wizard')).toBeVisible()
  const boxBefore = await countYellowPixels()
  await page.getByTestId('czb-halfwidth').fill('30')
  await expect.poll(() => countYellowPixels(), {
    timeout: 15_000, message: 'no centering-region box drawn on the DP',
  }).toBeGreaterThan(boxBefore)
  await shot('09-czb-region-box.png')
  ctx.assertNoJsErrors()
})

/** Count warm-yellow pixels (#ffcc00 / #ffd166 family) across all figure
 *  canvases — the CZB region box + the line-profile ROI colour. */
async function countYellowPixels(): Promise<number> {
  let total = 0
  for (const frame of page.frames()) {
    try {
      total += await frame.evaluate(() => {
        let n = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const ctx2 = (c as HTMLCanvasElement).getContext('2d')
          const el = c as HTMLCanvasElement
          if (!ctx2 || !el.width || !el.height) continue
          const d = ctx2.getImageData(0, 0, el.width, el.height).data
          for (let p = 0; p < d.length; p += 4) {
            const r = d[p], g = d[p + 1], b = d[p + 2]
            if (r > 200 && g > 150 && b < 130) n++
          }
        }
        return n
      })
    } catch { /* detached frame */ }
  }
  return total
}
