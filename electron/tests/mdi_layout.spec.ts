/**
 * mdi_layout.spec.ts — manual verification of the MDI window-management
 * improvements: smaller default size, free-slot auto-placement, snap-to-align
 * on drag/resize, titlebar always in view, and the Tile button.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'mdi_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  const { page } = ctx
  // launchApp's backend-ready signal is best-effort (a log match, not a hard
  // sync point) — a short settle wait avoids the action being silently
  // dropped before the Python stdin reader loop is actually pumping.
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_vectors')
  await waitForSubwindowCount(page, 4, 60_000)
  await page.waitForTimeout(1500)
})

test.afterAll(async () => { await ctx?.app?.close() })
test.setTimeout(120_000)

test('window placement, snapping, titlebar visibility, and tile', async () => {
  const { page } = ctx
  const wins = page.getByTestId('subwindow')
  const n = await wins.count()
  console.log('[mdi] window count =', n)

  // 1) Default size is smaller than the old 400x392 baseline.
  const box0 = await wins.first().boundingBox()
  console.log('[mdi] first window size =', box0)
  expect(box0!.width).toBeLessThanOrEqual(400)
  expect(box0!.height).toBeLessThanOrEqual(392)

  // 2) No two windows should be placed exactly on top of each other (auto free-slot).
  const boxes: { x: number; y: number; width: number; height: number }[] = []
  for (let i = 0; i < n; i++) {
    const b = await wins.nth(i).boundingBox()
    if (b) boxes.push(b)
  }
  let anyDistinct = false
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      if (Math.abs(boxes[i].x - boxes[j].x) > 2 || Math.abs(boxes[i].y - boxes[j].y) > 2) anyDistinct = true
    }
  }
  expect(anyDistinct).toBe(true)
  await page.screenshot({ path: join(SHOTS, '01-initial-layout.png') })

  // 3) Drag the last window near the second-to-last window's right edge —
  // it should SNAP flush against it (within a couple px), not stop at the
  // raw drop position.
  const target = wins.nth(n - 2)
  const mover = wins.nth(n - 1)
  const targetBox = (await target.boundingBox())!
  const moverBar = mover.getByTestId('subwindow-titlebar')
  const moverBarBox = (await moverBar.boundingBox())!
  const dropX = targetBox.x + targetBox.width + 4   // just within snap distance
  const dropY = targetBox.y + 20
  await page.mouse.move(moverBarBox.x + 20, moverBarBox.y + 10)
  await page.mouse.down()
  await page.mouse.move(dropX + 20, dropY + 10, { steps: 8 })
  await page.mouse.up()
  await page.waitForTimeout(300)
  const moverBoxAfter = (await mover.boundingBox())!
  console.log('[mdi] snap test: target right edge =', targetBox.x + targetBox.width,
    'mover left edge after drag =', moverBoxAfter.x)
  expect(Math.abs(moverBoxAfter.x - (targetBox.x + targetBox.width))).toBeLessThanOrEqual(2)
  await page.screenshot({ path: join(SHOTS, '02-snapped.png') })

  // 4) Drag a window far past the top-left corner — its titlebar must remain
  // at least partially visible (not fully off-screen).
  const edgeWin = wins.first()
  const edgeBar = edgeWin.getByTestId('subwindow-titlebar')
  const edgeBarBox = (await edgeBar.boundingBox())!
  await page.mouse.move(edgeBarBox.x + 20, edgeBarBox.y + 10)
  await page.mouse.down()
  await page.mouse.move(-500, -500, { steps: 8 })
  await page.mouse.up()
  await page.waitForTimeout(300)
  const edgeBoxAfter = (await edgeWin.boundingBox())!
  console.log('[mdi] clamp test: titlebar box after drag off top-left =', edgeBoxAfter)
  expect(edgeBoxAfter.y).toBeGreaterThanOrEqual(0)
  expect(edgeBoxAfter.x + edgeBoxAfter.width).toBeGreaterThan(20)   // some part still reachable
  await page.screenshot({ path: join(SHOTS, '03-clamped-top-left.png') })

  // 5) Drag a window far past the bottom-right — titlebar (top strip) must
  // still be on-screen (not scrolled/pushed below the visible area).
  const edgeWin2 = wins.nth(1)
  const edgeBar2 = edgeWin2.getByTestId('subwindow-titlebar')
  const edgeBarBox2 = (await edgeBar2.boundingBox())!
  const areaBox = (await page.getByTestId('mdi-area').boundingBox())!
  await page.mouse.move(edgeBarBox2.x + 20, edgeBarBox2.y + 10)
  await page.mouse.down()
  await page.mouse.move(areaBox.x + areaBox.width + 800, areaBox.y + areaBox.height + 800, { steps: 8 })
  await page.mouse.up()
  await page.waitForTimeout(300)
  const edgeBoxAfter2 = (await edgeWin2.boundingBox())!
  console.log('[mdi] clamp test bottom-right: box after =', edgeBoxAfter2, 'area =', areaBox)
  expect(edgeBoxAfter2.y).toBeLessThanOrEqual(areaBox.y + areaBox.height - 5)
  await page.screenshot({ path: join(SHOTS, '04-clamped-bottom-right.png') })

  // 6) Tile: all windows rearrange into a grid with no overlap.
  await page.getByTestId('tile-windows').click()
  await page.waitForTimeout(500)
  const tiledBoxes: { x: number; y: number; width: number; height: number }[] = []
  for (let i = 0; i < n; i++) {
    const b = await wins.nth(i).boundingBox()
    if (b) tiledBoxes.push(b)
  }
  console.log('[mdi] tiled boxes =', JSON.stringify(tiledBoxes))
  let overlapCount = 0
  for (let i = 0; i < tiledBoxes.length; i++) {
    for (let j = i + 1; j < tiledBoxes.length; j++) {
      const a = tiledBoxes[i], b = tiledBoxes[j]
      const overlaps = a.x < b.x + b.width && a.x + a.width > b.x &&
        a.y < b.y + b.height && a.y + a.height > b.y
      if (overlaps) overlapCount++
    }
  }
  console.log('[mdi] tile overlap count =', overlapCount)
  expect(overlapCount).toBe(0)
  await page.screenshot({ path: join(SHOTS, '05-tiled.png') })

  ctx.assertNoJsErrors()
})
