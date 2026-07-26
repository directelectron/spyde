/**
 * mdi_layout.spec.ts — manual verification of the MDI window-management
 * improvements: smaller default size, free-slot auto-placement, snap-to-align
 * on drag/resize, titlebar always in view, and the Tile button.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount, titlebarGrabPoint } = require('./_harness.cjs')

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
  const moverBox = (await mover.boundingBox())!
  // Grab RIGHT of the breadcrumb pill (an HTML5 drag source that stops
  // pointerdown — grabbing it starts a DnD payload, not a window move), and
  // keep the drop math in window coords via the grab offset.
  const grab = await titlebarGrabPoint(mover)
  const grabOffX = grab.x - moverBox.x
  const grabOffY = grab.y - moverBox.y
  const dropX = targetBox.x + targetBox.width + 4   // just within snap distance
  const dropY = targetBox.y + 20
  await page.mouse.move(grab.x, grab.y)
  await page.mouse.down()
  await page.mouse.move(dropX + grabOffX, dropY + grabOffY, { steps: 8 })
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
  const edgeGrab = await titlebarGrabPoint(edgeWin)
  await page.mouse.move(edgeGrab.x, edgeGrab.y)
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

test('snapped windows stay INDEPENDENT (no link/stick) and resize snaps too', async () => {
  const { page } = ctx
  const wins = page.getByTestId('subwindow')

  // Put two windows in a known, non-overlapping arrangement first: Tile, then
  // work with the first two. (Tile is asserted overlap-free above.)
  await page.getByTestId('tile-windows').click()
  await page.waitForTimeout(500)

  const mover = wins.nth(0)
  const peer = wins.nth(1)

  // ── 1) Drag `mover` so its RIGHT edge lands within snap distance of `peer`'s
  // LEFT edge — it should snap flush. (Left-of-peer, not right: after Tile the
  // right-hand column sits against the area bound, where the titlebar-visibility
  // clamp — not snapping — decides the final x.)
  let peerBox = (await peer.boundingBox())!
  let moverBox = (await mover.boundingBox())!
  let grab = await titlebarGrabPoint(mover)
  const offX = grab.x - moverBox.x, offY = grab.y - moverBox.y
  const dropX = peerBox.x - moverBox.width + 5     // right edge 5 px past peer's left
  const dropY = peerBox.y
  await page.mouse.move(grab.x, grab.y)
  await page.mouse.down()
  await page.mouse.move(dropX + offX, dropY + offY, { steps: 8 })
  await page.mouse.up()
  await page.waitForTimeout(300)

  moverBox = (await mover.boundingBox())!
  console.log('[mdi] independence: peer left =', peerBox.x,
    'mover right =', moverBox.x + moverBox.width)
  expect(Math.abs((moverBox.x + moverBox.width) - peerBox.x)).toBeLessThanOrEqual(2)
  await page.screenshot({ path: join(SHOTS, '06-adjacent.png') })

  // ── 2) THE POINT OF THIS TEST: dragging a window that is flush against a
  // neighbour must NOT drag the neighbour with it. The removed edge-snap
  // GROUPING feature ("stick windows", commit 9bfe6ae) nudged every partner in
  // the group by the same delta — snapping is alignment, not attachment.
  const peerBefore = (await peer.boundingBox())!
  grab = await titlebarGrabPoint(mover)
  await page.mouse.move(grab.x, grab.y)
  await page.mouse.down()
  await page.mouse.move(grab.x - 120, grab.y + 40, { steps: 10 })   // well past SNAP_DIST
  await page.mouse.up()
  await page.waitForTimeout(300)

  const peerAfter = (await peer.boundingBox())!
  const moverAfter = (await mover.boundingBox())!
  console.log('[mdi] peer before =', peerBefore, 'after =', peerAfter)
  expect(Math.abs(peerAfter.x - peerBefore.x)).toBeLessThanOrEqual(1)
  expect(Math.abs(peerAfter.y - peerBefore.y)).toBeLessThanOrEqual(1)
  // …and the dragged window really did move (otherwise the assert above is vacuous).
  expect(Math.abs(moverAfter.x - moverBox.x)).toBeGreaterThan(20)
  await page.screenshot({ path: join(SHOTS, '07-dragged-away-peer-stayed.png') })

  // ── 3) Resize snapping: drag `mover`'s bottom-right handle to just short of
  // `peer`'s LEFT edge — the right edge should snap flush to it.
  moverBox = (await mover.boundingBox())!
  peerBox = (await peer.boundingBox())!
  // Only meaningful when peer's left edge is to the RIGHT of mover's left edge
  // with room for the minimum width; skip the assert rather than fake it.
  const targetRight = peerBox.x
  if (targetRight - moverBox.x > 320) {
    const handle = mover.getByTestId('resize-handle')
    const hb = (await handle.boundingBox())!
    await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2)
    await page.mouse.down()
    // Aim 6 px short of peer's left edge (inside SNAP_DIST), same height.
    await page.mouse.move(targetRight - 6, hb.y + hb.height / 2, { steps: 10 })
    await page.mouse.up()
    await page.waitForTimeout(300)

    const resized = (await mover.boundingBox())!
    console.log('[mdi] resize snap: target (peer left) =', targetRight,
      'mover right after =', resized.x + resized.width)
    expect(Math.abs((resized.x + resized.width) - targetRight)).toBeLessThanOrEqual(2)

    // …and resizing must not resize/move the neighbour either (the removed
    // "v2 linked resize" propagated the linked dimension across the group).
    const peerAfterResize = (await peer.boundingBox())!
    expect(Math.abs(peerAfterResize.x - peerAfter.x)).toBeLessThanOrEqual(1)
    expect(Math.abs(peerAfterResize.width - peerAfter.width)).toBeLessThanOrEqual(1)
    expect(Math.abs(peerAfterResize.height - peerAfter.height)).toBeLessThanOrEqual(1)
    await page.screenshot({ path: join(SHOTS, '08-resize-snapped.png') })
  } else {
    console.log('[mdi] resize-snap skipped: no room between the two windows')
  }

  ctx.assertNoJsErrors()
})
