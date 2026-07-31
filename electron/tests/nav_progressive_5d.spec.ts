/**
 * nav_progressive_5d.spec.ts — the progressive navigator fill on a 5-D stack.
 *
 * A 5-D dataset (time × real-space y,x | ky,kx) opens TWO navigators: a 1-D time
 * line whose selector drives a 2-D real-space image. The fill used to be handed
 * the already-reduced 1-D time sum, so its "chunks" were whole time steps — it
 * read the entire dataset to paint one point of a short line and never filled
 * the real-space navigator at all. The fix computes the DEEP (t, y, x) nav-sum
 * progressively and DERIVES the time line from the real-space planes it already
 * has: one pass, two navigators.
 *
 * Real Dask (the distributed shm/poll fill is the path users hit) + the bundled
 * synthetic 5-D stack. Every wait is signal-based: the status bar's "Computing
 * navigator…" for the in-progress capture, the backend's "navigator fill
 * complete (recursive)" INFO line for completion. Screenshots land in
 * electron/nav5d_shots/ for the human eyes the project rule requires.
 */
import { test, expect } from '@playwright/test'
import * as fs from 'fs'
import * as path from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, navWindows, backendErrorLines,
} = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>

const SHOTS = path.join(__dirname, '..', 'nav5d_shots')
const shot = async (name: string) => {
  fs.mkdirSync(SHOTS, { recursive: true })
  await ctx.page.screenshot({ path: path.join(SHOTS, name) })
}

/** Mean luminance + stddev of a navigator window's rendered pixels. A black or
 *  never-filled navigator has ~0 of both; a filled structured one has both. */
async function navStats(page: any, index: number) {
  const win = navWindows(page).nth(index)
  const box = await win.boundingBox()
  if (!box) throw new Error(`navigator ${index} has no bounding box`)
  const buf = await page.screenshot({ clip: box })
  return page.evaluate(async (b64: string) => {
    const img = new Image()
    await new Promise((res, rej) => {
      img.onload = res; img.onerror = rej; img.src = 'data:image/png;base64,' + b64
    })
    const c = document.createElement('canvas')
    c.width = img.width; c.height = img.height
    const g = c.getContext('2d', { willReadFrequently: true })!
    g.drawImage(img, 0, 0)
    const d = g.getImageData(0, 0, c.width, c.height).data
    let sum = 0, sum2 = 0, n = 0
    for (let i = 0; i < d.length; i += 4) {
      const l = (d[i] + d[i + 1] + d[i + 2]) / 3
      sum += l; sum2 += l * l; n++
    }
    const mean = sum / n
    return { mean, std: Math.sqrt(Math.max(0, sum2 / n - mean * mean)) }
  }, buf.toString('base64'))
}

test.beforeAll(async () => {
  // INFO tees the backend logger to stderr so waitForLog sees the fill's own
  // completion line (PLOTAPP emit/status never reaches Playwright stdout).
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test.setTimeout(240_000)

test('a 5-D stack fills BOTH navigators from one progressive pass', async () => {
  const { page } = ctx
  await shot('00-empty.png')

  // Big enough that the fill takes seconds (6 t × 4×4 nav-chunk grid = 96
  // progressive steps), small enough for e2e.
  await backendAction(page, 'load_test_data_5d', { frames: 6, nav: 32, sig: 64 })

  // Two navigators (time + real space) and the diffraction pattern.
  await waitForSubwindowCount(page, 3, 120_000)
  await expect.poll(() => navWindows(page).count(), {
    timeout: 60_000, message: 'a 5-D stack must open TWO navigator windows',
  }).toBe(2)

  // IN PROGRESS — the fill's own throttled progress status is the signal.
  await expect.poll(
    async () => (await page.getByTestId('status-text').textContent()) ?? '',
    { timeout: 60_000, message: 'the navigator fill never reported progress' },
  ).toMatch(/Computing navigator/)
  await shot('01-fill-in-progress.png')

  // COMPLETE — the recursive fill's own completion line.
  await ctx.backend.waitForLog('navigator fill complete (recursive)', 180_000)
  // Let the final uniform repaint + histogram reach the renderer.
  await expect.poll(async () => (await navStats(page, 0)).std, {
    timeout: 30_000, message: 'the time navigator never drew any structure',
  }).toBeGreaterThan(2)
  await shot('02-fill-complete.png')

  // BOTH navigators must show real, structured content — a black or half-filled
  // navigator is the bug, and it reads as ~0 stddev.
  const timeNav = await navStats(page, 0)
  const spaceNav = await navStats(page, 1)
  expect(timeNav.std, 'time navigator is blank').toBeGreaterThan(2)
  expect(spaceNav.std, 'real-space navigator is blank').toBeGreaterThan(2)
  expect(spaceNav.mean, 'real-space navigator is black').toBeGreaterThan(5)

  // Moving the TIME selector must repaint the real-space navigator from the
  // recursive result. Driven through `test_nav_drag` (widget in DATA coords,
  // server-side) rather than the mouse: the 1-D navigator panel is small and
  // sub-frame pointer steps produce no-op reads — the same trap
  // movie_roi_drag_perf.spec.ts documents.
  const dragDone = ctx.backend.waitForLog('test_nav_drag:', 60_000)
  await backendAction(page, 'test_nav_drag',
    { targets: [[2, 0], [4, 0], [5, 0]] })
  const dragLine = await dragDone
  expect(dragLine, 'moving the time axis did not repaint the real-space navigator')
    .toContain('3/3 moves changed')
  await expect.poll(async () => (await navStats(page, 1)).mean, {
    timeout: 30_000,
    message: 'the real-space navigator did not follow the time axis',
  }).not.toBe(spaceNav.mean)
  await shot('03-time-moved.png')

  ctx.assertNoJsErrors()
  expect(backendErrorLines(ctx.backend), 'backend errors during the fill').toEqual([])
})
