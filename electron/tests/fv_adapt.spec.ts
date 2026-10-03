/**
 * fv_adapt.spec.ts — teaching the neural detector by double-clicking the pattern.
 *
 * With the Find Vectors caret open (neural), a double-click on a red preview
 * circle marks it "not a disk" (an orange ✕), on empty pattern "a disk is here"
 * (a green ring), and on a mark removes it. A second after the last mark (or on
 * Adapt) the backend refits a copy of the model; the caret switches its Model to
 * the taught one and reports how general it still is. Revert drops the marks and
 * the taught model.
 *
 * Only pixels prove this works: the marks have to appear where the clicks were,
 * so the spec finds a red circle by its pixels and double-clicks its centre.
 * Taught models go to a temporary SPYDE_MODELS_DIR, never the real home.
 */
import { test, expect, Page, Frame } from '@playwright/test'
import { mkdtempSync, mkdirSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'fv_adapt_shots'
mkdirSync(SHOTS, { recursive: true })

let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({
    dask: true,
    env: {
      SPYDE_LOG_LEVEL: 'INFO',
      SPYDE_MODELS_DIR: mkdtempSync(join(tmpdir(), 'spyde-e2e-models-')),
    },
  })
  await backendAction(ctx.page, 'load_test_data_si_grains')
  await waitForSubwindowCount(ctx.page, 2, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(240_000)

/** Pixels of one colour class in the diffraction window's figure, with their
 * connected blobs (canvas px) and the canvas→CSS scale. */
async function blobs(frame: Frame, kind: 'red' | 'orange' | 'green') {
  return frame.evaluate((k) => {
    const match = (r: number, g: number, b: number) =>
      k === 'red' ? (r > 200 && g < 80 && b < 80)
        : k === 'orange' ? (r > 220 && g > 120 && g < 190 && b < 100)
          : (g > 170 && r < 120 && b > 80 && b < 150)
    let best: { canvas: HTMLCanvasElement, mask: Uint8Array } | null = null
    let bestCount = 0
    for (const c of Array.from(document.querySelectorAll('canvas'))) {
      const g2 = c.getContext('2d')
      if (!g2 || !c.width || !c.height) continue
      const d = g2.getImageData(0, 0, c.width, c.height).data
      const mask = new Uint8Array(c.width * c.height)
      let n = 0
      for (let p = 0; p < mask.length; p++) {
        if (match(d[4 * p], d[4 * p + 1], d[4 * p + 2])) { mask[p] = 1; n++ }
      }
      if (n > bestCount) { bestCount = n; best = { canvas: c, mask } }
    }
    const none = { left: 0, top: 0, width: 0, height: 0 }
    if (!best) return { count: 0, blobs: [] as { x: number, y: number, n: number }[], rect: none }
    const { canvas, mask } = best
    const W = canvas.width
    const seen = new Uint8Array(mask.length)
    const out: { x: number, y: number, n: number }[] = []
    for (let p = 0; p < mask.length; p++) {
      if (!mask[p] || seen[p]) continue
      const stack = [p]; seen[p] = 1
      let sx = 0, sy = 0, n = 0
      while (stack.length) {
        const q = stack.pop()!
        const x = q % W, y = (q - x) / W
        sx += x; sy += y; n++
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]) {
          const xx = x + dx, yy = y + dy
          if (xx < 0 || yy < 0 || xx >= W || yy >= canvas.height) continue
          const r = yy * W + xx
          if (mask[r] && !seen[r]) { seen[r] = 1; stack.push(r) }
        }
      }
      const rect = canvas.getBoundingClientRect()
      out.push({ x: rect.left + (sx / n) * rect.width / W, y: rect.top + (sy / n) * rect.height / canvas.height, n })
    }
    const r = canvas.getBoundingClientRect()
    return { count: bestCount, blobs: out.filter((b) => b.n >= 6),
      rect: { left: r.left, top: r.top, width: r.width, height: r.height } }
  }, kind)
}

async function figureFrame(page: Page) {
  const handle = await sigWindow(page).locator('iframe').first().elementHandle()
  const frame = await handle!.contentFrame()
  const box = await sigWindow(page).locator('iframe').first().boundingBox()
  return { frame: frame!, box: box! }
}

async function setThreshold(page: Page, value: string) {
  await page.getByTestId('fv-threshold').evaluate((el: HTMLInputElement, v: string) => {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!
    setter.call(el, v)
    el.dispatchEvent(new Event('input', { bubbles: true }))
  }, value)
}

/** A screenshot of the pattern window and the caret only. */
async function shot(page: Page, name: string) {
  const a = (await sigWindow(page).boundingBox())!
  const b = (await page.getByTestId('find-vectors-wizard').boundingBox())!
  const x = Math.min(a.x, b.x) - 6, y = Math.min(a.y, b.y) - 6
  await page.screenshot({ path: join(SHOTS, name), clip: {
    x, y, width: Math.max(a.x + a.width, b.x + b.width) - x + 12,
    height: Math.max(a.y + a.height, b.y + b.height) - y + 12 } })
}

const near = (a: { x: number, y: number }, list: { x: number, y: number }[], r: number) =>
  list.some((b) => Math.hypot(a.x - b.x, a.y - b.y) < r)

test('double-click marks, adapt, revert', async () => {
  const { page, backend } = ctx
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await expect(page.getByTestId('fv-teach')).toBeVisible()
  await backend.waitForLog('neural calibration:', 90_000)
  await page.waitForTimeout(3000)        // the caret adopts the calibration and re-tunes

  const { frame, box } = await figureFrame(page)
  await expect.poll(async () => (await blobs(frame, 'red')).blobs.length, {
    timeout: 60_000, message: 'the preview drew no circles',
  }).toBeGreaterThan(1)
  await page.waitForTimeout(1500)
  const disks = (await blobs(frame, 'red')).blobs          // what the default threshold finds

  // A user hunting faint disks drops the threshold, and circles appear on
  // nothing: those, and only those, are what they mark as wrong.
  await setThreshold(page, '0.1')
  await expect.poll(async () => (await blobs(frame, 'red')).blobs.length, {
    timeout: 30_000, message: 'a lower threshold added no circles',
  }).toBeGreaterThan(disks.length + 1)
  await page.waitForTimeout(1500)
  const low = (await blobs(frame, 'red')).blobs
  const frameRect = (await blobs(frame, 'red')).rect
  await shot(page, '01-low-threshold.png')

  // "A disk is here" on empty pattern draws a green ring; double-clicking the
  // ring again removes it, and with no marks left nothing is refitted.
  let empty = { x: frameRect.left + frameRect.width / 2, y: frameRect.top + frameRect.height / 2 }
  let room = -1
  for (let i = 1; i < 10; i++) {
    for (let j = 1; j < 10; j++) {
      const p = { x: frameRect.left + frameRect.width * (0.1 + 0.08 * i),
        y: frameRect.top + frameRect.height * (0.1 + 0.08 * j) }
      const d = Math.min(...low.map((b) => Math.hypot(b.x - p.x, b.y - p.y)))
      if (d > room) { room = d; empty = p }
    }
  }
  await page.mouse.dblclick(box.x + empty.x, box.y + empty.y)
  await expect(page.getByTestId('fv-teach-count')).toContainText('0 wrong · 1 missed', { timeout: 20_000 })
  await expect.poll(async () => (await blobs(frame, 'green')).blobs, {
    timeout: 20_000, message: 'no green ring where empty pattern was double-clicked',
  }).toEqual(expect.arrayContaining([expect.anything()]))
  expect(near(empty, (await blobs(frame, 'green')).blobs, 12)).toBeTruthy()
  await shot(page, '02-disk-mark.png')
  await page.mouse.dblclick(box.x + empty.x, box.y + empty.y)
  await expect(page.getByTestId('fv-teach-count')).toContainText('double-click a wrong circle', { timeout: 20_000 })
  await expect.poll(async () => (await blobs(frame, 'green')).count, { timeout: 20_000 }).toBe(0)

  // Circles that only appear at the low threshold, on nothing, are what a user
  // marks wrong. The circles are read again before every click: a refit that
  // lands between marks repaints them.
  const inside = (b: { x: number, y: number }) => b.x > frameRect.left + 20
    && b.x < frameRect.left + frameRect.width - 20
    && b.y > frameRect.top + 20 && b.y < frameRect.top + frameRect.height - 20
  const marked: { x: number, y: number }[] = []
  for (let k = 0; k < 4; k++) {
    const now = (await blobs(frame, 'red')).blobs
    const sizes = now.map((b) => b.n).sort((a, b) => a - b)
    // One whole ring: not cut by the edge (smaller) and not touching another
    // ring (larger) — either way its pixel centroid is not the circle's centre.
    const ring = sizes[Math.floor(sizes.length / 2)]
    const next = now.find((b) => !near(b, disks, 10) && !near(b, marked, 10)
      && b.n >= 0.8 * ring && b.n <= 1.25 * ring && inside(b))
    if (!next) break
    await page.mouse.dblclick(box.x + next.x, box.y + next.y)
    marked.push(next)
    await expect(page.getByTestId('fv-teach-count')).toContainText(`${marked.length} wrong · 0 missed`,
      { timeout: 20_000 })
    if (marked.length === 1) await shot(page, '02-marked.png')
    // Let the refit this mark starts land and repaint before reading the circles
    // again (a user looks before the next click; a script has to wait for it).
    await expect(page.getByTestId('fv-teach-count')).toContainText('adapting', { timeout: 10_000 })
    if (marked.length === 1) await shot(page, '03-retraining.png')
    await expect(page.getByTestId('fv-teach-count')).not.toContainText('adapting', { timeout: 60_000 })
    await page.waitForTimeout(800)
  }
  expect(marked.length).toBeGreaterThan(0)
  const junk = marked
  await expect.poll(async () => (await blobs(frame, 'orange')).blobs.length, {
    timeout: 20_000, message: 'no orange ✕ where the circles were double-clicked',
  }).toBeGreaterThanOrEqual(junk.length)
  const crosses = (await blobs(frame, 'orange')).blobs
  for (const j of junk) expect(near(j, crosses, 18)).toBeTruthy()

  // The debounced refit lands on its own; the caret takes the taught model and
  // the marked circles are gone while the disks the default threshold found stay.
  await expect(page.getByTestId('fv-teach-gauge')).toBeVisible({ timeout: 120_000 })
  await expect(page.getByTestId('fv-status')).toContainText(/Adapted in/, { timeout: 30_000 })
  await expect(page.getByTestId('fv-teach-count')).not.toContainText('adapting', { timeout: 30_000 })
  await expect.poll(async () => {
    const now = (await blobs(frame, 'red')).blobs
    return junk.filter((j) => near(j, now, 6)).length
  }, { timeout: 30_000, message: 'the marked circles are still drawn after adapting' }).toBe(0)
  await page.waitForTimeout(1000)
  const after = (await blobs(frame, 'red')).blobs
  const kept = disks.filter((d) => near(d, after, 10)).length
  console.log(`circles: default ${disks.length}, low threshold ${low.length}, marked ${junk.length}, `
    + `after adapting ${after.length}, default-threshold disks kept ${kept}/${disks.length}`)
  // How many of the other disks survive is the MODEL's behaviour at a low
  // threshold (the synthetic grains' weak reflections look much like what
  // was marked), not the wiring this spec checks — so it is reported, not gated.
  await shot(page, '04-adapted.png')

  // The taught model is a saved model of this dataset's own, in the Model list.
  await page.getByTestId('fv-model').click()
  await expect(page.locator('[data-testid^="fv-model-opt-spotunet-taught-"]')).toHaveCount(1)
  {
    const a = (await sigWindow(page).boundingBox())!
    const list = (await page.locator('[data-testid^="fv-model-opt-"]').last().boundingBox())!
    const w = (await page.getByTestId('find-vectors-wizard').boundingBox())!
    const x = Math.min(a.x, w.x) - 6, y = a.y - 6
    await page.screenshot({ path: join(SHOTS, '05-saved-model-in-list.png'), clip: {
      x, y, width: Math.max(list.x + list.width, w.x + w.width) - x + 12,
      height: Math.max(list.y + list.height, w.y + w.height) - y + 12 } })
  }
  await page.keyboard.press('Escape')

  // Revert: no marks, no taught model, no ✕ left, and the low-threshold circles return.
  await page.getByTestId('fv-adapt-revert').click()
  await expect(page.getByTestId('fv-teach-count')).toContainText('double-click a wrong circle', { timeout: 20_000 })
  await expect(page.getByTestId('fv-teach-gauge')).toHaveCount(0)
  await expect.poll(async () => (await blobs(frame, 'orange')).count, { timeout: 20_000 }).toBe(0)
  await expect.poll(async () => (await blobs(frame, 'red')).blobs.length, {
    timeout: 30_000, message: 'reverting did not restore the original detector',
  }).toBeGreaterThanOrEqual(low.length - 1)
  await shot(page, '06-reverted.png')
  // The backend reaches Playwright only through its log; an adapt that failed
  // there would still leave this page looking reverted.
  const lines = backend.logBuffer as string[]
  console.log(lines.filter((l) => /WARNING|ERROR/.test(l)).slice(-10).join('\n'))
  const problems = lines.filter((l) => /WARNING|ERROR|Traceback/.test(l) && /adapt|teach|models/i.test(l))
  expect(problems, problems.join('\n')).toEqual([])
  ctx.assertNoJsErrors()
})
