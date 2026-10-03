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
// The second test continues from the first's app state (a named model exists).
test.describe.configure({ mode: 'serial' })

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

/** Raise the pattern's figure through the focus message: on CI a click can be
 * swallowed by another window's iframe otherwise (vom_quantem_overlay.spec.ts). */
async function raise(page: Page) {
  const id = await sigWindow(page).locator('iframe').first().getAttribute('data-testid')
  await page.evaluate((figId: string) => window.postMessage({ type: 'spyde_focus', figId }, '*'),
    id!.replace('figure-', ''))
  await page.waitForTimeout(300)
}

/** Double-click, and if no mark lands, say what the backend saw. */
async function mark(page: Page, backend: any, x: number, y: number, expected: string) {
  await raise(page)
  await page.mouse.dblclick(x, y)
  try {
    await expect(page.getByTestId('fv-teach-count')).toContainText(expected, { timeout: 20_000 })
  } catch (error) {
    console.log((backend.logBuffer as string[]).filter((l) => l.includes('fv-adapt')).slice(-10).join('\n'))
    throw error
  }
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

/** A screenshot of ``win`` (default: the pattern window) and the caret, grown to
 * take in the open model menu and its card when they are showing. */
async function shotAll(page: Page, name: string, win?: any) {
  const boxes = [(await (win ?? sigWindow(page)).boundingBox())!,
    (await page.getByTestId('find-vectors-wizard').boundingBox())!]
  for (const id of ['fv-model-card']) {
    const el = page.getByTestId(id)
    if (await el.count()) boxes.push((await el.boundingBox())!)
  }
  const menu = page.locator('[role="listbox"][aria-label="Model"]')
  if (await menu.count()) boxes.push((await menu.boundingBox())!)
  const x = Math.min(...boxes.map((b) => b.x)) - 6, y = Math.min(...boxes.map((b) => b.y)) - 6
  await page.screenshot({ path: join(SHOTS, name), clip: {
    x, y, width: Math.max(...boxes.map((b) => b.x + b.width)) - x + 12,
    height: Math.max(...boxes.map((b) => b.y + b.height)) - y + 12 } })
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
  // Inside the circles' own extent, so the click is on the image, not its margin.
  const xs = low.map((b) => b.x), ys = low.map((b) => b.y)
  const span = { left: Math.min(...xs), top: Math.min(...ys),
    width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) }
  let empty = { x: span.left + span.width / 2, y: span.top + span.height / 2 }
  let room = -1
  for (let i = 0; i <= 10; i++) {
    for (let j = 0; j <= 10; j++) {
      const p = { x: span.left + span.width * 0.1 * i, y: span.top + span.height * 0.1 * j }
      const d = Math.min(...low.map((b) => Math.hypot(b.x - p.x, b.y - p.y)))
      if (d > room) { room = d; empty = p }
    }
  }
  await mark(page, backend, box.x + empty.x, box.y + empty.y, '0 wrong · 1 missed')
  await expect.poll(async () => (await blobs(frame, 'green')).blobs, {
    timeout: 20_000, message: 'no green ring where empty pattern was double-clicked',
  }).toEqual(expect.arrayContaining([expect.anything()]))
  expect(near(empty, (await blobs(frame, 'green')).blobs, 12)).toBeTruthy()
  await shot(page, '02-disk-mark.png')
  await mark(page, backend, box.x + empty.x, box.y + empty.y, 'double-click a wrong circle')
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
    const free = (b: { x: number, y: number }) => !near(b, disks, 10) && !near(b, marked, 10) && inside(b)
    // Prefer a ring of the usual size; on a pattern with none, any free one will do.
    const next = now.find((b) => free(b) && b.n >= 0.8 * ring && b.n <= 1.25 * ring)
      ?? (marked.length === 0 ? now.find((b) => free(b) && b.n >= 0.5 * ring && b.n <= 1.6 * ring) : undefined)
    if (!next) break
    marked.push(next)
    await mark(page, backend, box.x + next.x, box.y + next.y, `${marked.length} wrong · 0 missed`)
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
  // Every marked circle is gone: training runs until each "not a disk" mark has
  // no detection on it (checked with the preview's own decode), and on these
  // patterns that is reached within the minimum 20 steps every time we measured.
  await expect(page.getByTestId('fv-status')).toContainText('all marks learned')
  await expect.poll(async () => {
    const now = (await blobs(frame, 'red')).blobs
    return junk.filter((j) => near(j, now, 6)).length
  }, { timeout: 30_000, message: 'a marked circle is still drawn after adapting' }).toBe(0)
  const nowDrawn = (await blobs(frame, 'red')).blobs
  const stillDrawn = junk.filter((j) => near(j, nowDrawn, 6)).length
  await page.waitForTimeout(1000)
  const after = (await blobs(frame, 'red')).blobs
  const kept = disks.filter((d) => near(d, after, 10)).length
  console.log(`circles: default ${disks.length}, low threshold ${low.length}, marked ${junk.length} `
    + `(still drawn ${stillDrawn}), `
    + `after adapting ${after.length}, default-threshold disks kept ${kept}/${disks.length}`)
  // How many of the other disks survive is the MODEL's behaviour at a low
  // threshold (the synthetic grains' weak reflections look much like what
  // was marked), not the wiring this spec checks — so it is reported, not gated.
  await shot(page, '04-adapted.png')

  // The fit is an unsaved draft: the Model dropdown shows it, a name field with
  // "<dataset> adapted" sits under it, and in the open menu it is a dashed row
  // after the divider. Enter in the name field keeps it as the user's model.
  const trigger = page.getByTestId('fv-model')
  const taught = page.locator('[data-testid^="fv-model-opt-spotunet-taught-"]')
  await expect(trigger).toContainText('Unsaved: test_data_si_grains adapted')
  await expect(page.getByTestId('fv-model-name-input')).toHaveValue('test_data_si_grains adapted')
  await shotAll(page, '05-draft-collapsed.png')
  await trigger.click()
  await expect(taught).toHaveCount(1)
  await expect(page.getByTestId('fv-model-divider')).toBeVisible()
  await expect(taught).toHaveAttribute('aria-selected', 'true')
  await taught.hover()
  await expect(page.getByTestId('fv-model-card')).toContainText('Not saved')
  await page.waitForTimeout(300)
  await shotAll(page, '06-draft-row-open.png')
  await page.keyboard.press('Escape')
  await expect(taught).toHaveCount(0)
  await page.getByTestId('fv-model-name-input').fill('Si grains, junk removed')
  await page.getByTestId('fv-model-name-input').press('Enter')
  await expect(trigger).toContainText('Si grains, junk removed', { timeout: 20_000 })
  await expect(page.getByTestId('fv-model-name-input')).toHaveCount(0)
  const taughtId = await trigger.getAttribute('data-value')

  // Its card: where it came from, what it was taught on, how general it still is.
  await trigger.click()
  await taught.hover()
  const card = page.getByTestId('fv-model-card')
  await expect(card).toContainText('Si grains, junk removed')
  await expect(card).toContainText('taught on')
  await expect(card).toContainText('general F1')
  await expect(card).toContainText('→')
  await expect(taught).toContainText('taught on test_data_si_grains')
  await page.waitForTimeout(300)
  await shotAll(page, '07-open-menu-card.png')
  await page.locator('[data-testid^="fv-model-opt-"]').first().hover()
  await expect(card).toContainText('SpotUNet')
  await expect(page.locator('[data-testid^="fv-model-opt-"]').first()).toContainText('built-in')
  await page.waitForTimeout(300)
  await shotAll(page, '08-vendored-card.png')

  // Keyboard: Esc closes; ↓ opens on the current model; ↑ moves above the
  // divider; Enter picks it.
  await page.keyboard.press('Escape')
  await expect(taught).toHaveCount(0)
  // Move the pointer off the menu first: a row that reappears under a resting
  // pointer gets its mouseenter, and the mouse sets the active row as it should.
  await page.mouse.move(0, 0)
  await trigger.focus()
  await page.keyboard.press('ArrowDown')
  await expect(taught).toHaveCount(1)
  await page.keyboard.press('ArrowUp')
  await page.keyboard.press('Enter')
  await expect(taught).toHaveCount(0)
  await expect(trigger).not.toHaveAttribute('data-value', taughtId!)
  await expect(trigger).toContainText('base16')          // the last built-in row, above the divider

  // Revert: no marks, no ✕ left, the original model and its low-threshold circles
  // return — and the NAMED model stays in the menu, it is the user's now.
  await page.getByTestId('fv-adapt-revert').click()
  await expect(page.getByTestId('fv-teach-count')).toContainText('double-click a wrong circle', { timeout: 20_000 })
  await expect(page.getByTestId('fv-teach-gauge')).toHaveCount(0)
  await trigger.click()
  await expect(taught).toHaveCount(1)
  await expect(taught).toHaveAttribute('aria-selected', 'false')
  await page.keyboard.press('Escape')
  await expect.poll(async () => (await blobs(frame, 'orange')).count, { timeout: 20_000 }).toBe(0)
  await expect.poll(async () => (await blobs(frame, 'red')).blobs.length, {
    timeout: 30_000, message: 'reverting did not restore the original detector',
  }).toBeGreaterThanOrEqual(low.length - 1)
  await shot(page, '09-reverted.png')
  // The backend reaches Playwright only through its log; an adapt that failed
  // there would still leave this page looking reverted.
  const lines = backend.logBuffer as string[]
  console.log(lines.filter((l) => /WARNING|ERROR/.test(l)).slice(-10).join('\n'))
  const problems = lines.filter((l) => /WARNING|ERROR|Traceback/.test(l) && /adapt|teach|models/i.test(l))
  expect(problems, problems.join('\n')).toEqual([])
  ctx.assertNoJsErrors()
})

test('a named model is offered on another dataset, renamed and deleted', async () => {
  const { page } = ctx
  await page.getByTestId('fv-close').click()
  await expect(page.getByTestId('find-vectors-wizard')).toHaveCount(0)
  await backendAction(page, 'load_test_data')
  await waitForSubwindowCount(page, 4, 120_000)
  const other = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-test_data$/ }) })
  await expect(other).toHaveCount(1)
  await other.getByTestId('subwindow-title').click()
  await other.getByTestId('subwindow-titlebar').hover()
  await other.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()

  // The model taught on the Si grains is here too, after the divider.
  const trigger = page.getByTestId('fv-model')
  const mine = page.locator('[data-testid^="fv-model-opt-spotunet-taught-"]')
  await expect(trigger).toBeVisible({ timeout: 30_000 })
  await trigger.click()
  await expect(mine).toHaveCount(1, { timeout: 30_000 })
  await expect(page.getByTestId('fv-model-divider')).toBeVisible()
  await mine.click()
  await expect(trigger).toContainText('Si grains, junk removed')
  const id = await trigger.getAttribute('data-value')
  await trigger.click()
  await mine.hover()
  await expect(page.getByTestId('fv-model-card')).toContainText('test_data_si_grains')
  await page.waitForTimeout(300)
  await shotAll(page, '10-used-on-another-dataset.png', other)

  // Rename and Delete live on the row, behind its ⋯.
  await page.getByTestId(`fv-model-more-${id}`).click()
  await page.getByTestId('fv-model-rename').click()
  await page.getByTestId('fv-model-rename-input').fill('Si grains v2')
  await shotAll(page, '11-rename.png', other)
  await page.getByTestId('fv-model-rename-input').press('Enter')
  await expect(mine).toContainText('Si grains v2', { timeout: 20_000 })
  await page.keyboard.press('Escape')
  await expect(trigger).toContainText('Si grains v2', { timeout: 20_000 })

  // Delete asks once; then the row, the divider and the selection go.
  await trigger.click()
  await page.getByTestId(`fv-model-more-${id}`).click()
  await page.getByTestId('fv-model-delete').click()
  await expect(page.getByTestId('fv-model-delete')).toHaveText('Delete?')
  await shotAll(page, '12-delete-confirm.png', other)
  await page.getByTestId('fv-model-delete').click()
  await expect(mine).toHaveCount(0, { timeout: 20_000 })
  await expect(page.getByTestId('fv-model-divider')).toHaveCount(0)
  await page.keyboard.press('Escape')
  await expect(trigger).not.toHaveAttribute('data-value', id!)
  await shotAll(page, '13-deleted.png', other)
  ctx.assertNoJsErrors()
})
