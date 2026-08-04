/**
 * present_layout.spec.ts — Present-mode slide LAYOUT, against the real app.
 *
 * Three things that a green typecheck and a passing selector cannot see, so each
 * one is screenshotted and the pixels are looked at:
 *
 *  1. STACKED SPLITS. `split_layout` has four values, but Present mode used to
 *     test it with a bare `!== 'text-right'` — so `text-top` / `text-bottom`
 *     round-tripped through the document perfectly and then rendered SIDE BY
 *     SIDE. The editor offered a layout the deck could not show.
 *  2. TEXT SLIDES FILL THE STAGE. A heading plus a few bullets used to render at
 *     prose size, vertically centred in a 60rem column: a small block adrift in
 *     a large dark rectangle. Sparse slides now scale up; dense ones must NOT
 *     (that is what keeps them from overflowing into the pager).
 *  3. A DROPPED PNG FILLS THE SLOT. `report_set_cell_image` puts a photo into an
 *     EXISTING split cell rather than appending a new cell below it.
 *
 * The deck is built through the real backend verbs rather than a fixture file,
 * so the wiring under test is the wiring a user drives.
 *
 * Run:
 *   npx playwright test tests/present_layout.spec.ts --project=electron \
 *     --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdirSync, readFileSync } from 'fs'
const { launchApp, backendAction, backendErrorLines } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'present_layout_shots')

// A 480×300 PNG — a blue disc in a bordered navy panel. Deliberately NOT a 1×1
// pixel: the screenshot is the actual verification here, and a 1-pixel image
// satisfies every assertion while showing nothing, so it would prove the slot
// was filled without proving anything was DRAWN in it.
const PHOTO = join(__dirname, 'fixtures', 'split-photo.png')
const PHOTO_URL =
  'data:image/png;base64,' + readFileSync(PHOTO).toString('base64')

// Slide 1: sparse — a heading and four short bullets. THE case from the report.
const SPARSE = '## Motivation\n\n- Code base solutions are good, but…\n' +
  '- Sometimes all you want is a black box\n' +
  '- Black boxes are fine — if you can open them\n' +
  '- Data analysis software has to be tested\n'
// Slide 2: dense — long enough that scaling it up would overflow the stage.
const DENSE = '## A dense slide\n\n' + Array.from({ length: 7 }, (_, i) =>
  `- Bullet ${i + 1}: a deliberately long line of body copy that exists purely ` +
  'to push this slide past the density threshold so it keeps prose sizing.\n').join('')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  await ctx.page.waitForTimeout(1500)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

test('stacked splits render, sparse text fills, a dropped PNG fills the slot', async () => {
  const { page } = ctx

  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', { type: 'presentation' })

  await backendAction(page, 'report_add_cell', { source: SPARSE })
  await backendAction(page, 'report_add_cell', { source: DENSE, slide_break: true })
  await backendAction(page, 'report_add_split_cell', {
    source: '## Stacked\n\nText above, picture below.\n',
    layout: 'text-top', slide_break: true,
  })

  await expect
    .poll(async () => page.getByTestId(/^report-slide-\d+$/).count(),
          { timeout: 60_000, message: 'the seeded deck produced no slides' })
    .toBe(3)

  // ── 3. the dropped-PNG path: fill the split's slot, don't append ───────────
  // The split cell carries its id in its testid; the empty figure side shows a
  // dropzone until something fills it.
  const splitTestId = await page.locator('[data-testid^="report-splitcell-"]')
    .first().getAttribute('data-testid')
  const splitId = (splitTestId || '').replace('report-splitcell-', '')
  expect(splitId, 'could not resolve the split cell id').not.toBe('')
  await expect(page.getByTestId(`report-split-dropzone-${splitId}`)).toBeVisible()

  const cellsBefore = await page.locator('[data-report-cell="1"]').count()
  await backendAction(page, 'report_set_cell_image', {
    cell_id: splitId, image_b64: PHOTO_URL, image_ext: 'png',
  })
  // The dropzone gives way to the picture, IN the same cell.
  await expect(page.getByTestId(`report-split-dropzone-${splitId}`))
    .toBeHidden({ timeout: 15_000 })
  expect(await page.locator('[data-report-cell="1"]').count(),
         'filling a slot must not append a new cell').toBe(cellsBefore)

  // ── Present ────────────────────────────────────────────────────────────────
  await page.getByTestId('report-present').click()
  const stage = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(stage).toBeVisible({ timeout: 30_000 })
  await page.waitForTimeout(1200)

  // ── 2. sparse fills, dense does not ───────────────────────────────────────
  await page.screenshot({ path: join(SHOTS, '01-sparse-fill.png') })
  expect(await stage.getAttribute('data-fill'),
         'a sparse text slide should take the large fill tier').toBe('lg')
  const sparsePx = await stage.locator('.present-md').first()
    .evaluate(el => parseFloat(getComputedStyle(el).fontSize))

  await page.keyboard.press('ArrowRight')
  await page.waitForTimeout(700)
  await page.screenshot({ path: join(SHOTS, '02-dense-base.png') })
  expect(await stage.getAttribute('data-fill'),
         'a dense text slide must keep prose sizing').toBe('base')
  const densePx = await stage.locator('.present-md').first()
    .evaluate(el => parseFloat(getComputedStyle(el).fontSize))

  // The whole point: the sparse slide is VISIBLY bigger, not marginally.
  expect(sparsePx, `sparse ${sparsePx}px vs dense ${densePx}px`)
    .toBeGreaterThan(densePx * 1.25)

  // A scaled-up slide that overflows is a regression, not a feature.
  const denseOverflow = await stage.evaluate((el: HTMLElement) =>
    el.scrollHeight - el.clientHeight)
  expect(denseOverflow, 'the dense slide overflows its stage').toBeLessThanOrEqual(2)

  // ── 1. the stacked split ──────────────────────────────────────────────────
  await page.keyboard.press('ArrowRight')
  await page.waitForTimeout(700)
  await page.screenshot({ path: join(SHOTS, '03-split-text-top.png') })

  const split = stage.locator('[data-testid^="present-split-"]')
  await expect(split).toBeVisible()
  expect(await split.getAttribute('data-layout'),
         'the split lost its layout on the way to the slide').toBe('text-top')

  // The photo really renders on the slide, at a size worth looking at — not a
  // collapsed 0-height box that every geometry assertion below would still pass.
  const img = split.locator('img')
  await expect(img).toBeVisible()
  const box = await img.boundingBox()
  expect(box!.height, 'the dropped photo rendered too small to see')
    .toBeGreaterThan(120)

  // ONE grid column is what "stacked" means. Two columns here is exactly the
  // old bug: the document said text-top and the slide drew text-left.
  const cols = await split.evaluate(
    (el: HTMLElement) => getComputedStyle(el).gridTemplateColumns)
  expect(cols.trim().split(/\s+/).length,
         `stacked split rendered ${cols} — expected a single column`).toBe(1)

  // The text really is ABOVE the picture, not merely in a one-column grid.
  const order = await split.evaluate((el: HTMLElement) => {
    const kids = Array.from(el.children) as HTMLElement[]
    const text = kids.find(k => k.querySelector('h2'))
    const fig = kids.find(k => k.querySelector('img') || k !== text)
    return { textTop: text?.getBoundingClientRect().top ?? 0,
             figTop: fig?.getBoundingClientRect().top ?? 0 }
  })
  expect(order.figTop, 'text-top must put the text above the figure')
    .toBeGreaterThan(order.textTop)

  await page.keyboard.press('Escape')
  await expect(stage).toBeHidden({ timeout: 15_000 })

  const errs = backendErrorLines(ctx.backend)
  expect(errs, `backend errors:\n${errs.join('\n')}`).toEqual([])
})
