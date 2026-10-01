/**
 * report_replace_image.spec.ts — swap a picture that is ALREADY in the report.
 *
 * The backend verbs are unit-tested; what only the app can show is the RENDERER
 * wiring, because that is what was missing. An image cell carried only its
 * reorder drag handlers and a split cell's drop zone existed solely while the
 * figure side was EMPTY, so a file dragged onto an existing picture fell through
 * to the sidebar body and was APPENDED as a new cell below it.
 *
 * The drop is synthesized with a real DataTransfer carrying a real File, and
 * dispatched at the element — the same events the browser fires — so the
 * handlers under test are the ones a user hits.
 *
 * Run:
 *   npx playwright test tests/report_replace_image.spec.ts --project=electron \
 *     --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdirSync, readFileSync } from 'fs'
const { launchApp, backendAction, backendErrorLines } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'replace_image_shots')
const PHOTO = join(__dirname, 'fixtures', 'split-photo.png')
const PHOTO_URL =
  'data:image/png;base64,' + readFileSync(PHOTO).toString('base64')

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

/** Dispatch a real dragover+drop carrying one PNG file at `selector`. */
async function dropPngOn(page: any, selector: string, tint: number) {
  await page.evaluate(async ({ sel, tint }: { sel: string; tint: number }) => {
    // A tiny but VALID png whose bytes differ per `tint`, so the resulting data
    // URL is provably a different image rather than the same one re-sent.
    const cv = document.createElement('canvas')
    cv.width = 24; cv.height = 24
    const g = cv.getContext('2d')!
    g.fillStyle = `rgb(${tint},20,40)`
    g.fillRect(0, 0, 24, 24)
    const blob: Blob = await new Promise(res => cv.toBlob(b => res(b!), 'image/png'))
    const file = new File([blob], 'replacement.png', { type: 'image/png' })

    const dt = new DataTransfer()
    dt.items.add(file)
    const el = document.querySelector(sel)
    if (!el) throw new Error(`no element for ${sel}`)
    for (const type of ['dragenter', 'dragover', 'drop']) {
      el.dispatchEvent(new DragEvent(type, {
        bubbles: true, cancelable: true, dataTransfer: dt,
      }))
    }
  }, { sel: selector, tint })
}

test('a dropped PNG replaces an existing picture instead of stacking below it',
     async () => {
  const { page } = ctx

  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', {})
  await backendAction(page, 'report_add_image_cell', {
    image_b64: PHOTO_URL, image_ext: 'png', caption: 'original',
  })

  const box = page.locator('[data-testid^="report-imgcell-box-"]').first()
  await expect(box).toBeVisible({ timeout: 30_000 })
  const cellId = (await box.getAttribute('data-testid'))!
    .replace('report-imgcell-box-', '')
  const img = page.getByTestId(`report-imgcell-img-${cellId}`)
  const before = await img.getAttribute('src')
  const cellsBefore = await page.locator('[data-report-cell="1"]').count()
  await page.screenshot({ path: join(SHOTS, '01-before.png') })

  await dropPngOn(page, `[data-testid="report-imgcell-box-${cellId}"]`, 200)

  // The SAME cell now shows a DIFFERENT image.
  await expect
    .poll(async () => img.getAttribute('src'),
          { timeout: 20_000, message: 'the picture never changed' })
    .not.toBe(before)
  expect(await page.locator('[data-report-cell="1"]').count(),
         'replacing must not append a second cell').toBe(cellsBefore)
  // Identity survives the swap — this is why it replaces rather than re-creates.
  await expect(page.getByTestId(`report-imgcell-caption-${cellId}`))
    .toContainText('original')
  await page.screenshot({ path: join(SHOTS, '02-after.png') })

  const errs = backendErrorLines(ctx.backend)
  expect(errs, `backend errors:\n${errs.join('\n')}`).toEqual([])
})

test('dropping a picture on a TEXT slide turns it into a split slide', async () => {
  const { page } = ctx

  if (!(await page.getByTestId('report-sidebar').isVisible())) {
    await page.getByTestId('toggle-report').click()
  }
  await backendAction(page, 'report_new', { type: 'presentation' })
  await page.waitForTimeout(600)
  await backendAction(page, 'report_add_cell', {
    source: '## Motivation\n\n- a point worth making\n',
  })
  const textCell = page.locator('[data-testid^="report-cell-"]').first()
  await expect(textCell).toBeVisible({ timeout: 30_000 })
  const cellId = (await textCell.getAttribute('data-testid'))!
    .replace('report-cell-', '')
  const cellsBefore = await page.locator('[data-report-cell="1"]').count()
  await page.screenshot({ path: join(SHOTS, '04-text-before.png') })

  await dropPngOn(page, `[data-testid="report-cell-${cellId}"]`, 140)

  // The markdown cell became a SPLIT cell — same id, so the slide is still one
  // slide and keeps whatever was riding on it.
  await expect(page.getByTestId(`report-splitcell-${cellId}`),
               'the text cell never became a split').toBeVisible({ timeout: 20_000 })
  await expect(page.getByTestId(`report-split-figure-${cellId}`).locator('img'))
    .toBeVisible({ timeout: 20_000 })
  // The prose came with it.
  await expect(page.getByTestId(`report-split-rendered-${cellId}`))
    .toContainText('a point worth making')
  expect(await page.locator('[data-report-cell="1"]').count(),
         'a layout change must not add a cell').toBe(cellsBefore)
  await page.screenshot({ path: join(SHOTS, '05-text-became-split.png') })

  const errs = backendErrorLines(ctx.backend)
  expect(errs, `backend errors:\n${errs.join('\n')}`).toEqual([])
})

// NOT COVERED HERE: dropping a file on a LIVE figure cell.
//
// The backend half is pinned by test_report_handlers.py
// (TestPhotoReplacesALiveFigure — the figure cell converts to a photo cell and
// its window is torn down). The RENDERER half — a shield mounting over the
// out-of-process iframe for an OS file drag, so the iframe cannot swallow the
// drop — is NOT verified end-to-end. An attempt using load_test_data_si_grains
// resolved a window id of 0 from the first breadcrumb (the NAVIGATOR window,
// not the signal one) and report_add_figure then produced no figure cell with
// no backend error, so the spec was measuring nothing. Getting a real painted
// signal window and a real figure cell into this fixture is the work; it is
// worth doing, and it is not done.

test("a split cell's FILLED photo side accepts a replacement too", async () => {
  const { page } = ctx

  // Self-sufficient: the sidebar may not be open when this test runs alone.
  if (!(await page.getByTestId('report-sidebar').isVisible())) {
    await page.getByTestId('toggle-report').click()
  }
  await backendAction(page, 'report_new', {})
  await page.waitForTimeout(600)
  await backendAction(page, 'report_add_split_cell', {
    source: '## Split\n\ntext side\n', layout: 'text-left',
  })
  const splitLoc = page.locator('[data-testid^="report-splitcell-"]').first()
  await expect(splitLoc).toBeVisible({ timeout: 30_000 })
  const splitTestId = await splitLoc.getAttribute('data-testid')
  const splitId = (splitTestId || '').replace('report-splitcell-', '')
  // Fill it first — the bug was specifically that a FILLED side had no target.
  await backendAction(page, 'report_set_cell_image', {
    cell_id: splitId, image_b64: PHOTO_URL, image_ext: 'png',
  })
  const figPane = page.getByTestId(`report-split-figure-${splitId}`)
  await expect(figPane.locator('img')).toBeVisible({ timeout: 20_000 })
  const before = await figPane.locator('img').getAttribute('src')
  const cellsBefore = await page.locator('[data-report-cell="1"]').count()

  await dropPngOn(page, `[data-testid="report-split-figure-${splitId}"] img`, 90)

  await expect
    .poll(async () => figPane.locator('img').getAttribute('src'),
          { timeout: 20_000, message: 'the split photo never changed' })
    .not.toBe(before)
  expect(await page.locator('[data-report-cell="1"]').count(),
         'replacing must not append a second cell').toBe(cellsBefore)
  await page.screenshot({ path: join(SHOTS, '03-split-replaced.png') })

  const errs = backendErrorLines(ctx.backend)
  expect(errs, `backend errors:\n${errs.join('\n')}`).toEqual([])
})
