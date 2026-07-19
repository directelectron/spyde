/**
 * report_image_cell.spec.ts — Report Builder PHOTO/IMAGE cells, end-to-end.
 *
 * No dataset needed (image cells only) → SPYDE_NO_DASK fast launch. Drives:
 *   • add an image cell (report_add_image_cell with a tiny base64 PNG) →
 *     it renders in the sidebar as an <img> with the data URL,
 *   • edit its caption (click → type → commit) → the caption persists,
 *   • enter Present mode → the image shows large on a slide.
 *
 * Screenshots to photo_cell_shots/ — each Read by the author (a blank panel
 * is a failure even when selectors pass).
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'photo_cell_shots')

// A 1×1 red PNG data URL — the smallest real PNG so the bytes round-trip
// through the backend + data URL without any faking. Its distinctive red pixel
// makes the rendered <img> visible in the screenshot.
const PNG_1x1 =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(120_000)

test.beforeAll(async () => {
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1000)
  // Dismiss the first-run welcome tour if it auto-opened (its overlay intercepts
  // pointer events, so any click through it fails otherwise).
  const tour = page.getByTestId('tour-close')
  if (await tour.count()) {
    await tour.click().catch(() => {})
    await expect(page.getByTestId('tour-overlay')).toHaveCount(0, { timeout: 5_000 }).catch(() => {})
  }
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', { type: 'presentation' })
  await expect(page.getByTestId('report-body')).toBeVisible()
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

test('1) add an image cell → it renders inline as an <img> with the data URL', async () => {
  const { page } = ctx
  await backendAction(page, 'report_add_image_cell', {
    image_b64: PNG_1x1, image_ext: 'png', caption: 'A red pixel',
  })

  // The image cell mounts with an <img> carrying the data URL.
  const img = page.locator('[data-testid^="report-imgcell-img-"]').first()
  await expect(img).toBeVisible({ timeout: 10_000 })
  const src = await img.getAttribute('src')
  expect(src?.startsWith('data:image/png;base64,')).toBe(true)

  // The caption reads what we set.
  const cap = page.locator('[data-testid^="report-imgcell-caption-"]').first()
  await expect(cap).toHaveText('A red pixel')

  await page.screenshot({ path: join(SHOTS, '01-image-cell-in-sidebar.png') })
  ctx.assertNoJsErrors()
})

test('2) edit the caption → it commits and persists', async () => {
  const { page } = ctx
  const cap = page.locator('[data-testid^="report-imgcell-caption-"]').first()
  await cap.click()
  const input = page.locator('[data-testid^="report-imgcell-caption-input-"]').first()
  await expect(input).toBeVisible()
  await input.fill('Edited caption')
  await input.press('Enter')

  // The rendered caption now shows the edit (round-tripped through the backend).
  const cap2 = page.locator('[data-testid^="report-imgcell-caption-"]').first()
  await expect(cap2).toHaveText('Edited caption', { timeout: 10_000 })
  await page.screenshot({ path: join(SHOTS, '02-caption-edited.png') })
  ctx.assertNoJsErrors()
})

test('3) Present mode shows the image on a slide', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await expect(page.getByTestId('present-mode')).toBeVisible({ timeout: 10_000 })

  // The image renders on the active slide.
  const slideImg = page.locator(
    '[data-testid="present-slide"][data-active="1"] [data-testid^="present-img-"] img')
  await expect(slideImg).toBeVisible({ timeout: 10_000 })
  const src = await slideImg.getAttribute('src')
  expect(src?.startsWith('data:image/png;base64,')).toBe(true)
  // The caption carried onto the slide.
  await expect(page.locator(
    '[data-testid="present-slide"][data-active="1"] figcaption'))
    .toHaveText('Edited caption')

  await page.screenshot({ path: join(SHOTS, '03-image-on-slide.png') })

  await page.keyboard.press('Escape')
  await expect(page.getByTestId('present-mode')).toHaveCount(0, { timeout: 5_000 })
  ctx.assertNoJsErrors()
})
