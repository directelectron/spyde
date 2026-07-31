/**
 * chunk_viewer.spec.ts — the "lazy" chip's block-layout viewer.
 *
 * dask's own `_repr_html_` block picture (what HyperSpy shows in a notebook)
 * restyled for the dock, plus the judgement dask can't make: whether a chunk
 * holds WHOLE signal frames. Driven on real lazy data because the whole feature
 * is a drawing — the numbers behind it can be right while the picture is empty.
 */
import { test, expect } from '@playwright/test'
const { launchApp, backendAction } = require('./_harness.cjs')

const SHOTS = 'chunk_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'INFO' } })
  // nav 24×24 in chunks of 8 (a 3×3 nav-chunk grid), signal axes unsplit — the
  // storage-aligned case, so the viewer should pass it.
  await backendAction(ctx.page, 'load_test_data_lazy_chunked')
  await expect(ctx.page.getByTestId('subwindow').first()).toBeVisible({ timeout: 60_000 })
  await expect(ctx.page.getByTestId('metadata-panel')).toBeVisible({ timeout: 30_000 })
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(120_000)

test('the lazy chip toggles a block diagram of the chunking', async () => {
  const { page } = ctx
  const chip = page.getByTestId('meta-Dataset-Chunks')
  await expect(chip).toHaveText('lazy')

  await chip.click()
  const viewer = page.getByTestId('chunk-viewer')
  await expect(viewer).toBeVisible()

  // The diagram is DRAWN, not just described: two spaces, and the navigation
  // one carries the 3×3 grid's interior boundaries (2 vertical + 2 horizontal).
  const svgs = viewer.locator('svg')
  await expect(svgs).toHaveCount(2)
  expect(await svgs.first().locator('line').count(),
    'the navigation grid has no chunk boundaries drawn').toBe(4)
  // …and the signal frame is ONE block — no interior lines at all.
  expect(await svgs.last().locator('line').count(),
    'the signal frame is drawn as split when it is not').toBe(0)

  // Signal axes are whole here, so the verdict must be the good one.
  await expect(page.getByTestId('chunk-verdict')).toContainText(/whole frames/i)
  await expect(viewer).toContainText('24 × 24 × 32 × 32')
  await expect(viewer).toContainText('8 × 8 × 32 × 32')
  await expect(viewer).toContainText('float32')
  await page.screenshot({ path: `${SHOTS}/01-viewer.png` })

  // Clicking the chip again closes it (the whole point of a toggle).
  await chip.click()
  await expect(viewer).toBeHidden()
  ctx.assertNoJsErrors()
})

test('an eager dataset has no viewer to open', async () => {
  const { page } = ctx
  await backendAction(page, 'load_test_data_si_grains')
  // The dock follows the ACTIVE window, so target the new dataset's own.
  await expect.poll(async () => page.getByTestId('subwindow').count(),
    { timeout: 60_000 }).toBeGreaterThan(2)
  await page.getByTestId('subwindow').last().getByTestId('subwindow-titlebar').click()
  await expect(page.getByTestId('meta-Dataset-Lazy')).toHaveText('eager', { timeout: 30_000 })

  // The chip still explains itself — through the plain field detail, not a
  // block diagram of chunks that do not exist.
  await page.getByTestId('meta-Dataset-Lazy').click()
  await expect(page.getByTestId('chunk-viewer')).toHaveCount(0)
  await expect(page.getByTestId('meta-detail')).toBeVisible()
  await page.keyboard.press('Escape')
  ctx.assertNoJsErrors()
})
