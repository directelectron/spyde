/**
 * metadata_panel.spec.ts — the dock's curated metadata summary + detail box.
 *
 * The panel deliberately does NOT show every field: duplicates (Dtype, Dim.)
 * and near-never-populated ones (Mode, Cam.) are hidden, and the Dataset group
 * collapses to a chip strip. That only works if the hidden information is still
 * reachable — which is what the caret detail box is for. So this checks both
 * halves: what the summary omits, and that clicking gets it back.
 */
import { test, expect } from '@playwright/test'
const { launchApp, backendAction } = require('./_harness.cjs')

const SHOTS = 'metadata_shots'
const ROOT = 'Root Experiment Details'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await backendAction(ctx.page, 'load_test_data_si_grains')
  await expect(ctx.page.getByTestId('subwindow').first()).toBeVisible({ timeout: 60_000 })
  await expect(ctx.page.getByTestId('metadata-panel')).toBeVisible({ timeout: 30_000 })
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(120_000)

test('the summary is curated: chips for the dataset, no duplicated fields', async () => {
  const { page } = ctx
  const panel = page.getByTestId('metadata-panel')

  // Dataset is a chip strip, not four labelled rows.
  await expect(page.getByTestId('dataset-strip')).toBeVisible()
  await expect(page.getByTestId(`meta-Dataset-Shape`)).toBeVisible()
  await expect(page.getByTestId(`meta-Dataset-Dtype`)).toBeVisible()

  // All four instrument fields, on ONE row (same y, four distinct x).
  const ys = await Promise.all(['Mag', 'Acc. Volt.', 'Cam. Len.', 'Conv. Angle'].map(
    async (p) => (await page.getByTestId(`meta-Instrument Metadata-${p}`).boundingBox())!))
  expect(new Set(ys.map((b) => Math.round(b.y))).size, 'instrument fields are not on one row').toBe(1)
  expect(new Set(ys.map((b) => Math.round(b.x))).size, 'instrument fields overlap').toBe(4)

  // Hidden from the summary: the dtype/shape duplicates and the rare detector
  // fields. (They are asserted present in the ⋯ box by the next test.)
  for (const prop of ['Dtype', 'Dim.', 'Mode', 'Cam.']) {
    await expect(page.getByTestId(`meta-${ROOT}-${prop}`),
      `${prop} is still in the summary`).toHaveCount(0)
  }
  // The Movie group folded into the strip — no heading of its own.
  await expect(panel).not.toContainText('Movie / In-Situ')

  // How tall the CONTENT is, not the box — the panel is the dock's elastic
  // child, so its box (and therefore scrollHeight) is just whatever height the
  // pinned sections left over.
  const content = await panel.evaluate((el) => {
    const top = el.getBoundingClientRect().top
    return Math.max(...Array.from(el.children).map(
      (k) => k.getBoundingClientRect().bottom)) - top
  })
  expect(content, 'the metadata summary grew back').toBeLessThan(140)
  await page.getByTestId('plot-control-dock').screenshot({ path: `${SHOTS}/01-summary.png` })
  ctx.assertNoJsErrors()
})

test('clicking a field opens a detail box with its description and key', async () => {
  const { page } = ctx
  await page.getByTestId('meta-Instrument Metadata-Mag').click()

  const detail = page.getByTestId('meta-detail')
  await expect(detail).toBeVisible()
  // The YAML description, shipped with the metadata message.
  await expect(detail).toContainText('magnification')
  // …and where hyperspy stores it.
  await expect(detail).toContainText('Acquisition_instrument.TEM.magnification')
  // A writable field offers the editor here, sized properly, instead of an
  // input crammed into a quarter-width cell.
  await expect(page.getByTestId('meta-Instrument Metadata-Mag-input')).toBeVisible()
  await page.getByTestId('plot-control-dock').screenshot({ path: `${SHOTS}/02-field-detail.png` })

  await page.keyboard.press('Escape')
  await expect(detail).toBeHidden()
  ctx.assertNoJsErrors()
})

test('the ⋯ box reaches the fields the summary hides', async () => {
  const { page } = ctx
  await page.getByTestId(`meta-more-${ROOT}`).click()

  const detail = page.getByTestId('meta-detail')
  await expect(detail).toBeVisible()
  for (const prop of ['Name', 'Dtype', 'Dim.', 'Mode', 'Cam.']) {
    await expect(page.getByTestId(`meta-detail-row-${prop}`),
      `${prop} is unreachable — hidden from the summary AND missing from the box`)
      .toBeVisible()
  }
  await page.getByTestId('plot-control-dock').screenshot({ path: `${SHOTS}/03-group-box.png` })

  // Drilling into one shows its detail; a derived field says so instead of
  // offering an editor that cannot commit.
  await page.getByTestId('meta-detail-row-Dim.').click()
  await expect(detail).toContainText('Dimensions of the collected data')
  await expect(page.getByTestId(`meta-${ROOT}-Dim.-input`)).toHaveCount(0)
  await expect(detail).toContainText('read only')

  await page.keyboard.press('Escape')
  await expect(detail).toBeHidden()
  ctx.assertNoJsErrors()
})

test('an edit made in the detail box commits', async () => {
  const { page } = ctx
  const cell = page.getByTestId('meta-Instrument Metadata-Cam. Len.')
  await cell.click()
  const input = page.getByTestId('meta-Instrument Metadata-Cam. Len.-input')
  await expect(input).toBeVisible()
  // Unset fields must pre-fill EMPTY, not with the "-- mm" display string.
  expect(await input.inputValue()).toBe('')
  await input.fill('150')
  await input.press('Enter')

  // Round-trips through the backend and comes back with its unit.
  await expect.poll(async () => (await cell.textContent())?.trim(),
    { message: 'the committed value never reached the cell' }).toContain('150')
  await expect(page.getByTestId('meta-detail')).toBeHidden()

  // Reopening pre-fills the RAW value — the display string carries the unit and
  // would fail the backend's float parse if sent back.
  await cell.click()
  expect(await page.getByTestId('meta-Instrument Metadata-Cam. Len.-input').inputValue())
    .toBe('150.0')
  await page.keyboard.press('Escape')
  await page.getByTestId('plot-control-dock').screenshot({ path: `${SHOTS}/04-after-edit.png` })
  ctx.assertNoJsErrors()
})
