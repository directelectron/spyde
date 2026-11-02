/**
 * find_vectors_center_beam.spec.ts — Find Vectors centers its result on the
 * zero beam ("Center on zero beam", on by default).
 *
 * The bundled Si-grains scan loads with the beam already at the origin, so the
 * spec first zeroes the detector offsets through the Axes table, the way a
 * scan straight off a detector reads (k = 0 at the corner pixel). Find Vectors
 * must then move its result so the beam reads (0, 0) again: the result
 * window's detector offsets come back negative, half the detector across.
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'fv_center_beam_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await backendAction(ctx.page, 'load_test_data_si_grains')
  await waitForSubwindowCount(ctx.page, 2, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(240_000)

/** Type *value* into an Axes-table cell and commit it with Enter. */
async function editAxisCell(page, testid: string, value: string) {
  await page.getByTestId(testid).click()
  const input = page.getByTestId(`${testid}-input`)
  await input.fill(value)
  await input.press('Enter')
}

const signalOffsets = (page) => Promise.all([2, 3].map((index) =>
  page.getByTestId(`axis-${index}-offset`).innerText()))

test('the found vectors come back with the zero beam at the origin', async () => {
  const { page } = ctx

  const source = sigWindow(page)
  await source.getByTestId('subwindow-title').click()
  await expect(page.getByTestId('axes-table')).toBeVisible()

  // An uncentered detector: k = 0 at the corner pixel.
  await editAxisCell(page, 'axis-2-offset', '0')
  await editAxisCell(page, 'axis-3-offset', '0')
  await expect.poll(() => signalOffsets(page)).toEqual(['0.00', '0.00'])
  await page.screenshot({ path: `${SHOTS}/01-uncentered-detector.png` })

  await source.getByTestId('subwindow-titlebar').hover()
  await source.getByTestId('action-btn-Find Diffraction Vectors').click()
  const wizard = page.getByTestId('find-vectors-wizard')
  await expect(wizard).toBeVisible()
  await expect(page.getByTestId('fv-center-beam')).toBeChecked()
  await expect(page.getByTestId('fv-compute')).toBeEnabled({ timeout: 60_000 })
  await wizard.screenshot({ path: `${SHOTS}/02-caret-center-on-zero-beam.png` })
  await page.screenshot({ path: `${SHOTS}/03-caret-in-context.png` })

  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'vectors result window never opened',
  }).toBeGreaterThan(before)
  await ctx.backend.waitForLog('[fv-batch] finalized', 120_000)
  await ctx.backend.waitForLog('[fv-batch] Zero beam centered', 30_000)
  await expect(page.getByTestId('status-text')).toContainText('Zero beam centered')
  await page.screenshot({ path: `${SHOTS}/04-status-zero-beam-centered.png` })

  // The result window's detector axes put the beam back at the origin: half
  // the detector across, negative.
  const result = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .last()
  await result.getByTestId('subwindow-title').click()
  await expect.poll(async () => (await signalOffsets(page))
    .every((text) => Number(text) < 0), {
    timeout: 15_000, message: 'result detector offsets did not move the beam to the origin',
  }).toBe(true)
  await page.screenshot({ path: `${SHOTS}/05-result-axes-origin-on-beam.png` })
  await page.getByTestId('axes-section').screenshot({ path: `${SHOTS}/06-result-axes-table.png` })

  ctx.assertNoJsErrors()
})
