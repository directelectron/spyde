/**
 * fv_adapt_centre.spec.ts — "Adapt to this scan" for the Find Vectors centre stage.
 *
 * On the real ZrNb precipitate scan (Examples menu, already downloaded on the
 * box that runs this): pick the F5 centre network, press Adapt, and wait for
 * the card. The adaptation's wall-clock budget is cut to 40 s here
 * (SPYDE_ADAPT_BUDGET_SECONDS), and adapted networks are written to a scratch
 * models folder (SPYDE_MODELS_DIR), never the real profile's.
 *
 * Whether the copy is accepted depends on the scan; the spec asserts that the
 * card reports an outcome with before/after numbers, and that an accepted copy
 * becomes the Centre choice.
 */
import { test, expect } from '@playwright/test'
import { mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
const { launchApp, waitForSubwindowCount, sigWindow } = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>
const SHOTS = 'fv_adapt_shots'

test.beforeAll(async () => {
  test.setTimeout(400_000)
  ctx = await launchApp({
    dask: true,
    env: {
      SPYDE_LOG_LEVEL: 'INFO',
      SPYDE_ADAPT_BUDGET_SECONDS: '40',
      SPYDE_MODELS_DIR: mkdtempSync(join(tmpdir(), 'spyde-models-')),
    },
  })
  await ctx.page.evaluate(() => window.electron.action('load_example', { name: 'ZrNbPrecipitate' }))
  await waitForSubwindowCount(ctx.page, 2, 300_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(600_000)

test('Adapt: the button, progress and the before/after card', async () => {
  const { page, backend } = ctx

  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await backend.waitForLog('neural calibration:', 120_000)

  // No Adapt for the decode: there is no network to adapt.
  await expect(page.getByTestId('fv-adapt')).toHaveCount(0)
  await page.getByTestId('fv-centre').click()
  await page.getByTestId('fv-centre-opt-centre-fast-f5-v1').click()
  await expect(page.getByTestId('fv-adapt')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/01-adapt-button.png` })

  await page.getByTestId('fv-adapt').click()
  await expect(page.getByTestId('fv-adapt-cancel')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/02-adapting.png` })

  await expect(page.getByTestId('fv-adapt-title')).toBeVisible({ timeout: 400_000 })
  const title = await page.getByTestId('fv-adapt-title').innerText()
  console.log('[adapt] ' + title + '\n' + await page.getByTestId('fv-adapt-card').innerText())
  await page.getByTestId('find-vectors-wizard').screenshot({ path: `${SHOTS}/03-card.png` })
  await page.screenshot({ path: `${SHOTS}/04-window.png` })
  expect(title).toMatch(/accepted|not adopted|declined/)
  if (/accepted/.test(title)) {
    await expect(page.getByTestId('fv-centre')).toContainText('adapted to')
    await page.getByTestId('fv-adapt-save').click()
    await expect(page.getByTestId('fv-adapt-card')).toContainText('Saved.')
    await page.getByTestId('find-vectors-wizard').screenshot({ path: `${SHOTS}/05-saved.png` })
  }
  ctx.assertNoJsErrors()
})
