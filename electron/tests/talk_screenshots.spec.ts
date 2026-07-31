/**
 * talk_screenshots.spec.ts — capture REAL SpyDE screenshots for the
 * "SpyDE — an overview" presentation (doc/presentations/).
 *
 * Not a regression test: a capture run. Each block loads bundled synthetic data
 * the way a user would, drives one differentiating feature, and screenshots the
 * whole window into `talk_shots/`. The presentation build script embeds those
 * PNGs as report IMAGE cells.
 *
 * Run:
 *   npx playwright test tests/talk_screenshots.spec.ts --project=electron \
 *     --reporter=line --retries=0
 */
import { test } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'
const {
  launchApp, backendAction, waitForSubwindowCount, countColorPixels, sigWindow,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'talk_shots')

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(() => { mkdirSync(SHOTS, { recursive: true }) })

/** Screenshot the whole app window under `talk_shots/<name>.png`. */
async function shot(page: any, name: string) {
  await page.screenshot({ path: join(SHOTS, `${name}.png`) })
}

test('4D-STEM overview + find vectors + virtual imaging', async () => {
  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'load_test_data_si_grains')
    await waitForSubwindowCount(page, 2, 120_000)
    await page.waitForTimeout(4000)               // let the DP paint

    // 1) The core two-window live view: navigator + diffraction pattern.
    await shot(page, '01-navigator-and-dp')

    // 2) Find Diffraction Vectors — wizard open, live red peak preview on the DP.
    const sig = sigWindow(page)
    await sig.getByTestId('subwindow-title').click()
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
    await page.getByTestId('find-vectors-wizard').waitFor({ timeout: 30_000 })
    await expectRed(page, 30_000)
    await page.waitForTimeout(1200)
    await shot(page, '02-find-vectors-wizard')

    // 3) Compute across the scan → the vectors result window opens.
    const before = await page.getByTestId('subwindow').count()
    await page.getByTestId('fv-compute').click()
    await waitForSubwindowCount(page, before + 1, 180_000)
    await page.waitForTimeout(6000)
    await shot(page, '03-find-vectors-result')
  } finally {
    try { await ctx?.app?.close() } catch { /* best effort */ }
  }
})

test('virtual imaging live ROI', async () => {
  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'load_test_data_si_grains')
    await waitForSubwindowCount(page, 2, 120_000)
    await page.waitForTimeout(4000)

    const sig = sigWindow(page)
    await sig.getByTestId('subwindow-title').click()
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-Virtual Imaging').click()
    await page.waitForTimeout(600)
    await page.getByTestId('subaction-add_virtual_image').click()
    await waitForSubwindowCount(page, 3, 120_000)
    await page.waitForTimeout(6000)
    await shot(page, '04-virtual-imaging')
  } finally {
    try { await ctx?.app?.close() } catch { /* best effort */ }
  }
})

test('EELS spectrum image', async () => {
  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'load_test_data_eels')
    await waitForSubwindowCount(page, 2, 120_000)
    await page.waitForTimeout(5000)
    await shot(page, '05-eels')
  } finally {
    try { await ctx?.app?.close() } catch { /* best effort */ }
  }
})

/** Poll until the DP shows saturated-red overlay pixels (the live peak preview). */
async function expectRed(page: any, timeout: number) {
  const deadline = Date.now() + timeout
  while (Date.now() < deadline) {
    if ((await countColorPixels(page, 'red')) > 0) return
    await page.waitForTimeout(500)
  }
}
