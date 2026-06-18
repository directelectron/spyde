/**
 * vector_vi_lazy.spec.ts — Vector Virtual Imaging end-to-end with a real Dask
 * backend. Reuses the multi-VI sub-toolbar UX on a vectors-image window:
 *   load_test_vectors (Find Vectors result tree) → open the Vector Virtual
 *   Imaging sub-toolbar → "＋" adds a detector ROI VI → a new output window
 *   opens and a coloured VI chip is listed.
 *
 * `load_test_vectors` is a test-only backend action that builds the vectors
 * tree directly (no picker / wizard), so the vectors-image window opens on top.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env },   // real LocalCluster + client
  })
  let daskReady = false
  app.process().stdout?.on('data', (d: Buffer) => {
    if (String(d).includes('Dask cluster ready')) daskReady = true
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  for (let i = 0; i < 80 && !daskReady; i++) await page.waitForTimeout(500)
  await page.evaluate(() => window.electron.action('load_test_vectors', {}))
  // source nav+sig (2) + vectors nav+sig (2) = 4 windows.
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 4,
    { timeout: 60_000 },
  )
  await page.waitForTimeout(2500)
})

test.afterAll(async () => { await app?.close() })

test('Vector Virtual Imaging: ＋ adds a vector VI window from the found vectors', async () => {
  // Both vectors windows share the title "— Vectors"; the SIGNAL one (not the
  // count-map navigator) is the only one carrying the Vector Virtual Imaging
  // action, so select by the presence of that button (it's in the DOM even while
  // the hover-reveal toolbar is hidden).
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Vector Virtual Imaging') }).first()
  await vsig.getByTestId('subwindow-titlebar').click()    // raise to front
  await vsig.getByTestId('subwindow-titlebar').hover()    // reveal toolbar
  await expect(vsig.getByTestId('action-btn-Vector Virtual Imaging'))
    .toBeVisible({ timeout: 15_000 })
  await vsig.getByTestId('action-btn-Vector Virtual Imaging').click()
  await expect(page.getByTestId('sub-toolbar')).toBeVisible()

  // "＋" adds a vector VI → a new output window opens + a coloured chip appears.
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('subaction-add_vector_virtual_image').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 30_000, message: 'vector VI output window never opened',
  }).toBeGreaterThan(before)
  await expect(page.getByTestId(/^vi-icon-Vector Image 1/))
    .toBeVisible({ timeout: 10_000 })
})
