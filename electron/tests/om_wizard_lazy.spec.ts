/**
 * om_wizard_lazy.spec.ts — the staged Orientation-Mapping WIZARD, end-to-end on
 * LAZY 4D data with a real Dask cluster and the real Silver .cif:
 *   open wizard → pick .cif → Generate Library (real library + live refine
 *   overlay) → Compute Map → IPF-Z window opens.
 * The native file dialog is mocked at the main process to return the bundled
 * Silver__0011135.cif.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

const CIF = join(__dirname, '..', '..', 'spyde', 'tests', 'Silver__0011135.cif')

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
  // Mock the native .cif picker → the bundled Silver cif.
  await app.evaluate(({ ipcMain }, cif) => {
    ipcMain.removeHandler('spyde:pick-file')
    ipcMain.handle('spyde:pick-file', async () => cif)
  }, CIF)
  await page.evaluate(() => window.electron.action('load_test_data_lazy', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 60_000 },
  )
  await page.waitForTimeout(2000)
})

test.afterAll(async () => { await app?.close() })

test('staged wizard: Generate Library → Compute Map opens the IPF window (lazy, real Dask)', async () => {
  const sig = page.getByTestId('subwindow').filter({ hasNotText: 'Navigator' }).first()
  await sig.getByTestId('subwindow-titlebar').hover()             // reveal toolbar
  await sig.getByTestId('action-btn-Orientation Mapping').click()
  await expect(page.getByTestId('orientation-wizard')).toBeVisible()

  // 1 Load → pick the real cif (mocked); wait for it to resolve.
  await page.getByTestId('om-pick-cif').click()
  await expect(page.getByTestId('om-cif-list')).toContainText('Silver__0011135.cif')

  // 2 Library → Generate (real diffsims library on the lazy dataset).
  await page.getByTestId('om-tab-Library').click()
  await page.getByTestId('om-generate').click()
  // Backend reports the built library in the status bar.
  await expect(page.getByTestId('status-text'))
    .toContainText('library ready', { timeout: 60_000 })

  // 4 Run → Compute Map → a new IPF-Z window opens.
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('om-tab-Run').click()
  await page.getByTestId('om-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 90_000, message: 'orientation IPF window never opened',
  }).toBeGreaterThan(before)
  await expect(
    page.getByTestId('subwindow').filter({ hasText: 'Orientation' }).first(),
  ).toBeVisible({ timeout: 10_000 })
})
