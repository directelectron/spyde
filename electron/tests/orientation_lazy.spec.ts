/**
 * orientation_lazy.spec.ts — the full Orientation Mapping workflow must run
 * end-to-end on LAZY 4-D data with a real Dask cluster: compute → open the IPF-Z
 * orientation-map window. Uses the test-only `run_test_orientation` action (a
 * built-in Al phase) so no CIF file dialog is needed.
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
  await page.evaluate(() => window.electron.action('load_test_data_lazy', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 60_000 },
  )
  await page.waitForTimeout(2000)
})

test.afterAll(async () => { await app?.close() })

// The OM compute (CIF→library build + numba-JIT'd template match over the lazy
// dataset) is slow and variable on a cold run — give it a generous budget.
test.setTimeout(180_000)

test('orientation mapping runs on lazy data and opens the IPF map window', async () => {
  const before = await page.getByTestId('subwindow').count()
  await page.evaluate(() => window.electron.action('run_test_orientation', {}))

  // The IPF-Z orientation-map window opens once compute finishes (library build
  // + template match over the lazy dataset).
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 150_000, message: 'orientation IPF window never opened',
  }).toBeGreaterThan(before)

  // Its title marks it as the orientation (IPF-Z) result.
  await expect(
    page.getByTestId('subwindow').filter({ hasText: 'Orientation' }).first(),
  ).toBeVisible({ timeout: 10_000 })

  // The IPF window has a 2D/3D toggle (a second `view:"3d"` explorer figure was
  // emitted). Switching to 3D shows the 3-D scatter iframe.
  const toggle = page.getByTestId(/^ipf-view-toggle-/).first()
  await expect(toggle).toBeVisible({ timeout: 15_000 })
  await page.getByTestId(/^ipf-view-3d-/).first().click()
  await expect(page.getByTestId(/^ipf-view-3d-/).first()).toBeVisible()

  // The native IPF density heatmap (inverse pole density function) is also
  // attached → a "PDF" toggle (view:"density"). Switching to it shows the
  // density iframe; screenshot it for the inverse-pole-density-function view.
  const pdf = page.getByTestId(/^ipf-view-density-/).first()
  await expect(pdf).toBeVisible({ timeout: 20_000 })
  await pdf.click()
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(__dirname, '..', 'ipf_density.png') })
})
