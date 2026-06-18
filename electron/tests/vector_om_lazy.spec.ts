/**
 * vector_om_lazy.spec.ts — Vector Orientation Mapping end-to-end with a real
 * Dask backend, reusing the staged OM wizard pattern on a vectors-image window:
 *   load_test_vectors → open the Vector Orientation Mapping wizard → pick the
 *   real Silver .cif (mocked) → Generate Library → Compute Maps → an
 *   Orientation (IPF-Z) window + 3 strain windows open.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

const CIF = join(__dirname, '..', '..', 'spyde', 'tests', 'Silver__0011135.cif')

let app: ElectronApplication
let page: Page

// Count strongly-coloured marker pixels across every figure iframe. The live
// refine overlay draws measured vectors red (#ff3030) and the fitted template
// green (#30ff60); the grayscale DP has neither.
async function colorPixels(kind: 'red' | 'green'): Promise<number> {
  let total = 0
  for (const frame of page.frames()) {
    try {
      total += await frame.evaluate((k) => {
        let n = 0
        for (const c of Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]) {
          const ctx = c.getContext('2d')
          if (!ctx || !c.width || !c.height) continue
          const d = ctx.getImageData(0, 0, c.width, c.height).data
          for (let p = 0; p < d.length; p += 4) {
            if (k === 'red' && d[p] > 120 && d[p + 1] < 90 && d[p + 2] < 90) n++
            if (k === 'green' && d[p + 1] > 120 && d[p] < 130 && d[p + 2] < 150) n++
          }
        }
        return n
      }, kind)
    } catch { /* detached frame */ }
  }
  return total
}

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
  await app.evaluate(({ ipcMain }, cif) => {
    ipcMain.removeHandler('spyde:pick-file')
    ipcMain.handle('spyde:pick-file', async () => cif)
  }, CIF)
  await page.evaluate(() => window.electron.action('load_test_vectors', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 4,
    { timeout: 60_000 },
  )
  await page.waitForTimeout(2500)
})

test.afterAll(async () => { await app?.close() })

test('Vector Orientation Mapping: Generate → Compute opens IPF + strain windows', async () => {
  // The vectors-image SIGNAL window carries the Vector Orientation Mapping action.
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Vector Orientation Mapping') }).first()
  await vsig.getByTestId('subwindow-titlebar').click()
  await vsig.getByTestId('subwindow-titlebar').hover()
  await vsig.getByTestId('action-btn-Vector Orientation Mapping').click()
  await expect(page.getByTestId('vector-orientation-wizard')).toBeVisible()

  // 1 Load → pick the real cif (mocked); wait for the async picker to resolve.
  await page.getByTestId('vom-pick-cif').click()
  await expect(page.getByTestId('vom-pick-cif')).toHaveText('Silver__0011135.cif')

  // 2 Library → Generate (real diffsims library).
  await page.getByTestId('vom-tab-Library').click()
  await page.getByTestId('vom-generate').click()
  await expect(page.getByTestId('status-text'))
    .toContainText('library ready', { timeout: 60_000 })

  // Generating the library activates the LIVE refine overlay: the measured
  // vectors (red) + the fitted template (green) appear on the vectors DP.
  await expect.poll(() => colorPixels('green'), {
    timeout: 30_000, message: 'fitted template (green) never drawn on the DP',
  }).toBeGreaterThan(0)
  expect(await colorPixels('red')).toBeGreaterThan(0)

  // 3 Run → Compute Maps → IPF-Z + εxx/εyy/εxy = 4 new windows.
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('vom-tab-Run').click()
  await page.getByTestId('vom-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'orientation/strain windows never opened',
  }).toBeGreaterThanOrEqual(before + 4)
  await expect(
    page.getByTestId('subwindow').filter({ hasText: 'Orientation' }).first(),
  ).toBeVisible({ timeout: 10_000 })
  await expect(
    page.getByTestId('subwindow').filter({ hasText: 'εxx' }).first(),
  ).toBeVisible({ timeout: 10_000 })

  // The IPF orientation window has the 2D/3D explorer toggle.
  const toggle = page.getByTestId(/^ipf-view-toggle-/).first()
  await expect(toggle).toBeVisible({ timeout: 15_000 })
  await page.getByTestId(/^ipf-view-3d-/).first().click()
})
