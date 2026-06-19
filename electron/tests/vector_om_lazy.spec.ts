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
  // Find Vectors runs on a background thread AFTER the windows open — wait for it
  // to finish attaching `diffraction_vectors` (else Generate Library races it and
  // errors "run Find Diffraction Vectors first").
  await expect(page.getByTestId('status-text'))
    .toContainText('Found', { timeout: 60_000 })
  await page.waitForTimeout(1500)
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
  // "library ready" is transient — on synthetic data the field fit completes
  // immediately and advances the status to "live IPF ready". Accept either.
  await expect(page.getByTestId('status-text'))
    .toContainText(/library ready|live IPF ready/, { timeout: 60_000 })

  // Generating the library activates the LIVE refine overlay: the measured
  // vectors (red) + the fitted template (green) appear on the vectors DP.
  await expect.poll(() => colorPixels('green'), {
    timeout: 30_000, message: 'fitted template (green) never drawn on the DP',
  }).toBeGreaterThan(0)
  expect(await colorPixels('red')).toBeGreaterThan(0)

  // Generate also fits the WHOLE field on the GPU and opens the live IPF
  // heatmap (the orientation map appears while you refine — the "super nice" bit).
  await expect(
    page.getByTestId('subwindow').filter({ hasText: 'Orientation' }).first(),
  ).toBeVisible({ timeout: 150_000 })

  // 3 Refine → the live single-pattern fit streams a strain readout to the
  // Refine tab (Qt parity), and the strain-cap slider re-fits live.
  await page.getByTestId('vom-tab-Refine').click()
  await expect(page.getByTestId('vom-strain-readout'))
    .toContainText('εxx', { timeout: 30_000 })

  // 4 Run → reuses the field, adds ONE unified Strain window (IPF already shown).
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('vom-tab-Run').click()
  await page.getByTestId('vom-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'strain window never opened',
  }).toBeGreaterThanOrEqual(before + 1)

  // The Strain window holds the unified chip strip: εxx / εyy / εxy as views of
  // one window. εxx is selected by default; ⌘-click εyy asks the backend to
  // rebuild ONE figure with the two views as side-by-side anyplotlib axes.
  await expect(page.getByTestId(/^view-chip-εxx-/).first()).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId(/^view-chip-εyy-/).first()).toBeVisible()
  await expect(page.getByTestId(/^view-chip-εxy-/).first()).toBeVisible()
  const strainWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(/^view-chip-εxx-/) }).first()
  await page.getByTestId(/^view-chip-εyy-/).first().click({ modifiers: ['Meta'] })
  // The combined side-by-side figure (title "εxx / εyy") arrives and is shown —
  // a single iframe with two axes, not two iframes.
  await expect(strainWin.locator('iframe[title="εxx / εyy"]')).toBeVisible({ timeout: 30_000 })
  await expect.poll(() =>
    strainWin.locator('iframe:visible').count(), { timeout: 5_000 },
  ).toBe(1)
  await page.waitForTimeout(800)   // let the two anyplotlib axes paint
  await strainWin.screenshot({ path: join(__dirname, '..', 'vom_combined_strain.png') })

  // The IPF orientation (vector-OM) window also carries the 2D/3D explorer
  // toggle + X/Y/Z (the actual 3-D switch is exercised in orientation_lazy.spec;
  // here the result windows cascade so the toggle can be covered — assert it's
  // wired, i.e. present on the vector-OM IPF window).
  const ipfWin = page.getByTestId('subwindow').filter({ hasText: 'Orientation' }).first()
  await expect(ipfWin.getByTestId(/^ipf-view-toggle-/)).toBeVisible({ timeout: 15_000 })
  await expect(ipfWin.getByTestId(/^ipf-view-3d-/)).toBeVisible()
})
