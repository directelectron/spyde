/**
 * om_two_phase_refine.spec.ts — the Orientation Mapping Refine step with TWO
 * phases, in the real app: the matched template's spots are drawn on the
 * diffraction pattern, the per-phase IPF heatmap is painted, and zooming the
 * heatmap does not blank it.
 *
 * Pixels are read from the figure canvases, not from screenshots of the
 * window, so window chrome cannot pass for a drawn overlay.
 */
import { test, expect, _electron as electron, ElectronApplication, Page, Frame, Locator } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'

const CIF = join(__dirname, '..', '..', 'spyde', 'tests', 'Silver__0011135.cif')
const SHOTS = join(__dirname, '..', 'om_two_phase_shots')

let app: ElectronApplication
let page: Page

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

const shot = (name: string) => page.screenshot({ path: join(SHOTS, `${name}.png`) })

const signalWindow = () => page.getByTestId('subwindow')
  .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) }).first()
const refineWindow = () => page.getByTestId('subwindow').filter({ hasText: 'IPF Refine' }).first()

async function frameOf(window: Locator): Promise<Frame> {
  const handle = await window.locator('iframe').first().elementHandle()
  const frame = await handle?.contentFrame()
  if (!frame) throw new Error('window has no figure frame')
  return frame
}

/** Pixel counts over the 2-D canvases of a figure: strongly chromatic pixels
 *  (a heatmap or a red marker), and the overlay's green spot colour. With
 *  *leftHalf*, only canvases in the left half of the figure count — the first
 *  of two side-by-side panels. */
async function canvasPixels(window: Locator, leftHalf = false) {
  const frame = await frameOf(window)
  return frame.evaluate((onlyLeft) => {
    let chromatic = 0
    let green = 0
    const middle = document.documentElement.clientWidth / 2
    for (const canvas of Array.from(document.querySelectorAll('canvas'))) {
      const context = (canvas as HTMLCanvasElement).getContext('2d')
      if (!context || !canvas.width || !canvas.height) continue
      const box = canvas.getBoundingClientRect()
      if (onlyLeft && box.left + box.width / 2 > middle) continue
      const { data } = context.getImageData(0, 0, canvas.width, canvas.height)
      for (let i = 0; i < data.length; i += 4) {
        if (data[i + 3] < 128) continue
        const red = data[i], greenChannel = data[i + 1], blue = data[i + 2]
        if (Math.max(red, greenChannel, blue) - Math.min(red, greenChannel, blue) > 80) chromatic++
        if (greenChannel > 200 && red < 120 && blue < 160) green++
      }
    }
    return { chromatic, green }
  }, leftHalf)
}

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_NO_DASK: '1', SPYDE_LOG_LEVEL: 'WARNING' },
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  await app.evaluate(({ ipcMain }, cif) => {
    ipcMain.removeHandler('spyde:pick-file')
    ipcMain.handle('spyde:pick-file', async () => cif)
  }, CIF)
  await page.evaluate(() => window.electron.action('load_test_data_si_grains', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 60_000 })
})

test.afterAll(async () => { await app?.close() })

test('two phases: spots on the pattern and a painted heatmap', async () => {
  await signalWindow().getByTestId('subwindow-titlebar').hover()
  await signalWindow().getByTestId('action-btn-Orientation Mapping').click()
  await expect(page.getByTestId('orientation-wizard')).toBeVisible()

  // Two phases, each given the (mocked) structure.
  await page.getByTestId('om-add-phase').click()
  await page.getByTestId('phase-0-cif').click()
  await expect(page.getByTestId('phase-0-structure')).toContainText('Silver__0011135')
  await page.getByTestId('ptable-add-phase').click()
  await page.getByTestId('phase-1-cif').click()
  await expect(page.getByTestId('phase-1-structure')).toContainText('Silver__0011135')
  await page.getByTestId('ptable-done').click()

  await page.getByTestId('om-tab-Library').click()
  await page.getByTestId('om-generate').click()
  // The wizard itself says when the library is ready.
  await expect(page.getByTestId('om-status')).toContainText('Library ready', { timeout: 120_000 })
  await expect(refineWindow()).toBeVisible({ timeout: 30_000 })

  await page.evaluate(() => window.electron.action('test_nav_drag', { targets: [[2, 2]] }))
  await expect.poll(async () => (await canvasPixels(signalWindow())).green, {
    timeout: 30_000, message: 'no template spots on the diffraction pattern',
  }).toBeGreaterThan(20)
  await expect.poll(async () => (await canvasPixels(refineWindow())).chromatic, {
    timeout: 30_000, message: 'the refine heatmap was never painted',
  }).toBeGreaterThan(200)
  expect((await canvasPixels(refineWindow(), true)).chromatic,
    'the first phase panel was never painted').toBeGreaterThan(100)
  await shot('01-refine')
})

// Needs the anyplotlib release that keeps a decoded raster out of the state a
// zoom writes back (anyplotlib fix/raster-cache-view-writeback).
test('zooming a heatmap panel keeps it painted', async () => {
  const painted = (await canvasPixels(refineWindow(), true)).chromatic
  const frameBox = (await refineWindow().locator('iframe').first().boundingBox())!
  await page.mouse.move(frameBox.x + frameBox.width * 0.25, frameBox.y + frameBox.height * 0.5)
  for (let step = 0; step < 4; step++) {
    await page.mouse.wheel(0, -240)
    await page.waitForTimeout(150)
  }
  await page.waitForTimeout(1500)
  await shot('02-zoomed')
  const zoomed = (await canvasPixels(refineWindow(), true)).chromatic
  console.log(`first panel chromatic pixels: before zoom ${painted}, after zoom ${zoomed}`)
  expect(zoomed, 'zooming blanked the refine heatmap').toBeGreaterThan(100)
})
