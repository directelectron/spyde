/**
 * phase_viewer.spec.ts — clicking a phase's structure in the Composition dock
 * expands the phase to show its Kikuchi band sphere beside the pattern the
 * detector sees — in the dock, as part of the phase, not as a window — and
 * dragging the sphere turns the crystal and repaints the pattern.
 *
 * Runs on the bundled synthetic EBSD scan, whose geometry (pattern centre,
 * detector shape) the viewer reads, with the bundled Ag structure.
 */
import { test, expect, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'
const {
  launchApp, backendAction, waitForSubwindowCount, navWindow,
} = require('./_harness.cjs')

const CIF = join(__dirname, '..', '..', 'spyde', 'tests', 'Silver__0011135.cif')
const SHOTS = join(__dirname, '..', 'phase_viewer_shots')

let app: ElectronApplication
let page: Page

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

const shot = async (name: string) =>
  page.screenshot({ path: join(SHOTS, `${name}.png`) })
const windowCount = () =>
  page.evaluate(() => document.querySelectorAll('[data-testid="subwindow"]').length)

/** A screenshot of a region of the screen, for "did these pixels change".
 *  Clipped to the plot area so window chrome (a pulsing chip, a hover
 *  button) cannot make it differ. */
const region = (clip: { x: number, y: number, width: number, height: number }) =>
  page.screenshot({ clip })

let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  test.setTimeout(240_000)
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({})
  app = ctx.app
  page = ctx.page
  await app.evaluate(({ ipcMain }, cif) => {
    ipcMain.removeHandler('spyde:pick-file')
    ipcMain.handle('spyde:pick-file', async () => cif)
  }, CIF)
  await backendAction(page, 'load_test_data_ebsd')
  await waitForSubwindowCount(page, 2, 120_000)
  await navWindow(page).getByTestId('subwindow-titlebar').click()
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await app?.close()
})

test('clicking the structure expands the phase with its viewer, in the dock', async () => {
  await page.getByTestId('composition-edit').click()
  await page.getByTestId('phase-0-cif').click()
  await expect(page.getByTestId('phase-0-structure')).toContainText('Silver__0011135')
  await page.getByTestId('ptable-done').click()
  await expect(page.getByTestId('composition-structure-0')).toContainText('Silver__0011135')

  const windows = await windowCount()
  await page.getByTestId('composition-structure-0').click()
  const viewer = page.getByTestId('composition-viewer-0')
  await expect(viewer.locator('iframe')).toBeVisible({ timeout: 60_000 })
  await page.waitForTimeout(1500)            // the texture decodes asynchronously
  expect(await windowCount()).toBe(windows)  // part of the phase, not a window
  await shot('01-expanded')
})

test('dragging the sphere turns the crystal and repaints the detector', async () => {
  const box = (await page.getByTestId('composition-viewer-0').locator('iframe').boundingBox())!
  const detector = { x: box.x + box.width * 0.55, y: box.y + 14,
                     width: box.width * 0.42, height: box.height - 18 }
  const before = await region(detector)

  const x = box.x + box.width * 0.25, y = box.y + box.height * 0.5
  await page.mouse.move(x, y)
  await page.mouse.down()
  await page.mouse.move(x + 40, y - 25, { steps: 15 })
  await page.mouse.up()
  await page.waitForTimeout(800)
  await shot('02-dragged')

  expect((await region(detector)).equals(before)).toBe(false)
})

test('clicking the structure again collapses the phase', async () => {
  await page.getByTestId('composition-structure-0').click()
  await expect(page.getByTestId('composition-viewer-0')).toHaveCount(0)
})
