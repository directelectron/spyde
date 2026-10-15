/**
 * phase_viewer.spec.ts — clicking a phase's structure in the Composition dock
 * pops its viewer out beside the sidebar: [master pattern] [unit cell]
 * [detector], the master-pattern choice, and the microscope geometry folded
 * away under them. Which microscope is read off the signal — an EBSD scan is
 * SEM, a diffraction dataset TEM (with spots). It is part of the phase, not a
 * window; several open phases stack.
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
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

const shot = async (name: string) =>
  page.screenshot({ path: join(SHOTS, `${name}.png`) })
const windowCount = () =>
  page.evaluate(() => document.querySelectorAll('[data-testid="subwindow"]').length)
const card = (index = 0) => page.getByTestId('phase-popout').locator('section').nth(index)

/** A screenshot of part of the screen, for "did these pixels change". Clipped
 *  to a plot area so chrome (a hover button, a pulsing chip) cannot differ. */
const region = async (index: number, pane: number) => {
  const box = (await card(index).locator('iframe').boundingBox())!
  const third = box.width / 3
  return page.screenshot({ clip: { x: box.x + third * pane + 6, y: box.y + 22,
                                   width: third - 12, height: box.height - 30 } })
}

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
  await page.getByTestId('composition-edit').click()
  await page.getByTestId('phase-0-cif').click()
  await expect(page.getByTestId('phase-0-structure')).toContainText('Silver__0011135')
  await page.getByTestId('ptable-done').click()
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await app?.close()
})

test('clicking a phase pops out its viewer beside the sidebar', async () => {
  const windows = await windowCount()
  await page.getByTestId('composition-structure-0').click()
  await expect(card(0).locator('iframe')).toBeVisible({ timeout: 60_000 })
  await expect(card(0)).toContainText('Fm-3m')
  await expect(card(0).getByTestId(/phase-geometry-/)).toContainText('SEM geometry')
  await expect(card(0).getByTestId(/phase-field-/)).toHaveCount(0)   // folded away
  await expect(card(0).getByTestId(/phase-spots-/)).toHaveCount(0)   // EBSD: no spots
  await page.waitForTimeout(1500)             // the texture decodes asynchronously
  expect(await windowCount()).toBe(windows)   // part of the phase, not a window
  await shot('01-popout')
})

test('dragging the sphere turns the crystal and repaints the detector', async () => {
  const before = await region(0, 2)
  const box = (await card(0).locator('iframe').boundingBox())!
  const x = box.x + box.width / 6, y = box.y + box.height * 0.55
  await page.mouse.move(x, y)
  await page.mouse.down()
  await page.mouse.move(x + 40, y - 25, { steps: 15 })
  await page.mouse.up()
  await page.waitForTimeout(800)
  await shot('02-dragged')
  expect((await region(0, 2)).equals(before)).toBe(false)
})

test('opening the geometry and editing a field moves the detector', async () => {
  await card(0).getByTestId(/phase-geometry-/).click()
  await expect(card(0).getByTestId(/phase-field-.*-twist$/)).toBeVisible()
  const before = await region(0, 2)
  const field = card(0).getByTestId(/phase-field-.*-pc_y$/)
  await field.fill('0.3')
  await field.press('Enter')
  await page.waitForTimeout(800)
  expect((await region(0, 2)).equals(before)).toBe(false)
})

test('the full simulation replaces the kinematic bands', async () => {
  const note = card(0).getByTestId(/phase-full-note-/)
  if (await note.count() && /Needs de_ebsd/.test(await note.innerText())) {
    test.skip(true, 'de_ebsd is not installed here')
  }
  const before = await region(0, 0)
  await card(0).getByTestId(/phase-master-.*-full$/).click()
  await expect(card(0).getByTestId(/phase-master-.*-full$/))
    .toHaveAttribute('aria-checked', 'true', { timeout: 120_000 })
  await page.waitForTimeout(1500)
  await shot('03-full-simulation')
  expect((await region(0, 0)).equals(before)).toBe(false)
})

test('solving the pattern centre from the scan reports its fit', async () => {
  const solve = card(0).getByTestId(/phase-solve-pc-/)
  if (!await solve.count()) test.skip(true, 'de_ebsd is not installed here')
  test.setTimeout(240_000)
  await solve.click()
  await expect(solve).toHaveText('Solving…')
  // The answer depends on how well the phase matches the data; either way
  // the note says what the fit was and whether the centre was written.
  await expect(card(0).getByTestId(/phase-solve-note-/))
    .toContainText(/patterns · fit/, { timeout: 200_000 })
  await expect(solve).toHaveText('Solve from data')
  await page.waitForTimeout(800)
  await shot('04-solved')
})

test('a second phase stacks under the first', async () => {
  await page.getByTestId('composition-edit').click()
  await page.getByTestId('ptable-add-phase').click()
  await page.getByTestId('phase-1-cif').click()
  await expect(page.getByTestId('phase-1-structure')).toContainText('Silver__0011135')
  await page.getByTestId('ptable-done').click()
  await page.getByTestId('composition-structure-1').click()
  await expect(card(1).locator('iframe')).toBeVisible({ timeout: 60_000 })
  const first = (await card(0).boundingBox())!, second = (await card(1).boundingBox())!
  expect(second.y).toBeGreaterThan(first.y + first.height - 1)
  await page.waitForTimeout(1500)
  await shot('05-stacked')

  await card(0).getByRole('button', { name: 'Close phase viewer' }).click()
  await expect(page.getByTestId('phase-popout').locator('section')).toHaveCount(1)
})

test('a diffraction dataset is simulated as TEM, with spots', async () => {
  await backendAction(page, 'load_test_data_si_grains')
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 4,
    { timeout: 120_000 })
  const navigators = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
  await navigators.last().getByTestId('subwindow-titlebar').click()
  await page.getByTestId('composition-edit').click()
  await page.getByTestId('phase-0-cif').click()
  await expect(page.getByTestId('phase-0-structure')).toContainText('Silver__0011135')
  await page.getByTestId('ptable-done').click()
  await page.getByTestId('composition-structure-0').click()
  const tem = page.getByTestId('phase-popout').locator('section').last()
  await expect(tem.getByTestId(/phase-geometry-/)).toContainText('TEM geometry', { timeout: 60_000 })
  await expect(tem.getByTestId(/phase-spots-/)).toBeChecked()
  await page.waitForTimeout(1500)
  await shot('06-tem-spots')
})
