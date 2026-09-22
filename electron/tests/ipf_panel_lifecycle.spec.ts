/**
 * ipf_panel_lifecycle.spec.ts — the wizard's IPF window follows the ACTION.
 *
 * Selecting Vector Orientation Mapping and giving a phase a structure opens
 * the IPF Refine window with the phase's empty triangle; Generate fills it
 * and then shows the sampled orientations; deselecting the action hides the
 * window; reselecting brings it back. Each stage is screenshot to
 * ipf_panel_lifecycle_shots/ for the author's eyes.
 *
 * The drive is the one ipf_two_window_vom.spec.ts uses (load_test_vectors →
 * the caret → the mocked .cif picker → Generate).
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'
const { raiseWindow } = require('./_harness.cjs')

const CIF = join(__dirname, '..', '..', 'spyde', 'tests', 'Silver__0011135.cif')
const SHOTS = join(__dirname, '..', 'ipf_panel_lifecycle_shots')

let app: ElectronApplication
let page: Page

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

const ipfWindow = () => page.getByTestId('subwindow')
  .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /IPF Refine/ }) })

test.beforeAll(async () => {
  test.setTimeout(300_000)
  mkdirSync(SHOTS, { recursive: true })
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env },
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
  await expect(page.getByTestId('status-text'))
    .toContainText('Found', { timeout: 60_000 })
  await page.waitForTimeout(1500)
})

test.afterAll(async () => { await app?.close() })

test('a phase with a structure opens the empty triangle', async () => {
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Vector Orientation Mapping') }).first()
  await vsig.getByTestId('subwindow-titlebar').click()
  await vsig.getByTestId('subwindow-titlebar').hover()
  await vsig.getByTestId('action-btn-Vector Orientation Mapping').click()
  await expect(page.getByTestId('vector-orientation-wizard')).toBeVisible()
  await expect(ipfWindow(), 'no phase yet, so no window yet').toHaveCount(0)

  await page.getByTestId('vom-add-phase').click()
  await expect(page.getByTestId('periodic-table')).toBeVisible()
  await expect(page.getByTestId('phase-row-0')).toBeVisible()
  await page.getByTestId('phase-0-cif').click()
  await expect(page.getByTestId('phase-0-structure')).toContainText('Silver__0011135')
  await page.getByTestId('ptable-done').click()
  await expect(page.getByTestId('vom-cif-list')).toContainText('Silver__0011135')

  await expect(ipfWindow(), 'the structure opens the IPF window').toHaveCount(1, { timeout: 30_000 })
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '01-empty-triangle.png') })
})

test('Generate fills the triangle, then shows the sampled orientations', async () => {
  await page.getByTestId('vom-tab-Library').click()
  await page.getByTestId('vom-generate').click()
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '02a-filling.png') })
  await page.waitForTimeout(2500)
  await page.screenshot({ path: join(SHOTS, '02b-filling.png') })
  await expect(page.getByTestId('status-text'))
    .toContainText(/Vector Orientation: ready/, { timeout: 120_000 })
  await expect(ipfWindow(), 'Generate reuses the window, it does not open a second').toHaveCount(1)
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '03-library-points.png') })
})

test('deselecting the action hides the window; reselecting brings it back', async () => {
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Vector Orientation Mapping') }).first()
  // The IPF window opened over the source window's titlebar; raise the
  // source first so the hover reveals its toolbar.
  await raiseWindow(vsig)
  await vsig.getByTestId('subwindow-titlebar').hover()
  await vsig.getByTestId('action-btn-Vector Orientation Mapping').click()
  await expect(page.getByTestId('vector-orientation-wizard')).toBeHidden()
  await expect(ipfWindow(), 'deselect hides the window').toHaveCount(0, { timeout: 15_000 })
  await page.screenshot({ path: join(SHOTS, '04-hidden.png') })

  await raiseWindow(vsig)
  await vsig.getByTestId('subwindow-titlebar').hover()
  await vsig.getByTestId('action-btn-Vector Orientation Mapping').click()
  await expect(page.getByTestId('vector-orientation-wizard')).toBeVisible()
  await expect(ipfWindow(), 'reselect shows it again').toHaveCount(1, { timeout: 15_000 })
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '05-shown-again.png') })
})
