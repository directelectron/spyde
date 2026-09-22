/**
 * phases_composition.spec.ts — building a sample's phases from the dock,
 * against the real backend, with the NAVIGATOR focused: on a scan the
 * navigator is usually the focused window, and the dock follows it.
 *
 * A sample is a list of phases; the periodic table edits the selected one.
 * The cases that have to hold: zirconia and alpha-Zr (Zr in both), a phase
 * whose elements come from its .cif, and a trace element kept out of the
 * structure search.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'

const CIF = join(__dirname, '..', '..', 'spyde', 'tests', 'Silver__0011135.cif')
const SHOTS = join(__dirname, '..', 'phases_shots')

let app: ElectronApplication
let page: Page

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

const shot = async (name: string) =>
  page.screenshot({ path: join(SHOTS, `${name}.png`) })
const dockPhase = (index: number) => page.getByTestId(`composition-phase-${index}`)

test.beforeAll(async () => {
  test.setTimeout(240_000)
  mkdirSync(SHOTS, { recursive: true })
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_NO_DASK: '1' },
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  await app.evaluate(({ ipcMain }, cif) => {
    ipcMain.removeHandler('spyde:pick-file')
    ipcMain.handle('spyde:pick-file', async () => cif)
  }, CIF)
  await page.evaluate(() => window.electron.action('load_test_data', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 60_000 })
  const navigator = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) }).first()
  await navigator.getByTestId('subwindow-titlebar').click()
})

test.afterAll(async () => { await app?.close() })

test('the first element clicked makes Phase 1, and the dock shows it', async () => {
  await expect(page.getByTestId('composition-empty')).toBeVisible()
  await page.getByTestId('composition-edit').click()
  await expect(page.getByTestId('periodic-table')).toBeVisible()
  await expect(page.getByTestId('phase-btn-0')).toHaveAttribute('data-active', 'true')
  await shot('01-empty')

  // Two clicks without waiting: both land in the one new phase.
  await page.getByTestId('ptable-el-Zr').click()
  await page.getByTestId('ptable-el-O').click()
  await expect(page.getByTestId('phase-0-el-Zr')).toBeVisible()
  await expect(page.getByTestId('phase-0-el-O')).toBeVisible()
  await expect(page.getByTestId('phase-btn-1')).toHaveCount(0)
  await expect(dockPhase(0)).toContainText('Zr')
  await expect(dockPhase(0)).toContainText('O')
})

test('an element can be in two phases', async () => {
  await page.getByTestId('ptable-add-phase').click()
  await expect(page.getByTestId('phase-btn-1')).toHaveAttribute('data-active', 'true')
  await expect(dockPhase(1)).toContainText('no elements')

  // Zr is already in the zirconia; this click adds it to alpha-Zr and leaves
  // the zirconia alone.
  await page.getByTestId('ptable-el-Zr').click()
  await expect(page.getByTestId('phase-1-el-Zr')).toBeVisible()
  await expect(page.getByTestId('ptable-el-O')).not.toHaveAttribute('data-in-phase', 'true')
  await expect(dockPhase(0)).toContainText('Zr')
  await expect(dockPhase(0)).toContainText('O')
  await expect(dockPhase(1)).toContainText('Zr')
  await expect(dockPhase(1)).not.toContainText('O')
  await shot('02-shared-element')
})

test('a percentage is kept as typed, in its own phase', async () => {
  await page.getByTestId('phase-btn-0').click()
  await page.getByTestId('phase-0-pct-Zr').fill('33.3')
  await page.getByTestId('phase-btn-1').click()
  await expect(page.getByTestId('phase-1-pct-Zr')).toHaveValue('')
  await expect(page.getByTestId('composition-chip-0-Zr')).toContainText('33.3%')
  await expect(page.getByTestId('composition-chip-1-Zr')).not.toContainText('%')
})

test('a phase takes its elements from a .cif, and the dock shows the structure', async () => {
  await page.getByTestId('ptable-add-phase').click()
  await expect(page.getByTestId('phase-row-2')).toContainText('Click elements above')
  await page.getByTestId('phase-2-cif').click()
  await expect(page.getByTestId('phase-2-structure')).toContainText('Silver__0011135')
  await expect(page.getByTestId('phase-2-el-Ag')).toBeVisible()
  await expect(page.getByTestId('composition-structure-2')).toContainText('Silver__0011135')
})

test('a trace element counts for the sample but not for the structure search', async () => {
  await page.getByTestId('ptable-el-O').click()
  await expect(page.getByTestId('phase-2-el-O')).toBeVisible()
  await expect(page.getByTestId('phase-2-cod')).toHaveAttribute('title', /Ag-O structures/)
  await page.getByTestId('phase-2-trace-O').click()
  await expect(page.getByTestId('phase-2-trace-O')).toHaveAttribute('data-on', 'true')
  await expect(page.getByTestId('phase-2-cod')).toHaveAttribute('title', /for Ag structures/)
  await expect(page.getByTestId('composition-chip-2-O')).toHaveAttribute('data-trace', 'true')
  await shot('03-trace')
})

test('removing a phase removes only that phase', async () => {
  await page.getByTestId('phase-btn-1').click()
  await page.getByTestId('phase-1-remove').click()
  await expect(page.getByTestId('phase-btn-2')).toHaveCount(0)
  await expect(dockPhase(1)).toContainText('Ag')
  await expect(page.getByTestId('composition-structure-1')).toContainText('Silver__0011135')
  await expect(dockPhase(0)).toContainText('Zr')

  await page.getByTestId('ptable-done').click()
  await expect(page.getByTestId('periodic-table')).toBeHidden()
  await expect(page.getByTestId('composition-section')).toContainText('&')
  await shot('04-dock')
})
