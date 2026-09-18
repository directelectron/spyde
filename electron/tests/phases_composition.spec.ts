/**
 * phases_composition.spec.ts — composition and phases in ONE popout.
 *
 * They used to be apart: the dock held a flat element list, the orientation
 * wizards held .cif paths, and the COD search silently read the former to fill
 * the latter. That cannot describe a two-phase sample — COD matches the
 * elements EXACTLY, so a Cu/Nb sample searched as one composition asks for a
 * Cu-Nb compound and gets nothing, while fcc Cu and bcc Nb are one query each.
 *
 * So clicking elements builds the sample, selecting a phase first makes those
 * clicks build THAT phase, and each phase carries the structure that indexes
 * it. An element can also belong to the sample without belonging to any phase.
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
  await page.waitForTimeout(1500)
})

test.afterAll(async () => { await app?.close() })

test('clicking elements with a phase selected builds that phase', async () => {
  // One popout, not two: the periodic table and the phases are the same widget,
  // because a phase is a SUBSET of the composition.
  await page.getByTestId('composition-edit').click()
  await expect(page.getByTestId('periodic-table')).toBeVisible()
  await expect(page.getByTestId('ptable-phases')).toBeVisible()
  await shot('01-editor')

  await page.getByTestId('ptable-add-phase').click()
  await expect(page.getByTestId('phase-btn-0')).toHaveAttribute('data-active', 'true')
  // Elements clicked now land in phase 1 AND in the sample.
  await page.getByTestId('ptable-el-Cu').click()
  await expect(page.getByTestId('phase-0-el-Cu')).toBeVisible()
  await expect(page.getByTestId('ptable-selected')).toContainText('Cu')

  // A second phase, built the same way.
  await page.getByTestId('ptable-add-phase').click()
  await expect(page.getByTestId('phase-btn-1')).toHaveAttribute('data-active', 'true')
  await page.getByTestId('ptable-el-Nb').click()
  await expect(page.getByTestId('phase-1-el-Nb')).toBeVisible()
  await shot('02-two-phases')

  // Cu belongs to phase 1 only — selecting phase 2 must not show it.
  await expect(page.getByTestId('phase-1-el-Cu')).toHaveCount(0)
})

test('an element can belong to the sample without belonging to a phase', async () => {
  // The extra oxygen that is in neither structure being indexed against.
  await page.getByTestId('phase-btn-1').click()          // deselect (toggle off)
  await expect(page.getByTestId('phase-btn-1')).not.toHaveAttribute('data-active', 'true')
  await page.getByTestId('ptable-el-O').click()
  await expect(page.getByTestId('ptable-selected')).toContainText('O')
  // …and it joined no phase.
  await page.getByTestId('phase-btn-1').click()
  await expect(page.getByTestId('phase-1-el-O')).toHaveCount(0)
  await shot('03-sample-only-element')
})

test('the COD search is scoped to the selected phase', async () => {
  // COD matches the elements EXACTLY, so the query has to be one phase's.
  await page.getByTestId('phase-btn-0').click()
  await expect(page.getByTestId('phase-0-cod')).toHaveAttribute('title', /Cu structures/)
  await page.getByTestId('phase-btn-1').click()
  await expect(page.getByTestId('phase-1-cod')).toHaveAttribute('title', /Nb structures/)
})

test('a phase takes its elements from a .cif, and the dock shows the structure', async () => {
  // The file knows what it is made of; a phase with a structure but no elements
  // reads as a mistake and has nothing to search COD with either.
  await page.getByTestId('ptable-add-phase').click()
  await expect(page.getByTestId('phase-row-2')).toContainText('Click elements above')
  await page.getByTestId('phase-2-cif').click()
  await expect(page.getByTestId('phase-2-structure')).toContainText('Silver__0011135')
  await expect(page.getByTestId('phase-2-el-Ag')).toBeVisible()
  await shot('04-elements-from-cif')

  await page.getByTestId('ptable-apply').click()
  await expect(page.getByTestId('periodic-table')).toBeHidden()
  // The structure is on the SAMPLE, so the dock shows it beside the chips —
  // previously a picked .cif was visible nowhere outside the wizard.
  await expect(page.getByTestId('composition-structure-2')).toContainText('Silver__0011135')
  await expect(page.getByTestId('composition-section')).toContainText('&')
  await shot('05-dock')
})
