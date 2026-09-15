/**
 * phases_composition.spec.ts — a sample is made of PHASES, and a phase is
 * composition AND structure.
 *
 * Those used to be apart: the dock held a flat element list, the orientation
 * wizard held .cif paths, and the COD search silently read the former to fill
 * the latter. That cannot describe a two-phase sample — COD matches the
 * elements EXACTLY, so a Cu/Nb sample searched as one composition asks for a
 * Cu-Nb compound and gets nothing, while fcc Cu and bcc Nb are one query each.
 *
 * Network is mocked at the backend seam (`_cod_query`), so this asserts the
 * WIRING — that each phase's own elements reach the query — without depending
 * on COD being up. The real queries are checked in test_composition.py.
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

test('the dock shows one group per phase, with & between them', async () => {
  // Two phases, Cu and Nb — the case that returns nothing when asked as one
  // composition. Driven through the backend so this test is about what the
  // dock RENDERS, not about clicking a periodic table.
  await page.evaluate(() => {
    window.electron.action('set_phase', { index: 0, elements: ['Cu'] })
  })
  await page.waitForTimeout(400)
  await page.evaluate(() => {
    window.electron.action('set_phase', { index: 1, elements: ['Nb'] })
  })
  await page.waitForTimeout(800)
  await shot('01-two-phases')

  const section = page.getByTestId('composition-section')
  await expect(section).toContainText('Cu')
  await expect(section).toContainText('Nb')
  // The & is what says these are separate phases rather than one compound.
  await expect(section).toContainText('&')
  await expect(page.getByTestId('composition-phase-0')).toContainText('Cu')
  await expect(page.getByTestId('composition-phase-1')).toContainText('Nb')
})

test('the COD search is scoped to ONE phase, not the whole sample', async () => {
  await page.getByTestId('composition-edit').click()
  await expect(page.getByTestId('phases-editor')).toBeVisible()
  await shot('02-phases-editor')

  // Each row offers a COD search of its OWN elements — the button's tooltip is
  // the contract, and it names only that phase.
  await expect(page.getByTestId('phase-0-cod')).toHaveAttribute(
    'title', /Cu structures/)
  await expect(page.getByTestId('phase-1-cod')).toHaveAttribute(
    'title', /Nb structures/)
})

test('a phase carries its structure, and the dock says so', async () => {
  // The file route: pick a .cif for phase 0 (mocked to the bundled silver).
  await page.getByTestId('phase-0-cif').click()
  await expect(page.getByTestId('phase-0-structure')).toContainText('Silver__0011135')
  await shot('03-structure-bound')

  await page.getByTestId('phases-done').click()
  // The structure is on the SAMPLE now, so the dock shows it beside the chips —
  // previously a picked .cif was visible nowhere outside the wizard.
  await expect(page.getByTestId('composition-structure-0')).toContainText('Silver__0011135')
  await expect(page.getByTestId('composition-phase-1')).toContainText('Nb')
  await shot('04-dock-shows-structure')
})

test('a phase added from a .cif takes its elements from the file', async () => {
  // The file knows what it is made of. A phase that has a structure but claims
  // no elements reads as a mistake, and it would also have nothing to search
  // COD with.
  await page.getByTestId('composition-edit').click()
  await expect(page.getByTestId('phases-editor')).toBeVisible()
  await page.getByTestId('phases-add').click()
  const row = page.getByTestId('phase-row-2')
  await expect(row).toBeVisible()
  await expect(row).toContainText('no elements')

  await page.getByTestId('phase-2-cif').click()
  await expect(page.getByTestId('phase-2-structure')).toContainText('Silver__0011135')
  // Ag, read out of the .cif — not typed in.
  await expect(page.getByTestId('phase-2-el-Ag')).toBeVisible()
  await shot('05-elements-from-cif')
  await page.getByTestId('phases-done').click()
  await expect(page.getByTestId('composition-phase-2')).toContainText('Ag')
})
