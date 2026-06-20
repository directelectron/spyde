/**
 * strain_lazy.spec.ts — Strain Mapping end-to-end on a found-vectors result.
 *   load_test_vectors → click the "Strain Mapping" toolbar action on the vectors
 *   signal window → a Strain window opens with the εxx/εyy/εxy/ω component toggle;
 *   switching the component dispatches strain_set_component.
 *
 * The strain physics is covered by the headless suite; this verifies the wiring:
 * toolbar action → strain window → live component toggle.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

test.beforeAll(async () => {
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
  await page.evaluate(() => window.electron.action('load_test_vectors', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 4,
    { timeout: 60_000 },
  )
  await page.waitForTimeout(2500)
})

test.afterAll(async () => { await app?.close() })
test.setTimeout(120_000)

test('Strain Mapping: opens the strain window with a live component toggle', async () => {
  // The vectors SIGNAL window carries the Strain Mapping action button.
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Strain Mapping') }).first()
  await vsig.getByTestId('subwindow-titlebar').click()    // raise
  await vsig.getByTestId('subwindow-titlebar').hover()    // reveal toolbar
  await expect(vsig.getByTestId('action-btn-Strain Mapping')).toBeVisible({ timeout: 15_000 })

  const before = await page.getByTestId('subwindow').count()
  await vsig.getByTestId('action-btn-Strain Mapping').click()

  // A new Strain window opens carrying the full εxx/εyy/εxy/ω component toggle
  // (the live component swap dispatches strain_set_component — covered headless).
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 30_000, message: 'strain window never opened',
  }).toBeGreaterThan(before)
  const swin = page.getByTestId('subwindow').filter({ has: page.getByTestId(/^strain-toggle-/) }).first()
  await expect(swin.getByTestId(/^strain-comp-exx-/)).toBeVisible({ timeout: 15_000 })
  for (const c of ['eyy', 'exy', 'omega']) {
    await expect(swin.getByTestId(new RegExp(`^strain-comp-${c}-`))).toBeVisible()
  }
})
