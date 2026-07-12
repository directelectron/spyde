/**
 * center_zero_beam.spec.ts — Center Zero Beam (Automatic) end-to-end on a real
 * backend: load 4D data with an off-centre beam → open the Center Zero Beam
 * wizard → Automatic → Center → the status reports the beam centred and a
 * "Centered" node is added to the signal tree.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_NO_DASK: '1' },   // eager test data
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  await page.waitForTimeout(1500)
  await page.evaluate(() => window.electron.action('load_test_data', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 30_000 },
  )
  await page.waitForTimeout(1000)
})

test.afterAll(async () => { await app?.close() })

test('Center Zero Beam (Automatic) centres the beam and records a tree node', async () => {
  const sig = page.getByTestId('subwindow').filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) }).first()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Center Zero Beam').click()
  await expect(page.getByTestId('center-zero-beam-wizard')).toBeVisible()

  await page.getByTestId('czb-center').click()
  await expect(page.getByTestId('status-text'))
    .toContainText('centered', { timeout: 30_000, ignoreCase: true })

  // The workflow view shows the new "Centered" step (re-emitted after the transform).
  await expect(page.getByTestId('signal-tree')).toBeVisible({ timeout: 10_000 })
  await expect(page.getByTestId(/^tree-node-Centered/).first())
    .toBeVisible({ timeout: 10_000 })
})
