/**
 * beta_ribbon.spec.ts — the `beta:` toolbar key, end to end.
 *
 * An action declaring `beta: True` in the toolbar schema must reach the user as
 * a mark, not as a hidden action: a badge on its toolbar button and a ribbon
 * across its caret. Vector Orientation Mapping is the one declaring it while it
 * is rebuilt on the quantem matcher; Orientation Mapping is the control — the
 * same WizardShell chrome, so an unconditional ribbon would show up there too.
 *
 * Screenshots go to electron/beta_ribbon_shots/ and are the point of the spec:
 * the assertions prove the elements exist, only the pixels prove they read. The
 * toolbar only exists while its window is hovered, so every capture follows a
 * titlebar hover — a screenshot taken after the pointer drifts shows nothing.
 */
import { test, expect, _electron as electron, ElectronApplication, Page, Locator } from '@playwright/test'
import { join } from 'path'

const SHOTS = join(__dirname, '..', 'beta_ribbon_shots')

let app: ElectronApplication
let page: Page

/** The subwindow whose toolbar offers `action`. */
function windowOffering(action: string): Locator {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(`action-btn-${action}`) }).first()
}

/** Raise that window's toolbar. It fades to `pointerEvents: none` once the
 *  pointer leaves, so this has to run immediately before each interaction —
 *  hovering once per test and then asserting for a few seconds loses the bar. */
async function raiseToolbar(win: Locator): Promise<void> {
  await win.getByTestId('subwindow-titlebar').hover()
  await page.waitForTimeout(250)
}

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_NO_DASK: '1' },   // no compute needed here
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  await page.evaluate(() => window.electron.action('load_test_vectors', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 4,
    { timeout: 60_000 },
  )
  await page.waitForTimeout(1500)
})

test.afterAll(async () => { await app?.close() })

test('a beta action is badged on its button and ribboned in its caret', async () => {
  const win = windowOffering('Vector Orientation Mapping')
  await raiseToolbar(win)

  // Offered, not hidden — beta is a label, never a gate.
  const button = win.getByTestId('action-btn-Vector Orientation Mapping')
  await expect(button).toBeVisible({ timeout: 30_000 })
  await expect(win.getByTestId('beta-badge-Vector Orientation Mapping')).toBeVisible()
  await button.screenshot({ path: join(SHOTS, '01-button-badge.png') })
  await page.screenshot({ path: join(SHOTS, '02-toolbar-in-context.png') })

  await raiseToolbar(win)
  await button.click()
  const wizard = page.getByTestId('vector-orientation-wizard')
  await expect(wizard).toBeVisible({ timeout: 15_000 })
  const ribbon = page.getByTestId('vector-orientation-wizard-beta')
  await expect(ribbon).toBeVisible()
  await expect(ribbon).toContainText('BETA')
  await wizard.screenshot({ path: join(SHOTS, '03-caret-ribbon.png') })

  await page.getByTestId('vom-close').click()
  await expect(wizard).toHaveCount(0)
})

test('an ordinary action gets no badge and no ribbon', async () => {
  // Crop is the control: same WizardShell chrome as the beta action, so an
  // unconditionally-rendered ribbon would appear here too.
  // The SAME window and toolbar as the beta action, so the action is the only
  // thing that differs. Picking any window offering Crop finds one stacked
  // under another, whose toolbar is occluded and cannot be clicked.
  const win = windowOffering('Vector Orientation Mapping')
  await raiseToolbar(win)
  const button = win.getByTestId('action-btn-Crop')
  await expect(button).toBeVisible({ timeout: 30_000 })
  await expect(win.getByTestId('beta-badge-Crop')).toHaveCount(0)

  await raiseToolbar(win)
  await button.click()
  const wizard = page.getByTestId('crop-wizard')
  await expect(wizard).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('crop-wizard-beta')).toHaveCount(0)
  await wizard.screenshot({ path: join(SHOTS, '04-control-no-ribbon.png') })
})
