/**
 * plot_control_node.spec.ts — the Plot Control dock follows the selected node.
 *
 *  1  the Rebin caret separates its scan fields from its diffraction fields
 *     with a "|": Scan X  Scan Y | Diffraction X  Diffraction Y;
 *  2  after binning the scan along x, the dock's shape chip and axes table
 *     describe the Binned node (nav 12 × 24, x scale 2), and switching back to
 *     the root describes the root again.
 *
 * Screenshots land in electron/<PLOT_CONTROL_SHOTS or plot_control_shots>/.
 */
import { test, expect, Page } from '@playwright/test'
import { join } from 'path'
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { launchApp, raiseWindow, backendAction, waitForSubwindowCount, sigWindow } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', process.env.PLOT_CONTROL_SHOTS || 'plot_control_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>
let page: Page

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  page = ctx.page
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_lazy_chunked')
  await waitForSubwindowCount(page, 2, 60_000)
  await page.waitForTimeout(2000)
})

test.afterAll(async () => { await ctx?.app?.close() })

const shot = (name: string) => page.screenshot({ path: join(SHOTS, name) })

async function openRebin() {
  const win = sigWindow(page)
  await win.getByTestId('subwindow-titlebar').click()
  await raiseWindow(win)
  await win.getByTestId('action-btn-Rebin').click()
  const flyout = page.getByTestId('action-flyout')
  await expect(flyout).toBeVisible()
  return flyout
}

/** The x-axis row of the axes table: array order is (y, x, ky, kx). */
const scanXScale = () => page.getByTestId('axis-1-scale')

test('1: the Rebin caret puts a "|" between the scan and diffraction fields', async () => {
  const flyout = await openRebin()
  await shot('01-rebin-caret.png')
  const order = await flyout.locator('[data-testid^="param-row-"], [data-testid="param-separator"]')
    .evaluateAll((elements) => elements.map((element) => element.getAttribute('data-testid')))
  expect(order).toEqual([
    'param-row-scan_x', 'param-row-scan_y', 'param-separator',
    'param-row-scale_x', 'param-row-scale_y'])
  await expect(flyout.getByTestId('param-separator')).toHaveText('|')
})

test('2: the dock describes the Binned node, then the root again', async () => {
  await expect(page.getByText('nav 24 × 24 · sig 32 × 32')).toBeVisible()
  await expect(scanXScale()).toHaveText('1.00')

  const flyout = page.getByTestId('action-flyout')
  if (!(await flyout.isVisible())) await openRebin()
  await flyout.getByTestId('param-scan_x').fill('2')
  await flyout.getByTestId('param-scan_y').fill('1')
  await flyout.getByTestId('param-scale_x').fill('1')
  await flyout.getByTestId('param-scale_y').fill('1')
  await page.getByTestId('action-run').click()
  await expect(page.getByTestId('tree-node-Binned')).toBeVisible({ timeout: 30_000 })
  await page.waitForTimeout(1500)
  await shot('02-dock-binned-node.png')

  await expect(page.getByText('nav 12 × 24 · sig 32 × 32')).toBeVisible({ timeout: 15_000 })
  await expect(scanXScale()).toHaveText('2.00')

  await page.getByTestId('tree-node-root').click()
  await expect(page.getByText('nav 24 × 24 · sig 32 × 32')).toBeVisible({ timeout: 15_000 })
  await expect(scanXScale()).toHaveText('1.00')
  await shot('03-dock-root-node.png')
})
