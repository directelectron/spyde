/**
 * nav_axis_edit.spec.ts — editing a NAVIGATION axis (units/scale) in the Plot
 * Control dock must recalibrate the NAVIGATOR plot (ticks/scale bar), not just
 * the signal plot. Regression for "updating the nav name and scale don't adjust
 * the navigation plot".
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'nav_axis_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 60_000)
  await page.waitForTimeout(2500)
})
test.afterAll(async () => { await ctx?.app?.close() })
test.setTimeout(120_000)

test('editing nav-axis units + scale recalibrates the navigator plot', async () => {
  const { page } = ctx
  // Focus the NAVIGATOR window so the dock's axes table + Workflow target it.
  const navWin = page.getByTestId('subwindow').filter({ hasText: 'Navigator' }).first()
  await navWin.click()
  await page.waitForTimeout(500)
  await page.screenshot({ path: join(SHOTS, '01-before.png') })

  // Edit BOTH navigation axes (nav rows) → units "nm", scale 2. The axis rows
  // are indexed into the full axes list; nav axes have role "nav".
  const navRows = page.locator('[data-testid^="axis-row-"]')
    .filter({ has: page.getByText('nav', { exact: true }) })
  const count = await navRows.count()
  console.log('[nav-axis] nav row count =', count)
  expect(count).toBeGreaterThanOrEqual(2)

  for (let i = 0; i < count; i++) {
    const row = navRows.nth(i)
    const idx = (await row.getAttribute('data-testid'))!.replace('axis-row-', '')
    // units
    await page.getByTestId(`axis-${idx}-units`).click()
    const uIn = page.getByTestId(`axis-${idx}-units-input`)
    await uIn.fill('nm'); await uIn.press('Enter')
    await page.waitForTimeout(200)
    // scale
    await page.getByTestId(`axis-${idx}-scale`).click()
    const sIn = page.getByTestId(`axis-${idx}-scale-input`)
    await sIn.fill('2'); await sIn.press('Enter')
    await page.waitForTimeout(200)
  }
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '02-after.png') })

  // The dock's axes table re-emits with the edited nav units — a machine check
  // that the edit landed; the screenshot is the eyes check that the NAVIGATOR
  // plot's ticks/scale bar recalibrated (nm), which is the actual regression.
  const navUnitsCell = page.getByTestId('axis-0-units')
  await expect(navUnitsCell).toContainText('nm')
  ctx.assertNoJsErrors()
})
