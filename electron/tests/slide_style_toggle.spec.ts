/**
 * slide_style_toggle.spec.ts — the per-slide BACKGROUND-style chrome toggle.
 *
 * The slide_style presets (default/plain/accent) already round-trip + render in
 * Present mode + export; this spec covers the NEW authoring control: a chrome
 * button on a slide-starting cell that cycles the preset. Asserts the button
 * cycles default → plain → accent (data-slide-style attr) and that the chosen
 * preset reaches the report doc + paints the Present-mode slide background.
 *
 * No-dask fast launch (a text-only deck needs no compute). Screenshots to
 * slide_style_shots/ and the author Reads them — an accent slide that doesn't
 * LOOK tinted is a failure even when the attribute flips.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'slide_style_shots')

async function reportDoc(page: any): Promise<any> {
  return await page.evaluate(() => (window as any)._spyde_test_report?.())
}

test.describe('slide background-style toggle', () => {
  let ctx: any
  test.beforeAll(async () => { ctx = await launchApp({ dask: false }) })
  test.afterAll(async () => { await ctx?.app?.close() })

  test('chrome toggle cycles default → plain → accent and paints Present mode', async () => {
    const { page } = ctx

    // Open the Report Builder dock so the cell + its chrome render.
    await page.getByTestId('toggle-report').click()
    await expect(page.getByTestId('report-sidebar')).toBeVisible()

    // A one-cell deck: the single markdown cell is the slide's starting cell,
    // so its chrome offers the slide-style toggle.
    await page.evaluate(() => (window as any).electron.action('report_new', {}))
    await page.evaluate(() => (window as any).electron.action('report_add_cell', {
      cell_type: 'markdown', source: '# A tinted slide\n\nBody text.', index: 0,
    }))

    // Resolve the cell id from the report doc.
    await expect.poll(async () => (await reportDoc(page))?.cells?.length ?? 0).toBeGreaterThan(0)
    const doc = await reportDoc(page)
    const cellId = doc.cells[0].id

    // Hover the cell to reveal its chrome, then read the style toggle.
    const cell = page.locator(`[data-testid^="report-cell-"]`).first()
    await cell.dispatchEvent('mouseover', { bubbles: true })
    const btn = page.getByTestId(`cell-slide-style-${cellId}`)
    await expect(btn).toBeVisible()
    await expect(btn).toHaveAttribute('data-slide-style', 'default')

    // Cycle: default → plain → accent.
    await btn.click()
    await expect(btn).toHaveAttribute('data-slide-style', 'plain')
    await btn.click()
    await expect(btn).toHaveAttribute('data-slide-style', 'accent')

    // The doc reflects the accent preset.
    await expect.poll(async () => (await reportDoc(page))?.cells?.[0]?.slide_style)
      .toBe('accent')

    // Present mode paints the accent background on the section.
    await page.getByTestId('report-present').click()
    const section = page.locator('[data-testid="present-slide"]').first()
    await expect(section).toBeVisible()
    // The accent preset adds a slide-style-accent class (export) / inline bg
    // (present). Assert the section's background is not the plain default.
    const bg = await section.evaluate((el) => getComputedStyle(el).backgroundImage + '|' + getComputedStyle(el).backgroundColor)
    console.log('[slide-style] accent section bg =', bg)
    await page.screenshot({ path: join(SHOTS, '01-accent-slide.png') })
    // A radial-gradient (accent) or a non-transparent tint distinguishes it.
    expect(bg).not.toBe('none|rgba(0, 0, 0, 0)')

    ctx.assertNoJsErrors?.()
  })
})
