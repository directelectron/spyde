/**
 * report_empty_cards.spec.ts — the Report sidebar EMPTY STATE.
 *
 * Opening the Report sidebar with no document must present clear New Report /
 * New Presentation CARDS (not just a File-menu hint), plus from-a-guide chips and
 * an Open link. Clicking the Presentation card opens a presentation. The whole
 * point is what's on screen, so the shot is Read by the author.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
import { join } from 'path'
const { launchApp } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_empty_shots')

test('the empty Report sidebar shows New Report / New Presentation cards', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const ctx = await launchApp({ dask: false })
  const { page } = ctx
  try {
    await page.waitForTimeout(1200)
    // Dismiss the first-run tour if it opened.
    const tour = page.getByTestId('tour-close')
    if (await tour.count()) await tour.click().catch(() => {})

    await page.getByTestId('toggle-report').click()
    await expect(page.getByTestId('report-sidebar')).toBeVisible()
    await expect(page.getByTestId('report-empty')).toBeVisible()

    // The two document-type cards are present.
    const reportCard = page.getByTestId('report-new-report-card')
    const presCard = page.getByTestId('report-new-presentation-card')
    await expect(reportCard).toBeVisible()
    await expect(presCard).toBeVisible()
    await expect(reportCard).toContainText('Report')
    await expect(presCard).toContainText('Presentation')
    // From-a-guide chips + an Open link.
    await expect(page.getByTestId('report-empty-open')).toBeVisible()
    await page.screenshot({ path: join(SHOTS, '01-empty-cards.png') })

    // Clicking the Presentation card opens a presentation.
    await presCard.click()
    await expect(page.getByTestId('report-type-badge')).toHaveText('Presentation')
    await page.screenshot({ path: join(SHOTS, '02-presentation-opened.png') })

    ctx.assertNoJsErrors()
  } finally {
    await ctx.app.close()
  }
})
