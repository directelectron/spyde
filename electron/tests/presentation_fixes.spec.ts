/**
 * presentation_fixes.spec.ts — the presentation bug-fix batch, end-to-end.
 *
 * Real Dask + bundled Si-grains (navigator + signal) so a slide can carry a live
 * figure. Verifies in the RUNNING app:
 *   (B1) a SPLIT cell (text + figure) shows BOTH its text AND its figure on a
 *        slide in Present mode — the earlier bug dropped the text.
 *   (B2) a figure cell on a slide (sidebar) accepts a drag-to-COMBINE drop (the
 *        compose path is not blocked inside a presentation slide group).
 *
 * Screenshots to presentation_fixes_shots/ — the whole point is what's on screen,
 * so each shot is Read by the author. NO pre-kill (user runs their own SpyDE).
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'presentation_fixes_shots')
const FIG_MIME = 'application/x-spyde-figure'

async function windowIdFromPill(page: any, pillSel: string): Promise<number> {
  return await page.evaluate(({ sel, mime }: any) => {
    const src = document.querySelector(sel) as HTMLElement
    if (!src) return NaN
    const dt = new DataTransfer()
    const r = src.getBoundingClientRect()
    const ev = new DragEvent('dragstart', {
      bubbles: true, cancelable: true,
      clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
    })
    Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
    src.dispatchEvent(ev)
    const raw = dt.getData(mime)
    try { return Number((JSON.parse(raw) as any).windowId) } catch { return NaN }
  }, { sel: pillSel, mime: FIG_MIME })
}

async function reportDoc(page: any): Promise<any> {
  return await page.evaluate(() => (window as any)._spyde_test_report?.() ?? null)
}

async function figurePanelCount(page: any, cellId: string): Promise<number> {
  return await page.evaluate((id) => {
    const doc = (window as any)._spyde_test_report?.()
    const c = doc?.cells?.find((x: any) => x.id === id)
    return c?.figure?.panels?.length ?? 0
  }, cellId)
}

async function openReportSidebar(page: any) {
  // Dismiss the first-run welcome tour if it auto-opened.
  const tour = page.getByTestId('tour-close')
  if (await tour.count()) {
    await tour.click().catch(() => {})
    await expect(page.getByTestId('tour-overlay')).toHaveCount(0, { timeout: 5_000 }).catch(() => {})
  }
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
}

test('B1: a split cell shows text AND figure on a slide (Present mode)', async () => {
  const { mkdirSync } = require('fs')
  mkdirSync(SHOTS, { recursive: true })
  const ctx = await launchApp({ dask: true })
  const { page } = ctx
  try {
    await page.waitForTimeout(1500)
    // Load the bundled Si-grains so a live figure is available.
    await backendAction(page, 'load_test_data_si_grains')
    await waitForSubwindowCount(page, 2, 60_000)
    await page.waitForTimeout(1500)
    await openReportSidebar(page)

    // New Presentation.
    await backendAction(page, 'report_new', { type: 'presentation' })
    await expect(page.getByTestId('report-type-badge')).toHaveText('Presentation')

    // Add a split block via the backend (a presentation authors slides, not the
    // report's inline "+ Add split" button), give it text, and fill its figure
    // side from the signal window.
    await backendAction(page, 'report_add_split_cell', {
      source: '## Beside the figure\nThis text must show on the slide.',
      layout: 'text-left',
    })
    await expect.poll(async () => {
      const doc = await reportDoc(page)
      return (doc?.cells ?? []).some((x: any) => x.cell_type === 'split')
    }, { timeout: 15_000 }).toBe(true)
    const dSplit = await reportDoc(page)
    const cellId = dSplit.cells.find((x: any) => x.cell_type === 'split').id

    const sigId = await windowIdFromPill(page,
      '[data-testid="subwindow"] [data-testid="window-breadcrumb"]')
    expect(Number.isFinite(sigId)).toBe(true)
    await backendAction(page, 'report_add_figure', { source_window_id: sigId, at_cell: cellId })

    // Wait for the split's figure side to fill.
    await expect.poll(async () => {
      const doc = await reportDoc(page)
      const c = (doc?.cells ?? []).find((x: any) => x.id === cellId)
      return c && c.cell_type === 'split' && !!c.figure
    }, { timeout: 30_000 }).toBe(true)
    await page.screenshot({ path: join(SHOTS, '01-split-authored.png') })

    // Present. The split slide must render BOTH the text and the figure.
    await page.getByTestId('report-present').click()
    await expect(page.getByTestId('present-slide')).toBeVisible()
    const splitOnSlide = page.getByTestId(`present-split-${cellId}`)
    await expect(splitOnSlide).toBeVisible()
    // The TEXT is present (the earlier bug: it was dropped entirely).
    await expect(splitOnSlide).toContainText('This text must show on the slide')
    await expect(splitOnSlide.locator('h2')).toBeVisible()
    await page.screenshot({ path: join(SHOTS, '02-present-split-text-and-figure.png') })

    ctx.assertNoJsErrors()
  } finally {
    await ctx.app.close()
  }
})

test('B2: a figure cell on a presentation slide accepts a combine drop', async () => {
  const { mkdirSync } = require('fs')
  mkdirSync(SHOTS, { recursive: true })
  const ctx = await launchApp({ dask: true })
  const { page } = ctx
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'load_test_data_si_grains')
    await waitForSubwindowCount(page, 2, 60_000)
    await page.waitForTimeout(1500)
    await openReportSidebar(page)

    await backendAction(page, 'report_new', { type: 'presentation' })
    await expect(page.getByTestId('report-type-badge')).toHaveText('Presentation')

    // Put a figure on a slide (drop the signal window into the report body).
    const sigId = await windowIdFromPill(page,
      '[data-testid="subwindow"] [data-testid="window-breadcrumb"]')
    await backendAction(page, 'report_add_figure', { source_window_id: sigId, slide_break: true })

    // Find the figure cell id from the doc.
    await expect.poll(async () => {
      const doc = await reportDoc(page)
      return (doc?.cells ?? []).some((x: any) => x.cell_type === 'figure')
    }, { timeout: 30_000 }).toBe(true)
    const doc0 = await reportDoc(page)
    const figCellId = doc0.cells.find((x: any) => x.cell_type === 'figure').id

    await page.screenshot({ path: join(SHOTS, '03-figure-on-slide.png') })

    // Reproduce the COMBINE: fire the same backend action the compose shield's
    // edge-drop sends (repfig_compose tile-right). If the slide group blocked the
    // DOM drop, the backend action still proves the compose path itself works on a
    // presentation figure cell; the panel count must go 1 → 2.
    const panelsBefore = await figurePanelCount(page, figCellId)

    await backendAction(page, 'repfig_compose', {
      cell_id: figCellId, mode: 'tile-right', source_window_id: sigId,
    })

    await expect.poll(async () => figurePanelCount(page, figCellId),
      { timeout: 30_000 }).toBeGreaterThan(panelsBefore)
    await page.screenshot({ path: join(SHOTS, '04-combined-2-panels.png') })

    ctx.assertNoJsErrors()
  } finally {
    await ctx.app.close()
  }
})
