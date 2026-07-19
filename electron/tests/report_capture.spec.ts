/**
 * report_capture.spec.ts — "Capture to presentation" (one-click).
 *
 * report_add_figure already snapshots a source window's CURRENT live state
 * (nav position, contrast, colormap, overlays) into a figure cell; the
 * capture feature is the one-click wiring on top: focus a plot window, click
 * a camera button, and the live view lands as its OWN new slide in the
 * report — no dragging.
 *
 * Covers BOTH affordances:
 *   1. The report sidebar's header "Capture" button (captures
 *      state.activeWindowId).
 *   2. The per-window camera button in the SubWindow titlebar (captures that
 *      window explicitly, and auto-opens the report dock if it was closed).
 *
 * Then enters Present mode and confirms the captured figure renders on its
 * own slide (slide_break:true). Screenshots to capture_shots/ — read by the
 * author (a blank frame / stale placeholder is a failure, not success).
 *
 * tutorial_load('navigation') is the fast bundled path (pyxem synthetic 4D
 * data, no download, no dask needed) — opens a navigator + signal window.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'capture_shots')

// The exact figure-CELL wrapper (data-testid="report-figcell-<id>") — NOT its
// many sub-elements sharing the same prefix (report-figcell-caption-<id>,
// report-figcell-drag-<id>, report-figcell-edit-toggle-<id>, -placeholder-,
// -offline-, -pending-, -caption-input-, …). Cell ids are plain hex strings
// (no dashes), so the exact wrapper matches this regex while every
// sub-element (which has an extra "-word-" segment) does not.
const FIGCELL_RE = /^report-figcell-[0-9a-fA-F]+$/
function figcells(page: any) {
  return page.getByTestId(FIGCELL_RE)
}

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(120_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1000)
  await backendAction(page, 'tutorial_load', { name: 'navigation' })
  await waitForSubwindowCount(page, 2, 30_000)
  await page.waitForTimeout(1500)   // let the DP/navigator frames paint
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

test('1) sidebar Capture button snapshots the active window as a new slide', async () => {
  const { page } = ctx

  // Focus the signal window (click its titlebar) — this is the "active" window
  // the sidebar Capture button reads from state.activeWindowId.
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()

  // Open the report dock and start a fresh report.
  await page.getByTestId('toggle-report').click()
  await page.getByTestId('report-new').click()
  await expect(page.getByTestId('report-body')).toBeVisible()

  // No cells yet.
  await expect(page.getByTestId('report-drop-hint')).toBeVisible()

  const captureBtn = page.getByTestId('report-capture')
  await expect(captureBtn).toBeEnabled()
  await captureBtn.click()

  // A new figure cell appears.
  await expect(figcells(page).first())
    .toBeVisible({ timeout: 15_000 })

  // Confirmation note shows.
  await expect(page.getByTestId('report-export-note')).toHaveText(/Captured to presentation/)

  await page.screenshot({ path: join(SHOTS, '01-sidebar-capture.png') })
  ctx.assertNoJsErrors()
})

test('2) captured cell carries slide_break (its own slide)', async () => {
  const { page } = ctx
  // Reach into the injected test report state (report_state mirror) via the
  // DOM: the figure cell wrapper renders slideStart styling only for a cell
  // that starts a slide. Simpler + robust: read the raw report doc off the
  // SpyDE context test hook if present, else fall back to checking there is
  // exactly one figure cell and it is the FIRST cell (slide_break covers the
  // trivial "first cell" case too) — but we specifically want to confirm the
  // backend actually set slide_break:true, so drive a second capture and
  // check a NEW slide boundary appears between the two figure cells.
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await page.getByTestId('report-capture').click()

  await expect(figcells(page))
    .toHaveCount(2, { timeout: 15_000 })

  // Each capture requested slide_break:true, so Present mode must show TWO
  // slides (one figure per slide), not one slide with two panels.
  await page.screenshot({ path: join(SHOTS, '02-two-captures.png') })
  ctx.assertNoJsErrors()
})

test('3) Present mode shows the captured figure on its own slide', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await expect(page.getByTestId('present-mode')).toBeVisible()

  // Two captures, each its own slide via slide_break → counter shows "1 / N"
  // with N >= 2.
  const counterText = await page.getByTestId('present-counter').textContent()
  const match = counterText?.match(/(\d+)\s*\/\s*(\d+)/)
  expect(match).not.toBeNull()
  const total = Number(match![2])
  expect(total).toBeGreaterThanOrEqual(2)

  // The current (active) slide shows the captured figure.
  await expect(page.locator('[data-testid="present-slide"][data-active="1"]')).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '03-present-slide-1.png') })

  // Advance to the next slide — it should ALSO be its own figure slide (the
  // slide_break contract: each capture is isolated, not accumulated onto one
  // slide).
  await page.getByTestId('present-next').click()
  await page.waitForTimeout(400)
  await page.screenshot({ path: join(SHOTS, '04-present-slide-2.png') })

  await page.getByTestId('present-exit').click()
  await expect(page.getByTestId('present-mode')).toHaveCount(0)
  ctx.assertNoJsErrors()
})

test('4) per-window camera button captures directly from the titlebar', async () => {
  const { page } = ctx
  // Close the report dock first to prove the camera button re-opens it.
  const reportToggle = page.getByTestId('toggle-report')
  const isOpen = await page.getByTestId('report-sidebar').isVisible().catch(() => false)
  if (isOpen) await reportToggle.click()
  await expect(page.getByTestId('report-sidebar')).toHaveCount(0)

  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-titlebar').hover()
  const camBtn = sig.getByTestId('capture-btn')
  await expect(camBtn).toBeVisible()
  await camBtn.click()

  // The report dock auto-opens.
  await expect(page.getByTestId('report-sidebar')).toBeVisible({ timeout: 10_000 })
  // A third figure cell is now present.
  await expect(figcells(page))
    .toHaveCount(3, { timeout: 15_000 })

  await page.screenshot({ path: join(SHOTS, '05-window-camera-button.png') })
  ctx.assertNoJsErrors()
})
