/**
 * report_redesign.spec.ts — Wave B Report/Presentation redesign, end-to-end.
 *
 * Real Dask + bundled-synthetic Si-grains (navigator + signal window) so a
 * split block's figure side can be filled from a live plot window. Drives the
 * NEW clean chrome the way a user would and screenshots each stage — the WHOLE
 * POINT is a de-cluttered top bar, so the shots (Read by the author) are the
 * real test, not just the selectors:
 *
 *   (a) New → the type picker: New Presentation → the type badge reads
 *       "Presentation" + a Present ▶ button is visible; New Report → no Present.
 *   (b) The compact top bar has NO Aa / Rich / Capture / Paste buttons (those
 *       testids are ABSENT) and a single "File ▾" menu IS present.
 *   (c) "+ Add split block" → a split cell renders with a text pane + a figure
 *       drop zone; filling the figure side (report_add_figure {at_cell}) shows
 *       the figure BESIDE the text; the layout switch swaps the two sides.
 *   (d) Ctrl+V still pastes an image (no header Paste button).
 *
 * Screenshots to redesign_shots/ — a still-cluttered bar is a failure even when
 * the selectors pass, so every shot is Read.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'redesign_shots')
const FIG_MIME = 'application/x-spyde-figure'

// A 1×1 red PNG — the smallest real PNG so the bytes round-trip through the
// clipboard without faking (report_image_cell.spec.ts's fixture).
const PNG_1x1 =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'

/** Resolve a window's id by firing a dragstart on its pill and reading the MIME
 *  payload (the windowId is stamped there, not as a DOM attribute). Proven in
 *  report_present.spec.ts / report_tiling.spec.ts. */
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
    const end = new DragEvent('dragend', { bubbles: true, cancelable: true })
    Object.defineProperty(end, 'dataTransfer', { value: dt, configurable: true })
    src.dispatchEvent(end)
    const raw = dt.getData(mime)
    try { return Number((JSON.parse(raw) as any).windowId) } catch { return NaN }
  }, { sel: pillSel, mime: FIG_MIME })
}

/** The live report document (the renderer test hook). */
async function reportDoc(page: any): Promise<any> {
  return await page.evaluate(() => (window as any)._spyde_test_report?.())
}

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  // Dismiss the first-run welcome tour if it auto-opened.
  const tour = page.getByTestId('tour-close')
  if (await tour.count()) {
    await tour.click().catch(() => {})
    await expect(page.getByTestId('tour-overlay')).toHaveCount(0, { timeout: 5_000 }).catch(() => {})
  }
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)   // navigator + signal
  await page.waitForTimeout(2500)                 // let the DP paint
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

// ── (a) New = type picker → badge + Present gating ────────────────────────────

test('a1) New Presentation → type badge = Presentation + Present ▶ visible', async () => {
  const { page } = ctx
  // The compact File menu is the single collapse point for New.
  await page.getByTestId('report-menu-toggle').click()
  await expect(page.getByTestId('report-menu')).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '01-file-menu.png') })
  // Both type options are offered — the New type picker.
  await expect(page.getByTestId('menu-new-report')).toBeVisible()
  await expect(page.getByTestId('menu-new-presentation')).toBeVisible()
  await page.getByTestId('menu-new-presentation').click()
  await expect(page.getByTestId('report-body')).toBeVisible()

  const badge = page.getByTestId('report-type-badge')
  await expect(badge).toHaveText(/Presentation/)
  await expect(badge).toHaveAttribute('data-type', 'presentation')
  // Present ▶ shows only for a presentation.
  await expect(page.getByTestId('report-present')).toBeVisible()
  const doc = await reportDoc(page)
  expect(doc?.type).toBe('presentation')
  await page.screenshot({ path: join(SHOTS, '02-presentation-badge-present.png') })
  ctx.assertNoJsErrors()
})

test('a2) New Report → badge = Report + NO Present button', async () => {
  const { page } = ctx
  await page.getByTestId('report-menu-toggle').click()
  await page.getByTestId('menu-new-report').click()
  await expect(page.getByTestId('report-body')).toBeVisible()

  const badge = page.getByTestId('report-type-badge')
  await expect(badge).toHaveText(/Report/)
  await expect(badge).toHaveAttribute('data-type', 'report')
  // Present ▶ is gated OFF for a scrolling report.
  await expect(page.getByTestId('report-present')).toHaveCount(0)
  const doc = await reportDoc(page)
  expect(doc?.type).toBe('report')
  await page.screenshot({ path: join(SHOTS, '03-report-no-present.png') })
  ctx.assertNoJsErrors()
})

// ── (b) The compact top bar has NONE of the removed chrome ────────────────────

test('b) the top bar is de-cluttered — no Aa / Rich / Capture / Paste', async () => {
  const { page } = ctx
  // Every removed control's testid is ABSENT.
  for (const removed of [
    'report-md-size', 'report-raw-toggle', 'report-capture', 'report-paste',
  ]) {
    await expect(page.getByTestId(removed), `${removed} should be gone`).toHaveCount(0)
  }
  // A single compact File menu IS present (the only chrome collapse point).
  await expect(page.getByTestId('report-menu-toggle')).toBeVisible()
  // The per-window camera Capture button is gone too.
  await expect(page.getByTestId('capture-btn')).toHaveCount(0)
  await page.screenshot({ path: join(SHOTS, '04-clean-top-bar.png') })
  ctx.assertNoJsErrors()
})

// ── (c) Split block: text pane + figure drop zone → fill → layout switch ──────

test('c) + Add split block → drop zone → fill figure beside text → swap sides', async () => {
  const { page } = ctx
  // Insert a split block. It has a text pane + an empty figure drop zone.
  await page.getByTestId('report-add-split').click()
  const split = page.locator('[data-testid^="report-splitcell-"]').first()
  await expect(split).toBeVisible()
  const cellId = await split.evaluate((el) =>
    (el.getAttribute('data-testid') || '').replace('report-splitcell-', ''))

  await expect(page.getByTestId(`report-split-text-${cellId}`)).toBeVisible()
  await expect(page.getByTestId(`report-split-dropzone-${cellId}`)).toBeVisible()
  // Default layout: text on the left.
  await expect(split).toHaveAttribute('data-layout', 'text-left')
  await page.screenshot({ path: join(SHOTS, '05-split-empty-dropzone.png') })

  // Give the text side some content (double-click → edit → commit).
  await page.getByTestId(`report-split-rendered-${cellId}`).dblclick()
  const ta = page.getByTestId(`report-split-textarea-${cellId}`)
  await expect(ta).toBeVisible()
  await ta.fill('## Si grains\nDiffraction beside the text.')
  await ta.press('Control+Enter')
  await expect(page.getByTestId(`report-split-rendered-${cellId}`).locator('h2'))
    .toBeVisible()

  // Fill the FIGURE side by snapshotting the SIGNAL window into the split's
  // figure slot — Wave A: report_add_figure with at_cell targeting a split fills
  // its figure side in place (the same payload the drop-zone drop sends).
  const sigId = await windowIdFromPill(page, '[data-testid="subwindow"] [data-testid="window-breadcrumb"]')
  expect(Number.isFinite(sigId)).toBe(true)
  await backendAction(page, 'report_add_figure', {
    source_window_id: sigId, at_cell: cellId,
  })

  // The figure side is no longer empty — a live figure iframe renders beside the
  // text. Poll the report doc for the split cell becoming non-empty.
  await expect.poll(async () => {
    const doc = await reportDoc(page)
    const c = (doc?.cells ?? []).find((x: any) => x.id === cellId)
    return c ? { empty: !!c.split_empty, hasFig: !!c.figure } : null
  }, { timeout: 30_000, message: 'split figure side never filled' })
    .toEqual({ empty: false, hasFig: true })

  // The drop zone is gone; a figure iframe now lives in the figure pane.
  await expect(page.getByTestId(`report-split-dropzone-${cellId}`)).toHaveCount(0)
  await expect(page.locator(`[data-testid="report-split-figure-${cellId}"] iframe`))
    .toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(1500)   // let the figure paint
  await page.screenshot({ path: join(SHOTS, '06-split-filled-text-left.png') })

  // The layout switch swaps text ↔ figure sides.
  await split.dispatchEvent('mouseover', { bubbles: true })
  await page.getByTestId(`report-split-layout-${cellId}`).click()
  await expect.poll(async () => {
    const doc = await reportDoc(page)
    const c = (doc?.cells ?? []).find((x: any) => x.id === cellId)
    return c?.split_layout
  }, { timeout: 10_000, message: 'layout did not swap' }).toBe('text-right')
  await expect(split).toHaveAttribute('data-layout', 'text-right')
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '07-split-swapped-text-right.png') })
  ctx.assertNoJsErrors()
})

// ── (d) Ctrl+V still pastes an image (no header Paste button) ─────────────────

test('d) Ctrl+V pastes an image cell (Paste button removed)', async () => {
  const { page } = ctx
  // No header Paste button exists anymore — the paste flow is Ctrl+V only.
  await expect(page.getByTestId('report-paste')).toHaveCount(0)
  const before = await page.locator('[data-testid^="report-imgcell-"]').count()
  // Focus the report body so the paste lands on the report chrome, then dispatch
  // ONE synthetic paste carrying a PNG File — the sidebar's paste listener reads
  // the first image/* clipboard item and adds a photo cell.
  await page.getByTestId('report-body').click({ position: { x: 20, y: 20 } })
  await page.evaluate((dataUrl: string) => {
    const dt = new DataTransfer()
    const bin = atob(dataUrl.split(',')[1])
    const arr = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i)
    const file = new File([arr], 'paste.png', { type: 'image/png' })
    dt.items.add(file)
    const ev = new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData: dt })
    const body = document.querySelector('[data-testid="report-body"]') as HTMLElement
    ;(body || document.body).dispatchEvent(ev)
  }, PNG_1x1)

  // At least one MORE image cell than before (the paste added a photo cell).
  await expect.poll(async () => page.locator('[data-testid^="report-imgcell-"]').count(), {
    timeout: 15_000, message: 'Ctrl+V did not add an image cell',
  }).toBeGreaterThan(before)
  await page.waitForTimeout(600)
  await page.screenshot({ path: join(SHOTS, '08-ctrlv-image.png') })
  ctx.assertNoJsErrors()
})
