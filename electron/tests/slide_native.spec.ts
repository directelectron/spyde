/**
 * slide_native.spec.ts — Wave C: the SLIDE-NATIVE presentation authoring surface
 * in the Report sidebar, end-to-end.
 *
 * Re-surfaces the per-slide controls Wave B ripped out (title / background /
 * notes) as a clean, slide-native design. This spec DRIVES THE NEW UI the way a
 * user would (clicking the labeled per-slide header controls — not raw backend
 * actions where a UI control exists) and asserts:
 *
 *   1) New Presentation → add 2 slides → the sidebar reads as labeled SLIDE groups
 *      (Slide 1 / Slide 2), each with a per-slide header.
 *   2) Toggle Slide 1 → Title slide via the labeled "Title slide" control →
 *      slide_kind='title' + Present mode renders it big-centered.
 *   3) Set Slide 1 background → Accent via the labeled Background picker →
 *      slide_style='accent' + Present mode paints the accent stage.
 *   4) Add speaker notes below Slide 1 (the notes area under the slide, reusing
 *      SlideNotesEditor) → notes persist + the presenter view shows them.
 *   5) A REPORT (New Report) shows NO slide grouping / headers / notes — a flat
 *      cell list.
 *
 * Screenshots the slide-grouped presentation sidebar (per-slide headers + notes)
 * AND a flat report sidebar to slide_native_shots/ and the author Reads them — a
 * cluttered result is a failure even when the selectors pass.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'slide_native_shots')

const NOTES_1 = 'Open with the beamstop, then dwell 5s before advancing.'

/** The live report document (the renderer test hook). */
async function reportDoc(page: any): Promise<any> {
  return await page.evaluate(() => (window as any)._spyde_test_report?.())
}

/** The active slide's counter text ("n / N"). */
async function counterText(page: any): Promise<string> {
  return (await page.getByTestId('present-counter').textContent())?.trim() ?? ''
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
  await page.waitForTimeout(2000)
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally {
    await ctx?.app?.close()
  }
})

// ── 1) Presentation → two labeled SLIDE groups with per-slide headers ─────────

test('1) New Presentation + 2 slides → labeled Slide 1 / Slide 2 groups', async () => {
  const { page } = ctx
  await backendAction(page, 'report_new', { type: 'presentation' })
  await backendAction(page, 'report_set_title', { title: 'Slide Native Demo' })
  await expect(page.getByTestId('report-body')).toBeVisible()

  // Slide 1 — a text cell (starts slide 0).
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '# Orientation Mapping\n\nA SpyDE presentation',
    html: '<h1>Orientation Mapping</h1><p>A SpyDE presentation</p>',
  })
  // Slide 2 — a second text cell with a slide_break (a NEW slide).
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '## Results\n\nWhat we found.',
    html: '<h2>Results</h2><p>What we found.</p>',
    slide_break: true,
  })

  // The sidebar renders TWO labeled slide groups.
  await expect(page.getByTestId('report-slide-0')).toBeVisible({ timeout: 10_000 })
  await expect(page.getByTestId('report-slide-1')).toBeVisible()
  // Each has a per-slide header with a "Slide N" label.
  await expect(page.getByTestId('report-slide-header-0')).toContainText('Slide 1')
  await expect(page.getByTestId('report-slide-header-1')).toContainText('Slide 2')
  // The header carries the labeled per-slide controls (not cryptic glyphs).
  await expect(page.getByTestId('report-slide-title-toggle-0')).toContainText('Title slide')
  await expect(page.getByTestId('report-slide-bg-toggle-0')).toContainText('Default')
  // A collapsible Speaker-notes affordance sits under the slide.
  await expect(page.getByTestId('report-slide-notes-toggle-0')).toContainText('Speaker notes')

  await page.screenshot({ path: join(SHOTS, '01-two-slide-groups.png') })
  ctx.assertNoJsErrors()
})

// ── 2) Title-slide toggle (labeled control) → slide_kind + Present renders big ─

test('2) toggle Slide 1 → Title slide → slide_kind set + big-centered in Present', async () => {
  const { page } = ctx
  const toggle = page.getByTestId('report-slide-title-toggle-0')
  await expect(toggle).toHaveAttribute('data-active', '0')
  await toggle.click()
  // The pill flips to active and the backend records slide_kind='title' on the
  // slide's FIRST cell.
  await expect(toggle).toHaveAttribute('data-active', '1')
  await expect.poll(async () => {
    const doc = await reportDoc(page)
    return doc?.cells?.[0]?.slide_kind
  }, { timeout: 8_000, message: 'slide_kind never became title' }).toBe('title')
  // The group is marked as a title slide.
  await expect(page.getByTestId('report-slide-0')).toHaveAttribute('data-slide-kind', 'title')

  await page.screenshot({ path: join(SHOTS, '02-title-toggled.png') })
  ctx.assertNoJsErrors()
})

// ── 3) Background picker (labeled) → slide_style='accent' + Present paints it ──

test('3) set Slide 1 background → Accent → slide_style set', async () => {
  const { page } = ctx
  await page.getByTestId('report-slide-bg-toggle-0').click()
  await expect(page.getByTestId('report-slide-bg-menu-0')).toBeVisible()
  // The picker offers clearly labeled options with swatches (Default/Plain/Accent).
  await expect(page.getByTestId('report-slide-bg-0-default')).toContainText('Default')
  await expect(page.getByTestId('report-slide-bg-0-plain')).toContainText('Plain')
  await page.getByTestId('report-slide-bg-0-accent').click()
  await expect.poll(async () => {
    const doc = await reportDoc(page)
    return doc?.cells?.[0]?.slide_style
  }, { timeout: 8_000, message: 'slide_style never became accent' }).toBe('accent')
  await expect(page.getByTestId('report-slide-0')).toHaveAttribute('data-slide-style', 'accent')
  // The toggle now reflects the chosen background label.
  await expect(page.getByTestId('report-slide-bg-toggle-0')).toContainText('Accent')

  await page.screenshot({ path: join(SHOTS, '03-accent-set.png') })
  ctx.assertNoJsErrors()
})

// ── 4) Speaker notes below the slide → persist + presenter view shows them ────

test('4) add speaker notes below Slide 1 → persist + shown in presenter view', async () => {
  const { page } = ctx
  // Expand the notes area under Slide 1 (collapsible, reusing SlideNotesEditor).
  await page.getByTestId('report-slide-notes-toggle-0').click()
  const firstId: string = await page.evaluate(async () =>
    (window as any)._spyde_test_report?.()?.cells?.[0]?.id)
  expect(firstId).toBeTruthy()
  const ta = page.getByTestId(`slide-notes-textarea-${firstId}`)
  await expect(ta).toBeVisible({ timeout: 8_000 })
  await ta.fill(NOTES_1)
  // Blur flushes the debounced commit.
  await ta.blur()
  await expect.poll(async () => {
    const doc = await reportDoc(page)
    return doc?.cells?.[0]?.notes
  }, { timeout: 8_000, message: 'notes never committed' }).toBe(NOTES_1)
  // The collapsed toggle shows a "has notes" affordance (data flag).
  await page.getByTestId(`slide-notes-close-${firstId}`).click()
  await expect(page.getByTestId('report-slide-notes-toggle-0'))
    .toHaveAttribute('data-has-notes', '1')

  await page.screenshot({ path: join(SHOTS, '04-notes-and-header.png') })
  ctx.assertNoJsErrors()
})

// ── 5) Present mode reflects the header choices (title + accent + notes) ───────

test('5) Present mode + presenter view reflect the slide-native header', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await expect(page.getByTestId('present-mode')).toBeVisible({ timeout: 10_000 })

  const active = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(active).toBeVisible()
  // Slide 1 was set to a TITLE slide with an ACCENT background via the header.
  await expect(active).toHaveAttribute('data-kind', 'title')
  await expect(active).toHaveAttribute('data-style', 'accent')
  expect(await counterText(page)).toBe('1 / 2')
  // The title heading is markedly larger + centered (present-title-md).
  const h1 = active.locator('h1')
  await expect(h1).toHaveText(/Orientation Mapping/)
  const info = await h1.evaluate((el: HTMLElement) => {
    const cs = getComputedStyle(el)
    return { fontPx: parseFloat(cs.fontSize), textAlign: cs.textAlign }
  })
  expect(info.fontPx).toBeGreaterThan(64)
  expect(info.textAlign).toBe('center')
  await page.screenshot({ path: join(SHOTS, '05-present-title-accent.png') })

  // Toggle the presenter view (S) → the speaker notes set from the header show up.
  await page.keyboard.press('S')
  await expect(page.getByTestId('presenter-view')).toBeVisible({ timeout: 8_000 })
  const notesBody = page.getByTestId('presenter-notes-body')
  await expect(notesBody).toBeVisible()
  await expect(notesBody).toContainText('beamstop')
  await page.screenshot({ path: join(SHOTS, '06-presenter-notes.png') })

  // Exit Present mode back to the authoring surface.
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('present-mode')).toHaveCount(0, { timeout: 8_000 })
  ctx.assertNoJsErrors()
})

// ── 6) "+ Add slide" starter menu creates a new slide with slide_break ────────

test('6) "+ Add slide" → Title slide starter → a THIRD labeled slide group', async () => {
  const { page } = ctx
  await page.getByTestId('report-add-slide').click()
  await expect(page.getByTestId('report-add-slide-menu')).toBeVisible()
  // The starter menu offers slide-native layouts.
  await expect(page.getByTestId('add-slide-text')).toBeVisible()
  await expect(page.getByTestId('add-slide-split')).toBeVisible()
  await page.getByTestId('add-slide-title').click()

  // A third slide group appears; its added cell carries slide_break + is a title.
  await expect(page.getByTestId('report-slide-2')).toBeVisible({ timeout: 8_000 })
  await expect.poll(async () => {
    const doc = await reportDoc(page)
    const c = doc?.cells?.[doc.cells.length - 1]
    return c ? { brk: !!c.slide_break, kind: c.slide_kind } : null
  }, { timeout: 8_000, message: '+ Add slide did not create a title slide' })
    .toEqual({ brk: true, kind: 'title' })
  await expect(page.getByTestId('report-slide-2')).toHaveAttribute('data-slide-kind', 'title')

  await page.screenshot({ path: join(SHOTS, '07-added-title-slide.png') })
  ctx.assertNoJsErrors()
})

// ── 7) A REPORT stays flat — no slide grouping / headers / notes ──────────────

test('7) New Report → flat cell list, NO slide groups / headers / notes', async () => {
  const { page } = ctx
  await backendAction(page, 'report_new', { type: 'report' })
  await backendAction(page, 'report_set_title', { title: 'Flat Report' })
  await expect(page.getByTestId('report-body')).toBeVisible()
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '# A scrolling article\n\nJust cells, no slides.',
    html: '<h1>A scrolling article</h1><p>Just cells, no slides.</p>',
  })
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: 'A second paragraph cell.',
    html: '<p>A second paragraph cell.</p>',
  })

  // The report is a FLAT list — no slide grouping / per-slide header / notes / add
  // slide button appears.
  await expect(page.locator('[data-testid^="report-cell-rendered-"]').first())
    .toBeVisible({ timeout: 10_000 })
  await expect(page.getByTestId('report-slide-0')).toHaveCount(0)
  await expect(page.getByTestId('report-slide-header-0')).toHaveCount(0)
  await expect(page.getByTestId('report-slide-notes-toggle-0')).toHaveCount(0)
  await expect(page.getByTestId('report-add-slide')).toHaveCount(0)
  // The report add-buttons ARE present (cell-native).
  await expect(page.getByTestId('report-add-text')).toBeVisible()
  await expect(page.getByTestId('report-add-split')).toBeVisible()
  await expect(page.getByTestId('report-add-image')).toBeVisible()
  // No Present button on a scrolling report.
  await expect(page.getByTestId('report-present')).toHaveCount(0)

  await page.screenshot({ path: join(SHOTS, '08-flat-report.png') })
  ctx.assertNoJsErrors()
})

// ── 8) No Python tracebacks in the backend log ────────────────────────────────

test('8) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[slide_native] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
