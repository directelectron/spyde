/**
 * report_callouts.spec.ts — Report Builder Phase 3 fresh-slice zoom-inset
 * callouts, driven end-to-end on the synthetic in-situ MOVIE (1-D time nav —
 * the "+ Time callouts" path; the 4-D / marker-drag paths are covered by
 * spyde/tests/migrated/test_report_callouts.py).
 *
 * Flow (launch pattern from report_slimbar.spec.ts):
 *   1. load_test_data_movie → drag the SIGNAL window pill into the report →
 *      figure cell mounts.
 *   2. Edit mode → the slim bar shows "+ Callout" AND "+ Time callouts"
 *      (nav_dims === 1 stamped on the shipped panel dict gates both).
 *   3. Click "+ Time callouts" → poll report_state for THREE insets carrying
 *      time_index 0 / n//2 / n-1 with spread top anchors → screenshot.
 *      The movie frames carry a per-frame index band + moving content, so
 *      three insets showing IDENTICAL images = a fresh-slice failure (the
 *      author READS the screenshot).
 *
 * Interaction notes (proven patterns):
 *   - figcell chrome only mounts on a bubbling `mouseover` dispatch (the OOPIF
 *     iframe eats real hover) — never .hover().
 *   - report_state is authoritative — poll it via window._spyde_test_report,
 *     never a fixed sleep.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_callouts_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_movie')
  await waitForSubwindowCount(page, 2, 120_000)   // 1-D time navigator + frame
  await page.waitForTimeout(2500)                 // let the first frame paint
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

// ── shared helpers (proven shapes from report_slimbar) ─────────────────────────

/** Native HTML5 drag src→report body, one shared DataTransfer. */
async function dragToBody(page: any, srcSel: string) {
  await page.evaluate(({ srcSel }: any) => {
    const src = document.querySelector(srcSel) as HTMLElement
    const dst = document.querySelector('[data-testid="report-body"]') as HTMLElement
    if (!src || !dst) throw new Error('drag src/report-body not found')
    const dt = new DataTransfer()
    const fire = (target: HTMLElement, type: string) => {
      const r = target.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
    }
    fire(src, 'dragstart')
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
  }, { srcSel })
}

/** The single figure cell's id. */
async function figCellId(page: any): Promise<string> {
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  return await figCell.evaluate((el: HTMLElement) =>
    (el.getAttribute('data-testid') || '').replace('report-figcell-', ''))
}

/** The anyplotlib figId of the (single) report figure iframe. */
async function reportFigId(page: any): Promise<string | null> {
  return await page.evaluate(() => {
    const cell = document.querySelector('[data-testid^="report-figcell-"]')
    const ifr = cell?.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
    if (!ifr) return null
    return (ifr.getAttribute('data-testid') || '').replace('figure-', '')
  })
}

/** The report doc's figure cell (panels incl. nav_dims/insets) via the hook. */
async function docCell(page: any, cellId: string): Promise<any> {
  return await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    return d?.cells?.find((c: any) => c.id === cid) ?? null
  }, cellId)
}

/** A fresh single-panel figure cell in a NEW report from the SIGNAL window. */
async function makeFigureCell(page: any): Promise<{ cellId: string }> {
  if (!(await page.getByTestId('report-sidebar').count())) {
    await page.getByTestId('toggle-report').click()
  }
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new')
  await expect(page.getByTestId('report-body')).toBeVisible({ timeout: 10_000 })
  await expect(page.locator('[data-testid^="report-figcell-"]')).toHaveCount(0)

  const sig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  await sig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-callout-sig', '1'))
  await dragToBody(page, '[data-callout-sig="1"]')
  await sig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.removeAttribute('data-callout-sig'))

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2500)   // let the figure paint
  return { cellId: await figCellId(page) }
}

/** Report-scoped backend-error assertion. */
async function assertNoBackendErrors(tag: string) {
  const errs = backendErrorLines(ctx.backend)
    .filter((l: string) => /report|repfig|callout|inset|panel|figure/i.test(l))
  if (errs.length) console.log(`[${tag}] backend error lines:\n` + errs.join('\n'))
  expect(errs, 'report-related Python tracebacks/errors in backend log').toEqual([])
}

// Shared across the serial tests.
let cellId = ''
let panelId = ''

// ── 1: slim bar shows the callout buttons on a 1-D-nav panel ───────────────────

test('1) movie figure cell: slim bar gates + Callout and + Time callouts on nav_dims', async () => {
  const { page } = ctx
  ;({ cellId } = await makeFigureCell(page))
  await page.screenshot({ path: join(SHOTS, '01-figure-cell.png') })

  // Enter edit mode (bubbling mouseover mounts the chrome; never hover()).
  const figCell = page.locator(`[data-testid="report-figcell-${cellId}"]`)
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const toggle = page.getByTestId(`report-figcell-edit-toggle-${cellId}`)
  await expect(toggle).toBeVisible()
  await toggle.click()
  await expect(page.getByTestId(`figcell-edit-${cellId}`)).toBeVisible({ timeout: 10_000 })

  // nav_dims === 1 must be stamped on the shipped panel dict.
  await expect.poll(async () => {
    const cell = await docCell(page, cellId)
    panelId = cell?.figure?.panels?.[0]?.id ?? ''
    return cell?.figure?.panels?.[0]?.nav_dims ?? null
  }, { timeout: 15_000, message: 'panel never carried nav_dims=1' }).toBe(1)
  expect(panelId).toBeTruthy()

  await expect(page.getByTestId(`figcell-add-callout-${panelId}`)).toBeVisible()
  await expect(page.getByTestId(`figcell-add-time-callouts-${panelId}`)).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '02-slim-bar-callout-buttons.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('callouts-1')
})

// ── 2: + Time callouts → three fresh-slice insets at t=0 / mid / end ───────────

test('2) + Time callouts → 3 insets (t=0/mid/end, spread anchors) render', async () => {
  const { page } = ctx
  const figIdBefore = await reportFigId(page)

  await page.getByTestId(`figcell-add-time-callouts-${panelId}`).click()

  // report_state is authoritative: three insets with time_index + anchor.
  await expect.poll(async () => {
    const cell = await docCell(page, cellId)
    return (cell?.figure?.panels?.[0]?.insets ?? []).length
  }, { timeout: 30_000, message: '+ Time callouts did not append 3 insets' }).toBe(3)

  const cell = await docCell(page, cellId)
  const insets = cell.figure.panels[0].insets
  const times = insets.map((i: any) => Number(i.time_index))
  // Movie is 6 frames → t = 0, 3, 5 (start / middle / end).
  expect(times[0]).toBe(0)
  expect(times[2]).toBeGreaterThan(times[1])
  expect(times[1]).toBeGreaterThan(times[0])
  for (const ins of insets) {
    expect(Array.isArray(ins.anchor)).toBe(true)
    expect(ins.connector).toBeNull()
  }
  // Distinct spread anchors (top-left / top-center / top-right).
  const xs = insets.map((i: any) => i.anchor[0])
  expect(new Set(xs).size).toBe(3)

  // The rebuild swaps the iframe; wait for the promotion to settle.
  await expect.poll(async () => await reportFigId(page), {
    timeout: 30_000, message: 'time callouts did not rebuild the figure',
  }).not.toBe(figIdBefore)
  await expect.poll(async () => await page.locator(
    `[data-testid="report-figcell-${cellId}"] iframe[data-testid^="figure-"]`).count(),
    { timeout: 15_000, message: 'seamless iframe swap never settled' })
    .toBe(1)
  await page.waitForTimeout(3500)   // let the three inset images paint

  // READ THIS SHOT: three insets along the top edge, each a DIFFERENT movie
  // frame (index band / content differs) — identical or blank insets = failure.
  await page.screenshot({ path: join(SHOTS, '03-time-callouts.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('callouts-2')
})

test('3) final: no callout-related Python tracebacks in the backend log', async () => {
  await assertNoBackendErrors('callouts-final')
})
