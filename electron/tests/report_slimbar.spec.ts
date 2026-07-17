/**
 * report_slimbar.spec.ts — the Report Builder's slim figure-edit bar + floating
 * AnnotationPopover (the Phase-2 redesign that retired the below-plot edit dock).
 *
 * Verifies, in the REAL Electron app:
 *   1. Edit mode shows the SLIM BAR (figcell-edit-<cellId>) with the add-
 *      annotation palette + the per-layer row — and NONE of the retired dock
 *      sections (chips wrapper, panel block, figure-edit block, grid summary).
 *      A single-panel figure renders no targeting chips at all.
 *   2. "+ Circle" appends a spec annotation; the floating AnnotationPopover
 *      opens on a widget pointer_up re-dispatched as the spyde:figure_event
 *      CustomEvent (widget id resolved from report_state's ann_widgets map);
 *      a preset color dot click round-trips into the report doc (edgecolors).
 *
 * Interaction notes (proven patterns from report_edit2/report_annotations):
 *   - figcell chrome only mounts on a bubbling `mouseover` dispatch (the OOPIF
 *     iframe eats real hover) — never .hover().
 *   - The popover-open event is dispatched RENDERER-ONLY via the CustomEvent
 *     mirror (SpyDEContext re-dispatches every awi_event as spyde:figure_event);
 *     posting a real awi_event postMessage would ALSO hit the backend's drag-
 *     persist handler with whatever geometry the blob carried.
 *   - report_state is authoritative: ann_widgets ships on the figure cell only
 *     while in edit mode, populated by the interactive rebuild — poll for it.
 *
 * Real Dask + bundled si_grains. Screenshots to report_slimbar_shots/ (the
 * author READS them — a blank or mis-positioned popover is a failure).
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_slimbar_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)   // navigator + signal
  await page.waitForTimeout(2500)                 // let the DP paint
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

// ── shared helpers (proven shapes from report_edit2) ────────────────────────────

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

/** The report doc's figure cell (panels + ann_widgets) via the test hook. */
async function docCell(page: any, cellId: string): Promise<any> {
  return await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    return d?.cells?.find((c: any) => c.id === cid) ?? null
  }, cellId)
}

/** A fresh single-panel figure cell in a NEW report. Returns {cellId}. */
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
    .evaluate((el: HTMLElement) => el.setAttribute('data-slim-sig', '1'))
  await dragToBody(page, '[data-slim-sig="1"]')
  await sig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.removeAttribute('data-slim-sig'))

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2500)   // let the figure paint
  return { cellId: await figCellId(page) }
}

/** Report-scoped backend-error assertion. */
async function assertNoBackendErrors(tag: string) {
  const errs = backendErrorLines(ctx.backend)
    .filter((l: string) => /report|repfig|annotation|panel|figure|layout/i.test(l))
  if (errs.length) console.log(`[${tag}] backend error lines:\n` + errs.join('\n'))
  expect(errs, 'report-related Python tracebacks/errors in backend log').toEqual([])
}

// Shared across the serial tests (test 2 continues on test 1's cell).
let cellId = ''
let panelId = ''

// ── 1: slim bar renders (palette + layer row, no legacy dock) ───────────────────

test('1) slim bar: add palette + layer row, no retired dock sections, no chips on single panel', async () => {
  const { page } = ctx
  ;({ cellId } = await makeFigureCell(page))

  // Enter edit mode: mount the chrome with a bubbling mouseover (NEVER hover()
  // — the OOPIF iframe eats it), then click the ✎ toggle.
  const figCell = page.locator(`[data-testid="report-figcell-${cellId}"]`)
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const toggle = page.getByTestId(`report-figcell-edit-toggle-${cellId}`)
  await expect(toggle).toBeVisible()
  await toggle.click()

  const bar = page.getByTestId(`figcell-edit-${cellId}`)
  await expect(bar).toBeVisible({ timeout: 10_000 })

  // Panel id from the authoritative report doc.
  const cell = await docCell(page, cellId)
  panelId = cell?.figure?.panels?.[0]?.id
  expect(panelId, 'no panel id in the report doc').toBeTruthy()

  // The add-annotation palette targets the ONLY panel directly (no chips).
  for (const k of ['text', 'circle', 'rect', 'arrow']) {
    await expect(page.getByTestId(`figcell-add-${k}-${panelId}`)).toBeVisible()
  }
  // The per-layer row (base layer): cmap select + alpha slider.
  await expect(bar.locator('[data-testid^="figcell-layer-cmap-"]').first()).toBeVisible()
  await expect(bar.locator('[data-testid^="figcell-layer-alpha-"]').first()).toBeVisible()

  // NO retired dock sections and NO targeting chips (single-panel figure).
  await expect(page.getByTestId(`figcell-chips-${cellId}`)).toHaveCount(0)
  await expect(page.locator(`[data-testid="figcell-panel-${panelId}"]`)).toHaveCount(0)
  await expect(page.getByTestId(`figcell-figure-edit-${cellId}`)).toHaveCount(0)
  await expect(page.getByTestId(`figcell-grid-summary-${cellId}`)).toHaveCount(0)
  await expect(page.locator('[data-testid^="figcell-chip-"]')).toHaveCount(0)
  // No annotation rows outside the (closed) popover.
  await expect(page.locator('[data-testid^="figcell-annotation-"]')).toHaveCount(0)

  await page.screenshot({ path: join(SHOTS, '01-slim-bar.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('slimbar-1')
})

// ── 2: + Circle → popover via spyde:figure_event → preset color round-trip ─────

test('2) + Circle → popover opens over the figure → preset dot recolors the spec', async () => {
  const { page } = ctx
  const figIdBefore = await reportFigId(page)

  await page.getByTestId(`figcell-add-circle-${panelId}`).click()

  // The annotation lands in the spec AND the interactive rebuild populates
  // ann_widgets (report_state is authoritative — poll it, not a sleep).
  const widgetIdOf = async (): Promise<string | null> => {
    const cell = await docCell(page, cellId)
    const m = cell?.ann_widgets ?? {}
    for (const wid of Object.keys(m)) {
      const v = m[wid]
      if (v && v.panel_id === panelId && Number(v.index) === 0) return wid
    }
    return null
  }
  await expect.poll(async () =>
    (await docCell(page, cellId))?.figure?.panels?.[0]?.annotations?.length ?? 0,
    { timeout: 15_000, message: '+ Circle did not append a spec annotation' })
    .toBe(1)
  await expect.poll(widgetIdOf, {
    timeout: 15_000, message: 'ann_widgets never mapped the circle widget (edit rebuild)',
  }).not.toBeNull()

  // Wait for the rebuilt figure's iframe to be PROMOTED (seamless swap done):
  // the figId changes and exactly one iframe remains in the cell.
  await expect.poll(async () => await reportFigId(page), {
    timeout: 15_000, message: 'add-circle did not rebuild the figure',
  }).not.toBe(figIdBefore)
  await expect.poll(async () => await page.locator(
    `[data-testid="report-figcell-${cellId}"] iframe[data-testid^="figure-"]`).count(),
    { timeout: 15_000, message: 'seamless iframe swap never settled' })
    .toBe(1)
  await page.waitForTimeout(1200)   // let the widget paint

  // Open the popover: re-dispatch the widget pointer_up as the CustomEvent
  // mirror (renderer-only). Retry with a re-read figId in case a swap raced.
  const widgetId = (await widgetIdOf())!
  const popover = page.getByTestId(`figcell-annotation-${panelId}-0`)
  for (let attempt = 0; attempt < 3; attempt++) {
    const figId = (await reportFigId(page))!
    await page.evaluate(({ fid, pid, wid }: any) => {
      window.dispatchEvent(new CustomEvent('spyde:figure_event', {
        detail: { figId: fid, event: { panel_id: pid, event_type: 'pointer_up', widget_id: wid } },
      }))
    }, { fid: figId, pid: panelId, wid: widgetId })
    try { await expect(popover).toBeVisible({ timeout: 4_000 }); break }
    catch { /* stale figId — retry */ }
  }
  await expect(popover).toBeVisible({ timeout: 4_000 })
  // Popover carries the old row controls: color swatch + preset dots + width.
  await expect(page.getByTestId(`figcell-annotation-color-${panelId}-0`)).toBeVisible()
  await expect(page.getByTestId(`figcell-annotation-${panelId}-0-width`)).toBeVisible()

  // Screenshot with the popover OPEN over the figure (read by the author:
  // the popover must be visibly positioned over the figure box, not stranded).
  await page.screenshot({ path: join(SHOTS, '02-popover-open.png') })

  // Click the green preset dot (#a6e3a1) → repfig_update_annotation → the
  // spec's edgecolors round-trips (poll report_state; the popover closes on
  // the state-driven cell.figure change — by design).
  await page.getByTestId(`figcell-annotation-${panelId}-0-preset-a6e3a1`).click()
  await expect.poll(async () => {
    const cell = await docCell(page, cellId)
    const ann = cell?.figure?.panels?.[0]?.annotations?.[0]
    return ann ? String(ann.edgecolors ?? '') : null
  }, { timeout: 10_000, message: 'preset dot did not persist edgecolors=#a6e3a1' })
    .toBe('#a6e3a1')

  await page.waitForTimeout(1500)   // in-place recolor settles
  await page.screenshot({ path: join(SHOTS, '03-after-preset-recolor.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('slimbar-2')
})

test('3) final: no report-related Python tracebacks in the backend log', async () => {
  await assertNoBackendErrors('slimbar-final')
})
