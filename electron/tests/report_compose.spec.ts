/**
 * report_compose.spec.ts — Report Builder Phase 2, CENTER-drop compose prompts.
 *
 * The probe spec (report_phase2_probe) already covers an EDGE drop → tile. This
 * spec covers the two RICHER center-drop compose modes that open the "Combine
 * figure…" popover (spyde/actions/report/compose.py repfig_query_compose):
 *
 *   a. OVERLAY — embed signal-A in the report, drag signal-B's pill onto the cell
 *      CENTER. Both are si_grains 128×128 DPs (same shape) → the prompt offers
 *      "Overlay". Click it → the cell figure becomes TWO layers (the edit toolbar
 *      lists 2 layers on the panel) and colored (magma-blended) pixels appear in
 *      the cell iframe.
 *   b. CALLOUT — embed the SIGNAL cell, drag its OWN NAVIGATOR (same tree) onto the
 *      center → nav↔signal pair → the prompt offers "Callout". Click it → the cell
 *      rebuilds with an inset panel (the edit toolbar lists a 2nd panel; the cell
 *      iframe repaints).
 *
 * Real Dask + bundled si_grains, loaded TWICE (two trees) so part (a) has two
 * same-shape signals from different trees and part (b) has a signal + its own
 * navigator. Screenshots each stage to report_phase2_shots/ (a blank cell iframe
 * is a failure). Final: assertNoJsErrors + a backend traceback scan.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_phase2_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2000)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 4, 120_000)
  await page.waitForTimeout(3000)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

function sigWindows(page: any) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
}
function navWindows(page: any) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
}

/** Full native HTML5 drag src→dst, shared DataTransfer (report_sidebar pattern). */
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

/**
 * Drop a window pill onto the CENTER of the figure cell's compose shield. Same
 * split as the probe's edge-tile: dragstart + dragover over the report body to
 * promote dragKind='window' (mounts the figcell-compose-shield over the iframe),
 * yield to React, then dragover+drop at the shield CENTER (fx=fy=0.5 → the center
 * zone → repfig_query_compose → the "Combine figure…" prompt).
 */
async function dropOnCellCenter(page: any, srcSel: string) {
  await page.evaluate(({ srcSel }: any) => {
    const src = document.querySelector(srcSel) as HTMLElement
    const body = document.querySelector('[data-testid="report-body"]') as HTMLElement
    if (!src || !body) throw new Error('drag src/report-body not found')
    const dt = new DataTransfer()
    ;(window as any).__cmpdt = dt
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
    fire(body, 'dragover')   // promote dragKind='window' → the shield mounts
  }, { srcSel })
  await page.waitForTimeout(300)

  await page.evaluate(({ srcSel }: any) => {
    const dt = (window as any).__cmpdt as DataTransfer
    const shield = document.querySelector('[data-testid^="figcell-compose-shield-"]') as HTMLElement
    const src = document.querySelector(srcSel) as HTMLElement
    if (!shield) throw new Error('compose shield not mounted')
    const fire = (target: HTMLElement, type: string, fx = 0.5, fy = 0.5) => {
      const r = target.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width * fx, clientY: r.top + r.height * fy,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
    }
    // CENTER of the shield → the 'center' zone (repfig_query_compose).
    fire(shield, 'dragenter'); fire(shield, 'dragover'); fire(shield, 'drop')
    if (src) fire(src, 'dragend')
  }, { srcSel })
}

// Colored (non-gray) pixels inside the REPORT figure cell iframe (a magma overlay
// blend tints the grayscale base). Scoped to the report cell, not the MDI windows.
async function reportCellColoredPixels(page: any): Promise<number> {
  const src: string | null = await page.evaluate(() => {
    const cell = document.querySelector('[data-testid^="report-figcell-"]')
    const ifr = cell?.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
    return ifr?.src || null
  })
  if (!src) return -1
  const frame = page.frames().find((f: any) => f.url() === src)
  if (!frame) return -1
  try {
    return await frame.evaluate(() => {
      let n = 0
      for (const c of Array.from(document.querySelectorAll('canvas'))) {
        const cv = c as HTMLCanvasElement
        const cctx = cv.getContext('2d')
        if (!cctx || !cv.width || !cv.height) continue
        const d = cctx.getImageData(0, 0, cv.width, cv.height).data
        for (let p = 0; p < d.length; p += 4) {
          const r = d[p], g = d[p + 1], b = d[p + 2]
          if ((r > 24 || g > 24 || b > 24) &&
              (Math.max(r, g, b) - Math.min(r, g, b) > 28)) n++
        }
      }
      return n
    })
  } catch { return -1 }
}

/** The report doc's cell via the authoritative test hook (report_slimbar
 *  pattern) — report_state is the source of truth for panels/layers. */
async function docCell(page: any, cellId: string): Promise<any> {
  return await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    return d?.cells?.find((c: any) => c.id === cid) ?? null
  }, cellId)
}

// Open the SLIM edit bar for the (single) figure cell; return {cellId, panelId}.
// Post-Phase-2 there is no `figcell-panel-<id>` block and chips render only on
// multi-panel cells — the panel id comes from report_state instead (the same
// source report_slimbar.spec.ts reads).
async function openEditToolbar(page: any): Promise<{ cellId: string; panelId: string }> {
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  const cellId = await figCell.evaluate((el) =>
    (el.getAttribute('data-testid') || '').replace('report-figcell-', ''))
  // Reveal the hover chrome (React synthesizes onMouseEnter from a delegated
  // mouseover — dispatch a bubbling mouseover, not a raw mouseenter).
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const toggle = page.getByTestId(`report-figcell-edit-toggle-${cellId}`)
  await expect(toggle).toBeVisible()
  // Only open if not already open.
  if (!(await page.getByTestId(`figcell-edit-${cellId}`).count())) await toggle.click()
  await expect(page.getByTestId(`figcell-edit-${cellId}`)).toBeVisible()
  const cell = await docCell(page, cellId)
  const panelId = cell?.figure?.panels?.[0]?.id
  expect(panelId, 'no panel id in the report doc').toBeTruthy()
  return { cellId, panelId }
}

test('setup: open a new report + embed signal-A as a live figure cell', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await page.getByTestId('report-new').click()
  await expect(page.getByTestId('report-body')).toBeVisible()

  // Embed the FIRST signal window (tree A) — drag its pill into the report body.
  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-cmp-sigA', '1'))
  await dragToBody(page, '[data-cmp-sigA="1"]')

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2000)
  await page.screenshot({ path: join(SHOTS, '20-report-embed.png') })
  ctx.assertNoJsErrors()
})

test('a) center-drop signal-B → Overlay prompt → cell becomes two-layer', async () => {
  const { page } = ctx
  const before = await reportCellColoredPixels(page)
  console.log('[compose] report cell colored pixels BEFORE overlay =', before)

  const oldFigId = await page.locator('[data-testid^="report-figcell-"] iframe[data-testid^="figure-"]')
    .first().getAttribute('data-testid')

  // Drag the SECOND signal window's pill (tree B, same 128×128 shape) onto the
  // cell center.
  const sigB = sigWindows(page).nth(1)
  await sigB.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-cmp-sigB', '1'))
  await dropOnCellCenter(page, '[data-cmp-sigB="1"]')

  // The "Combine figure…" prompt opens with an Overlay option (same_shape).
  const prompt = page.getByTestId('figcell-compose-prompt')
  await expect(prompt).toBeVisible({ timeout: 10_000 })
  const overlayBtn = page.getByTestId('compose-overlay')
  await expect(overlayBtn).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '21-overlay-prompt.png') })
  await overlayBtn.click()

  // The cell figure rebuilds (new figId) with the overlay composited.
  await expect.poll(async () =>
    page.locator('[data-testid^="report-figcell-"] iframe[data-testid^="figure-"]')
      .first().getAttribute('data-testid').catch(() => null), {
    timeout: 15_000, message: 'overlay compose did not rebuild the cell figure',
  }).not.toBe(oldFigId)
  await page.waitForTimeout(2000)   // let the rebuilt figure paint

  // The edit toolbar's panel lists TWO layers now (base + overlay).
  const { panelId } = await openEditToolbar(page)
  const layerRows = page.locator(`[data-testid^="figcell-layer-${panelId}-"]`)
  await expect.poll(async () => layerRows.count(), {
    timeout: 10_000, message: 'panel did not gain a second (overlay) layer',
  }).toBe(2)

  // Colored (magma-blended) pixels appear in the cell iframe.
  await expect.poll(async () => reportCellColoredPixels(page), {
    timeout: 20_000,
    message: 'report cell gained no colored pixels after overlay compose',
  }).toBeGreaterThan(Math.max(500, before + 500))

  await page.waitForTimeout(500)
  await page.screenshot({ path: join(SHOTS, '22-overlay-two-layer.png') })
  ctx.assertNoJsErrors()
})

test('b) new signal cell + its OWN navigator center-drop → Callout prompt → inset', async () => {
  const { page } = ctx
  // Fresh report so the callout cell starts from a single-panel signal figure
  // (the previous cell is now a 2-layer overlay). Re-embed signal-A.
  await backendAction(page, 'report_new')
  await expect(page.getByTestId('report-body')).toBeVisible({ timeout: 10_000 })
  await expect(page.locator('[data-testid^="report-figcell-"]')).toHaveCount(0)

  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-cmp-sigA2', '1'))
  await dragToBody(page, '[data-cmp-sigA2="1"]')
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2000)

  const oldFigId = await figCell.locator('iframe[data-testid^="figure-"]')
    .first().getAttribute('data-testid')

  // Panel count BEFORE the callout (single panel). The slim bar's chips list
  // GRID panels only (a callout's hidden inset panel never gets a chip — the
  // Phase-3 chip fix), so the true spec-panel count comes from report_state.
  const before = await openEditToolbar(page)
  const panelCount = async (cellId: string) =>
    (await docCell(page, cellId))?.figure?.panels?.length ?? 0
  const panelsBefore = await panelCount(before.cellId)
  console.log('[compose] panels before callout =', panelsBefore)
  expect(panelsBefore).toBe(1)
  // Close the editor so the compose shield isn't obstructed by the edit panel.
  await page.getByTestId(`report-figcell-edit-toggle-${before.cellId}`).click()
  await expect(page.getByTestId(`figcell-edit-${before.cellId}`)).toHaveCount(0)

  // Drag signal-A's OWN NAVIGATOR (tree A) onto the cell center → nav↔signal pair
  // of one tree → the prompt offers Callout.
  const navA = navWindows(page).nth(0)
  await navA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-cmp-navA', '1'))
  await dropOnCellCenter(page, '[data-cmp-navA="1"]')

  const prompt = page.getByTestId('figcell-compose-prompt')
  await expect(prompt).toBeVisible({ timeout: 10_000 })
  const calloutBtn = page.getByTestId('compose-callout')
  await expect(calloutBtn).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '23-callout-prompt.png') })
  await calloutBtn.click()

  // The cell figure rebuilds with the inset panel.
  await expect.poll(async () =>
    figCell.locator('iframe[data-testid^="figure-"]').first()
      .getAttribute('data-testid').catch(() => null), {
    timeout: 15_000, message: 'callout compose did not rebuild the cell figure',
  }).not.toBe(oldFigId)
  await page.waitForTimeout(2000)

  // The spec now carries a SECOND panel (the hidden inset callout panel) —
  // report_state is authoritative (the inset panel deliberately gets no chip).
  const after = await openEditToolbar(page)
  await expect.poll(async () => panelCount(after.cellId), {
    timeout: 10_000, message: 'callout did not add an inset panel to the cell',
  }).toBeGreaterThanOrEqual(2)

  // The cell iframe still paints real pixels (inset rendered, not a blank frame).
  await expect.poll(async () => {
    // Reuse the report_sidebar bright-pixel check (grayscale base + inset).
    const s: string | null = await page.evaluate(() => {
      const cell = document.querySelector('[data-testid^="report-figcell-"]')
      const ifr = cell?.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
      return ifr?.src || null
    })
    if (!s) return -1
    const fr = page.frames().find((f: any) => f.url() === s)
    if (!fr) return -1
    try {
      return await fr.evaluate(() => {
        let n = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const cv = c as HTMLCanvasElement
          const cc = cv.getContext('2d')
          if (!cc || !cv.width || !cv.height) continue
          const d = cc.getImageData(0, 0, cv.width, cv.height).data
          for (let p = 0; p < d.length; p += 4)
            if (d[p] > 20 || d[p + 1] > 20 || d[p + 2] > 20) n++
        }
        return n
      })
    } catch { return -1 }
  }, { timeout: 20_000, message: 'callout cell drew no pixels (blank rebuild)' })
    .toBeGreaterThan(500)

  await page.waitForTimeout(500)
  await page.screenshot({ path: join(SHOTS, '24-callout-inset.png') })
  ctx.assertNoJsErrors()
})

test('c) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[compose] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
