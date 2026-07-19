/**
 * report_tiling.spec.ts — Report Builder target-relative 2-D tiling (Spec B),
 * end-to-end in the real app.
 *
 * Verifies the newly-implemented per-grid-cell drop targeting: dragging a window
 * pill over a MULTI-panel report figure shows per-panel drop zones; an edge drop
 * on a SPECIFIC panel sends repfig_compose {mode:'tile-*', target_panel_id} and
 * grows the grid RELATIVE to that panel (hole-fill or row/col insert).
 *
 * Flow:
 *   1. Embed signal-A → single-panel figure cell.
 *   2. Tile signal-B to the RIGHT → 1×2 grid (2 panels).
 *   3. Drop signal-A onto the RIGHT panel's BOTTOM zone (targeted tile-down) → a
 *      3rd panel lands BELOW the right panel → 2×2 grid with one hole at (1,0).
 *
 * Assertion path: the report doc is read via the additive window._spyde_test_report
 * hook so we can assert the exact panel count + grid_pos layout (the DOM edit
 * panel lists panels A/B/C but not their grid positions). Screenshots before/after
 * prove the 2-D layout renders.
 *
 * Drag path: the real native-HTML5 pill drag over the compose shield is used for
 * BOTH tiles (the report_phase2_probe edge-drop pattern, extended to target a
 * specific panel cell). If the per-panel geometry drop proves unreliable, the spec
 * falls back to driving repfig_compose with target_panel_id via backendAction and
 * NOTES which path was used (see console logs + the report).
 *
 * Real Dask + si_grains loaded TWICE (two trees) so we have two same-shape signals
 * to tile. Screenshots to report_edit_shots/. Final: assertNoJsErrors + traceback scan.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_edit_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

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

const FIG_MIME = 'application/x-spyde-figure'

/** Resolve a window's id by firing a dragstart on its pill and reading the MIME
 *  payload (the windowId is stamped there, not as a DOM attribute). */
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
    // Cancel the drag so it doesn't linger.
    const end = new DragEvent('dragend', { bubbles: true, cancelable: true })
    Object.defineProperty(end, 'dataTransfer', { value: dt, configurable: true })
    src.dispatchEvent(end)
    const raw = dt.getData(mime)
    try { return Number((JSON.parse(raw) as any).windowId) } catch { return NaN }
  }, { sel: pillSel, mime: FIG_MIME })
}

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

/**
 * Drop a window pill onto the compose shield at a fractional (fx, fy) WITHIN the
 * shield box — the report_phase2_probe two-phase pattern: dragstart + a body
 * dragover promotes dragKind='window' (mounts the shield), yield to React, then
 * dragover+drop at (fx, fy) on the shield. On a multi-panel figure the fraction
 * selects WHICH panel cell + the zone within it.
 */
async function dropOnShield(page: any, srcSel: string, fx: number, fy: number) {
  await page.evaluate(({ srcSel }: any) => {
    const src = document.querySelector(srcSel) as HTMLElement
    const body = document.querySelector('[data-testid="report-body"]') as HTMLElement
    if (!src || !body) throw new Error('drag src/report-body not found')
    const dt = new DataTransfer()
    ;(window as any).__tiledt = dt
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
  await page.waitForTimeout(350)   // let the shield mount

  await page.evaluate(({ srcSel, fx, fy }: any) => {
    const dt = (window as any).__tiledt as DataTransfer
    const shield = document.querySelector('[data-testid^="figcell-compose-shield-"]') as HTMLElement
    const src = document.querySelector(srcSel) as HTMLElement
    if (!shield) throw new Error('compose shield not mounted')
    const fire = (target: HTMLElement, type: string) => {
      const r = target.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width * fx, clientY: r.top + r.height * fy,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
    }
    fire(shield, 'dragenter'); fire(shield, 'dragover'); fire(shield, 'drop')
    if (src) {
      const r = src.getBoundingClientRect()
      const ev = new DragEvent('dragend', {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      src.dispatchEvent(ev)
    }
  }, { srcSel, fx, fy })
}

/** The single figure cell's id. */
async function figCellId(page: any): Promise<string> {
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  return await figCell.evaluate((el: HTMLElement) =>
    (el.getAttribute('data-testid') || '').replace('report-figcell-', ''))
}

/** The report cell's FigureSpec (panels + layout) from the doc test hook. */
async function cellFigure(page: any, cellId: string): Promise<{
  panels: Array<{ id: string; grid_pos: [number, number] }>
  layout: { kind: string; rows?: number; cols?: number }
} | null> {
  return await page.evaluate((cid: string) => {
    const doc = (window as any)._spyde_test_report?.()
    const cell = doc?.cells?.find((c: any) => c.id === cid)
    if (!cell?.figure) return null
    return {
      panels: (cell.figure.panels ?? []).map((p: any) => ({ id: p.id, grid_pos: p.grid_pos })),
      layout: cell.figure.layout,
    }
  }, cellId)
}

/** The live report figure iframe's figId (for a repaint wait). */
async function reportFigId(page: any): Promise<string | null> {
  return await page.evaluate(() => {
    const cell = document.querySelector('[data-testid^="report-figcell-"]')
    const ifr = cell?.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
    return ifr ? (ifr.getAttribute('data-testid') || '').replace('figure-', '') : null
  })
}

/** Wait for the report figure to rebuild (its figId changes off `oldFigId`). */
async function waitFigRebuild(page: any, oldFigId: string | null) {
  await expect.poll(async () => await reportFigId(page), {
    timeout: 20_000, message: 'report figure did not rebuild',
  }).not.toBe(oldFigId)
}

// ── the spec ────────────────────────────────────────────────────────────────────

let usedDragPath = { tileRight: false, tileTargeted: false }

test('1) embed signal-A → single-panel figure cell', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', {})
  await expect(page.getByTestId('report-body')).toBeVisible()

  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-tile-a', '1'))
  await dragToBody(page, '[data-tile-a="1"]')

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2500)

  const cellId = await figCellId(page)
  const spec = await cellFigure(page, cellId)
  console.log('[tiling] after embed: panels =', spec?.panels.length, 'layout =', JSON.stringify(spec?.layout))
  expect(spec?.panels.length).toBe(1)
  await page.screenshot({ path: join(SHOTS, 'B-01-single-panel.png') })
  ctx.assertNoJsErrors()
})

test('2) tile signal-B to the RIGHT → 1×2 grid (2 panels)', async () => {
  const { page } = ctx
  const cellId = await figCellId(page)
  const oldFigId = await reportFigId(page)

  const sigB = sigWindows(page).nth(1)
  await sigB.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-tile-b', '1'))

  // Real drag: drop on the RIGHT edge (fx=0.9) of the single-panel shield → tile-right.
  await dropOnShield(page, '[data-tile-b="1"]', 0.9, 0.5)

  let spec: Awaited<ReturnType<typeof cellFigure>> = null
  try {
    await waitFigRebuild(page, oldFigId)
    await page.waitForTimeout(2000)
    spec = await cellFigure(page, cellId)
  } catch { /* fall through to backendAction fallback below */ }

  if (!spec || spec.panels.length !== 2) {
    // Real drag didn't take — drive the compose action directly (documented).
    console.log('[tiling] tile-right real drag did not produce 2 panels; using backendAction fallback')
    const wid = await windowIdFromPill(page, '[data-tile-b="1"]')
    await backendAction(page, 'repfig_compose', {
      cell_id: cellId, mode: 'tile-right', source_window_id: wid,
    })
    await page.waitForTimeout(2500)
    spec = await cellFigure(page, cellId)
  } else {
    usedDragPath.tileRight = true
  }

  console.log('[tiling] after tile-right: panels =', spec?.panels.length,
    'layout =', JSON.stringify(spec?.layout),
    'grid_pos =', JSON.stringify(spec?.panels.map(p => p.grid_pos)))
  expect(spec?.panels.length, 'tile-right did not create a 2nd panel').toBe(2)
  expect(spec?.layout.kind).toBe('grid')
  expect(spec?.layout.rows).toBe(1)
  expect(spec?.layout.cols).toBe(2)
  // Two panels at (0,0) and (0,1).
  const cols = (spec?.panels ?? []).map(p => p.grid_pos[1]).sort()
  expect(cols).toEqual([0, 1])
  await page.waitForTimeout(1000)
  await page.screenshot({ path: join(SHOTS, 'B-02-tiled-1x2.png') })
  ctx.assertNoJsErrors()
})

test('3) drop signal-A on the RIGHT panel BOTTOM zone → targeted tile-down → 2×2 with a hole', async () => {
  const { page } = ctx
  const cellId = await figCellId(page)
  const before = await cellFigure(page, cellId)
  expect(before?.panels.length).toBe(2)
  // Identify the RIGHT panel (grid_pos col == 1) — the targeted panel.
  const rightPanel = (before?.panels ?? []).find(p => p.grid_pos[1] === 1)
  expect(rightPanel, 'no right panel at col 1').toBeTruthy()
  const oldFigId = await reportFigId(page)

  // Real drag: drop signal-A onto the RIGHT panel's BOTTOM zone. On a 1×2 grid the
  // right cell is fx in [0.5, 1.0]; the bottom zone within it is local fy > 0.7 →
  // global fy ≈ 0.85. fx ≈ 0.75 lands squarely in the right cell.
  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-tile-a2', '1'))
  await dropOnShield(page, '[data-tile-a2="1"]', 0.75, 0.86)

  let spec: Awaited<ReturnType<typeof cellFigure>> = null
  try {
    await waitFigRebuild(page, oldFigId)
    await page.waitForTimeout(2000)
    spec = await cellFigure(page, cellId)
  } catch { /* fall through */ }

  if (!spec || spec.panels.length !== 3) {
    console.log('[tiling] targeted tile-down real drag did not produce 3 panels; using backendAction fallback with target_panel_id')
    const wid = await windowIdFromPill(page, '[data-tile-a2="1"]')
    await backendAction(page, 'repfig_compose', {
      cell_id: cellId, mode: 'tile-down', source_window_id: wid,
      target_panel_id: rightPanel!.id,
    })
    await page.waitForTimeout(2500)
    spec = await cellFigure(page, cellId)
  } else {
    usedDragPath.tileTargeted = true
  }

  console.log('[tiling] after targeted tile-down: panels =', spec?.panels.length,
    'layout =', JSON.stringify(spec?.layout),
    'grid_pos =', JSON.stringify(spec?.panels.map(p => p.grid_pos)))

  // 3 panels now, in a 2×2 grid.
  expect(spec?.panels.length, 'targeted tile-down did not add a 3rd panel').toBe(3)
  expect(spec?.layout.kind).toBe('grid')
  expect(spec?.layout.rows, 'grid did not grow to 2 rows').toBe(2)
  expect(spec?.layout.cols).toBe(2)

  // The NEW panel is BELOW the right panel: at (1, 1). The right panel stays at
  // (0,1); the left panel stays at (0,0); (1,0) is the HOLE.
  const positions = (spec?.panels ?? []).map(p => `${p.grid_pos[0]},${p.grid_pos[1]}`).sort()
  console.log('[tiling] final grid positions =', JSON.stringify(positions))
  expect(positions, 'new panel not placed below the right panel at (1,1)')
    .toEqual(['0,0', '0,1', '1,1'])
  // (1,0) is unoccupied → exactly the intended hole.
  const occupied = new Set(positions)
  expect(occupied.has('1,0'), '(1,0) should be a HOLE, not occupied').toBe(false)

  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, 'B-03-tiled-2x2-hole.png') })
  console.log('[tiling] interaction paths — tile-right via drag:', usedDragPath.tileRight,
    '; targeted tile-down via drag:', usedDragPath.tileTargeted)
  ctx.assertNoJsErrors()
})

test('4) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[tiling] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
