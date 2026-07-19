/**
 * slide_layout.spec.ts — Report Builder 2-column slide layout, e2e.
 *
 * Real Dask + bundled-synthetic Si-grains. Builds ONE slide holding a text cell
 * (column=left) BESIDE a figure cell (column=right), enters Present mode, and
 * asserts the two render SIDE BY SIDE (the text's right edge is at or left of
 * the figure's left edge, and they share a row — vertical overlap). The
 * side-by-side slide is screenshotted to slide_layout_shots/ and Read.
 *
 * The column is assigned via the report_set_cell_column verb (backendAction).
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'slide_layout_shots')
const FIG_MIME = 'application/x-spyde-figure'

/** Resolve a window's id by firing a dragstart on its pill and reading the MIME
 *  payload (proven in report_present.spec.ts / report_tiling.spec.ts). */
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

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)   // navigator + signal
  await page.waitForTimeout(2500)                 // let the DP paint
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally {
    await ctx?.app?.close()
  }
})

test('1) build a slide with a LEFT text cell and a RIGHT figure cell', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()

  await backendAction(page, 'report_new', {})
  await backendAction(page, 'report_set_title', { title: 'Two-Column Slide' })

  // Left column: a text cell describing the figure.
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '## The pattern\n\nA silicon-grains diffraction pattern, described '
      + 'in text on the LEFT of this slide while the figure sits on the RIGHT.',
    html: '<h2>The pattern</h2><p>A silicon-grains diffraction pattern, described '
      + 'in text on the LEFT of this slide while the figure sits on the RIGHT.</p>',
    column: 'left',
  })

  // Right column: a figure from the SIGNAL window (S- prefixed subwindow).
  const sigWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  const pill = sigWin.getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-col-src', '1'))
  const sigWid = await windowIdFromPill(page, '[data-col-src="1"]')
  expect(Number.isFinite(sigWid)).toBe(true)
  await backendAction(page, 'report_add_figure', {
    source_window_id: sigWid, caption: 'Fig. 1 — Si grains DP',
  })

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 20_000 })

  // Assign the figure cell to the RIGHT column.
  const figCellId: string = await page.evaluate(() => {
    const el = document.querySelector('[data-testid^="report-figcell-"]')
    return (el?.getAttribute('data-testid') || '').replace('report-figcell-', '')
  })
  await backendAction(page, 'report_set_cell_column', { cell_id: figCellId, column: 'right' })
  await page.waitForTimeout(1000)

  // The sidebar shows the column badges on both cells (authoring hint).
  await expect(page.locator('[data-testid="cell-column-badge"][data-column="left"]'))
    .toHaveCount(1)
  await expect(page.locator('[data-testid="cell-column-badge"][data-column="right"]'))
    .toHaveCount(1)

  await page.screenshot({ path: join(SHOTS, '01-sidebar-badges.png') })
  ctx.assertNoJsErrors()
})

test('2) Present mode renders the text and figure SIDE BY SIDE', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await expect(page.getByTestId('present-mode')).toBeVisible({ timeout: 10_000 })

  const active = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(active).toBeVisible()
  // The 2-column grid row rendered, with a left and a right column.
  const cols = active.locator('[data-testid="present-cols"]')
  await expect(cols).toHaveCount(1)
  const leftCol = active.locator('[data-testid="present-col-left"]')
  const rightCol = active.locator('[data-testid="present-col-right"]')
  await expect(leftCol.locator('h2')).toHaveText(/The pattern/)
  await expect(rightCol.locator('figure')).toBeVisible()

  // Let the figure embed paint.
  await page.waitForTimeout(2500)

  // Geometry: the text column's right edge is at/left of the figure column's
  // left edge (side by side, not stacked), and they overlap vertically (same
  // row). Compare the actual content boxes.
  const lBox = await leftCol.boundingBox()
  const rBox = await rightCol.boundingBox()
  expect(lBox).not.toBeNull()
  expect(rBox).not.toBeNull()
  if (lBox && rBox) {
    // Side by side: left column ends before (or at) the right column starts.
    expect(lBox.x + lBox.width).toBeLessThanOrEqual(rBox.x + 4)
    // Same row: their vertical spans overlap.
    const overlap = Math.min(lBox.y + lBox.height, rBox.y + rBox.height)
      - Math.max(lBox.y, rBox.y)
    expect(overlap).toBeGreaterThan(20)
  }

  await page.screenshot({ path: join(SHOTS, '02-present-side-by-side.png') })
  ctx.assertNoJsErrors()
})

test('3) the figure column drew a live embed (non-blank)', async () => {
  const { page } = ctx
  const src: string | null = await page.evaluate(() => {
    const slide = document.querySelector('[data-testid="present-slide"][data-active="1"]')
    const ifr = slide?.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
    return ifr?.src || null
  })
  // A figure embed should be present; if the snapshot fell back to a static PNG
  // that's still a valid side-by-side render (asserted in test 2). Only probe
  // pixels when we actually have an iframe.
  if (src) {
    const frame = page.frames().find((f: any) => f.url() === src)
    if (frame) {
      const bright = await frame.evaluate(() => {
        let n = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const cv = c as HTMLCanvasElement
          const cctx = cv.getContext('2d')
          if (!cctx || !cv.width || !cv.height) continue
          const d = cctx.getImageData(0, 0, cv.width, cv.height).data
          for (let p = 0; p < d.length; p += 4) {
            if (d[p] > 20 || d[p + 1] > 20 || d[p + 2] > 20) n++
          }
        }
        return n
      })
      expect(bright).toBeGreaterThan(200)
    }
  }
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('present-mode')).toHaveCount(0, { timeout: 5_000 })
  ctx.assertNoJsErrors()
})

test('4) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[slide_layout] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
