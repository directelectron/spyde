/**
 * report_present.spec.ts — Report Builder Phase 6 (Present mode / slides), e2e.
 *
 * Real Dask + bundled-synthetic Si-grains (navigator + signal window). Builds a
 * small report (a markdown title slide, a second markdown slide, a figure slide),
 * marks slide breaks, then drives PRESENT MODE the way a presenter would:
 *   • click the sidebar "Present" button → the full-screen present overlay,
 *   • assert one big slide is shown + a slide counter,
 *   • arrow-key ADVANCES slides (and ← goes back),
 *   • a figure slide's interactive embed renders (non-blank iframe),
 *   • ESC exits.
 *
 * Screenshots at every stage to present_shots/ — a blank overlay is a failure
 * even when selectors pass, so each shot is Read by the author.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'present_shots')
const FIG_MIME = 'application/x-spyde-figure'

/** Resolve a window's id by firing a dragstart on its pill and reading the MIME
 *  payload (the windowId is stamped there, not as a DOM attribute). Proven in
 *  report_tiling.spec.ts. */
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

/** Count bright canvas pixels INSIDE the active present slide's figure iframe. */
async function presentFigurePixels(page: any): Promise<number> {
  const src: string | null = await page.evaluate(() => {
    const slide = document.querySelector('[data-testid="present-slide"][data-active="1"]')
    if (!slide) return null
    const ifr = slide.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
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
          if (d[p] > 20 || d[p + 1] > 20 || d[p + 2] > 20) n++
        }
      }
      return n
    })
  } catch { return -1 }
}

/** The active slide's counter text ("n / N"). */
async function counterText(page: any): Promise<string> {
  return (await page.getByTestId('present-counter').textContent())?.trim() ?? ''
}

test('1) open the report sidebar and build a 3-slide deck', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()

  // Build the deck via the backend actions (no OS dialogs). A markdown title
  // slide, a second markdown slide, and a figure slide from the signal window.
  await backendAction(page, 'report_new', {})
  await backendAction(page, 'report_set_title', { title: 'Phase 6 Demo Deck' })
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '# Presenting SpyDE\n\nA slide deck built from a report.',
    html: '<h1>Presenting SpyDE</h1><p>A slide deck built from a report.</p>',
  })
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '## The method\n\n- Load data\n- Find vectors\n- Map orientation',
    html: '<h2>The method</h2><ul><li>Load data</li><li>Find vectors</li><li>Map orientation</li></ul>',
    slide_break: true,
  })

  // A figure slide: resolve the SIGNAL window's id from its breadcrumb pill
  // (the S- prefixed subwindow) and drive report_add_figure against it.
  const sigWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  const pill = sigWin.getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-present-src', '1'))
  const sigWid = await windowIdFromPill(page, '[data-present-src="1"]')
  expect(Number.isFinite(sigWid)).toBe(true)
  await backendAction(page, 'report_add_figure', {
    source_window_id: sigWid, caption: 'Fig. 1 — Si grains DP',
  })

  // Wait for the figure cell to mount + its iframe to appear in the sidebar.
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 20_000 })

  // Mark the figure cell as its own slide (3rd slide). Resolve its cell id.
  const figCellId: string = await page.evaluate(() => {
    const el = document.querySelector('[data-testid^="report-figcell-"]')
    const t = el?.getAttribute('data-testid') || ''
    return t.replace('report-figcell-', '')
  })
  await backendAction(page, 'report_toggle_slide_break', { cell_id: figCellId, value: true })
  await page.waitForTimeout(1500)

  await page.screenshot({ path: join(SHOTS, '01-deck-built.png') })
  ctx.assertNoJsErrors()
})

test('2) click Present → full-screen overlay shows the first slide', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await expect(page.getByTestId('present-mode')).toBeVisible({ timeout: 10_000 })
  // The active slide shows the big title heading.
  const active = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(active).toBeVisible()
  await expect(active.locator('h1')).toHaveText(/Presenting SpyDE/)
  // Slide counter reads 1 / 3.
  expect(await counterText(page)).toBe('1 / 3')
  await page.screenshot({ path: join(SHOTS, '02-present-slide1.png') })
  ctx.assertNoJsErrors()
})

test('3) arrow key advances to the next slide', async () => {
  const { page } = ctx
  await page.keyboard.press('ArrowRight')
  const active = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(active.locator('h2')).toHaveText(/The method/, { timeout: 5_000 })
  expect(await counterText(page)).toBe('2 / 3')
  await page.screenshot({ path: join(SHOTS, '03-present-slide2.png') })

  // Space advances again → the figure slide.
  await page.keyboard.press('Space')
  await expect(page.locator('[data-testid="present-slide"][data-active="1"] figure'))
    .toBeVisible({ timeout: 5_000 })
  expect(await counterText(page)).toBe('3 / 3')
  ctx.assertNoJsErrors()
})

test('4) the figure slide renders its interactive embed (non-blank)', async () => {
  const { page } = ctx
  // The figure slide's iframe drew real pixels (a live embed, not a blank frame).
  await expect.poll(async () => presentFigurePixels(page), {
    timeout: 30_000, message: 'present figure iframe drew no pixels (blank embed)',
  }).toBeGreaterThan(500)
  await page.waitForTimeout(1000)
  await page.screenshot({ path: join(SHOTS, '04-present-figure-slide.png') })
  ctx.assertNoJsErrors()
})

test('5) left arrow goes back, ESC exits', async () => {
  const { page } = ctx
  await page.keyboard.press('ArrowLeft')
  expect(await counterText(page)).toBe('2 / 3')
  await page.screenshot({ path: join(SHOTS, '05-present-back-to-slide2.png') })

  // ESC exits Present mode entirely.
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('present-mode')).toHaveCount(0, { timeout: 5_000 })
  // The report sidebar is still there behind it.
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '06-exited.png') })
  ctx.assertNoJsErrors()
})

test('6) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[present] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
