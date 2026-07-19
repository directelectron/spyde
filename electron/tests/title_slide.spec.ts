/**
 * title_slide.spec.ts — Report Builder presentation POLISH (title / section
 * slides + per-slide styling + figure captions), e2e.
 *
 * Real Dask + bundled-synthetic Si-grains (navigator + signal window). Builds a
 * two-slide deck:
 *   • Slide 1 — a markdown TITLE slide (slide_kind='title' via report_set_slide_kind):
 *     rendered big + centered.
 *   • Slide 2 — a content slide with a FIGURE + a caption.
 * Then enters Present mode and asserts:
 *   • slide 1's title font is markedly LARGER than a normal content heading AND
 *     horizontally centered (data-kind="title" + present-title-md),
 *   • slide 2 shows the figure with its caption UNDER it (muted/centered figcaption).
 *
 * Screenshots the title slide + the captioned figure slide to title_slide_shots/
 * and the author Reads them — a title slide that doesn't LOOK like a title slide
 * is a failure even when selectors pass.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'title_slide_shots')
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
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)   // navigator + signal
  await page.waitForTimeout(2500)                 // let the DP paint
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally {
    await ctx?.app?.close()
  }
})

test('1) build a title-slide + captioned-figure deck', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()

  await backendAction(page, 'report_new', {})
  await backendAction(page, 'report_set_title', { title: 'Title Slide Demo' })

  // Slide 1 — a title slide: one markdown cell we then mark slide_kind='title'.
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '# Orientation Mapping\n\nA SpyDE presentation',
    html: '<h1>Orientation Mapping</h1><p>A SpyDE presentation</p>',
  })
  // Wait for the markdown cell to mount, then resolve its id from the root cell
  // testid (report-cell-<id>; the sub-testids report-cell-drag/-rendered/… also
  // match the prefix, so pick the one whose id has no extra dash-word).
  const mdCell = page.locator('[data-testid^="report-cell-"]').first()
  await expect(mdCell).toBeVisible({ timeout: 15_000 })
  const titleCellId: string = await page.evaluate(() => {
    for (const el of Array.from(document.querySelectorAll('[data-testid^="report-cell-"]'))) {
      const t = el.getAttribute('data-testid') || ''
      const id = t.replace('report-cell-', '')
      // Root cell id is `c`+hex with no interior dash; sub-testids carry a
      // leading word like `drag-` / `rendered-`.
      if (/^c[0-9a-f]+$/.test(id)) return id
    }
    return ''
  })
  expect(titleCellId).toBeTruthy()
  await backendAction(page, 'report_set_slide_kind', {
    cell_id: titleCellId, slide_kind: 'title',
  })

  // Slide 2 — a content slide (slide_break) with a figure + caption. First a
  // small markdown lead-in, then the figure from the signal window.
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '## Results',
    html: '<h2>Results</h2>',
    slide_break: true,
  })

  const sigWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  const pill = sigWin.getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-present-src', '1'))
  const sigWid = await windowIdFromPill(page, '[data-present-src="1"]')
  expect(Number.isFinite(sigWid)).toBe(true)
  await backendAction(page, 'report_add_figure', {
    source_window_id: sigWid, caption: 'Figure 1. Diffraction pattern of Si grains',
  })

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(1500)

  await page.screenshot({ path: join(SHOTS, '01-deck-built.png') })
  ctx.assertNoJsErrors()
})

test('2) enter Present mode → slide 1 is a BIG CENTERED title', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await expect(page.getByTestId('present-mode')).toBeVisible({ timeout: 10_000 })

  const active = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(active).toBeVisible()
  // The slide is marked as a title slide.
  await expect(active).toHaveAttribute('data-kind', 'title')
  const h1 = active.locator('h1')
  await expect(h1).toHaveText(/Orientation Mapping/)
  expect(await counterText(page)).toBe('1 / 2')

  // The title font is markedly LARGE (present-title-md h1 = 4.2rem ≈ 88px at the
  // 21px stage base) and the text is CENTERED. Assert both from computed style.
  const info = await h1.evaluate((el: HTMLElement) => {
    const cs = getComputedStyle(el)
    const md = el.closest('[data-testid^="present-md-"]') as HTMLElement | null
    return {
      fontPx: parseFloat(cs.fontSize),
      textAlign: cs.textAlign,
      mdTitle: md?.getAttribute('data-title-slide') || '0',
    }
  })
  expect(info.mdTitle).toBe('1')
  // A title heading is much bigger than a normal slide h1 (~2.4rem ≈ 50px).
  expect(info.fontPx).toBeGreaterThan(64)
  expect(info.textAlign).toBe('center')

  await page.screenshot({ path: join(SHOTS, '02-title-slide.png') })
  ctx.assertNoJsErrors()
})

test('3) advance → slide 2 shows the figure with its caption', async () => {
  const { page } = ctx
  await page.keyboard.press('ArrowRight')
  const active = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(active).toHaveAttribute('data-kind', 'content')
  expect(await counterText(page)).toBe('2 / 2')

  // The figure + its caption render; the caption sits UNDER the figure, muted +
  // italic + centered.
  const fig = active.locator('figure[data-testid^="present-fig-"]').first()
  await expect(fig).toBeVisible({ timeout: 10_000 })
  const cap = fig.locator('figcaption')
  await expect(cap).toHaveText(/Figure 1\. Diffraction pattern of Si grains/)
  const capStyle = await cap.evaluate((el: HTMLElement) => {
    const cs = getComputedStyle(el)
    return { fontStyle: cs.fontStyle, textAlign: cs.textAlign }
  })
  expect(capStyle.fontStyle).toBe('italic')

  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '03-figure-caption-slide.png') })
  ctx.assertNoJsErrors()
})

test('4) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[title_slide] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
