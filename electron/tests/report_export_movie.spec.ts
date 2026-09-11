/**
 * report_export_movie.spec.ts — a MOVIE cell survives every HTML export, and an
 * exported figure box is sized by its shape rather than by one fixed height.
 *
 * A movie cell exported as nothing at all: no poster, no caption, in static
 * HTML, interactive HTML and the PDF that renders from the static file. A
 * headless test cannot see that, because the export reports success either way.
 * So this drives the real app, exports for real, and loads the exported files in
 * a throwaway Chromium to measure what a reader would actually get.
 *
 *   1. Build a report on the synthetic in-situ movie: a markdown cell, a live
 *      figure cell, and a movie cell rendered to a real .gif (so the manager
 *      holds a poster AND a file to inline).
 *   2. Static HTML — poster, caption and play badge, rendering as real pixels.
 *   3. Interactive HTML — the movie itself inlined, and the figure iframe sized
 *      from the figure's shape instead of the retired fixed 480 px box.
 *
 * Screenshots land in report_fixes_shots/ — a blank frame is a failure, not a
 * pass. SPYDE_LOG_LEVEL=WARNING tees backend logging to stderr so the final
 * audit can scan for Python tracebacks.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdtempSync, existsSync, rmSync, readFileSync, statSync } from 'fs'
import { tmpdir } from 'os'
import { chromium } from 'playwright'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines, sigWindow,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_fixes_shots')
const FIG_MIME = 'application/x-spyde-figure'
// The height every exported figure used to get, whatever its shape.
const RETIRED_FIXED_HEIGHT = 480

let ctx: Awaited<ReturnType<typeof launchApp>>
let workDir: string
let htmlStaticPath: string
let htmlInteractivePath: string
let moviePath: string

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  // A movie cell needs a time-series source.
  await backendAction(page, 'load_test_data_movie')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2500)

  workDir = mkdtempSync(join(tmpdir(), 'spyde-export-movie-'))
  htmlStaticPath = join(workDir, 'report-static.html')
  htmlInteractivePath = join(workDir, 'report-interactive.html')
  // .gif, not .mp4: the GIF encoder is Pillow, so this needs no ffmpeg.
  moviePath = join(workDir, 'clip.gif')
})

test.afterAll(async () => {
  try {
    const bad = ctx?.backend ? backendErrorLines(ctx.backend) : []
    expect(bad, `backend errors during the run:\n${bad.join('\n')}`).toEqual([])
    ctx?.assertNoJsErrors()
  } finally {
    await ctx?.app?.close()
    if (workDir && existsSync(workDir)) {
      try { rmSync(workDir, { recursive: true, force: true }) } catch { /* */ }
    }
  }
})

/** The backend-assigned window id, read out of the pill's own drag payload:
 *  there is no window-id attribute in the DOM. */
async function windowIdFromPill(page: any, pillSelector: string): Promise<number> {
  return await page.evaluate(({ selector, mime }: any) => {
    const source = document.querySelector(selector) as HTMLElement
    if (!source) return NaN
    const transfer = new DataTransfer()
    const rect = source.getBoundingClientRect()
    const event = new DragEvent('dragstart', {
      bubbles: true, cancelable: true,
      clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2,
    })
    Object.defineProperty(event, 'dataTransfer', { value: transfer, configurable: true })
    source.dispatchEvent(event)
    try { return Number((JSON.parse(transfer.getData(mime)) as any).windowId) }
    catch { return NaN }
  }, { selector: pillSelector, mime: FIG_MIME })
}

/** The id of the last movie cell in the open report ('' when there is none). */
async function lastMovieCellId(page: any): Promise<string> {
  return await page.evaluate(() => {
    const cells = ((window as any)._spyde_test_report?.()?.cells ?? []) as any[]
    const movies = cells.filter((c) => c.cell_type === 'movie')
    return movies.length ? String(movies[movies.length - 1].id) : ''
  })
}

/** A full native HTML5 drag, in-page so one DataTransfer is shared across
 *  dragstart/dragover/drop the way a real user drag is. */
async function dragAndDrop(page: any, srcSelector: string, dstSelector: string) {
  return await page.evaluate(({ srcSelector, dstSelector }: any) => {
    const src = document.querySelector(srcSelector) as HTMLElement
    const dst = document.querySelector(dstSelector) as HTMLElement
    if (!src || !dst) throw new Error('drag src/dst not found')
    const transfer = new DataTransfer()
    const fire = (target: HTMLElement, type: string) => {
      const rect = target.getBoundingClientRect()
      const event = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2,
      })
      Object.defineProperty(event, 'dataTransfer', { value: transfer, configurable: true })
      target.dispatchEvent(event)
    }
    fire(src, 'dragstart')
    const types = Array.from(transfer.types)
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
    return { types }
  }, { srcSelector, dstSelector })
}

/** Render an exported HTML file in a throwaway Chromium and measure it: a
 *  caption in the source is not proof that a reader sees anything. */
async function measureExport(path: string, shot: string) {
  const browser = await chromium.launch()
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 1400 } })
    await page.goto('file://' + path.replace(/\\/g, '/'))
    // 5 s, not 2.5: an iframe figure was still blank at 2.5 s and the screenshot
    // read as a broken cell when the export was fine.
    await page.waitForTimeout(5000)
    await page.screenshot({ path: join(SHOTS, shot), fullPage: true })
    return await page.evaluate(() => {
      const figure = document.querySelector('figure.report-movie')
      const media = figure
        ? figure.querySelector('img, video') as HTMLImageElement | HTMLVideoElement | null
        : null
      const rect = media ? media.getBoundingClientRect() : { width: 0, height: 0 }
      const frames = Array.from(document.querySelectorAll(
        'figure.report-figure iframe')) as HTMLIFrameElement[]
      return {
        found: !!figure,
        area: Math.round(rect.width * rect.height),
        caption: figure?.querySelector('figcaption')?.textContent ?? '',
        broken: !!(media && media.tagName === 'IMG'
          && !(media as HTMLImageElement).naturalWidth),
        frameHeights: frames.map((f) => Math.round(f.getBoundingClientRect().height)),
        boxHeights: Array.from(document.querySelectorAll('figure.report-figure .fig-box'))
          .map((b) => Math.round(b.getBoundingClientRect().height)),
      }
    })
  } finally {
    await browser.close()
  }
}

// ── 1) Build a report with a figure cell AND a rendered movie cell ───────────

test('1) build a report: markdown + live figure + a rendered movie cell', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', {})
  await expect(page.getByTestId('report-body')).toBeVisible()

  await page.getByTestId('report-add-text').click()
  const rendered = page.locator('[data-testid^="report-cell-rendered-"]').first()
  await expect(rendered).toBeVisible()
  await rendered.dblclick()
  const textarea = page.locator('[data-testid^="report-cell-textarea-"]').first()
  await textarea.fill('# Movie export\nEvery cell kind must survive every export.')
  await textarea.press('Control+Enter')

  // A live figure cell, via the proven breadcrumb-pill drag.
  const pill = sigWindow(page).getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-fig-src', '1'))
  const dropped = await dragAndDrop(page, '[data-fig-src="1"]',
                                    '[data-testid="report-body"]')
  expect(dropped.types).toContain(FIG_MIME)
  await expect(page.locator('[data-report-cell="1"] > [data-testid^="report-figcell-"]')
    .first()).toBeVisible({ timeout: 20_000 })

  // A movie cell from the same in-situ tree, then RENDERED so the manager holds
  // both a poster and a movie file.
  const windowId: number = await windowIdFromPill(page, '[data-fig-src="1"]')
  expect(Number.isFinite(windowId), 'no window id from the signal pill').toBe(true)
  await backendAction(page, 'report_add_movie_cell', {})
  await expect.poll(() => lastMovieCellId(page), {
    timeout: 20_000, message: 'movie cell never appeared in the report',
  }).not.toBe('')
  const cellId = await lastMovieCellId(page)

  await backendAction(page, 'report_set_movie_source',
                      { cell_id: cellId, source_window_id: windowId })
  await page.waitForTimeout(1500)
  // movie_export renders from the cell's EDITOR session, not from the spec, so
  // the editor has to be open first.
  await backendAction(page, 'movie_open', { cell_id: cellId })
  await page.waitForTimeout(2500)
  await backendAction(page, 'movie_export', { cell_id: cellId, path: moviePath })
  await expect.poll(() => existsSync(moviePath) && statSync(moviePath).size > 1000, {
    timeout: 120_000, message: 'movie never rendered to disk',
  }).toBe(true)
  await page.waitForTimeout(1500)

  await page.screenshot({ path: join(SHOTS, '01-report-with-movie-cell.png') })
  ctx.assertNoJsErrors()
})

// ── 2) Static export keeps the movie ─────────────────────────────────────────

test('2) static HTML keeps the movie poster, caption and badge', async () => {
  const { page } = ctx
  await backendAction(page, 'report_export_html',
                      { mode: 'static', path: htmlStaticPath })
  await expect.poll(
    () => existsSync(htmlStaticPath) && statSync(htmlStaticPath).size > 2000, {
      timeout: 60_000, message: 'static export never written',
    }).toBe(true)
  const html = readFileSync(htmlStaticPath, 'utf-8')

  expect(html, 'movie cell missing from the static export').toContain('report-movie')
  expect(html, 'the still is not badged as a movie').toContain('movie-badge')
  expect(html).not.toContain('\x00bin:')

  const measured = await measureExport(htmlStaticPath, '02-static-rendered.png')
  expect(measured.found, 'no movie figure in the rendered static page').toBe(true)
  expect(measured.broken, 'movie poster rendered as a broken image').toBe(false)
  expect(measured.area, 'movie poster occupies no screen area').toBeGreaterThan(5000)
})

// ── 3) Interactive export inlines the movie and sizes its figure box ─────────

test('3) interactive HTML inlines the movie and sizes the box by shape', async () => {
  const { page } = ctx
  await backendAction(page, 'report_export_html',
                      { mode: 'interactive', path: htmlInteractivePath })
  await expect.poll(
    () => existsSync(htmlInteractivePath) && statSync(htmlInteractivePath).size > 2000, {
      timeout: 120_000, message: 'interactive export never written',
    }).toBe(true)
  const html = readFileSync(htmlInteractivePath, 'utf-8')

  // The rendered movie itself travels, not just a still of it.
  expect(html, 'movie was not inlined into the interactive export')
    .toContain('data:image/gif;base64,')
  expect(html).not.toContain('\x00bin:')
  // The figure box is shaped, not pinned.
  expect(html, 'the figure box carries no aspect ratio').toContain('aspect-ratio:')
  expect(html).not.toContain(`height:${RETIRED_FIXED_HEIGHT}px`)

  const measured = await measureExport(htmlInteractivePath, '03-interactive-rendered.png')
  expect(measured.found).toBe(true)
  expect(measured.broken, 'inlined movie rendered as a broken image').toBe(false)
  expect(measured.area, 'inlined movie occupies no screen area').toBeGreaterThan(5000)
  console.log('[export-movie] figure box heights =',
              JSON.stringify(measured.boxHeights),
              'iframe heights =', JSON.stringify(measured.frameHeights))
  expect(measured.boxHeights.length, 'no shaped figure box in the rendered page')
    .toBeGreaterThan(0)
  for (const height of measured.boxHeights) {
    expect(height, 'the figure box is still the retired fixed height')
      .not.toBe(RETIRED_FIXED_HEIGHT)
    expect(height, 'the figure box collapsed to nothing').toBeGreaterThan(100)
  }
})
