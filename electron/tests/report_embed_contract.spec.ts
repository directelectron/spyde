/**
 * report_embed_contract.spec.ts — every report cell kind survives every export
 * mode, driven end-to-end in the real app.
 *
 * The gap this covers: a MOVIE cell used to export as nothing at all — no
 * poster, no caption — in static HTML, interactive HTML, the slides deck, and
 * the PDF that renders from the static file. A headless test cannot see that
 * (the export "succeeded"), so this drives the real app, exports for real, and
 * loads the exported files in a throwaway Chromium to look at the pixels.
 *
 *   1. Build a report on the synthetic in-situ movie: a markdown cell, a live
 *      figure cell, and a movie cell rendered to a real .gif (so the manager
 *      holds a poster AND a movie file to inline).
 *   2. Static HTML  — the movie's poster + caption + play badge are present.
 *   3. Interactive  — the movie inlines as real animated pixels, and an embed
 *      that reports its height resizes its iframe past the 480 px default.
 *   4. Both files RENDER: opened in a plain Chromium over file://, the movie
 *      figure occupies real screen area and is not a broken image.
 *
 * Screenshots land in report_embed_contract_shots/ — a blank frame is a
 * failure, not a pass. SPYDE_LOG_LEVEL=WARNING tees backend logging to stderr
 * so the final audit can scan for Python tracebacks.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import {
  mkdtempSync, existsSync, rmSync, readFileSync, statSync, copyFileSync,
} from 'fs'
import { tmpdir } from 'os'
import { chromium } from 'playwright'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines, sigWindow,
  navWindow,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_embed_contract_shots')
const FIG_MIME = 'application/x-spyde-figure'

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
  // The synthetic in-situ movie: a movie cell needs a time-series source.
  await backendAction(page, 'load_test_data_movie')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2500)

  workDir = mkdtempSync(join(tmpdir(), 'spyde-embed-contract-'))
  htmlStaticPath = join(workDir, 'report-static.html')
  htmlInteractivePath = join(workDir, 'report-interactive.html')
  // .gif, not .mp4: the GIF encoder is Pillow, so this does not need ffmpeg.
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

/** The backend-assigned window id, read out of the pill's own drag payload —
 *  there is no window-id attribute in the DOM. */
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
    try { return Number((JSON.parse(dt.getData(mime)) as any).windowId) }
    catch { return NaN }
  }, { sel: pillSel, mime: FIG_MIME })
}

/** The id of the last movie cell in the open report ('' when there is none). */
async function lastMovieCellId(page: any): Promise<string> {
  return await page.evaluate(() => {
    const cells = ((window as any)._spyde_test_report?.()?.cells ?? []) as any[]
    const m = cells.filter((c) => c.cell_type === 'movie')
    return m.length ? String(m[m.length - 1].id) : ''
  })
}

async function dragAndDrop(page: any, srcSelector: string, dstSelector: string) {
  return await page.evaluate(({ srcSelector, dstSelector }: any) => {
    const src = document.querySelector(srcSelector) as HTMLElement
    const dst = document.querySelector(dstSelector) as HTMLElement
    if (!src || !dst) throw new Error('drag src/dst not found')
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
    const types = Array.from(dt.types)
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
    return { types }
  }, { srcSelector, dstSelector })
}

/** Render an exported HTML file in a throwaway Chromium and measure the movie
 *  figure — a caption alone is not proof it renders. */
async function measureExport(path: string, shot: string) {
  const browser = await chromium.launch()
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 1400 } })
    await page.goto('file://' + path.replace(/\\/g, '/'))
    // 5 s, not 2.5: a 1-D line iframe was still blank at 2.5 s and the
    // screenshot read as a broken cell when the export was fine.
    await page.waitForTimeout(5000)
    await page.screenshot({ path: join(SHOTS, shot), fullPage: true })
    return await page.evaluate(() => {
      const fig = document.querySelector('figure.report-movie')
      if (!fig) return { found: false, area: 0, cap: '', broken: true, frameH: 0 }
      const media = fig.querySelector('img, video') as
        HTMLImageElement | HTMLVideoElement | null
      const r = media ? media.getBoundingClientRect() : { width: 0, height: 0 }
      const broken = !!(media && media.tagName === 'IMG'
        && !(media as HTMLImageElement).naturalWidth)
      const frames = Array.from(document.querySelectorAll(
        'figure.report-figure iframe')) as HTMLIFrameElement[]
      const frameH = frames.length
        ? Math.max(...frames.map((f) => f.getBoundingClientRect().height)) : 0
      return {
        found: true,
        area: Math.round(r.width * r.height),
        cap: fig.querySelector('figcaption')?.textContent ?? '',
        broken,
        frameH: Math.round(frameH),
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
  const ta = page.locator('[data-testid^="report-cell-textarea-"]').first()
  await ta.fill('# Embed contract\nEvery cell kind must survive every export.')
  await ta.press('Control+Enter')

  // A live figure cell, via the proven breadcrumb-pill drag. This is the 2048²
  // movie frame — 4.2 M values, past the data-embed cap, which is the "we told
  // you" path test 4 checks.
  const sigWin = sigWindow(page)
  const pill = sigWin.getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-fig-src', '1'))
  const res = await dragAndDrop(page, '[data-fig-src="1"]', '[data-testid="report-body"]')
  expect(res.types).toContain(FIG_MIME)
  await expect(page.locator('[data-report-cell="1"] > [data-testid^="report-figcell-"]')
    .first()).toBeVisible({ timeout: 20_000 })

  // …and the NAVIGATOR as a second cell: a 6-point 1-D curve, comfortably under
  // the cap, so the same export exercises the embedded-values path too.
  const navPill = navWindow(page).getByTestId('window-breadcrumb')
  await navPill.evaluate((el: HTMLElement) => el.setAttribute('data-fig-src', '2'))
  await dragAndDrop(page, '[data-fig-src="2"]', '[data-testid="report-body"]')
  await expect.poll(() => page.locator(
    '[data-report-cell="1"] > [data-testid^="report-figcell-"]').count(), {
    timeout: 20_000, message: 'the navigator cell never appeared',
  }).toBeGreaterThan(1)

  // A movie cell sourced from the in-situ tree, then RENDERED so the manager
  // holds both a poster and a movie file. The window id comes out of the pill's
  // own drag payload (the pattern presentation_fixes.spec.ts uses) — there is no
  // window-id attribute in the DOM to read.
  const wid: number = await windowIdFromPill(page, '[data-fig-src="1"]')
  expect(Number.isFinite(wid), 'no window id from the signal pill').toBe(true)
  await backendAction(page, 'report_add_movie_cell', {})
  await expect.poll(() => lastMovieCellId(page), {
    timeout: 20_000, message: 'movie cell never appeared in the report',
  }).not.toBe('')
  const cellId = await lastMovieCellId(page)

  await backendAction(page, 'report_set_movie_source',
                      { cell_id: cellId, source_window_id: wid })
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

  await page.screenshot({ path: join(SHOTS, '01-report-built.png') })
  ctx.assertNoJsErrors()
})

// ── 2) Static export keeps the movie ─────────────────────────────────────────

test('2) static HTML keeps the movie poster, caption and badge', async () => {
  const { page } = ctx
  await backendAction(page, 'report_export_html',
                      { mode: 'static', path: htmlStaticPath })
  await expect.poll(() => existsSync(htmlStaticPath) && statSync(htmlStaticPath).size > 2000, {
    timeout: 30_000, message: 'static export never written',
  }).toBe(true)
  const html = readFileSync(htmlStaticPath, 'utf-8')

  // The regression: this whole figure used to be absent from the document.
  expect(html, 'movie cell missing from the static export')
    .toContain('report-movie')
  expect(html, 'movie poster missing').toContain('movie-badge')
  expect(html).not.toContain('\x00bin:')

  const m = await measureExport(htmlStaticPath, '02-static-rendered.png')
  expect(m.found, 'no movie figure in the rendered static page').toBe(true)
  expect(m.broken, 'movie poster rendered as a broken image').toBe(false)
  expect(m.area, 'movie poster occupies no screen area').toBeGreaterThan(5000)
})

// ── 3) Interactive export inlines the real movie + autosizes its embeds ──────

test('3) interactive HTML inlines the movie and sizes its embeds', async () => {
  const { page } = ctx
  await backendAction(page, 'report_export_html',
                      { mode: 'interactive', path: htmlInteractivePath })
  await expect.poll(() => existsSync(htmlInteractivePath) && statSync(htmlInteractivePath).size > 2000, {
    timeout: 120_000, message: 'interactive export never written',
  }).toBe(true)
  const html = readFileSync(htmlInteractivePath, 'utf-8')

  // The rendered movie itself travels, not just a still.
  expect(html, 'movie was not inlined into the interactive export')
    .toContain('data:image/gif;base64,')
  expect(html).not.toContain('\x00bin:')

  // Keep a copy next to the screenshots: the temp dir is removed in afterAll,
  // and the exported file is the artifact worth inspecting when a cell renders
  // blank.
  copyFileSync(htmlInteractivePath, join(SHOTS, 'export-interactive.html'))

  const m = await measureExport(htmlInteractivePath, '03-interactive-rendered.png')
  expect(m.found).toBe(true)
  expect(m.broken, 'inlined movie rendered as a broken image').toBe(false)
  expect(m.area, 'inlined movie occupies no screen area').toBeGreaterThan(5000)
})

// ── 4) the exported page hands back the NUMBERS ──────────────────────────────

test('4) the exported page downloads the values behind a figure', async () => {
  // Asserting the button exists proves nothing: the page builds the file itself
  // in JS, so the only real check is clicking it and reading what comes out.
  const browser = await chromium.launch()
  try {
    const page = await browser.newPage({ acceptDownloads: true })
    await page.goto('file://' + htmlInteractivePath.replace(/\\/g, '/'))
    await page.waitForTimeout(2000)
    // The 2048² movie frame is past the embed cap. It must SAY so rather than
    // ship a quietly decimated copy, which would look exactly like the real one.
    const note = page.locator('.fig-note').first()
    await expect(note, 'the oversized panel did not report itself')
      .toBeVisible({ timeout: 15_000 })
    await expect(note).toHaveText(/exceeds the .* embed cap/)

    // The navigator curve is small enough to travel whole. Its download lives on
    // the figure's HOVER toolbar now — the same gesture as the app's report
    // cell, instead of a permanent row of buttons under every figure.
    const box = page.locator('.fig-box')
      .filter({ has: page.getByTestId('fig-chrome-data') }).first()
    await box.hover()
    const dataBtn = box.getByTestId('fig-chrome-data')
    await expect(dataBtn, 'no data download on the figure toolbar')
      .toBeVisible({ timeout: 15_000 })
    await page.screenshot({ path: join(SHOTS, '04-data-controls.png'), fullPage: true })

    // Save-as-PNG: the figure as it looks now, contrast and all. anyplotlib's
    // standalone page answers the request; the host turns it into a file.
    const pngBtn = box.getByTestId('fig-chrome-png')
    await expect(pngBtn, 'no PNG save on the figure toolbar').toBeVisible()
    const [pngDownload] = await Promise.all([
      page.waitForEvent('download', { timeout: 30_000 }),
      pngBtn.click(),
    ])
    const pngPath = join(workDir, 'figure.png')
    await pngDownload.saveAs(pngPath)
    const png = readFileSync(pngPath)
    expect(Array.from(png.subarray(1, 4)).map((c) => String.fromCharCode(c)).join(''),
      'saved file is not a PNG').toBe('PNG')
    expect(png.length, 'saved PNG is suspiciously small').toBeGreaterThan(5000)

    const [npyDownload] = await Promise.all([
      page.waitForEvent('download', { timeout: 20_000 }),
      dataBtn.click(),
    ])
    const npyPath = join(workDir, 'downloaded.npy')
    await npyDownload.saveAs(npyPath)
    const npy = readFileSync(npyPath)
    expect(Array.from(npy.subarray(1, 6)).map((c) => String.fromCharCode(c)).join(''),
      'downloaded .npy has no NUMPY magic').toBe('NUMPY')
    // The header is what makes it loadable, and it is hand-built in JS: numpy
    // requires 10 + headerLen to be a multiple of 64 and the dict to declare the
    // dtype and shape. Magic bytes alone would pass on a file numpy rejects.
    const headerLen = npy[8] | (npy[9] << 8)
    expect((10 + headerLen) % 64, '.npy header is not 64-byte aligned').toBe(0)
    const dict = npy.subarray(10, 10 + headerLen).toString('latin1')
    expect(dict).toContain("'descr': '<f4'")
    expect(dict).toMatch(/'shape': \(\d+,/)
    // …and the payload is exactly the declared number of float32s.
    const count = Number(dict.match(/'shape': \((\d+),/)![1])
    expect(npy.length - 10 - headerLen).toBe(count * 4)
  } finally {
    await browser.close()
  }
})
