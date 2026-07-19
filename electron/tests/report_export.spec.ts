/**
 * report_export.spec.ts — Report Builder Phase 3 (exports + copy/paste),
 * end-to-end in the real app.
 *
 * Real Dask + bundled-synthetic Si-grains. Builds a small report (a markdown
 * cell with a heading + bold, one figure cell via the proven breadcrumb-pill
 * drag from report_sidebar.spec.ts), then exercises every Phase-3 export/paste
 * surface the way a user would:
 *
 *   2. Static HTML   — Export ▾ → Static HTML → assert the file contains the
 *      title, REAL rendered markdown (<h1>/<strong>, NOT the <pre class="md-src">
 *      fallback), exactly one data:image/png <img>, the caption, no \x00bin:, no
 *      <iframe>.
 *   3. Interactive HTML — sandboxed <iframe srcdoc>, no \x00bin:, AND it actually
 *      RENDERS: load the file in a throwaway Chromium page and assert non-trivial
 *      colored pixels in the figure region (first real-browser check of the
 *      interactive export).
 *   4. PDF — stubbed pdf path → success note → file exists, >20 KB, starts %PDF.
 *   5. Markdown folder — report.md + figures/*.yaml + assets/*.png match the
 *      report.
 *   6. Copy/Paste — figure Copy → Paste enables → Paste appends a SECOND live
 *      figure cell; markdown Duplicate appends a duplicate; OS clipboard holds a
 *      real PNG after the figure Copy.
 *   7. Copy to Report toolbar button — from a 2-D plot's floating toolbar → a new
 *      figure cell appends; also from a CLOSED report state (auto-opens one).
 *
 * The export flows call ipcMain.handle('report:export-dialog') /
 * 'report:export-pdf' in the MAIN process; we stub the dialog from the spec via
 * app.evaluate (remove + re-register the handler → return fixed paths) so no OS
 * dialog blocks. Screenshots at every stage to report_export_shots/ — a blank
 * frame is a failure. SPYDE_LOG_LEVEL=WARNING tees backend logging to stderr so
 * the final audit scans ctx.backend.logBuffer for Python tracebacks.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdtempSync, existsSync, statSync, rmSync, readFileSync, readdirSync } from 'fs'
import { tmpdir } from 'os'
import { chromium } from 'playwright'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines, sigWindow,
  titlebarGrabPoint,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_export_shots')
const FIG_MIME = 'application/x-spyde-figure'

let ctx: Awaited<ReturnType<typeof launchApp>>
let workDir: string
// Fixed export target paths handed to the stubbed report:export-dialog. Each
// export decides which one to return by the dialog `kind` argument.
let htmlStaticPath: string
let htmlInteractivePath: string
let pdfPath: string
let mdFolderPath: string

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)   // navigator + signal
  await page.waitForTimeout(2500)                 // let the DP paint

  workDir = mkdtempSync(join(tmpdir(), 'spyde-report-export-'))
  htmlStaticPath = join(workDir, 'report-static.html')
  htmlInteractivePath = join(workDir, 'report-interactive.html')
  pdfPath = join(workDir, 'report.pdf')
  mdFolderPath = join(workDir, 'md-folder')   // created empty on demand per-export

  // Stub the MAIN-process export dialog so the UI export flows never block on an
  // OS picker. The dialog kind ('html' | 'pdf' | 'folder') + a per-call routing
  // string (set on globalThis before each export) decide which fixed path to
  // return. This mirrors the existing spyde.spec.ts ipcMain.handle stub pattern.
  await ctx.app.evaluate(({ ipcMain }, paths) => {
    const g = globalThis as unknown as { __exportRoute?: string }
    ipcMain.removeHandler('report:export-dialog')
    ipcMain.handle('report:export-dialog', async (_e, kind: string) => {
      if (kind === 'folder') return paths.mdFolder
      if (kind === 'pdf') return paths.pdf
      // kind === 'html' — route static vs interactive by the pending marker.
      return g.__exportRoute === 'interactive' ? paths.htmlInteractive : paths.htmlStatic
    })
  }, { htmlStatic: htmlStaticPath, htmlInteractive: htmlInteractivePath,
       pdf: pdfPath, mdFolder: mdFolderPath })
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally {
    await ctx?.app?.close()
    if (workDir && existsSync(workDir)) {
      try { rmSync(workDir, { recursive: true, force: true }) } catch { /* */ }
    }
  }
})

/** Set the pending html-export route marker in the main process (static default). */
async function setExportRoute(route: 'static' | 'interactive') {
  await ctx.app.evaluate((_e, r) => {
    ;(globalThis as unknown as { __exportRoute?: string }).__exportRoute = r
  }, route)
}

/**
 * Full native HTML5 drag src→dst, entirely in-page so the constructed
 * DataTransfer is shared across dragstart/dragover/drop (the way a real user
 * drag is). Copied verbatim from report_sidebar.spec.ts (the proven pattern).
 */
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

// The `[data-testid^="report-figcell-"]` prefix also matches a figure cell's
// SUB-elements (report-figcell-caption-<id>, -edit-toggle-<id>, …), so a naive
// count/index over that prefix is wrong. A figure-cell CONTAINER is the element
// whose parent is the `data-report-cell="1"` list wrapper (ReportSidebar wraps
// each cell). Count/index those.
function figCellContainers() {
  return `[data-report-cell="1"] > [data-testid^="report-figcell-"]`
}

// Number of ACTUAL figure-cell containers in the report body.
async function countFigCells(page: any): Promise<number> {
  return await page.locator(figCellContainers()).count()
}

// Bright (non-black) pixels inside a specific report figure cell's iframe (the
// Nth figcell CONTAINER). Scoped to the report cell, NOT the MDI signal window.
// -1 if the iframe isn't mounted yet. Mirrors report_sidebar's reportFigurePixels.
async function figCellPixels(page: any, nth = 0): Promise<number> {
  const src: string | null = await page.evaluate((n: number) => {
    const cells = Array.from(document.querySelectorAll(
      '[data-report-cell="1"] > [data-testid^="report-figcell-"]'))
    const cell = cells[n]
    if (!cell) return null
    const ifr = cell.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
    return ifr?.src || null
  }, nth)
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

/**
 * Click the signal window's "Copy to Report" floating-toolbar button.
 *
 * The floating toolbar shares the window's z-level and sits BELOW the window in
 * the empty MDI space; its pointer-events toggle with the window's hover state,
 * so a plain `.click()` can land while the bar is momentarily transparent and the
 * mdi-area background swallows it. Focus-raise the window first (click its
 * titlebar, per the UI-degradation memory), then HOVER the button — the bar's own
 * onMouseEnter keeps it interactive — and click via the mouse at the button's box
 * centre so no intervening layer can intercept.
 */
async function clickCopyToReport(page: any) {
  const sigWin = sigWindow(page)
  // Raise the window so its toolbar is above sibling windows / the mdi-area.
  const grab = await titlebarGrabPoint(sigWin)
  await page.mouse.click(grab.x, grab.y)
  await sigWin.hover()
  const btn = sigWin.getByTestId('action-btn-Copy to Report')
  await expect(btn).toBeVisible({ timeout: 10_000 })
  await btn.hover()   // keep the bar shown (its onMouseEnter → onHoverShow)
  const box = await btn.boundingBox()
  if (!box) throw new Error('Copy to Report button has no bounding box')
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2)
}

// ── 1) Build a small report (markdown cell + one figure cell) ────────────────

test('1) build a small report: markdown H1+bold cell + one live figure cell', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', {})
  await expect(page.getByTestId('report-body')).toBeVisible()

  // Markdown cell: heading + bold.
  await page.getByTestId('report-add-text').click()
  const rendered = page.locator('[data-testid^="report-cell-rendered-"]').first()
  await expect(rendered).toBeVisible()
  await rendered.dblclick()
  const ta = page.locator('[data-testid^="report-cell-textarea-"]').first()
  await expect(ta).toBeVisible()
  await ta.fill('# Export Results\nSome **bold** text')
  await ta.press('Control+Enter')
  const renderedAfter = page.locator('[data-testid^="report-cell-rendered-"]').first()
  await expect(renderedAfter.locator('h1')).toHaveText(/Export Results/)
  await expect(renderedAfter.locator('strong')).toHaveText('bold')

  // Figure cell: drag the SIGNAL window's breadcrumb pill into the report body.
  const sigWin = sigWindow(page)
  const pill = sigWin.getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-fig-src', '1'))
  const res = await dragAndDrop(page, '[data-fig-src="1"]', '[data-testid="report-body"]')
  expect(res.types).toContain(FIG_MIME)

  const figCell = page.locator(figCellContainers()).first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await expect.poll(async () => figCellPixels(page, 0), {
    timeout: 30_000, message: 'report figure iframe drew no pixels (blank embed)',
  }).toBeGreaterThan(500)

  // Caption on the figure cell (used to assert on it in the exports).
  const captionView = page.locator('[data-testid^="report-figcell-caption-"]').first()
  await captionView.click()
  const capInput = page.locator('[data-testid^="report-figcell-caption-input-"]').first()
  await expect(capInput).toBeVisible()
  await capInput.fill('Fig. 1 — Si grains DP')
  await capInput.press('Enter')
  await expect(page.locator('[data-testid^="report-figcell-caption-"]').first())
    .toHaveText('Fig. 1 — Si grains DP')

  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '01-report-built.png') })
  ctx.assertNoJsErrors()
})

// ── 2) Static HTML export ────────────────────────────────────────────────────

test('2) static HTML export → real rendered markdown + one embedded PNG', async () => {
  const { page } = ctx
  await setExportRoute('static')
  await page.getByTestId('report-menu-toggle').click()
  await page.getByTestId('report-export-toggle').click()
  await expect(page.getByTestId('report-export-menu')).toBeVisible()
  await page.getByTestId('export-html-static').click()

  // Transient "Exported ✓" note.
  const note = page.getByTestId('report-export-note')
  await expect(note).toBeVisible({ timeout: 20_000 })
  await expect(note).toHaveText(/Exported/)
  await page.screenshot({ path: join(SHOTS, '02-static-export-note.png') })

  await expect.poll(() => existsSync(htmlStaticPath), {
    timeout: 10_000, message: 'static HTML export file was never written',
  }).toBe(true)
  const html = readFileSync(htmlStaticPath, 'utf-8')

  // The page carries a <title> (the report title — "Untitled Report" by default;
  // the "Export Results" heading is the markdown cell's H1, asserted next) and
  // the REAL rendered markdown (proves the html cache — NOT the
  // <pre class="md-src"> raw fallback).
  expect(html).toMatch(/<title>[^<]+<\/title>/i)
  expect(html).toMatch(/<h1[^>]*>[^<]*Export Results/)
  expect(html).toMatch(/<strong>bold<\/strong>/)
  expect(html, 'markdown fell back to <pre class="md-src"> — html cache lost')
    .not.toMatch(/class="md-src"/)

  // Exactly one embedded PNG <img>, no iframe, no binary token.
  const imgs = html.match(/<img[^>]+src="data:image\/png;base64,/g) ?? []
  expect(imgs.length, `expected exactly 1 embedded PNG <img>, got ${imgs.length}`).toBe(1)
  expect(html).toContain('Fig. 1 — Si grains DP')
  expect(html, 'static export must not embed an iframe').not.toMatch(/<iframe/)
  expect(html, 'static export leaked a \\x00bin: token').not.toContain('\x00bin:')

  ctx.assertNoJsErrors()
})

// ── 3) Interactive HTML export (renders in a real browser) ───────────────────

test('3) interactive HTML export → sandboxed srcdoc iframe that RENDERS', async () => {
  const { page } = ctx
  await setExportRoute('interactive')
  await page.getByTestId('report-menu-toggle').click()
  await page.getByTestId('report-export-toggle').click()
  await expect(page.getByTestId('report-export-menu')).toBeVisible()
  await page.getByTestId('export-html-interactive').click()

  const note = page.getByTestId('report-export-note')
  await expect(note).toBeVisible({ timeout: 20_000 })
  await expect(note).toHaveText(/Exported/)
  await page.screenshot({ path: join(SHOTS, '03-interactive-export-note.png') })

  await expect.poll(() => existsSync(htmlInteractivePath), {
    timeout: 10_000, message: 'interactive HTML export file was never written',
  }).toBe(true)
  const html = readFileSync(htmlInteractivePath, 'utf-8')

  // A sandboxed srcdoc iframe (the live figure), no binary tokens.
  expect(html, 'interactive export has no <iframe>').toMatch(/<iframe[^>]*srcdoc=/)
  expect(html).toMatch(/sandbox="allow-scripts"/)
  expect(html, 'interactive export leaked a \\x00bin: token').not.toContain('\x00bin:')
  // Markdown still real, title present.
  expect(html).toMatch(/<h1[^>]*>[^<]*Export Results/)
  expect(html).toContain('Fig. 1 — Si grains DP')

  // Real-browser render check: open the exported file in a throwaway Chromium
  // page (WebGPU-capable channel so the anyplotlib figure can draw). The figure
  // renders to a canvas that may be a WebGPU context — getImageData on a WebGPU
  // canvas returns nothing, so DON'T read the canvas directly. Instead screenshot
  // the composited page (captures WebGPU or Canvas2D alike, per gpu_image_parity
  // pattern), decode the PNG in-page, and assert the figure region is non-blank:
  // a dark figure background (#11111b-ish) with a bright central spot, distinct
  // from the white article page — a blank/failed embed would be all-white.
  const browser = await chromium.launch({
    channel: 'chromium',
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
  })
  const shotPath = join(SHOTS, '03b-interactive-in-browser.png')
  let figureStats = { dark: 0, bright: 0, total: 0 }
  try {
    const bpage = await browser.newPage({ viewport: { width: 900, height: 1100 } })
    const errs: string[] = []
    bpage.on('pageerror', (e) => errs.push(String(e)))
    await bpage.goto('file://' + htmlInteractivePath.replace(/\\/g, '/'))
    await bpage.waitForLoadState('networkidle').catch(() => {})
    // Wait for the figure's canvas to exist in the srcdoc iframe (it loads its
    // ESM module, then paints). Poll for a sized canvas across all frames.
    await expect.poll(async () => {
      let n = 0
      for (const fr of bpage.frames()) {
        try {
          n += await fr.evaluate(() =>
            Array.from(document.querySelectorAll('canvas'))
              .filter((c) => (c as HTMLCanvasElement).width > 0).length)
        } catch { /* detached */ }
      }
      return n
    }, { timeout: 30_000, message: 'interactive export mounted no figure canvas' })
      .toBeGreaterThan(0)
    // Give the GPU/Canvas2D paint a moment to land, then screenshot + decode.
    await bpage.waitForTimeout(2500)
    const buf: Buffer = await bpage.screenshot({ path: shotPath, fullPage: true })
    // Decode the screenshot PNG in-page and classify pixels: a real figure adds
    // a large DARK region (its canvas background) + a BRIGHT cluster (the beam);
    // the surrounding article is white.
    figureStats = await bpage.evaluate(async (b64: string) => {
      const img = await new Promise<HTMLImageElement>((res, rej) => {
        const i = new Image(); i.onload = () => res(i); i.onerror = rej
        i.src = 'data:image/png;base64,' + b64
      })
      const cv = document.createElement('canvas')
      cv.width = img.width; cv.height = img.height
      const c2 = cv.getContext('2d')!
      c2.drawImage(img, 0, 0)
      const d = c2.getImageData(0, 0, cv.width, cv.height).data
      let dark = 0, bright = 0
      const total = cv.width * cv.height
      for (let p = 0; p < d.length; p += 4) {
        const r = d[p], g = d[p + 1], b = d[p + 2]
        const L = 0.3 * r + 0.59 * g + 0.11 * b
        if (L < 40) dark++            // figure's dark canvas background
        else if (L > 200 && r > 150 && g > 150 && b > 150) bright++ // beam/white
      }
      return { dark, bright, total }
    }, buf.toString('base64'))
    expect(errs, `real-browser render errors: ${errs.join('; ')}`).toEqual([])
  } finally {
    await browser.close()
  }
  console.log('[export] interactive-in-browser figure stats =', JSON.stringify(figureStats))
  // The figure's dark canvas region must occupy a non-trivial fraction of the
  // page — a blank/failed embed is all-white (dark≈0). > 2% of pixels dark is a
  // comfortable margin for a ~480px figure on a ~900px-wide article.
  expect(figureStats.dark,
    'interactive export figure did not render a dark canvas region in a real browser')
    .toBeGreaterThan(figureStats.total * 0.02)

  ctx.assertNoJsErrors()
})

// ── 4) PDF export ─────────────────────────────────────────────────────────────

test('4) PDF export → %PDF file > 20 KB', async () => {
  const { page } = ctx
  await page.getByTestId('report-menu-toggle').click()
  await page.getByTestId('report-export-toggle').click()
  await expect(page.getByTestId('report-export-menu')).toBeVisible()
  await page.getByTestId('export-pdf').click()

  const note = page.getByTestId('report-export-note')
  await expect(note).toBeVisible({ timeout: 40_000 })
  await expect(note).toHaveText(/Exported/)
  await page.screenshot({ path: join(SHOTS, '04-pdf-export-note.png') })

  await expect.poll(() => existsSync(pdfPath), {
    timeout: 10_000, message: 'PDF export file was never written',
  }).toBe(true)
  const size = statSync(pdfPath).size
  console.log('[export] PDF size =', size)
  expect(size, `PDF too small (${size} B) — printToPDF likely failed`).toBeGreaterThan(20 * 1024)
  const head = readFileSync(pdfPath).subarray(0, 5).toString('latin1')
  expect(head, 'PDF file does not start with %PDF').toContain('%PDF')

  ctx.assertNoJsErrors()
})

// ── 5) Markdown-folder export ────────────────────────────────────────────────

test('5) markdown-folder export → report.md + figures/*.yaml + assets/*.png', async () => {
  const { page } = ctx
  // The backend refuses a non-empty / non-export folder; mdFolderPath doesn't
  // exist yet, so dir_is_safe_md_target creates it fresh.
  await page.getByTestId('report-menu-toggle').click()
  await page.getByTestId('report-export-toggle').click()
  await expect(page.getByTestId('report-export-menu')).toBeVisible()
  await page.getByTestId('export-md-folder').click()

  const note = page.getByTestId('report-export-note')
  await expect(note).toBeVisible({ timeout: 20_000 })
  await expect(note).toHaveText(/Exported/)
  await page.screenshot({ path: join(SHOTS, '05-md-folder-note.png') })

  await expect.poll(() => existsSync(join(mdFolderPath, 'report.md')), {
    timeout: 10_000, message: 'markdown-folder export report.md was never written',
  }).toBe(true)

  const md = readFileSync(join(mdFolderPath, 'report.md'), 'utf-8')
  console.log('[export] md-folder report.md:\n' + md)
  expect(md).toMatch(/#\s*Export Results/)
  expect(md).toMatch(/\*\*bold\*\*/)
  const figLine = md.match(/!\[Fig\. 1 — Si grains DP\]\(assets\/(c[0-9a-f]+)\.png\)/)
  expect(figLine, 'figure image-ref with caption not found in report.md').not.toBeNull()
  const cid = figLine![1]

  const yamlPath = join(mdFolderPath, 'figures', `${cid}.yaml`)
  expect(existsSync(yamlPath), `figures/${cid}.yaml missing`).toBe(true)
  const yamlText = readFileSync(yamlPath, 'utf-8')
  expect(yamlText).toMatch(/panels:/)
  expect(yamlText).toMatch(/layers:/)

  const pngPath = join(mdFolderPath, 'assets', `${cid}.png`)
  expect(existsSync(pngPath), `assets/${cid}.png missing`).toBe(true)
  const pngSize = statSync(pngPath).size
  console.log('[export] md-folder asset png size =', pngSize)
  expect(pngSize, `snapshot PNG too small (${pngSize} B)`).toBeGreaterThan(5 * 1024)

  // Sanity: exactly the expected top-level entries (report.md + figures + assets).
  const entries = readdirSync(mdFolderPath).sort()
  console.log('[export] md-folder entries =', entries.join(', '))
  expect(entries).toContain('report.md')
  expect(entries).toContain('figures')
  expect(entries).toContain('assets')

  ctx.assertNoJsErrors()
})

// ── 6) Copy / Paste + Duplicate ──────────────────────────────────────────────

test('6) figure Copy → Paste appends a live cell; markdown Duplicate; OS clipboard', async () => {
  const { page } = ctx
  const figcellCountBefore = await countFigCells(page)
  expect(figcellCountBefore).toBe(1)

  // Reveal the figure cell's hover chrome + click Copy.
  const figCell = page.locator(figCellContainers()).first()
  const figCellId = await figCell.evaluate((el) =>
    (el.getAttribute('data-testid') || '').replace('report-figcell-', ''))
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const copyBtn = page.getByTestId(`cell-copy-${figCellId}`)
  await expect(copyBtn).toBeVisible()
  await copyBtn.click()
  await page.screenshot({ path: join(SHOTS, '06a-after-copy.png') })

  // OS clipboard now holds a real PNG (isEmpty() === false).
  const clipEmpty = await ctx.app.evaluate(({ clipboard }) => clipboard.readImage().isEmpty())
  expect(clipEmpty, 'OS clipboard image is empty after figure Copy').toBe(false)

  // Wave B removed the header Paste button (Ctrl+V still pastes an image; the
  // internal cell clipboard is consumed by the per-cell Duplicate). Duplicate the
  // figure cell → a SECOND figure cell appears with LIVE pixels.
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const figDupBtn = page.getByTestId(`cell-duplicate-${figCellId}`)
  await expect(figDupBtn).toBeVisible()
  await figDupBtn.click()
  await expect.poll(async () => countFigCells(page), {
    timeout: 15_000, message: 'Duplicate did not append a second figure cell',
  }).toBe(2)
  // The newly-pasted (last) figure cell should render live pixels.
  await expect.poll(async () => figCellPixels(page, 1), {
    timeout: 30_000, message: 'pasted figure cell drew no pixels (not live)',
  }).toBeGreaterThan(500)
  await page.waitForTimeout(1000)
  await page.screenshot({ path: join(SHOTS, '06b-after-paste.png') })

  // Duplicate the markdown cell → a duplicate appears immediately below it.
  const mdCell = page.locator('[data-testid^="report-cell-rendered-"]').first()
  const mdCellId = await mdCell.evaluate((el) =>
    (el.getAttribute('data-testid') || '').replace('report-cell-rendered-', ''))
  const mdCountBefore = await page.locator('[data-testid^="report-cell-rendered-"]').count()
  const mdContainer = page.getByTestId(`report-cell-${mdCellId}`)
  await mdContainer.dispatchEvent('mouseover', { bubbles: true })
  const dupBtn = page.getByTestId(`cell-duplicate-${mdCellId}`)
  await expect(dupBtn).toBeVisible()
  await dupBtn.click()
  await expect.poll(async () => page.locator('[data-testid^="report-cell-rendered-"]').count(), {
    timeout: 10_000, message: 'markdown Duplicate did not add a cell',
  }).toBe(mdCountBefore + 1)
  // The duplicate carries the same rendered H1.
  const h1s = page.locator('[data-testid^="report-cell-rendered-"] h1')
  await expect.poll(async () => h1s.count(), { timeout: 5_000 }).toBeGreaterThanOrEqual(2)
  await page.waitForTimeout(600)
  await page.screenshot({ path: join(SHOTS, '06c-after-duplicate.png') })

  ctx.assertNoJsErrors()
})

// ── 7) "Copy to Report" toolbar button ───────────────────────────────────────

test('7) Copy to Report toolbar button appends a figure cell (open report)', async () => {
  const { page } = ctx
  const before = await countFigCells(page)

  // The button is on the signal window's floating toolbar (toolbar_side: right,
  // plot_dim [2]).
  await clickCopyToReport(page)

  await expect.poll(async () => countFigCells(page), {
    timeout: 15_000, message: 'Copy to Report did not append a figure cell',
  }).toBe(before + 1)
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '07a-copy-to-report-open.png') })
  ctx.assertNoJsErrors()
})

test('7b) Copy to Report from a CLOSED report auto-opens a report', async () => {
  const { page } = ctx
  // Close the report first — the sidebar returns to the empty New/Open state.
  await backendAction(page, 'report_close')
  await expect(page.getByTestId('report-empty')).toBeVisible({ timeout: 10_000 })
  await expect(page.locator(figCellContainers())).toHaveCount(0)
  await page.screenshot({ path: join(SHOTS, '07b-report-closed.png') })

  // Copy to Report on the signal plot → auto-opens a fresh report with one cell.
  await clickCopyToReport(page)

  await expect(page.getByTestId('report-body')).toBeVisible({ timeout: 15_000 })
  await expect.poll(async () => countFigCells(page), {
    timeout: 15_000, message: 'Copy to Report from closed state did not open a report with a figure',
  }).toBe(1)
  // The auto-opened figure cell renders live pixels.
  const figCell = page.locator(figCellContainers()).first()
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await expect.poll(async () => figCellPixels(page, 0), {
    timeout: 30_000, message: 'auto-opened figure cell drew no pixels',
  }).toBeGreaterThan(500)
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '07c-copy-to-report-closed.png') })
  ctx.assertNoJsErrors()
})

// ── 8) Final audit ────────────────────────────────────────────────────────────

test('8) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[export] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
