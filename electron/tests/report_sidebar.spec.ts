/**
 * report_sidebar.spec.ts — Report Builder Phase 1, end-to-end in the real app.
 *
 * Real Dask + bundled-synthetic Si-grains (navigator + signal window). Drives
 * the Report sidebar the way a user would: toggle open, add a markdown cell +
 * edit it, drag the SIGNAL window's breadcrumb pill into the report body (native
 * HTML5 DnD with a shared DataTransfer), edit the caption, toggle raw mode, save
 * to an explicit path (no OS dialog), inspect the written .spyde-report zip, then
 * close + reopen and confirm the figure REBINDS live (tree still open).
 *
 * Screenshots at every stage to report_sidebar_shots/ — a blank panel is a
 * failure even when selectors pass, so each shot is Read by the author.
 *
 * Backend emit/emit_error do NOT reach Playwright stdout (PLOTAPP line protocol);
 * SPYDE_LOG_LEVEL=WARNING tees logging to stderr → ctx.backend.logBuffer, which
 * the final audit scans for Python tracebacks.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdtempSync, existsSync, statSync, rmSync, readFileSync } from 'fs'
import { tmpdir } from 'os'
import { execFileSync } from 'child_process'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_sidebar_shots')
const FIG_MIME = 'application/x-spyde-figure'

let ctx: Awaited<ReturnType<typeof launchApp>>
let workDir: string
let reportPath: string

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)   // navigator + signal
  await page.waitForTimeout(2500)                 // let the DP paint
  workDir = mkdtempSync(join(tmpdir(), 'spyde-report-'))
  reportPath = join(workDir, 'phase1.spyde-report')
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally {
    await ctx?.app?.close()
    if (workDir && existsSync(workDir)) {
      try { rmSync(workDir, { recursive: true, force: true }) } catch { /* */ }
    }
  }
})

/**
 * Full native HTML5 drag src→dst, entirely in-page so the constructed
 * DataTransfer is shared across dragstart/dragover/drop (the way a real user
 * drag is). Copied from breadcrumb_header.spec.ts — this is the proven pattern.
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

// Count bright (non-black) canvas pixels INSIDE the report figure cell's iframe
// specifically — NOT the MDI signal window (which is always bright). Resolves
// the report iframe's src, matches the Playwright frame by URL, sums its canvas
// pixels. Returns -1 if the report iframe isn't mounted yet.
async function reportFigurePixels(page: any): Promise<number> {
  const src: string | null = await page.evaluate(() => {
    const cell = document.querySelector('[data-testid^="report-figcell-"]')
    if (!cell) return null
    const ifr = cell.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
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

test('1) toggle the report sidebar open', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '01-sidebar-open.png') })
  ctx.assertNoJsErrors()
})

test('2) add a markdown cell, edit it, render H1 + bold', async () => {
  const { page } = ctx
  // New report (the empty sidebar shows New/Open only; create a doc).
  const newBtn = page.getByTestId('report-new')
  await newBtn.click()
  await expect(page.getByTestId('report-body')).toBeVisible()

  await page.getByTestId('report-add-text').click()
  // One markdown cell now exists — find its rendered view + double-click to edit.
  const rendered = page.locator('[data-testid^="report-cell-rendered-"]').first()
  await expect(rendered).toBeVisible()
  await rendered.dblclick()
  const ta = page.locator('[data-testid^="report-cell-textarea-"]').first()
  await expect(ta).toBeVisible()
  await ta.fill('# Results\nSome **bold** text')
  await ta.press('Control+Enter')
  // Rendered again — assert the H1 + bold appear.
  const renderedAfter = page.locator('[data-testid^="report-cell-rendered-"]').first()
  await expect(renderedAfter.locator('h1')).toHaveText(/Results/)
  await expect(renderedAfter.locator('strong')).toHaveText('bold')
  await page.screenshot({ path: join(SHOTS, '02-markdown-cell.png') })
  ctx.assertNoJsErrors()
})

// Spec A (report_edit): Shift+Enter commits a markdown cell — the textarea
// closes and the rendered markdown reflects the new source. (Test 2 exercised
// Ctrl+Enter; the ReportCell keydown handler also commits on Shift+Enter.)
test('2b) Shift+Enter commits the markdown cell (textarea closes, content updates)', async () => {
  const { page } = ctx
  // Re-enter edit on the same (only) markdown cell.
  const rendered = page.locator('[data-testid^="report-cell-rendered-"]').first()
  await expect(rendered).toBeVisible()
  await rendered.dblclick()
  const ta = page.locator('[data-testid^="report-cell-textarea-"]').first()
  await expect(ta).toBeVisible()
  await ta.fill('## Updated via Shift+Enter\nNew *content* here')
  // The load-bearing assertion: Shift+Enter (NOT Ctrl+Enter) commits + closes.
  await ta.press('Shift+Enter')

  // The textarea is gone (edit committed → back to rendered view).
  await expect(page.locator('[data-testid^="report-cell-textarea-"]')).toHaveCount(0, {
    timeout: 10_000,
  })
  // The rendered markdown reflects the new source: an H2 with the new heading +
  // emphasised (em) text — and NOT the old "Results" H1.
  const renderedAfter = page.locator('[data-testid^="report-cell-rendered-"]').first()
  await expect(renderedAfter).toBeVisible()
  await expect(renderedAfter.locator('h2')).toHaveText(/Updated via Shift\+Enter/)
  await expect(renderedAfter.locator('em')).toHaveText('content')
  await expect(renderedAfter.locator('h1')).toHaveCount(0)
  await page.screenshot({ path: join(SHOTS, '02b-markdown-shift-enter.png') })
  ctx.assertNoJsErrors()

  // Restore the cell to its original H1+bold content so downstream tests
  // (save/reopen asserts on "# Results" + **bold**) still hold.
  await renderedAfter.dblclick()
  const ta2 = page.locator('[data-testid^="report-cell-textarea-"]').first()
  await expect(ta2).toBeVisible()
  await ta2.fill('# Results\nSome **bold** text')
  await ta2.press('Shift+Enter')
  await expect(page.locator('[data-testid^="report-cell-rendered-"]').first().locator('h1'))
    .toHaveText(/Results/, { timeout: 10_000 })
  ctx.assertNoJsErrors()
})

test('3) drag the signal window pill into the report → live figure cell', async () => {
  const { page } = ctx
  // The signal window's breadcrumb pill stamps FIGURE_DRAG_MIME (windowId). Tag
  // the source + the report body as drop target, then native-DnD between them.
  const sigWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  const pill = sigWin.getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-fig-src', '1'))

  const res = await dragAndDrop(page, '[data-fig-src="1"]', '[data-testid="report-body"]')
  console.log('[report] drop types =', JSON.stringify(res.types))
  expect(res.types).toContain(FIG_MIME)

  // A figure cell appears. Its live iframe (data-testid=figure-<figId>) mounts
  // and paints; assert non-trivial pixels inside it.
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  const iframe = figCell.locator('iframe[data-testid^="figure-"]')
  await expect(iframe).toBeVisible({ timeout: 15_000 })

  // Scope the pixel check to the REPORT figure iframe (not the MDI signal DP —
  // that's always bright). A blank report figure fails here.
  await expect.poll(async () => reportFigurePixels(page), {
    timeout: 30_000, message: 'report figure iframe drew no pixels (blank embed)',
  }).toBeGreaterThan(500)

  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '03-figure-cell.png') })
  ctx.assertNoJsErrors()
})

test('4) edit the caption', async () => {
  const { page } = ctx
  const captionView = page.locator('[data-testid^="report-figcell-caption-"]').first()
  await expect(captionView).toBeVisible()
  await captionView.click()
  const input = page.locator('[data-testid^="report-figcell-caption-input-"]').first()
  await expect(input).toBeVisible()
  await input.fill('Fig. 1 — Si grains DP')
  await input.press('Enter')
  await expect(page.locator('[data-testid^="report-figcell-caption-"]').first())
    .toHaveText('Fig. 1 — Si grains DP')
  await page.screenshot({ path: join(SHOTS, '04-caption.png') })
  ctx.assertNoJsErrors()
})

test('5) toggle raw mode (source textareas) and back', async () => {
  const { page } = ctx
  await page.getByTestId('report-raw-toggle').click()
  // In raw mode every markdown cell shows a textarea.
  await expect(page.locator('[data-testid^="report-cell-textarea-"]').first())
    .toBeVisible()
  await page.screenshot({ path: join(SHOTS, '05-raw-mode.png') })
  // Toggle back to rich.
  await page.getByTestId('report-raw-toggle').click()
  await expect(page.locator('[data-testid^="report-cell-rendered-"]').first())
    .toBeVisible()
  ctx.assertNoJsErrors()
})

test('6) save to an explicit path and validate the zip contents', async () => {
  const { page } = ctx
  // Drive report_save with an explicit path (no OS dialog). The renderer save
  // button would open a dialog; the backend action accepts {path} directly.
  await backendAction(page, 'report_save', { path: reportPath })

  // Poll the DOM for the saved (non-dirty) state: the dirty dot disappears.
  await expect.poll(async () => {
    return await page.getByTestId('report-dirty').count()
  }, { timeout: 30_000, message: 'report never reached saved (non-dirty) state' })
    .toBe(0)

  // The zip must exist on disk.
  await expect.poll(() => existsSync(reportPath), {
    timeout: 10_000, message: 'report file was never written',
  }).toBe(true)

  // A .spyde-report is a ZIP. Extract it into the work dir. Picking the right
  // extractor is per-platform: on Windows bsdtar reads zip, but we must resolve
  // the System32 binary explicitly — a bare 'tar' can resolve to git's GNU tar
  // depending on the spawning shell's PATH, and GNU tar treats the drive-letter
  // in an absolute 'C:\...' path as a remote-host prefix. On Linux/macOS the
  // system 'tar' is GNU tar, which CANNOT read a zip ("does not look like a tar
  // archive") — use 'unzip' (preinstalled on the GitHub Ubuntu runners) there.
  if (process.platform === 'win32') {
    const tarExe = join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'tar.exe')
    execFileSync(tarExe, ['-xf', reportPath, '-C', workDir], { stdio: 'pipe' })
  } else {
    execFileSync('unzip', ['-o', '-q', reportPath, '-d', workDir], { stdio: 'pipe' })
  }
  // The extractor writes report.md / figures/ / assets/ under workDir.
  const mdPath = join(workDir, 'report.md')
  expect(existsSync(mdPath)).toBe(true)
  const md = readFileSync(mdPath, 'utf-8')
  console.log('[report] report.md:\n' + md)

  // report.md contains the markdown + the figure image ref with the caption.
  expect(md).toMatch(/#\s*Results/)
  expect(md).toMatch(/\*\*bold\*\*/)
  const figLine = md.match(/!\[Fig\. 1 — Si grains DP\]\(assets\/(c[0-9a-f]+)\.png\)/)
  expect(figLine, 'figure image-ref line with caption not found in report.md').not.toBeNull()
  const cid = figLine![1]

  // figures/<id>.yaml exists and parses as YAML with a layers entry.
  const yamlPath = join(workDir, 'figures', `${cid}.yaml`)
  expect(existsSync(yamlPath), `figures/${cid}.yaml missing`).toBe(true)
  const yamlText = readFileSync(yamlPath, 'utf-8')
  console.log('[report] figure yaml:\n' + yamlText)
  expect(yamlText).toMatch(/panels:/)
  expect(yamlText).toMatch(/layers:/)

  // assets/<id>.png exists and is a REAL WYSIWYG snapshot (>5 KB), not a 1px
  // placeholder / bake fallback of a tiny frame.
  const pngPath = join(workDir, 'assets', `${cid}.png`)
  expect(existsSync(pngPath), `assets/${cid}.png missing`).toBe(true)
  const pngSize = statSync(pngPath).size
  console.log('[report] asset png size =', pngSize)
  expect(pngSize, `snapshot PNG too small (${pngSize} B) — export harvest likely failed`)
    .toBeGreaterThan(5 * 1024)

  await page.screenshot({ path: join(SHOTS, '06-after-save.png') })
  ctx.assertNoJsErrors()
})

test('7) close, reopen, figure rebinds live', async () => {
  const { page } = ctx
  await backendAction(page, 'report_close')
  // Sidebar empties → the empty-state (New/Open) shows, no cells / no body.
  await expect.poll(async () => page.locator('[data-testid^="report-figcell-"]').count(), {
    timeout: 10_000, message: 'figure cell persisted after close',
  }).toBe(0)
  // A closed report shows the "No report open" empty-state (New/Open only), not
  // an open-but-empty body with dangling Save/dirty chrome.
  await expect(page.getByTestId('report-empty')).toBeVisible({ timeout: 10_000 })
  await expect(page.getByTestId('report-body')).toHaveCount(0)
  await page.screenshot({ path: join(SHOTS, '07a-closed.png') })

  // Reopen the same path (tree is still open → figure should rebind LIVE).
  await backendAction(page, 'report_open', { path: reportPath })
  await expect(page.getByTestId('report-body')).toBeVisible({ timeout: 10_000 })

  // Markdown cell came back.
  await expect(page.locator('[data-testid^="report-cell-rendered-"]').first().locator('h1'))
    .toHaveText(/Results/, { timeout: 10_000 })

  // Figure cell came back and REBOUND live (iframe, not the offline badge).
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 10_000 })
  // No "data offline" badge on a live rebind.
  await expect(page.locator('[data-testid^="report-figcell-offline-"]')).toHaveCount(0)
  const iframe = figCell.locator('iframe[data-testid^="figure-"]')
  await expect(iframe).toBeVisible({ timeout: 15_000 })
  await expect.poll(async () => reportFigurePixels(page), {
    timeout: 30_000, message: 'reopened figure iframe drew no pixels (rebind failed)',
  }).toBeGreaterThan(500)

  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '07b-reopened.png') })
  ctx.assertNoJsErrors()
})

test('8) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[report] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
