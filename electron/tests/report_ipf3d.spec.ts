/**
 * report_ipf3d.spec.ts — Phase 5: 3-D IPF report cells, end-to-end.
 *
 * Real Dask + bundled-synthetic Si-grains, then a full dense OM run via the
 * test-only `run_test_orientation` (the cheapest deterministic path to a 3-D
 * IPF view — same recipe as orientation_workflow/orientation_lazy; there is no
 * bundled OM loader). Then, the Phase-5 surface the way a user hits it:
 *
 *   1. switch the Orientation window to its 3-D view (2D⇄3D toggle),
 *   2. drag its breadcrumb pill into the Report sidebar — the FIGURE_DRAG_MIME
 *      payload must carry view:'3d' (the drag-time activeFigure registry) and
 *      the backend must build a kind='scene3d' cell with a LIVE 3-D iframe,
 *   3. static HTML export → the cell's <img> holds REAL 3-D pixels (Phase A2
 *      exportPNG re-renders WebGPU 3-D panels in-task),
 *   4. interactive HTML export → the cell's srcdoc iframe renders the live
 *      rotatable sphere in a throwaway WebGPU-capable Chromium.
 *
 * Pixel proof: the IPF sphere is a cloud of strongly CHROMATIC dots on a dark
 * figure background. WebGPU canvases refuse getImageData, so every probe here
 * decodes a composited SCREENSHOT (page.screenshot clip / full page — captures
 * WebGPU per the gpu_image_parity pattern) and counts colorful pixels
 * (max−min channel spread > 40). A blank/black sphere fails.
 *
 * Screenshots to report_ipf3d_shots/ — each one is Read by the author.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdtempSync, mkdirSync, existsSync, rmSync, readFileSync } from 'fs'
import { tmpdir } from 'os'
import { chromium } from 'playwright'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_ipf3d_shots')
const FIG_MIME = 'application/x-spyde-figure'

let ctx: Awaited<ReturnType<typeof launchApp>>
let workDir: string
let htmlStaticPath: string
let htmlInteractivePath: string
let omId = ''                     // the Orientation window's id (from its toggle)

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)   // navigator + signal
  await page.waitForTimeout(2500)                 // let the DP paint

  workDir = mkdtempSync(join(tmpdir(), 'spyde-report-ipf3d-'))
  htmlStaticPath = join(workDir, 'ipf3d-static.html')
  htmlInteractivePath = join(workDir, 'ipf3d-interactive.html')

  // Stub the MAIN-process export dialog (report_export.spec.ts pattern) so the
  // Export menu never blocks on an OS picker; route static/interactive by a
  // marker set before each export.
  await ctx.app.evaluate(({ ipcMain }, paths) => {
    const g = globalThis as unknown as { __exportRoute?: string }
    ipcMain.removeHandler('report:export-dialog')
    ipcMain.handle('report:export-dialog', async (_e, kind: string) => {
      if (kind !== 'html') return null
      return g.__exportRoute === 'interactive' ? paths.htmlInteractive : paths.htmlStatic
    })
  }, { htmlStatic: htmlStaticPath, htmlInteractive: htmlInteractivePath })
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally {
    await ctx?.app?.close()
    if (workDir && existsSync(workDir)) {
      try { rmSync(workDir, { recursive: true, force: true }) } catch { /* */ }
    }
  }
})

async function setExportRoute(route: 'static' | 'interactive') {
  await ctx.app.evaluate((_e, r) => {
    ;(globalThis as unknown as { __exportRoute?: string }).__exportRoute = r
  }, route)
}

/**
 * Full native HTML5 drag src→dst with a SHARED DataTransfer (the proven
 * report_sidebar.spec.ts pattern) — also returns the FIGURE_DRAG_MIME payload
 * the pill stamped, so the spec can assert view:'3d' rode the drag.
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
    const figPayload = dt.getData('application/x-spyde-figure') || null
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
    return { types, figPayload }
  }, { srcSelector, dstSelector })
}

/**
 * Decode a screenshot PNG (base64) IN-PAGE and classify its pixels:
 *   colorful — channel spread > 40 (the chromatic IPF dots),
 *   dark     — luminance < 40 (the figure's dark canvas background),
 *   total.
 * Works for WebGPU content because the input is the COMPOSITED screenshot,
 * never a canvas getImageData.
 */
async function shotStats(page: any, buf: Buffer):
    Promise<{ colorful: number; dark: number; total: number }> {
  return await page.evaluate(async (b64: string) => {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image(); i.onload = () => res(i); i.onerror = rej
      i.src = 'data:image/png;base64,' + b64
    })
    const cv = document.createElement('canvas')
    cv.width = img.width; cv.height = img.height
    const c2 = cv.getContext('2d')!
    c2.drawImage(img, 0, 0)
    const d = c2.getImageData(0, 0, cv.width, cv.height).data
    let colorful = 0, dark = 0
    for (let p = 0; p < d.length; p += 4) {
      const r = d[p], g = d[p + 1], b = d[p + 2]
      if (Math.max(r, g, b) - Math.min(r, g, b) > 40) colorful++
      if (0.3 * r + 0.59 * g + 0.11 * b < 40) dark++
    }
    return { colorful, dark, total: cv.width * cv.height }
  }, buf.toString('base64'))
}

test('1) OM run → Orientation window → 3-D IPF view shown', async () => {
  const { page } = ctx
  const before = await page.getByTestId('subwindow').count()
  await backendAction(page, 'run_test_orientation')

  // The IPF result window opens when compute finishes (signal-based wait).
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 180_000, message: 'orientation compute never opened the IPF window',
  }).toBeGreaterThan(before)

  // The OM RESULT window is the one that OWNS the 2D/3D toggle (do NOT match
  // by 'Orientation' text — the source window has an action button with it).
  const toggle = page.getByTestId(/^ipf-view-toggle-/).first()
  await expect(toggle).toBeAttached({ timeout: 90_000 })
  omId = (await toggle.getAttribute('data-testid'))!.replace('ipf-view-toggle-', '')
  console.log('[ipf3d] orientation window id =', omId)
  const omWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(`ipf-view-toggle-${omId}`) }).first()
  await expect(omWin).toBeVisible({ timeout: 20_000 })

  // Switch to the 3-D explorer and give the GPU probe + first draw a moment.
  await page.getByTestId(`ipf-view-3d-${omId}`).click({ force: true })
  await page.waitForTimeout(4000)

  const bb = await omWin.boundingBox()
  expect(bb).not.toBeNull()
  const shot = await page.screenshot({ clip: bb!, path: join(SHOTS, '01-ipf-3d-window.png') })
  const stats = await shotStats(page, shot)
  console.log('[ipf3d] 3-D window stats =', JSON.stringify(stats))
  // The sphere is a cloud of chromatic IPF dots — a blank/black panel fails.
  expect(stats.colorful, '3-D IPF window shows no chromatic sphere points')
    .toBeGreaterThan(150)
  ctx.assertNoJsErrors()
})

test('2) drag the 3-D pill into the report → scene3d cell with a LIVE 3-D iframe', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await page.getByTestId('report-new').click()
  await expect(page.getByTestId('report-body')).toBeVisible()

  // Drag the ORIENTATION window's breadcrumb pill (its 3-D view is up).
  const omWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(`ipf-view-toggle-${omId}`) }).first()
  const pill = omWin.getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-ipf-src', '1'))
  const res = await dragAndDrop(page, '[data-ipf-src="1"]', '[data-testid="report-body"]')
  console.log('[ipf3d] drop types =', JSON.stringify(res.types),
    'figPayload =', res.figPayload)
  expect(res.types).toContain(FIG_MIME)
  // THE Phase-5 drop contract: the pill stamped the ACTIVE view.
  const payload = JSON.parse(res.figPayload!)
  expect(payload.view).toBe('3d')

  // A figure cell appears and the backend built it as a scene3d panel.
  const figCell = page.locator('[data-report-cell="1"] > [data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect.poll(async () => await page.evaluate(() => {
    const rep = (window as any)._spyde_test_report?.()
    const cell = (rep?.cells ?? []).find((c: any) => c.cell_type === 'figure')
    return cell?.figure?.panels?.[0]?.kind ?? null
  }), { timeout: 15_000, message: 'report cell never became scene3d' })
    .toBe('scene3d')

  // The live iframe mounts and draws the sphere (screenshot-decode probe —
  // the 3-D canvas is WebGPU, getImageData is unavailable).
  const iframe = figCell.locator('iframe[data-testid^="figure-"]')
  await expect(iframe).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(4000)                 // GPU probe + first draw
  await expect.poll(async () => {
    const bb = await figCell.boundingBox()
    if (!bb) return -1
    const shot = await page.screenshot({ clip: bb })
    return (await shotStats(page, shot)).colorful
  }, { timeout: 30_000, message: 'report scene3d cell drew no chromatic sphere' })
    .toBeGreaterThan(150)

  const bb = await figCell.boundingBox()
  await page.screenshot({ clip: bb!, path: join(SHOTS, '02-scene3d-cell.png') })
  await page.screenshot({ path: join(SHOTS, '02b-full-app.png') })
  ctx.assertNoJsErrors()
})

test('3) static HTML export → the scene3d cell <img> holds real 3-D pixels', async () => {
  const { page } = ctx
  await setExportRoute('static')
  await page.getByTestId('report-export-toggle').click()
  await expect(page.getByTestId('report-export-menu')).toBeVisible()
  await page.getByTestId('export-html-static').click()

  const note = page.getByTestId('report-export-note')
  await expect(note).toBeVisible({ timeout: 30_000 })
  await expect(note).toHaveText(/Exported/)
  await expect.poll(() => existsSync(htmlStaticPath), {
    timeout: 10_000, message: 'static HTML export file was never written',
  }).toBe(true)
  const html = readFileSync(htmlStaticPath, 'utf-8')
  expect(html).toMatch(/<img[^>]+src="data:image\/png;base64,/)
  expect(html, 'static export must not embed an iframe').not.toMatch(/<iframe/)

  // Open the export in a plain throwaway Chromium and decode the figure <img>:
  // real 3-D pixels = a dark figure background + chromatic sphere dots. This is
  // the Phase-A2 proof (exportPNG re-rendered the WebGPU panel for harvest).
  const browser = await chromium.launch({ channel: 'chromium' })
  let stats = { colorful: 0, dark: 0, total: 0 }
  try {
    const bpage = await browser.newPage({ viewport: { width: 900, height: 1100 } })
    await bpage.goto('file://' + htmlStaticPath.replace(/\\/g, '/'))
    await bpage.waitForLoadState('load')
    stats = await bpage.evaluate(() => {
      const img = document.querySelector('figure.report-figure img') as HTMLImageElement
      if (!img || !img.naturalWidth) return { colorful: -1, dark: -1, total: 0 }
      const cv = document.createElement('canvas')
      cv.width = img.naturalWidth; cv.height = img.naturalHeight
      const c2 = cv.getContext('2d')!
      c2.drawImage(img, 0, 0)
      const d = c2.getImageData(0, 0, cv.width, cv.height).data
      let colorful = 0, dark = 0
      for (let p = 0; p < d.length; p += 4) {
        const r = d[p], g = d[p + 1], b = d[p + 2]
        if (Math.max(r, g, b) - Math.min(r, g, b) > 40) colorful++
        if (0.3 * r + 0.59 * g + 0.11 * b < 40) dark++
      }
      return { colorful, dark, total: cv.width * cv.height }
    })
    await bpage.screenshot({ path: join(SHOTS, '03-static-export.png'), fullPage: true })
  } finally {
    await browser.close()
  }
  console.log('[ipf3d] static export img stats =', JSON.stringify(stats))
  expect(stats.total, 'export <img> did not decode').toBeGreaterThan(0)
  // Non-blank: >1% of the image is not the white page background (the dark
  // figure bg dominates a real 3-D shot), plus chromatic sphere dots exist.
  expect(stats.dark, 'static 3-D image has no dark figure background (blank/white)')
    .toBeGreaterThan(stats.total * 0.01)
  expect(stats.colorful, 'static 3-D image has no chromatic sphere points')
    .toBeGreaterThan(150)
  ctx.assertNoJsErrors()
})

test('4) interactive HTML export → the scene3d iframe renders live 3-D', async () => {
  const { page } = ctx
  await setExportRoute('interactive')
  await page.getByTestId('report-export-toggle').click()
  await expect(page.getByTestId('report-export-menu')).toBeVisible()
  await page.getByTestId('export-html-interactive').click()

  const note = page.getByTestId('report-export-note')
  await expect(note).toBeVisible({ timeout: 30_000 })
  await expect(note).toHaveText(/Exported/)
  await expect.poll(() => existsSync(htmlInteractivePath), {
    timeout: 10_000, message: 'interactive HTML export file was never written',
  }).toBe(true)
  const html = readFileSync(htmlInteractivePath, 'utf-8')
  expect(html).toMatch(/<iframe[^>]*srcdoc=/)
  expect(html, 'interactive export leaked a \\x00bin: token').not.toContain('\x00bin:')

  // Render in a WebGPU-capable throwaway Chromium (gpu_image_parity pattern) —
  // the sphere must draw; getImageData is useless on the GPU canvas, so decode
  // the composited screenshot.
  const browser = await chromium.launch({
    channel: 'chromium',
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
  })
  let stats = { colorful: 0, dark: 0, total: 0 }
  try {
    const bpage = await browser.newPage({ viewport: { width: 900, height: 1100 } })
    const errs: string[] = []
    bpage.on('pageerror', (e) => errs.push(String(e)))
    await bpage.goto('file://' + htmlInteractivePath.replace(/\\/g, '/'))
    await bpage.waitForLoadState('networkidle').catch(() => {})
    // The srcdoc iframe loads its inlined ESM, then paints; wait for a sized
    // canvas in any frame, then give the GPU probe + first draw a moment.
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
    await bpage.waitForTimeout(3500)
    const buf: Buffer = await bpage.screenshot({
      path: join(SHOTS, '04-interactive-export.png'), fullPage: true })
    stats = await shotStats(bpage, buf)
    expect(errs, `real-browser render errors: ${errs.join('; ')}`).toEqual([])
  } finally {
    await browser.close()
  }
  console.log('[ipf3d] interactive export stats =', JSON.stringify(stats))
  expect(stats.dark, 'interactive 3-D iframe rendered no dark figure region')
    .toBeGreaterThan(stats.total * 0.02)
  expect(stats.colorful, 'interactive 3-D iframe rendered no chromatic sphere')
    .toBeGreaterThan(150)
  ctx.assertNoJsErrors()
})

test('5) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[ipf3d] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
