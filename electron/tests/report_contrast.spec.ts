/**
 * report_contrast.spec.ts — display range (contrast) on a report figure layer,
 * end-to-end in the real app.
 *
 * A report figure's contrast was fixed at drop time: LayerSpec.clim was captured
 * from the live plot and then unreachable, because no UI ever sent the `clim`
 * key repfig_set_layer had always accepted. This drives the control a user
 * drives — open the cell's editor, open contrast, drag the black point — and
 * asserts on the FIGURE'S PIXELS, because a spec value round-tripping proves
 * only that a number moved.
 *
 * Screenshots land in report_contrast_shots/. SPYDE_LOG_LEVEL=WARNING tees
 * backend logging to stderr for the traceback audit.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { existsSync, statSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { chromium } from 'playwright'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines, sigWindow,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_contrast_shots')
const FIG_MIME = 'application/x-spyde-figure'

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2500)
})

test.afterAll(async () => {
  try {
    const bad = ctx?.backend ? backendErrorLines(ctx.backend) : []
    expect(bad, `backend errors:\n${bad.join('\n')}`).toEqual([])
    ctx?.assertNoJsErrors()
  } finally {
    await ctx?.app?.close()
  }
})

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

/** Mean luminance of the report figure cell's iframe canvases. The contrast
 *  assertion has to be on PIXELS: a clim value in the spec proves a number
 *  moved, not that the picture changed. */
async function figureLuminance(page: any): Promise<number> {
  const src: string | null = await page.evaluate(() => {
    const cell = document.querySelector(
      '[data-report-cell="1"] > [data-testid^="report-figcell-"]')
    const ifr = cell?.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
    return ifr?.src || null
  })
  if (!src) return -1
  const frame = page.frames().find((f: any) => f.url() === src)
  if (!frame) return -1
  try {
    return await frame.evaluate(() => {
      let sum = 0, n = 0
      for (const c of Array.from(document.querySelectorAll('canvas'))) {
        const cv = c as HTMLCanvasElement
        const cctx = cv.getContext('2d')
        if (!cctx || !cv.width || !cv.height) continue
        const d = cctx.getImageData(0, 0, cv.width, cv.height).data
        for (let p = 0; p < d.length; p += 4) { sum += d[p] + d[p + 1] + d[p + 2]; n += 3 }
      }
      return n ? sum / n : -1
    })
  } catch { return -1 }
}

/** The id of the report's single figure cell. */
async function cellId(page: any): Promise<string> {
  return await page.evaluate(() => {
    const cells = ((window as any)._spyde_test_report?.()?.cells ?? []) as any[]
    return String(cells.find((c) => c.cell_type === 'figure')?.id ?? '')
  })
}

/** The base layer's stored clim, straight off the report doc. */
async function baseClim(page: any): Promise<number[] | null> {
  return await page.evaluate(() => {
    const cells = ((window as any)._spyde_test_report?.()?.cells ?? []) as any[]
    const fig = cells.find((c) => c.cell_type === 'figure')
    return fig?.figure?.panels?.[0]?.layers?.[0]?.clim ?? null
  })
}

test('1) a report figure layer has a working contrast histogram', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', {})
  await expect(page.getByTestId('report-body')).toBeVisible()

  // Drop the signal window in as a figure cell.
  const pill = sigWindow(page).getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-fig-src', '1'))
  const res = await dragAndDrop(page, '[data-fig-src="1"]', '[data-testid="report-body"]')
  expect(res.types).toContain(FIG_MIME)
  const figCell = page.locator(
    '[data-report-cell="1"] > [data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 20_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]'))
    .toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(2500)
  await page.screenshot({ path: join(SHOTS, '01-figure-dropped.png') })

  // Hover the cell to raise its chrome, then click ◐. Deliberately NOT in edit
  // mode: noticing a figure is too dark is not something you do with the layer
  // editor open, so the control has to be reachable without it.
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const openBtn = page.getByTestId(`report-figcell-contrast-${await cellId(page)}`)
  await expect(openBtn, 'no contrast button on the figure hover chrome')
    .toBeVisible({ timeout: 10_000 })
  await openBtn.click()

  // The histogram must actually arrive from the backend (bars, not "measuring…").
  const histo = page.locator('[data-testid^="figcell-"][data-testid$="-histogram"]').first()
  await expect(histo, 'contrast histogram never rendered').toBeVisible({ timeout: 15_000 })
  await page.screenshot({ path: join(SHOTS, '02-contrast-open.png') })

  const before = await figureLuminance(page)
  expect(before, 'figure drew no pixels to measure').toBeGreaterThan(0)

  // Drag the BLACK point far to the right: everything below it clips to black,
  // so the frame must get darker.
  const box = await histo.boundingBox()
  expect(box).not.toBeNull()
  const b = box!
  await page.mouse.move(b.x + 4, b.y + b.height - 4)
  await page.mouse.down()
  await page.mouse.move(b.x + b.width * 0.72, b.y + b.height - 4, { steps: 12 })
  await page.mouse.up()
  await page.waitForTimeout(3000)
  await page.screenshot({ path: join(SHOTS, '03-black-point-raised.png') })

  const clim = await baseClim(page)
  expect(clim, 'dragging the handle never reached LayerSpec.clim').not.toBeNull()

  const after = await figureLuminance(page)
  expect(after, 'figure pixels did not change with the contrast')
    .toBeLessThan(before)

  // Auto hands the range back to the backend's robust levels.
  await page.locator('[data-testid^="figcell-"][data-testid$="-clim-auto"]').first().click()
  await page.waitForTimeout(2500)
  await page.screenshot({ path: join(SHOTS, '04-auto.png') })
  const autoClim = await baseClim(page)
  expect(autoClim).not.toBeNull()
  expect(autoClim![0]).toBeLessThan(clim![0])
})

test('2) the EXPORTED page lets a reader re-window the contrast', async () => {
  // The reader's half. Exported pixels are 8-bit codes: if they were quantised
  // over the display window there is nothing left to widen into, so this has to
  // be checked on the real file, by dragging, and by reading canvas pixels.
  //
  // Needs anyplotlib's set_display_window (0.7.3+, CSSFrancis/anyplotlib#61).
  // On an older pin the export still quantises over the display window, the
  // page correctly offers no control, and there is nothing here to test —
  // SKIP rather than fail, so the pinned build stays green.
  const { page } = ctx
  const out = join(tmpdir(), `spyde-contrast-${Date.now()}.html`)
  await backendAction(page, 'report_export_html', { mode: 'interactive', path: out })
  await expect.poll(() => existsSync(out) && statSync(out).size > 2000, {
    timeout: 120_000, message: 'interactive export never written',
  }).toBe(true)

  const browser = await chromium.launch()
  try {
    const rp = await browser.newPage({ viewport: { width: 1100, height: 1200 } })
    await rp.goto('file://' + out.replace(/\\/g, '/'))
    await rp.waitForTimeout(5000)
    await rp.screenshot({ path: join(SHOTS, '05-export-contrast.png'), fullPage: true })

    // Hover the figure to raise its toolbar, then ◐ — the same gesture as the
    // app's report cell, which is the point: one document, one way to drive it.
    const box = rp.locator('.fig-box')
      .filter({ has: rp.getByTestId('fig-chrome-contrast') }).first()
    await box.hover()
    await rp.getByTestId('fig-chrome-contrast').first().click()
    const strip = rp.locator('.fig-caret svg').first()
    await expect(strip, 'the exported page has no contrast caret')
      .toBeVisible({ timeout: 15_000 })

    const before = await exportFigureLuminance(rp)
    expect(before, 'exported figure drew no pixels').toBeGreaterThan(0)

    // Drag the black point right: the frame must darken. On the old export this
    // was impossible — the codes were already saturated outside the window.
    const b = (await strip.boundingBox())!
    await rp.mouse.move(b.x + 3, b.y + b.height / 2)
    await rp.mouse.down()
    await rp.mouse.move(b.x + b.width * 0.75, b.y + b.height / 2, { steps: 14 })
    await rp.mouse.up()
    await rp.waitForTimeout(2500)
    await rp.screenshot({ path: join(SHOTS, '06-export-darkened.png'), fullPage: true })

    const after = await exportFigureLuminance(rp)
    expect(after, 'dragging the exported contrast changed nothing')
      .toBeLessThan(before)

    // …and Auto puts it back to the window the report was written with, so the
    // control is not one-way.
    await rp.locator('.fig-caret [data-act="auto"]').first().click()
    await rp.waitForTimeout(2500)
    const reset = await exportFigureLuminance(rp)
    expect(Math.abs(reset - before) / before,
      'Auto did not restore the authored window').toBeLessThan(0.05)
  } finally {
    await browser.close()
    try { rmSync(out, { force: true }) } catch { /* */ }
  }
})

/** Mean luminance across the exported page's figure iframes. */
async function exportFigureLuminance(rp: any): Promise<number> {
  let sum = 0, n = 0
  for (const fr of rp.frames()) {
    if (fr === rp.mainFrame()) continue
    try {
      const s = await fr.evaluate(() => {
        let t = 0, c = 0
        for (const cv of Array.from(document.querySelectorAll('canvas'))) {
          const el = cv as HTMLCanvasElement
          if (el.width < 100 || el.height < 100) continue
          const ctx = el.getContext('2d')
          if (!ctx) continue
          const d = ctx.getImageData(0, 0, el.width, el.height).data
          for (let p = 0; p < d.length; p += 4) { t += d[p] + d[p+1] + d[p+2]; c += 3 }
          break
        }
        return c ? [t, c] : null
      })
      if (s) { sum += s[0]; n += s[1] }
    } catch { /* torn-down frame */ }
  }
  return n ? sum / n : -1
}
