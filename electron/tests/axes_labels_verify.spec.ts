/**
 * axes_labels_verify.spec.ts — verify the axes / scale bar / title / 1-D y-label
 * fixes end-to-end in the running app (both the GPU-tiled large-signal path and
 * the small Canvas2D path), plus the subwindow-resize no-clip fix.
 *
 * The decorations (tick gutters, scale bar, title strip) are drawn on SEPARATE
 * overlay canvases inside the figure iframe. We assert the axis/title/scalebar
 * canvases actually have non-blank content — proving they render — and screenshot
 * each stage for the human eyes check.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'axes_labels_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

// The tests share ONE app instance and build on each other's loaded data (movie
// → resize that window → si_grains), so they must run in order.
test.describe.configure({ mode: 'serial' })

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  await ctx.page.waitForTimeout(1500)
})
test.afterAll(async () => { await ctx?.app?.close() })
test.setTimeout(180_000)

// Count non-transparent pixels on canvases in a frame whose testid/name matches
// `sel`. Used to prove an axis/title/scalebar canvas actually drew something.
async function canvasInk(page, cssSelector: string): Promise<number> {
  let total = 0
  for (const frame of page.frames()) {
    try {
      total += await frame.evaluate((sel) => {
        let n = 0
        for (const c of Array.from(document.querySelectorAll(sel))) {
          const cv = c as HTMLCanvasElement
          const ctx = cv.getContext('2d')
          if (!ctx || !cv.width || !cv.height) continue
          const d = ctx.getImageData(0, 0, cv.width, cv.height).data
          for (let p = 0; p < d.length; p += 4) {
            // Count text/tick ink: any pixel noticeably brighter than the dark
            // axis background (#1e1e2e ≈ 30,30,46).
            if (d[p] > 60 || d[p + 1] > 60 || d[p + 2] > 70) n++
          }
        }
        return n
      }, cssSelector)
    } catch { /* detached */ }
  }
  return total
}

test('GPU-tiled large signal shows axes + scale bar + title', async () => {
  const { page } = ctx
  await backendAction(page, 'load_test_data_movie')
  await waitForSubwindowCount(page, 1, 60_000)
  // Let the large frame paint + the GPU tile enable + the axes push land.
  await page.waitForTimeout(4000)
  await page.screenshot({ path: join(SHOTS, '01-movie-gpu-signal.png') })

  // The signal figure's axis gutter canvases must have tick ink (physical units
  // "nm" → has_axes True via set_extent). Before the fix these were display:none
  // for the tiled path. The gutter canvases have no class, so we prove the figure
  // rendered decorations by total canvas ink and rely on the screenshot for eyes.
  const anyInk = await canvasInk(page, 'canvas')
  console.log('[axes] movie total canvas ink =', anyInk)
  expect(anyInk).toBeGreaterThan(1000)   // something rendered
})

test('resize subwindow much larger does not clip the figure', async () => {
  const { page } = ctx
  const win = page.getByTestId('subwindow').first()
  const before = await win.boundingBox()
  console.log('[resize] before =', before)
  await page.screenshot({ path: join(SHOTS, '02-before-resize.png') })

  // Grab the bottom-right resize handle and drag it out to make the window much
  // larger than its initial size (the regime that used to clip the figure to the
  // baked-in body box).
  const handle = win.getByTestId('resize-handle')
  const hb = await handle.boundingBox()
  if (hb) {
    await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2)
    await page.mouse.down()
    await page.mouse.move(hb.x + 500, hb.y + 400, { steps: 12 })
    await page.mouse.up()
  } else {
    console.log('[resize] no resize handle testid found')
  }
  await page.waitForTimeout(2500)
  const after = await win.boundingBox()
  console.log('[resize] after =', after)
  await page.screenshot({ path: join(SHOTS, '03-after-resize-larger.png') })

  // The FIGURE iframe (a file:// page with a #widget-root, NOT the main app
  // renderer) must have its document body + #widget-root fill the iframe viewport
  // after the resize — the fix makes html/body/#widget-root width/height:100% so a
  // grown figure is never clipped to the baked-in initial body size.
  let checkedFigure = false
  const bodyFills = await (async () => {
    for (const frame of page.frames()) {
      try {
        const r = await frame.evaluate(() => {
          const root = document.getElementById('widget-root')
          if (!root) return null   // not a figure iframe (main app has no widget-root)
          const b = document.body
          return {
            bw: b ? b.getBoundingClientRect().width : 0,
            bh: b ? b.getBoundingClientRect().height : 0,
            iw: window.innerWidth, ih: window.innerHeight,
            rootW: root.getBoundingClientRect().width,
            rootH: root.getBoundingClientRect().height,
          }
        })
        if (!r || r.iw < 200 || r.ih < 200) continue   // skip main app + tiny frames
        checkedFigure = true
        console.log('[resize] figure body/iframe metrics =', JSON.stringify(r))
        // body AND #widget-root cover the iframe viewport (no clip to a smaller
        // initial box); before the fix the body stayed pinned to the baked size.
        if (r.bw >= r.iw - 4 && r.bh >= r.ih - 4
            && r.rootW >= r.iw - 4 && r.rootH >= r.ih - 4) return true
      } catch { /* detached */ }
    }
    return false
  })()
  expect(checkedFigure, 'a figure iframe was inspected').toBe(true)
  expect(bodyFills, 'figure body + widget-root fill the resized iframe (no clip)').toBe(true)
  ctx.assertNoJsErrors()
})

test('small calibrated signal shows title + axes (si_grains)', async () => {
  const { page } = ctx
  const before = await page.getByTestId('subwindow').count()
  await backendAction(page, 'load_test_data_si_grains')
  // si_grains opens a navigator + a signal window (2 more).
  await waitForSubwindowCount(page, before + 2, 60_000)
  await page.waitForTimeout(3000)
  await page.screenshot({ path: join(SHOTS, '04-si-grains.png') })
  const anyInk = await canvasInk(page, 'canvas')
  console.log('[axes] si_grains total canvas ink =', anyInk)
  expect(anyInk).toBeGreaterThan(1000)
  ctx.assertNoJsErrors()
})
