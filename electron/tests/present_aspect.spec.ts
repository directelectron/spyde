/**
 * present_aspect.spec.ts — a navigator and a signal on the same slide must keep
 * their own aspect ratios.
 *
 * The bug: `PresentMode`'s `figBox` is `width:100%; height:58vh` — a FIXED box
 * that ignores the figure's natural shape, and SeamlessFigureFrame's host is
 * `position:absolute; inset:0`, so the figure is stretched to fill it. A 6x6
 * navigator and a 128x128 diffraction pattern are both square, but the box is
 * wide, so both come out stretched — and by different amounts once the two
 * shapes differ.
 *
 * Drives BOTH the 4-D (si_grains) and 5-D (test_data_5d) datasets, because 5-D
 * adds a second navigator (a 1-D time line) whose aspect is nothing like the
 * other two.
 *
 * Screenshots to present_aspect_shots/ — each Read by the author.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'present_aspect_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  await ctx.page.waitForTimeout(1500)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

/** Native HTML5 drag with a shared DataTransfer (report_reorder's pattern). */
async function dragAndDrop(page: any, srcSelector: string, dstSelector: string) {
  return await page.evaluate(({ srcSelector, dstSelector }: any) => {
    const src = document.querySelector(srcSelector) as HTMLElement
    const dst = document.querySelector(dstSelector) as HTMLElement
    if (!src || !dst) throw new Error(`drag src/dst not found: ${!!src}/${!!dst}`)
    const dt = new DataTransfer()
    const fire = (el: HTMLElement, type: string) => {
      const ev = new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: dt })
      el.dispatchEvent(ev)
      return ev
    }
    fire(src, 'dragstart')
    fire(dst, 'dragenter')
    fire(dst, 'dragover')
    fire(dst, 'drop')
    fire(src, 'dragend')
    return { types: Array.from(dt.types) }
  }, { srcSelector, dstSelector })
}

/** Drag a window (matched by its breadcrumb prefix) into the report body. */
async function addWindowAsCell(page: any, prefix: RegExp, tag: string) {
  const win = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: prefix }) })
    .first()
  await win.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement, t: string) => el.setAttribute('data-fig-src', t), tag)
  await dragAndDrop(page, `[data-fig-src="${tag}"]`, '[data-testid="report-body"]')
}

/** The on-screen box of every figure frame on the active slide, plus the
 *  aspect of the CANVAS inside it — the two must agree, or the figure is
 *  stretched. */
async function slideFigureAspects(page: any) {
  return await page.evaluate(() => {
    const out: any[] = []
    const active = document.querySelector('[data-testid="present-mode"]')
      ?? document.body
    for (const host of Array.from(active.querySelectorAll('iframe'))) {
      const el = host as HTMLIFrameElement
      const r = el.getBoundingClientRect()
      if (r.width < 20 || r.height < 20) continue
      let inner: { w: number; h: number } | null = null
      try {
        const d = el.contentDocument
        const c = d?.querySelector('canvas') as HTMLCanvasElement | null
        if (c) inner = { w: c.width, h: c.height }
      } catch { /* cross-origin */ }
      out.push({ boxW: Math.round(r.width), boxH: Math.round(r.height),
                 boxAspect: +(r.width / r.height).toFixed(3), inner })
    }
    return out
  })
}

test('4D: build a presentation with the navigator AND the signal', async () => {
  const { page } = ctx
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2500)

  await page.getByTestId('toggle-report').click()
  await backendAction(page, 'report_new', { type: 'presentation' })
  await expect(page.getByTestId('report-body')).toBeVisible()

  await addWindowAsCell(page, /^N-/, 'nav4d')
  await page.waitForTimeout(1500)
  await addWindowAsCell(page, /^S-/, 'sig4d')
  await page.waitForTimeout(2500)

  // NB `report-figcell-<id>` is the cell ROOT, but `report-figcell-drag-<id>`
  // etc. share the prefix — count distinct cell ids, not raw matches.
  const ids = await page.evaluate(() => {
    const set = new Set<string>()
    for (const el of Array.from(document.querySelectorAll('[data-testid^="report-figcell-"]'))) {
      const t = el.getAttribute('data-testid') || ''
      const rest = t.slice('report-figcell-'.length)
      if (!rest.includes('-')) set.add(rest)
    }
    return Array.from(set)
  })
  expect(ids.length, 'both the navigator and the signal should be cells').toBe(2)
  await page.screenshot({ path: join(SHOTS, '01-4d-report.png') })
})

test('4D: present it and measure the figure boxes', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await page.waitForTimeout(3500)
  await page.screenshot({ path: join(SHOTS, '02-4d-present-slide1.png') })
  const a = await slideFigureAspects(page)
  console.log('[present-aspect] 4D slide figures =', JSON.stringify(a))

  // Advance through the deck, shooting each slide.
  for (let i = 0; i < 2; i++) {
    await page.keyboard.press('ArrowRight')
    await page.waitForTimeout(2000)
    await page.screenshot({ path: join(SHOTS, `03-4d-present-slide${i + 2}.png`) })
    console.log(`[present-aspect] 4D slide ${i + 2} =`,
                JSON.stringify(await slideFigureAspects(page)))
  }
  await page.keyboard.press('Escape')
  await page.waitForTimeout(1000)
})

test('5D: a stack adds a SECOND navigator (time) — three figures on a slide', async () => {
  const { page } = ctx
  // Fresh document + the 5-D stack: time line -> real-space nav -> DP.
  await backendAction(page, 'load_test_data_5d', { frames: 4, nav: 24, sig: 32 })
  await waitForSubwindowCount(page, 3, 120_000)
  await page.waitForTimeout(4000)
  await page.screenshot({ path: join(SHOTS, '10-5d-windows.png') })

  await backendAction(page, 'report_new', { type: 'presentation' })
  await expect(page.getByTestId('report-body')).toBeVisible()

  // Only the windows THIS dataset opened — the 4-D test's windows are still on
  // screen, and stacking eight figures on one slide is not the case under test.
  // The 5-D stack gives three of genuinely different aspect: a 1-D time line, a
  // 24x24 real-space map, and a 32x32 DP.
  const five = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /test_data_5d/ }) })
  const wins = await five.count()
  expect(wins, 'the 5-D stack should open its own windows').toBeGreaterThanOrEqual(2)
  for (let i = 0; i < wins; i++) {
    const w = five.nth(i)
    await w.getByTestId('window-breadcrumb')
      .evaluate((el: HTMLElement, t: string) => el.setAttribute('data-fig-src', t), `w5d${i}`)
    await dragAndDrop(page, `[data-fig-src="w5d${i}"]`, '[data-testid="report-body"]')
    await page.waitForTimeout(1200)
  }
  await page.waitForTimeout(2500)
  await page.screenshot({ path: join(SHOTS, '11-5d-report.png') })
})

test('5D: present it — every figure stays on screen', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await page.waitForTimeout(4000)
  await page.screenshot({ path: join(SHOTS, '12-5d-present.png') })

  const boxes = await slideFigureAspects(page)
  console.log('[present-aspect] 5D slide figures =', JSON.stringify(boxes))

  // THE regression this file exists for: the slide must not overflow, so the
  // figures' combined height has to fit the stage.
  const overflow = await page.evaluate(() => {
    const s = document.querySelector('[data-testid="present-slide"][data-active="1"]') as HTMLElement
    if (!s) return null
    return { scroll: s.scrollHeight, client: s.clientHeight,
             figVh: s.getAttribute('data-fig-vh') }
  })
  console.log('[present-aspect] 5D slide overflow =', JSON.stringify(overflow))
  expect(overflow, 'no active slide').not.toBeNull()
  expect(overflow!.scroll, 'the slide overflows — figures are clipped off-screen')
    .toBeLessThanOrEqual(overflow!.client + 4)

  await page.keyboard.press('Escape')
  await page.waitForTimeout(800)
  ctx.assertNoJsErrors()
})
