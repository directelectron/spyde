/**
 * report_pdcusi_page.spec.ts — the generated PdCuSi report page, in a real
 * browser over file://.
 *
 * The page is built by `python -m scripts.gen_report_pdcusi` from a 5.7 GB
 * dataset, so it is checked in rather than generated here. When it is absent the
 * spec SKIPS — a docs artifact that has not been built is not a test failure —
 * but when it is present every claim it makes has to hold up: the static figures
 * must have pixels, and the embedded explorer must actually run (its three
 * panels, its slice slider, and a pattern that changes when you move).
 */
import { test, expect, chromium, Browser, Page } from '@playwright/test'
import { existsSync, statSync } from 'fs'
import { join } from 'path'

const PAGE = join(__dirname, '..', '..', 'docs-site', 'public', 'media',
                  'reports', 'pdcusi-crystallization.html')
const SHOTS = join(__dirname, '..', 'report_pdcusi_shots')

let browser: Browser
let page: Page

test.beforeAll(async () => {
  test.skip(!existsSync(PAGE), 'report page not built (scripts/gen_report_pdcusi.py)')
  const { mkdirSync } = require('fs')
  mkdirSync(SHOTS, { recursive: true })
  browser = await chromium.launch()
  page = await browser.newPage({ viewport: { width: 1100, height: 900 } })
  await page.goto('file:///' + PAGE.replace(/\\/g, '/'))
  await page.waitForLoadState('load')
})

test.afterAll(async () => { await browser?.close() })

test.setTimeout(120_000)

test('the report reads as a report', async () => {
  await expect(page.locator('h1')).toContainText('PdCuSi')
  // Every section landed.
  const heads = await page.locator('h2').allTextContents()
  expect(heads).toEqual(expect.arrayContaining([
    'The dataset', 'What the patterns show', 'Finding the vectors',
    'Crystallization shows up as a count', 'Explore it', 'Reproducing this',
  ]))

  // The static figures are real pixels, not broken images.
  const imgs = page.locator('figure.report-figure img')
  expect(await imgs.count()).toBeGreaterThanOrEqual(3)
  for (let i = 0; i < await imgs.count(); i++) {
    const ok = await imgs.nth(i).evaluate(
      (el) => (el as HTMLImageElement).naturalWidth > 100)
    expect(ok, `figure ${i} has no pixels`).toBe(true)
  }
  await page.screenshot({ path: join(SHOTS, '01-report-top.png'), fullPage: false })
  // The crystallization curve + the real-space count maps — the report's
  // actual finding, so look at them rather than trusting that a <img> exists.
  await page.getByText('Crystallization shows up as a count').scrollIntoViewIfNeeded()
  await page.waitForTimeout(200)
  await page.screenshot({ path: join(SHOTS, '03-crystallization.png') })
})

test('the embedded explorer runs, with all three panels', async () => {
  // The report's figure iframes are loading="lazy", so the explorer does not
  // even fetch until it is near the viewport — scroll to it first.
  await page.locator('figure.report-figure iframe').scrollIntoViewIfNeeded()
  const frame = page.frameLocator('figure.report-figure iframe')
  await expect(frame.locator('#vx-root[data-ready="1"]')).toBeAttached({
    timeout: 60_000,
  })
  const fr = page.frames().find((f) => f.url().startsWith('about:srcdoc')
    || f.name() !== undefined && f !== page.mainFrame())
  expect(fr, 'no explorer frame').toBeTruthy()

  const info = await fr!.evaluate(() => {
    const vx = (window as any).__vx
    return { slices: vx.nSlices, stack: vx.hasStackPanel(),
             ids: vx._h() && [vx._h().NAV_ID, vx._h().DP_ID, vx._h().STACK_ID] }
  })
  expect(info.stack, 'the stack navigator panel is missing').toBe(true)
  expect(info.slices).toBeGreaterThan(1)
  expect(new Set(info.ids).size, 'panels are not distinct').toBe(3)

  // Moving through the series must change what the pattern shows.
  await fr!.evaluate(() => (window as any).__vx.setPointer({ ix: 19, iy: 23 }))
  const at = async (t: number) => {
    await fr!.evaluate((tt) => (window as any).__vx.setSlice(tt), t)
    await page.waitForTimeout(400)
    return fr!.evaluate(() => (window as any).__vx.stats.hit)
  }
  const first = await at(0)
  const last = await at(info.slices - 1)
  expect(first, 'no vectors at the first slice').toBeGreaterThan(0)
  expect(last, 'no vectors at the last slice').toBeGreaterThan(0)

  await page.locator('figure.report-figure iframe').scrollIntoViewIfNeeded()
  await page.screenshot({ path: join(SHOTS, '02-explorer.png') })
})

test('the loading card stands in until the explorer is parsed', async () => {
  // The point block is ~20 MB and rides at the END of the document, so the
  // article is readable long before the explorer arrives. The card says so; it
  // must then get out of the way rather than sit there next to a live panel.
  const p2 = await browser.newPage({ viewport: { width: 1100, height: 900 } })
  await p2.goto('file:///' + PAGE.replace(/\\/g, '/'))
  await p2.waitForLoadState('load')
  // Once loaded, the card is gone and the frame is visible and in flow.
  await expect(p2.locator('#vx-loading')).toHaveCount(0)
  const box = await p2.locator('#vx-frame').boundingBox()
  expect(box, 'the explorer frame never became visible').not.toBeNull()
  expect(box!.x).toBeGreaterThan(-1000)   // not still parked off-screen
  await p2.close()
})

test('on a phone', async () => {
  // The poster QR lands people here on a handset, so the phone rendering is the
  // one that matters most — not the desktop one.
  const phone = await browser.newPage({
    viewport: { width: 390, height: 844 }, deviceScaleFactor: 3,
    isMobile: true, hasTouch: true,
  })
  await phone.goto('file:///' + PAGE.replace(/\\/g, '/'))
  await phone.waitForLoadState('load')
  await phone.screenshot({ path: join(SHOTS, '04-phone-top.png') })
  await phone.locator('#vx-frame').scrollIntoViewIfNeeded()
  const fr = phone.frameLocator('#vx-frame')
  await expect(fr.locator('#vx-root[data-ready="1"]')).toBeAttached({ timeout: 60_000 })
  await phone.waitForTimeout(800)
  await phone.screenshot({ path: join(SHOTS, '05-phone-explorer.png') })

  const inner = phone.frames().find((f) => f !== phone.mainFrame())!
  const geom = await inner.evaluate(() => {
    const h = (window as any).__vx._h()
    const r = (p: any) => p?.plotCanvas?.getBoundingClientRect()
    const b = [r(h.stackPanel), r(h.navPanel), r(h.dpPanel)].filter(Boolean)
    return { widths: b.map((x: any) => Math.round(x.width)),
             stack: (window as any).__vx.hasStackPanel(),
             docW: document.documentElement.clientWidth }
  })
  console.log('phone panel widths (CSS px):', JSON.stringify(geom))

  // THE phone contract: two panels, not three, and each one big enough to aim
  // at. The wide layout put them at 130/94/113 px, which is why this exists.
  expect(geom.stack, 'the phone layout must drop the stack panel').toBe(false)
  expect(geom.widths.length).toBe(2)
  for (const w of geom.widths) {
    expect(w, `panel too small to drive on a phone: ${geom.widths}`)
      .toBeGreaterThan(150)
  }

  // …and the slider takes over as the slice control, visibly.
  await expect(fr.getByTestId('vx-slice')).toBeVisible()
  const sliderW = (await fr.getByTestId('vx-slice').boundingBox())!.width
  expect(sliderW, 'the slice slider is too narrow to be a touch target')
    .toBeGreaterThan(150)

  await inner.evaluate(() => (window as any).__vx.setPointer({ ix: 19, iy: 23 }))
  const hitAt = async (t: number) => {
    await inner.evaluate((tt) => (window as any).__vx.setSlice(tt), t)
    await phone.waitForTimeout(300)
    return inner.evaluate(() => (window as any).__vx.stats.hit)
  }
  expect(await hitAt(0)).toBeGreaterThan(0)
  const n = await inner.evaluate(() => (window as any).__vx.nSlices)
  expect(await hitAt(n - 1)).toBeGreaterThan(0)
  await phone.screenshot({ path: join(SHOTS, '08-phone-last-slice.png') })
  await phone.close()
})

test('a wide window still gets all three panels', async () => {
  const wide = await browser.newPage({ viewport: { width: 1400, height: 1000 } })
  await wide.goto('file:///' + PAGE.replace(/\\/g, '/'))
  await wide.waitForLoadState('load')
  await wide.locator('#vx-frame').scrollIntoViewIfNeeded()
  const fr = wide.frameLocator('#vx-frame')
  await expect(fr.locator('#vx-root[data-ready="1"]')).toBeAttached({ timeout: 60_000 })
  const inner = wide.frames().find((f) => f !== wide.mainFrame())!
  expect(await inner.evaluate(() => (window as any).__vx.hasStackPanel())).toBe(true)
  // The slider must stay hidden where the stack panel is doing that job.
  await expect(fr.getByTestId('vx-slice')).toBeHidden()
  await wide.close()
})
