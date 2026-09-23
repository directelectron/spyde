/**
 * tiled_navigator_crosshair.spec.ts — GitHub #171.
 *
 * ⇧-clicking two navigator chips of a 4D-STEM tree tiles the navigator images
 * side by side. Every panel carries a copy of each navigation selector's
 * crosshair, and the selector's own crosshair on the live navigator is the one
 * master holding the position. So:
 *
 *   1. both panels' crosshairs sit at the same image position when tiled;
 *   2. dragging the crosshair in one panel moves the one in the other panel to
 *      the same place, and neither snaps back once the drag is released;
 *   3. adding a selector while tiled puts its crosshair (in its own colour) on
 *      every panel.
 *
 * Screenshots go to tiled_crosshair_shots/ — read them; they are the test.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines, navWindow, raiseWindow,
} = require('./_harness.cjs')

const SHOTS = 'tiled_crosshair_shots'
mkdirSync(SHOTS, { recursive: true })

// SELECTOR_COLORS[0] and [1] in multiplot_manager.py.
const GREEN = [0x00, 0xe6, 0x76]
const BLUE = [0x40, 0xc4, 0xff]

type Crosshair = { panel: number, x: number, y: number, u: number, v: number, n: number }

/** Crosshair centres of one colour on every visible canvas of the window's
 *  shown figure. (u, v) is the position as a fraction of its canvas, which is
 *  what must agree between panels that show the same navigation grid. */
async function crosshairs(win: any, rgb: number[]): Promise<Crosshair[]> {
  const found: Crosshair[] = []
  const frames = win.locator('iframe')
  for (let i = 0; i < await frames.count(); i++) {
    const el = await frames.nth(i).elementHandle()
    const box = el ? await el.boundingBox() : null
    if (!el || !box || box.width < 10) continue
    const frame = await el.contentFrame()
    if (!frame) continue
    const hits: any[] = await frame.evaluate((target: number[]) => {
      const out: any[] = []
      const canvases = Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]
      canvases.forEach((c, index) => {
        const g = c.getContext('2d')
        if (!g || !c.width || !c.height) return
        const rect = c.getBoundingClientRect()
        if (rect.width < 10) return
        const d = g.getImageData(0, 0, c.width, c.height).data
        const cols = new Int32Array(c.width), rows = new Int32Array(c.height)
        let n = 0
        for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
          const p = (y * c.width + x) * 4
          if (d[p + 3] > 40 && Math.abs(d[p] - target[0]) < 40
              && Math.abs(d[p + 1] - target[1]) < 40 && Math.abs(d[p + 2] - target[2]) < 40) {
            cols[x]++; rows[y]++; n++
          }
        }
        if (n < 20) return
        let bx = 0, by = 0
        for (let x = 1; x < c.width; x++) if (cols[x] > cols[bx]) bx = x
        for (let y = 1; y < c.height; y++) if (rows[y] > rows[by]) by = y
        out.push({
          panel: index,
          x: rect.left + (bx + 0.5) * rect.width / c.width,
          y: rect.top + (by + 0.5) * rect.height / c.height,
          u: (bx + 0.5) / c.width, v: (by + 0.5) / c.height, n,
        })
      })
      return out
    }, rgb).catch(() => [])  // the iframe may be reloading
    for (const h of hits) found.push({ ...h, x: h.x + box.x, y: h.y + box.y })
  }
  return found.sort((a, b) => a.x - b.x)
}

function describe(list: Crosshair[]) {
  return list.map(c => `panel${c.panel}@(${c.u.toFixed(3)},${c.v.toFixed(3)}) n=${c.n}`).join('  ')
}

test('tiled navigators: one master crosshair per selector, copies follow it', async () => {
  test.setTimeout(240_000)
  const { app, page, backend, assertNoJsErrors } = await launchApp({
    dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' },
  })
  const shot = (name: string) => page.screenshot({ path: `${SHOTS}/${name}.png` })
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'load_test_data', {})
    await waitForSubwindowCount(page, 2, 60_000)
    await page.waitForTimeout(2000)
    const nav = navWindow(page)
    await expect(nav).toBeVisible()

    // A second navigator image, then ⇧-click both chips → tiled figure.
    await backendAction(page, 'test_add_second_navigator', {})
    const chips = nav.locator('button[data-testid^="nav-chip-"]')
    await expect(chips).toHaveCount(2, { timeout: 15_000 })
    await raiseWindow(nav)
    await chips.nth(0).click()
    await page.waitForTimeout(400)
    await chips.nth(1).click({ modifiers: ['Shift'] })
    await expect.poll(async () => (await crosshairs(nav, GREEN)).length,
      { timeout: 15_000, message: 'a green crosshair on each tiled panel' }).toBe(2)
    await page.waitForTimeout(500)
    await shot('01-tiled')

    // 1. Both panels agree on where the one selector is.
    let tiled = await crosshairs(nav, GREEN)
    console.log('tiled:', describe(tiled))
    expect(Math.abs(tiled[0].u - tiled[1].u)).toBeLessThan(0.02)
    expect(Math.abs(tiled[0].v - tiled[1].v)).toBeLessThan(0.02)
    const before = tiled

    // 2. Drag the RIGHT panel's crosshair with the mouse.
    const grab = tiled[1]
    await page.mouse.move(grab.x, grab.y)
    await page.mouse.down()
    for (let i = 1; i <= 6; i++) {
      await page.mouse.move(grab.x - 12 * i, grab.y + 8 * i, { steps: 3 })
      await page.waitForTimeout(120)
    }
    await shot('02-mid-drag')
    await page.mouse.up()
    await page.waitForTimeout(600)
    await shot('03-released')
    const released = await crosshairs(nav, GREEN)
    console.log('released:', describe(released))
    await page.waitForTimeout(2500)
    await shot('04-settled')
    const settled = await crosshairs(nav, GREEN)
    console.log('settled:', describe(settled))

    expect(settled.length).toBe(2)
    expect(Math.abs(settled[1].u - before[1].u), 'the dragged crosshair moved').toBeGreaterThan(0.05)
    for (const c of [0, 1]) {
      expect(Math.abs(settled[c].u - released[c].u), `panel ${c} snapped after release`).toBeLessThan(0.02)
      expect(Math.abs(settled[c].v - released[c].v), `panel ${c} snapped after release`).toBeLessThan(0.02)
    }
    expect(Math.abs(settled[0].u - settled[1].u), 'panels disagree').toBeLessThan(0.02)
    expect(Math.abs(settled[0].v - settled[1].v), 'panels disagree').toBeLessThan(0.02)

    // 3. Add a selector while tiled → its (blue) crosshair on both panels.
    await raiseWindow(nav)
    await nav.hover()
    const addSelector = nav.getByTestId('action-btn-Add Selector')
    await expect(addSelector).toBeVisible({ timeout: 10_000 })
    await addSelector.click()
    await expect.poll(async () => (await crosshairs(nav, BLUE)).length,
      { timeout: 20_000, message: 'a blue crosshair on each tiled panel' }).toBe(2)
    await page.waitForTimeout(800)
    await shot('05-second-selector')
    const blue = await crosshairs(nav, BLUE)
    const green = await crosshairs(nav, GREEN)
    console.log('after add — green:', describe(green), ' blue:', describe(blue))
    expect(green.length).toBe(2)
    expect(Math.abs(green[0].u - green[1].u)).toBeLessThan(0.02)
    expect(Math.abs(blue[0].u - blue[1].u)).toBeLessThan(0.02)

    assertNoJsErrors()
    const errors = backendErrorLines(backend)
    if (errors.length) console.log(errors.join('\n'))
    expect(errors.length).toBe(0)
  } finally {
    await app.close()
  }
})
