/**
 * crop_panel.spec.ts — Crop on 2-D, 3-D, 4-D and 5-D data, in the real app.
 *
 * The panel holds one range per axis and every plot drawing that axis carries
 * a box: the image / detector rectangle, the scan rectangle, the time span.
 * Per dataset this checks, by what is on screen:
 *   - the panel offers each kind of axis the dataset has;
 *   - typing a range puts a box on EVERY plot of that axis;
 *   - dragging a box with the mouse moves the panel's fields;
 *   - Crop opens a NEW dataset of the cut shape, with its own windows.
 *
 * Boxes are found by colour (#f9e2af) inside each window's figure iframe — the
 * only place a box exists is the pixels.
 */
import { test, expect, Page, Locator } from '@playwright/test'
import { mkdirSync } from 'fs'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'crop_panel_shots')
mkdirSync(SHOTS, { recursive: true })
test.setTimeout(180_000)

interface Case {
  name: string
  loader: string
  payload: Record<string, unknown>
  windows: number                    // windows the dataset opens
  groups: string[]                   // panel groups, in order
  ranges: Record<string, [number, number]>  // first / last kept, inclusive
  // The navigator box the mouse moves: which navigator window, which axis.
  drag?: { axis: string; navigator: number }
}

const CASES: Case[] = [
  { name: '2d', loader: 'load_test_data_image', payload: {}, windows: 1,
    groups: ['image'], ranges: { sig0: [64, 319], sig1: [32, 223] } },
  { name: '3d', loader: 'load_test_data_movie', payload: { size: 512, frames: 12 }, windows: 2,
    groups: ['time', 'image'], ranges: { nav0: [2, 8], sig0: [64, 447], sig1: [0, 255] },
    drag: { axis: 'nav0', navigator: 0 } },
  { name: '4d', loader: 'load_test_data_si_grains', payload: {}, windows: 2,
    groups: ['scan', 'detector'], ranges: { nav0: [1, 3], nav1: [1, 4], sig0: [16, 111], sig1: [16, 111] },
    drag: { axis: 'nav0', navigator: 0 } },
  { name: '5d', loader: 'load_test_data_5d', payload: { frames: 6, nav: 24, sig: 32 }, windows: 3,
    groups: ['time', 'scan', 'detector'],
    ranges: { nav2: [1, 4], nav0: [4, 15], nav1: [6, 17], sig0: [4, 27], sig1: [4, 27] },
    drag: { axis: 'nav0', navigator: 1 } },
]

/** The crop box's bounding rect inside *win*'s figure, in page CSS px. */
async function cropBox(win: Locator) {
  const element = await win.locator('iframe').first().elementHandle()
  const frame = await element?.contentFrame()
  const frameBox = await win.locator('iframe').first().boundingBox()
  if (!frame || !frameBox) return null
  const found = await frame.evaluate(() => {
    let best: { x0: number; y0: number; x1: number; y1: number; n: number } | null = null
    for (const canvas of Array.from(document.querySelectorAll('canvas'))) {
      const context = canvas.getContext('2d')
      if (!context || !canvas.width || !canvas.height) continue
      const data = context.getImageData(0, 0, canvas.width, canvas.height).data
      const rect = canvas.getBoundingClientRect()
      const sx = rect.width / canvas.width, sy = rect.height / canvas.height
      let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1, n = 0
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const p = (y * canvas.width + x) * 4
          // #f9e2af: red and green high, blue clearly lower — not grey/white.
          if (data[p] > 220 && data[p + 1] > 190 && data[p + 2] > 130 && data[p + 2] < 205
              && data[p] - data[p + 2] > 40) {
            n++; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y)
          }
        }
      }
      if (n > 20 && (!best || n > best.n)) {
        best = { x0: rect.left + x0 * sx, y0: rect.top + y0 * sy,
                 x1: rect.left + (x1 + 1) * sx, y1: rect.top + (y1 + 1) * sy, n }
      }
    }
    return best
  })
  if (!found) return null
  return { x0: frameBox.x + found.x0, y0: frameBox.y + found.y0,
           x1: frameBox.x + found.x1, y1: frameBox.y + found.y1 }
}

const windowsOf = (page: Page, prefix: string) => page.getByTestId('subwindow')
  .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: new RegExp(`^${prefix}`) }) })

async function setRange(page: Page, key: string, [first, last]: [number, number]) {
  const to = page.getByTestId(`crop-${key}-to`)
  const from = page.getByTestId(`crop-${key}-from`)
  await to.fill(String(last)); await to.blur()
  await from.fill(String(first)); await from.blur()
  await expect(from).toHaveValue(String(first))
  await expect(to).toHaveValue(String(last))
}

for (const c of CASES) {
  test(`crop ${c.name}`, async () => {
    const ctx = await launchApp({})
    const { page } = ctx
    const shot = (step: string) => page.screenshot({ path: join(SHOTS, `${c.name}-${step}.png`) })
    try {
      await backendAction(page, c.loader, c.payload)
      await waitForSubwindowCount(page, c.windows, 120_000)
      await page.getByTestId('tile-windows').click()
      await page.waitForTimeout(1500)

      const source = sigWindow(page)
      await source.getByTestId('subwindow-titlebar').hover()
      await source.getByTestId('action-btn-Crop').click()
      const panel = page.getByTestId('crop-wizard')
      await expect(panel).toBeVisible()
      for (const group of c.groups) {
        await expect(panel.getByTestId(`crop-group-${group}`)).toBeVisible()
      }
      expect(await panel.locator('[data-testid^="crop-group-"]').count()).toBe(c.groups.length)
      await shot('01-open')

      for (const [key, range] of Object.entries(c.ranges)) await setRange(page, key, range)
      await page.waitForTimeout(800)
      // A box on every window of the dataset: the signal window and each navigator.
      const windows = page.getByTestId('subwindow')
      for (let i = 0; i < c.windows; i++) {
        await expect.poll(() => cropBox(windows.nth(i)).then(Boolean), {
          timeout: 10_000, message: `no crop box on window ${i}`,
        }).toBe(true)
      }
      await shot('02-ranges')

      if (c.drag) {
        const from = page.getByTestId(`crop-${c.drag.axis}-from`)
        const before = await from.inputValue()
        const navigator = windowsOf(page, 'N-').nth(c.drag.navigator)
        const box = await cropBox(navigator)
        expect(box, 'no crop box on the navigator to drag').not.toBeNull()
        const width = box!.x1 - box!.x0, height = box!.y1 - box!.y0
        // Not the middle: a scan navigator's crosshair sits there, and
        // grabbing it moves the crosshair instead of the box.
        const x = box!.x0 + width * 0.3, y = box!.y0 + height * 0.7
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.mouse.move(x - width / 2, y, { steps: 12 })
        await page.mouse.up()
        await expect(from).not.toHaveValue(before)
        await shot('03-dragged')
        // Dragging raised the navigator, which may now cover the panel; the
        // panel may in turn cover its own title bar, so raise by the event.
        await source.dispatchEvent('mousedown')
      }

      const summary = await page.getByTestId('crop-summary').innerText()
      const promised = summary.split('→')[1].trim()
      await page.getByTestId('crop-run').click()
      await waitForSubwindowCount(page, c.windows * 2, 120_000)
      await page.getByTestId('tile-windows').click()
      await page.waitForTimeout(2000)
      const result = page.getByTestId('subwindow').filter({ hasText: '(cropped)' }).first()
      await expect(result).toBeVisible()
      await result.getByTestId('subwindow-titlebar').click({ position: { x: 8, y: 8 } })
      await expect(page.getByTestId('dataset-strip')).toContainText(promised)
      // The source is untouched.
      await expect(windowsOf(page, 'S-').filter({ hasNotText: '(cropped)' })).toHaveCount(1)
      await shot('04-result')
      ctx.assertNoJsErrors()
    } finally {
      await ctx.app.close()
    }
  })
}
