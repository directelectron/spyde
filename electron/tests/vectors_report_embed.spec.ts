/**
 * vectors_report_embed.spec.ts — the embedded-vectors explorer in an exported
 * HTML report, driven in a REAL browser (plain chromium over file://; the page
 * must work with no app, no backend, no network — that's its whole point).
 *
 * The fixture (spyde/tests/gen_vectors_embed.py) plants cluster A at
 * k=(-0.5,0) only in the LEFT nav half and cluster B at k=(+0.5,0) only in the
 * RIGHT half — dragging the detector between the clusters must flip which half
 * of the virtual image lights up, computed entirely by the page's own script.
 */
import { test, expect, chromium, Browser, Page } from '@playwright/test'
import { execFileSync } from 'child_process'
import { join } from 'path'
import { tmpdir } from 'os'

let browser: Browser
let page: Page
const htmlPath = join(tmpdir(), 'spyde-vectors-embed-test.html')

test.beforeAll(async () => {
  // Cross-platform venv interpreter (Windows Scripts/ vs POSIX bin/), falling
  // back to whatever python is on PATH (CI layouts vary).
  const root = join(__dirname, '..', '..')
  const { existsSync } = require('fs')
  const candidates = [
    join(root, '.venv', 'Scripts', 'python.exe'),
    join(root, '.venv', 'bin', 'python'),
  ]
  const py = candidates.find((p) => existsSync(p))
    ?? (process.platform === 'win32' ? 'python' : 'python3')
  execFileSync(py, ['-m', 'spyde.tests.gen_vectors_embed', htmlPath],
    { cwd: root })
  browser = await chromium.launch()
  page = await browser.newPage()
  await page.goto('file:///' + htmlPath.replace(/\\/g, '/'))
  await page.waitForSelector('#vx-root[data-ready="1"]', { timeout: 20_000 })
})

test.afterAll(async () => { await browser?.close() })

/** Mean pixel value of the left / right half of the virtual-image canvas. */
async function viHalves(): Promise<{ left: number; right: number }> {
  return page.evaluate(() => {
    const c = document.getElementById('vx-vi') as HTMLCanvasElement
    const ctx = c.getContext('2d')!
    const img = ctx.getImageData(0, 0, c.width, c.height).data
    let left = 0, right = 0, nl = 0, nr = 0
    for (let y = 0; y < c.height; y++) {
      for (let x = 0; x < c.width; x++) {
        const v = img[4 * (y * c.width + x)]
        if (x < c.width / 2) { left += v; nl++ } else { right += v; nr++ }
      }
    }
    return { left: left / nl, right: right / nr }
  })
}

/** Click the k-space canvas at DATA coords (extent is [-1,1] in the fixture). */
async function clickK(kx: number, ky: number) {
  const box = (await page.locator('#vx-k').boundingBox())!
  await page.mouse.click(
    box.x + ((kx - -1) / 2) * (box.width - 1),
    box.y + ((ky - -1) / 2) * (box.height - 1),
  )
}

test('detector position selects which nav half lights up', async () => {
  // Detector on cluster A (k = -0.5, 0) → LEFT half bright, right dark.
  await clickK(-0.5, 0)
  await expect.poll(async () => (await viHalves()).left, { timeout: 5_000 })
    .toBeGreaterThan(50)
  let h = await viHalves()
  expect(h.left).toBeGreaterThan(10 * Math.max(1, h.right))
  await expect(page.locator('#vx-readout')).toContainText('of 768 vectors')
  await page.screenshot({ path: 'vectors_embed_shots/01-left-cluster.png' })

  // Detector on cluster B (k = +0.5, 0) → flips to the RIGHT half.
  await clickK(0.5, 0)
  await expect.poll(async () => (await viHalves()).right, { timeout: 5_000 })
    .toBeGreaterThan(50)
  h = await viHalves()
  expect(h.right).toBeGreaterThan(10 * Math.max(1, h.left))
  await page.screenshot({ path: 'vectors_embed_shots/02-right-cluster.png' })

  // Counts mode still computes (readout keeps reporting hits).
  await page.locator('input[name=vx-mode][value=count]').check()
  await expect(page.locator('#vx-readout')).toContainText('vectors')
})
