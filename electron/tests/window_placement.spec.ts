/**
 * window_placement.spec.ts — new windows are packed into free space (first-fit)
 * instead of cascading on top of each other, so result windows (IPF / strain /
 * refine / vectors) don't bury one another.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_NO_DASK: '1' },
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  // A realistically-sized window so several result windows can actually tile
  // (the default test window is small).
  await app.evaluate(({ BrowserWindow }) => {
    BrowserWindow.getAllWindows()[0]?.setBounds({ x: 0, y: 0, width: 1900, height: 1180 })
  })
  await page.waitForTimeout(300)
})
test.afterAll(async () => { await app?.close() })

test.beforeEach(async () => {
  await page.reload()
  await page.waitForSelector('[data-testid="mdi-area"]')
})

async function inject(msg: Record<string, unknown>) {
  await page.evaluate((m) => { (window as Window & { _spyde_test_inject?: (m: unknown) => void })._spyde_test_inject?.(m) }, msg)
}

const bg = ['#1f4d7a', '#7a1f4d', '#4d7a1f', '#7a5a1f', '#1f7a6a', '#5a1f7a']

test('new windows pack into free space instead of stacking', async () => {
  // Inject three result windows; the packer puts them side by side rather than
  // cascading on top of one another (the old behaviour stacked them at +40px).
  for (let i = 1; i <= 3; i++) {
    await inject({
      type: 'figure', window_id: i, fig_id: `w${i}`,
      html: `<!doctype html><html><body style="margin:0;height:100vh;background:${bg[i % bg.length]}"></body></html>`,
      title: `Window ${i}`, is_navigator: false,
    })
  }
  await expect(page.getByTestId('subwindow')).toHaveCount(3)
  await page.waitForTimeout(400)

  const boxes = await page.getByTestId('subwindow').evaluateAll((els) =>
    els.map((e) => e.getBoundingClientRect()).map(r => ({ x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) })))

  // No two windows overlap (they tiled into free space).
  const SLOP = 6
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j]
      const overlap = a.x < b.x + b.w - SLOP && a.x + a.w - SLOP > b.x &&
                      a.y < b.y + b.h - SLOP && a.y + a.h - SLOP > b.y
      expect(overlap, `windows ${i} and ${j} overlap`).toBeFalsy()
    }
  }
  await page.screenshot({ path: join(__dirname, '..', 'window_placement.png') })
})
