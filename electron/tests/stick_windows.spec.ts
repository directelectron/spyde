/**
 * stick_windows.spec.ts — edge-snap + "stick" window grouping.
 *   • dragging a window so an edge nears another's snaps it into alignment,
 *   • staying edge-aligned for ~1s forms a stick group (link badge appears),
 *   • dragging one stuck window moves the whole group together.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

async function inject(msg: Record<string, unknown>) {
  await page.evaluate((m) => { (window as any)._spyde_test_inject?.(m) }, msg)
}

// Drag an element's titlebar by (dx, dy) using real mouse events.
async function dragBy(tb: ReturnType<Page['getByTestId']>, dx: number, dy: number) {
  const box = (await tb.boundingBox())!
  const x = box.x + 40, y = box.y + box.height / 2
  await page.mouse.move(x, y)
  await page.mouse.down()
  await page.mouse.move(x + dx, y + dy, { steps: 15 })
  await page.mouse.up()
}

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_NO_DASK: '1' },
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  await page.waitForTimeout(1200)
  // Two independent windows.
  await inject({ type: 'figure', window_id: 1, fig_id: 'a',
    html: '<html><body>A</body></html>', title: 'Win A', is_navigator: false })
  await inject({ type: 'figure', window_id: 2, fig_id: 'b',
    html: '<html><body>B</body></html>', title: 'Win B', is_navigator: false })
  await expect(page.getByTestId('subwindow')).toHaveCount(2)
  await page.waitForTimeout(300)
})

test.afterAll(async () => { await app?.close() })

test('windows edge-snap, stick after a dwell, and move as a group', async () => {
  const a = page.getByTestId('subwindow').nth(0)
  const b = page.getByTestId('subwindow').nth(1)
  const aBox = (await a.boundingBox())!
  const bBox = (await b.boundingBox())!
  const aRight = aBox.x + aBox.width

  // Drag B so its LEFT edge approaches A's RIGHT edge → it should snap to align.
  const want = aRight + 4              // a hair past A's right edge → inside snap zone
  await dragBy(b.getByTestId('subwindow-titlebar'), want - bBox.x, 0)
  const bSnap = (await b.boundingBox())!
  expect(Math.abs(bSnap.x - aRight)).toBeLessThanOrEqual(9)   // snapped to the shared edge

  // Stay aligned for the dwell → the stick group forms (link badge appears).
  await expect(page.getByTestId('stuck-badge').first()).toBeVisible({ timeout: 4_000 })

  // Drag A right — B (stuck to it) moves by the same delta.
  const aBefore = (await a.boundingBox())!
  const bBefore = (await b.boundingBox())!
  await dragBy(a.getByTestId('subwindow-titlebar'), 90, 30)
  const aAfter = (await a.boundingBox())!
  const bAfter = (await b.boundingBox())!
  expect(aAfter.x - aBefore.x).toBeGreaterThan(40)   // A moved
  expect(bAfter.x - bBefore.x).toBeGreaterThan(40)   // B moved WITH it
  expect(bAfter.y - bBefore.y).toBeGreaterThan(15)   // …in both axes
})
