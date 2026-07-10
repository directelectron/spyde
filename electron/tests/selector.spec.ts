/**
 * selector.spec.ts — E2E: dragging the navigator selector updates the signal plot.
 *
 * Drives the REAL Python backend (synthetic 4D data, no Dask/download via
 * SPYDE_NO_DASK + the load_test_data action), then drags the crosshair on the
 * navigator and asserts the diffraction-pattern canvas actually changes.
 *
 * This exercises the full live loop: iframe drag → awi_event → IPC →
 * figure_event → anyplotlib dispatch_event → selector callback → 4D slice →
 * signal-plot push → state_update → signal iframe repaint. It is the test that
 * catches the fig_id-mismatch bug (events from the iframe must route back to the
 * right figure).
 *
 * Self-contained (Node 23 + Playwright 1.61 break on cross-file .ts imports).
 */
import { test, expect, _electron as electron, ElectronApplication, Page, Frame } from '@playwright/test'
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
  // Real backend load of synthetic 4D data.
  await page.evaluate(() => window.electron.action('load_test_data', {}))
  // Wait for navigator + signal windows.
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 30_000 },
  )
  await page.waitForTimeout(1500) // let figures render + initial DP push
})

test.afterAll(async () => { await app?.close() })

// Resolve the iframe Frame inside the subwindow whose title (does not) match.
async function frameForWindow(isNav: boolean): Promise<{ frame: Frame; box: { x: number; y: number; width: number; height: number } }> {
  const subs = page.getByTestId('subwindow')
  const n = await subs.count()
  for (let i = 0; i < n; i++) {
    const title = (await subs.nth(i).getByTestId('subwindow-title').textContent()) || ''
    const matches = /^N-/.test(title)
    if (matches === isNav) {
      const handle = await subs.nth(i).locator('iframe').elementHandle()
      const frame = await handle!.contentFrame()
      const box = await subs.nth(i).locator('iframe').boundingBox()
      return { frame: frame!, box: box! }
    }
  }
  throw new Error(`no ${isNav ? 'navigator' : 'signal'} subwindow`)
}

// Weighted centroid of bright pixels on the LARGEST canvas (the image, not the
// axis/overlay canvases). The centroid moves when the diffraction pattern's
// bright spot moves, so it's a robust "did the DP change" signature.
async function dpSignature(frame: Frame): Promise<{ cx: number; cy: number; total: number }> {
  for (let i = 0; i < 40; i++) {
    const r = await frame.evaluate(() => {
      const canvases = Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]
      if (!canvases.length) return null
      // Largest by area = the image canvas.
      const c = canvases.sort((a, b) => b.width * b.height - a.width * a.height)[0]
      const ctx = c.getContext('2d')!
      const { data, width, height } = ctx.getImageData(0, 0, c.width, c.height)
      let sx = 0, sy = 0, tot = 0
      for (let p = 0; p < data.length; p += 4) {
        const v = data[p] + data[p + 1] + data[p + 2]
        if (v > 120) {
          const idx = p / 4
          sx += (idx % width) * v
          sy += Math.floor(idx / width) * v
          tot += v
        }
      }
      return tot > 0 ? { cx: Math.round(sx / tot), cy: Math.round(sy / tot), total: tot } : null
    })
    if (r) return r
    await page.waitForTimeout(100)
  }
  return { cx: 0, cy: 0, total: 0 }
}

// fig_id of the navigator / signal iframe (from its data-testid).
async function figIdForWindow(isNav: boolean): Promise<string> {
  const subs = page.getByTestId('subwindow')
  const n = await subs.count()
  for (let i = 0; i < n; i++) {
    const title = (await subs.nth(i).getByTestId('subwindow-title').textContent()) || ''
    if (/^N-/.test(title) === isNav) {
      const tid = await subs.nth(i).locator('iframe').getAttribute('data-testid')
      return (tid || '').replace('figure-', '')
    }
  }
  throw new Error(`no ${isNav ? 'navigator' : 'signal'} iframe`)
}
const navFigId = () => figIdForWindow(true)

test('moving the navigator selector updates the diffraction pattern', async () => {
  const sig = await frameForWindow(false)
  // Visual: the signal renders a non-black diffraction pattern to begin with.
  const before = await dpSignature(sig.frame)
  expect(before.total).toBeGreaterThan(0)

  const figId = await navFigId()
  // Find the crosshair selector that lives on the navigator panel.
  const widgets = await page.evaluate(
    (fid) => (window as any)._spyde_test_widgets(fid),
    figId,
  )
  const crosshair = widgets.find((w: any) => w.type === 'crosshair')
  expect(crosshair, `no crosshair widget on nav; got ${JSON.stringify(widgets.map((w:any)=>w.type))}`).toBeTruthy()

  // Post the exact awi_event the crosshair posts when dragged to a new nav
  // cell — this drives the real spyde chain (IPC → dispatch_event → selector
  // → 4D slice → signal push) and verifies events route back by fig_id.
  const moveTo = (cx: number, cy: number) => page.evaluate(
    ({ fid, wid, panel, x, y }) => {
      const ev = {
        source: 'js', panel_id: panel, widget_id: wid,
        event_type: 'pointer_move', cx: x, cy: y,
      }
      window.postMessage({ type: 'awi_event', figId: fid, data: JSON.stringify(ev) }, '*')
    },
    { fid: figId, wid: crosshair.id, panel: crosshair.panel_id, x: cx, y: cy },
  )

  // Image-data signature of the SIGNAL figure before the move (verifies the
  // data path, independent of canvas rendering).
  const sigFig = await figIdForWindow(false)
  const sigImg = (fid: string) => page.evaluate((f) => (window as any)._spyde_test_image_sig(f), fid)
  const imgBefore = await sigImg(sigFig)

  await moveTo(7, 7)   // far corner of the 8x8 navigation grid

  // The signal's image data must change — proves the full live chain:
  // selector move → awi_event → IPC → dispatch_event → 4D slice → signal push.
  let imgAfter = imgBefore
  for (let i = 0; i < 20 && imgAfter === imgBefore; i++) {
    await page.waitForTimeout(250)
    imgAfter = await sigImg(sigFig)
  }
  expect(imgAfter, 'signal image data did not change after selector move').not.toEqual(imgBefore)

  // And the signal still renders a non-black DP at the new position.
  const after = await dpSignature(sig.frame)
  expect(after.total).toBeGreaterThan(0)
})

test('a pointer_move (not just release) updates the signal live', async () => {
  // The selector must respond on pointer_move so the DP tracks the drag — not
  // only on pointer_up.
  const figId = await navFigId()
  const widgets = await page.evaluate((fid) => (window as any)._spyde_test_widgets(fid), figId)
  const crosshair = widgets.find((w: any) => w.type === 'crosshair')
  const sigFig = await figIdForWindow(false)
  const sigImg = (fid: string) => page.evaluate((f) => (window as any)._spyde_test_image_sig(f), fid)

  const post = (type: string, cx: number, cy: number) => page.evaluate(
    ({ fid, wid, panel, t, x, y }) => window.postMessage({
      type: 'awi_event', figId: fid,
      data: JSON.stringify({ source: 'js', panel_id: panel, widget_id: wid, event_type: t, cx: x, cy: y }),
    }, '*'),
    { fid: figId, wid: crosshair.id, panel: crosshair.panel_id, t: type, x: cx, y: cy },
  )

  await post('pointer_move', 1, 1)
  await page.waitForTimeout(400)
  const a = await sigImg(sigFig)
  // A pointer_move (no pointer_up) to a different cell must already change it.
  await post('pointer_move', 6, 6)
  let b = a
  for (let i = 0; i < 20 && b === a; i++) { await page.waitForTimeout(200); b = await sigImg(sigFig) }
  expect(b, 'pointer_move did not update the signal live').not.toEqual(a)
})

// Locate a real subwindow (navigator) by title.
function navSubwindow() {
  return page.getByTestId('subwindow').filter({ has: page.getByText('Navigator', { exact: false }) }).first()
}

test('a REAL figure window drags by its title bar (drag crosses the iframe)', async () => {
  const sub = navSubwindow()
  await expect(sub).toBeVisible()
  const before = await sub.boundingBox()
  const bar = sub.getByTestId('subwindow-titlebar')
  const bb = await bar.boundingBox()

  // Drag the title bar DOWN-RIGHT — the path crosses over the anyplotlib iframe,
  // whose canvas does setPointerCapture and would otherwise steal the drag.
  await page.mouse.move(bb!.x + bb!.width / 2, bb!.y + bb!.height / 2)
  await page.mouse.down()
  for (let s = 1; s <= 10; s++) {
    await page.mouse.move(
      bb!.x + bb!.width / 2 + 14 * s,
      bb!.y + bb!.height / 2 + 12 * s,
    )
    await page.waitForTimeout(15)
  }
  await page.mouse.up()

  const after = await sub.boundingBox()
  expect(Math.round(after!.x - before!.x), 'window did not move horizontally').toBeGreaterThan(80)
  expect(Math.round(after!.y - before!.y), 'window did not move vertically').toBeGreaterThan(70)
})

test('toggling Point→Integrate swaps the widget shown on the navigator image', async () => {
  const navFig = await navFigId()
  const widgets = (fid: string) => page.evaluate((f) => (window as any)._spyde_test_widgets(f), fid)

  const vis = (ws: any[], type: string) => {
    const w = ws.find((x: any) => x.type === type)
    // visible defaults to true when the key is absent.
    return w ? (w.data.visible !== false) : undefined
  }

  const before = await widgets(navFig)
  expect(vis(before, 'crosshair'), 'crosshair should start visible').toBe(true)
  expect(vis(before, 'rectangle'), 'rectangle should start hidden').toBe(false)

  // Toggle to Integrate via the dock.
  await page.getByTestId('selector-integrate').click()

  // The overlay must swap: crosshair hidden, rectangle visible.
  let after = before
  for (let i = 0; i < 25; i++) {
    after = await widgets(navFig)
    if (vis(after, 'crosshair') === false && vis(after, 'rectangle') === true) break
    await page.waitForTimeout(150)
  }
  expect(vis(after, 'crosshair'), 'crosshair should be hidden after Integrate').toBe(false)
  expect(vis(after, 'rectangle'), 'rectangle should be visible after Integrate').toBe(true)

  // Toggle back.
  await page.getByTestId('selector-crosshair').click()
  let back = after
  for (let i = 0; i < 25; i++) {
    back = await widgets(navFig)
    if (vis(back, 'crosshair') === true && vis(back, 'rectangle') === false) break
    await page.waitForTimeout(150)
  }
  expect(vis(back, 'crosshair')).toBe(true)
  expect(vis(back, 'rectangle')).toBe(false)
})

test('drag shield covers the iframe on mousedown (prevents native pointer capture)', async () => {
  // The real-world freeze is caused by the out-of-process figure iframe
  // capturing the native pointer mid-drag. Playwright's synthetic mouse can't
  // reproduce that capture, so we instead assert the FIX invariant: the moment
  // the title bar is pressed — BEFORE any movement — a shield is raised that
  // fully covers the iframe, so no native pointer event can reach it.
  const sub = navSubwindow()
  const bar = sub.getByTestId('subwindow-titlebar')
  const bb = await bar.boundingBox()
  const iframeBox = await sub.locator('iframe').first().boundingBox()

  await page.mouse.move(bb!.x + bb!.width / 2, bb!.y + bb!.height / 2)
  await page.mouse.down()
  try {
    const shield = sub.getByTestId('drag-shield')
    await expect(shield, 'shield not raised on mousedown').toBeVisible()
    const sBox = (await shield.boundingBox())!
    // Shield must cover the whole iframe rect (within 1px).
    expect(sBox.x).toBeLessThanOrEqual(iframeBox!.x + 1)
    expect(sBox.y).toBeLessThanOrEqual(iframeBox!.y + 1)
    expect(sBox.x + sBox.width).toBeGreaterThanOrEqual(iframeBox!.x + iframeBox!.width - 1)
    expect(sBox.y + sBox.height).toBeGreaterThanOrEqual(iframeBox!.y + iframeBox!.height - 1)
  } finally {
    await page.mouse.up()
  }
})

test('axes table renders and editing scale writes back (real backend)', async () => {
  // Activate a window so the dock shows its axes.
  await page.getByTestId('subwindow-titlebar').first().click()
  await expect(page.getByTestId('axes-table')).toBeVisible({ timeout: 10_000 })
  // Synthetic 4D STEM → 2 nav + 2 signal = 4 axis rows.
  await expect(page.locator('[data-testid^="axis-row-"]')).toHaveCount(4)

  // Edit axis 0 scale: click the cell text → it becomes an input. After commit
  // the backend writes it to the real axes_manager and re-emits axes_info, so the
  // cell re-renders the round-tripped value. The display formats the scale to 2dp
  // (PlotControlDock fmt = n.toFixed(2)), so "2.50" → "2.50". That round trip to
  // Python and back proves the write-back.
  await page.getByTestId('axis-0-scale').click()
  const scale0 = page.getByTestId('axis-0-scale-input')
  await scale0.fill('2.50')
  await scale0.blur()
  await expect(page.getByTestId('axis-0-scale')).toHaveText('2.50', { timeout: 10_000 })
})

test('Virtual Imaging: Add (real backend) creates a VI + lists a colored chip', async () => {
  // Full chain on the REAL Python backend: open the Virtual Imaging sub-toolbar
  // on the signal window, click "Add Virtual Image" → a new VI output window opens
  // AND a colour-coded chip appears in the sub-toolbar (the wired-up multi-VI).
  const sig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) }).first()
  await sig.getByTestId('subwindow-title').click()   // raise it
  const before = await page.getByTestId('subwindow').count()

  await sig.getByTestId('subwindow-titlebar').hover()   // toolbar reveals on hover
  await sig.getByTestId('action-btn-Virtual Imaging').click()
  await expect(page.getByTestId('sub-toolbar')).toBeVisible()
  await page.getByTestId('subaction-add_virtual_image').click()

  // A new output window opened…
  await expect.poll(() => page.getByTestId('subwindow').count(),
    { timeout: 20_000 }).toBe(before + 1)
  // …and a (red, first in the cycle) detector-shape icon is listed in the sub-toolbar.
  await expect(page.getByTestId('vi-icon-Virtual Image 1 (red)')).toBeVisible({ timeout: 20_000 })

  // The VI output must actually DISPLAY (not the black 10x10 placeholder): read
  // the new window's largest canvas and require non-black pixels.
  const viWin = page.getByTestId('subwindow').last()
  const maxPix = async () => {
    const h = await viWin.locator('iframe').first().elementHandle()
    const frame = h ? await h.contentFrame() : null
    if (!frame) return 0
    return frame.evaluate(() => {
      const cs = Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]
      if (!cs.length) return 0
      const c = cs.sort((a, b) => b.width * b.height - a.width * a.height)[0]
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
      let mx = 0
      for (let p = 0; p < d.length; p += 4) mx = Math.max(mx, d[p] + d[p + 1] + d[p + 2])
      return mx
    })
  }
  await expect.poll(maxPix, { timeout: 20_000 }).toBeGreaterThan(20)
})

// NOTE: keep this LAST — it tears down the shared backend load (the navigator's
// X closes the whole tree).
test('navigator close button (real backend) removes EVERY window of the tree', async () => {
  // Regression: the backend used to emit an unhandled `windows_closed`/`tree_id`
  // message, so clicking close did nothing in the real app. Closing the
  // NAVIGATOR closes the whole tree (nav + signal + any virtual-image windows
  // the previous test added).
  expect(await page.getByTestId('subwindow').count()).toBeGreaterThanOrEqual(2)
  // The Navigator Selector toggle is present while the tree is open.
  await expect(page.getByTestId('selector-control')).toBeVisible()
  // Raise the navigator first — an overlapping sibling can cover its controls
  // (windows share z-levels; there is no hover-raise).
  await navSubwindow().getByTestId('subwindow-titlebar').click()
  await navSubwindow().getByTestId('close-btn').click()
  await expect(page.getByTestId('subwindow')).toHaveCount(0, { timeout: 15_000 })
  // …and it must disappear when the tree closes (per-window state is dropped).
  await expect(page.getByTestId('selector-control')).toHaveCount(0)
})
