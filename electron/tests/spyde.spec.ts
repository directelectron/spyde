/**
 * spyde.spec.ts — SpyDE Electron UI tests (self-contained).
 *
 * NOTE: kept as a single self-contained file on purpose. Node 23 + Playwright
 * 1.61 crash (`context.conditions?.includes is not a function`) on any
 * cross-file relative `.ts` import, so the electron launch + helpers live
 * inline here rather than in a shared fixtures module. See README.
 *
 * These assert on the REAL rendered DOM given the PLOTAPP messages Python
 * sends — the class of bug ("no sidebar", "no toolbars") that eyeballing
 * stdout misses. Messages are injected via window._spyde_test_inject so tests
 * are deterministic and don't need the Python backend (launched with
 * SPYDE_NO_DASK=1).
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
})

test.afterAll(async () => { await app?.close() })

// Fresh renderer state per test (re-registers _spyde_test_inject on mount).
test.beforeEach(async () => {
  await page.reload()
  await page.waitForSelector('[data-testid="mdi-area"]')
})

async function inject(msg: Record<string, unknown>) {
  await page.evaluate((m) => { (window as any)._spyde_test_inject?.(m) }, msg)
}

// Capture outgoing actions in the MAIN process. The renderer's
// window.electron is a contextBridge object (immutable — can't be wrapped),
// so we observe the ipcMain 'spyde:action' channel instead.
async function trackActions() {
  await app.evaluate(({ ipcMain }) => {
    ;(globalThis as any).__sent = []
    // Replace any prior listeners (incl. ones added by earlier tests) so the
    // capture array isn't double-counted. The real send-to-Python handler is
    // not needed for these renderer-wiring assertions.
    ipcMain.removeAllListeners('spyde:action')
    ipcMain.on('spyde:action', (_e, action, payload, windowId) => {
      ;(globalThis as any).__sent.push({ action, payload, windowId })
    })
  })
}
const sentActions = () => app.evaluate(() => (globalThis as any).__sent)

// The per-window floating toolbar lives BELOW the window and reveals on hover.
// Hover the owning window so its toolbar becomes visible + interactive before
// clicking a toolbar button. `match` selects the window when there are several.
async function reveal(match?: string) {
  const win = match
    ? page.getByTestId('subwindow').filter({ hasText: match })
    : page.getByTestId('subwindow').first()
  // Hover the title bar (on top, not the figure iframe which intercepts events).
  await win.getByTestId('subwindow-titlebar').hover()
}

// Reproduces a 4D STEM open: toolbar_config arrives BEFORE the signal figure
// (the real ordering from PlotState.__init__) — the bug that hid toolbars.
async function openFourD() {
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [
      { name: 'Reset', icon: '', side: 'left', toggle: false, parameters: {} },
      { name: 'Virtual Image', icon: '', side: 'left', toggle: false, parameters: {} },
      { name: 'FFT', icon: '', side: 'left', toggle: true, parameters: {} },
    ],
  })
  await inject({ type: 'figure', window_id: 0, fig_id: 'nav',
    html: '<html><body>navigator</body></html>', title: 'Navigator', is_navigator: true })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>signal</body></html>', title: 'Diffraction', is_navigator: false })
}

// ── Launch ────────────────────────────────────────────────────────────────────

test('app launches with MDI area and sidebar', async () => {
  await expect(page.getByTestId('mdi-area')).toBeVisible()
  await expect(page.getByTestId('plot-control-dock')).toBeVisible()
})

test('top bar is a window-drag region', async () => {
  const bar = page.getByTestId('app-bar')
  await expect(bar).toBeVisible()
  // The whole bar must be a drag region so a frameless window can be moved.
  const region = await bar.evaluate(
    (el) => getComputedStyle(el as HTMLElement).getPropertyValue('-webkit-app-region'))
  expect(region).toBe('drag')
})

test('top bar toggle hides and shows the sidebar', async () => {
  await expect(page.getByTestId('plot-control-dock')).toBeVisible()
  await page.getByTestId('toggle-sidebar').click()
  await expect(page.getByTestId('plot-control-dock')).toHaveCount(0)
  await page.getByTestId('toggle-sidebar').click()
  await expect(page.getByTestId('plot-control-dock')).toBeVisible()
})

// ── Windows ───────────────────────────────────────────────────────────────────

test('navigator + signal windows both render', async () => {
  await openFourD()
  await expect(page.getByTestId('subwindow')).toHaveCount(2)
})

test('focusing a figure iframe raises its window (click-to-front)', async () => {
  await openFourD()
  const navWin = page.getByTestId('subwindow').filter({ hasText: 'Navigator' })
  const sigWin = page.getByTestId('subwindow').filter({ hasText: 'Diffraction' })
  const zOf = (w: ReturnType<typeof page.getByTestId>) =>
    w.evaluate((el) => parseInt(getComputedStyle(el as HTMLElement).zIndex || '0', 10))

  // Raise the navigator so the signal window is below it.
  await navWin.getByTestId('subwindow-title').click()
  expect(await zOf(navWin)).toBeGreaterThan(await zOf(sigWin))

  // A pointerdown INSIDE the signal figure posts a `spyde_focus` message to the
  // parent (injected by Plot._ensure_figure) — which our handler uses to raise
  // that window. Simulate that message here.
  await page.evaluate(() => {
    window.postMessage({ type: 'spyde_focus', figId: 'sig' }, '*')
  })
  await expect.poll(async () => (await zOf(sigWin)) > (await zOf(navWin)),
    { timeout: 5_000 }).toBe(true)
})

test('a wide navigator is sized to its image aspect (fills, not letterboxed)', async () => {
  // A non-square navigator (e.g. sped_ag 208×64) reports aspect so the window is
  // sized wide → the image fills it and the crosshair/axes line up. Otherwise it
  // would be aspect-letterboxed into a thin strip in a square window.
  await inject({ type: 'figure', window_id: 7, fig_id: 'navw', is_navigator: true,
    aspect: 3.25, html: '<html><body>nav</body></html>', title: 'Navigator' })
  const win = page.getByTestId('subwindow').filter({ hasText: 'Navigator' })
  const bb = (await win.boundingBox())!
  expect(bb.width).toBeGreaterThan(bb.height * 1.8)   // clearly wide, not square
  await inject({ type: 'window_closed', window_id: 7 })
})

test('window close removes the subwindow', async () => {
  await inject({ type: 'figure', window_id: 5, fig_id: 'c',
    html: '<html><body>x</body></html>', title: 'Closeable', is_navigator: false })
  await expect(page.getByTestId('subwindow')).toHaveCount(1)
  await page.getByTestId('close-btn').first().click()
  // close button sends close_window to Python; the renderer removes it on
  // window_closed — simulate Python's reply.
  await inject({ type: 'window_closed', window_id: 5 })
  await expect(page.getByTestId('subwindow')).toHaveCount(0)
})

test('a subwindow can be dragged by its title bar', async () => {
  await inject({ type: 'figure', window_id: 9, fig_id: 'drag',
    html: '<html><body>drag me</body></html>', title: 'Draggable', is_navigator: false })
  const sub = page.getByTestId('subwindow').first()
  const before = await sub.boundingBox()
  const bar = page.getByTestId('subwindow-titlebar').first()
  const bb = await bar.boundingBox()
  // Drag the title bar 120px right, 90px down.
  await page.mouse.move(bb!.x + bb!.width / 2, bb!.y + bb!.height / 2)
  await page.mouse.down()
  await page.mouse.move(bb!.x + bb!.width / 2 + 120, bb!.y + bb!.height / 2 + 90, { steps: 8 })
  await page.mouse.up()
  const after = await sub.boundingBox()
  expect(Math.round(after!.x - before!.x)).toBeGreaterThan(60)
  expect(Math.round(after!.y - before!.y)).toBeGreaterThan(40)
})

test('a subwindow can be resized by its corner handle', async () => {
  await inject({ type: 'figure', window_id: 12, fig_id: 'rsz',
    html: '<html><body>resize me</body></html>', title: 'Resizable', is_navigator: false })
  const sub = page.getByTestId('subwindow').first()
  const before = await sub.boundingBox()
  const handle = page.getByTestId('resize-handle').first()
  const hb = await handle.boundingBox()
  await page.mouse.move(hb!.x + hb!.width / 2, hb!.y + hb!.height / 2)
  await page.mouse.down()
  await page.mouse.move(hb!.x + 140, hb!.y + 110, { steps: 8 })
  await page.mouse.up()
  const after = await sub.boundingBox()
  expect(Math.round(after!.width - before!.width)).toBeGreaterThan(90)
  expect(Math.round(after!.height - before!.height)).toBeGreaterThan(70)
})

test('MDI area has no background grid', async () => {
  const bgImage = await page.getByTestId('mdi-area').evaluate(
    (el) => getComputedStyle(el).backgroundImage,
  )
  expect(bgImage).toBe('none')
})

// ── Toolbars (the reported "no toolbars" bug) ─────────────────────────────────

test('floating toolbar renders even when toolbar_config precedes the figure', async () => {
  await openFourD()
  await expect(page.getByTestId('floating-toolbar').first()).toBeVisible()
  await expect(page.getByTestId('action-btn-Virtual Image').first()).toBeVisible()
  await expect(page.getByTestId('action-btn-FFT').first()).toBeVisible()
})

test('zoom/reset actions (+, −, full-scale) are hidden from the toolbar', async () => {
  // anyplotlib zooms natively, so these view-only actions are dropped even if
  // the backend still sends them in toolbar_config.
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [
      { name: 'Reset', icon: '', side: 'right', toggle: false, parameters: {}, subfunctions: [] },
      { name: 'Zoom In', icon: '', side: 'right', toggle: false, parameters: {}, subfunctions: [] },
      { name: 'Zoom Out', icon: '', side: 'right', toggle: false, parameters: {}, subfunctions: [] },
      { name: 'FFT', icon: '', side: 'right', toggle: false, parameters: {}, subfunctions: [] },
    ],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })
  await expect(page.getByTestId('action-btn-FFT')).toBeVisible()
  await expect(page.getByTestId('action-btn-Reset')).toHaveCount(0)
  await expect(page.getByTestId('action-btn-Zoom In')).toHaveCount(0)
  await expect(page.getByTestId('action-btn-Zoom Out')).toHaveCount(0)
})

test('hovering an action button highlights it', async () => {
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{ name: 'FFT', icon: '', side: 'right', toggle: false, parameters: {}, subfunctions: [] }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })
  const btn = page.getByTestId('action-btn-FFT')
  const bg = () => btn.evaluate(el => getComputedStyle(el).backgroundColor)
  const before = await bg()
  await reveal()             // toolbar is hover-revealed
  await btn.hover()
  await expect.poll(bg).not.toBe(before)   // :hover paints a highlight
})

test('the floating toolbar tracks the window when it moves', async () => {
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{ name: 'FFT', icon: '', side: 'right', toggle: false, parameters: {}, subfunctions: [] }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  const tb = page.getByTestId('floating-toolbar')
  const before = await tb.boundingBox()
  const bar = page.getByTestId('subwindow-titlebar')
  const bb = await bar.boundingBox()
  await page.mouse.move(bb!.x + bb!.width / 2, bb!.y + bb!.height / 2)
  await page.mouse.down()
  for (let s = 1; s <= 8; s++) {
    await page.mouse.move(bb!.x + bb!.width / 2 + 12 * s, bb!.y + bb!.height / 2 + 10 * s)
    await page.waitForTimeout(15)
  }
  await page.mouse.up()
  const after = await tb.boundingBox()
  // Toolbar is parented to the window, so it moved with it.
  expect(Math.round(after!.x - before!.x), 'toolbar did not track horizontally').toBeGreaterThan(70)
  expect(Math.round(after!.y - before!.y), 'toolbar did not track vertically').toBeGreaterThan(60)
})

test('the floating toolbar reveals on hover and hides when the cursor leaves', async () => {
  await openFourD()
  const tb = page.getByTestId('floating-toolbar').first()
  const opacity = () => tb.evaluate(el => getComputedStyle(el).opacity)
  // Park the cursor away from any window → toolbar hidden.
  await page.mouse.move(4, 120)
  await expect.poll(opacity, { timeout: 4000 }).toBe('0')
  // Hovering the owning window fades it in…
  await reveal('Diffraction')
  await expect.poll(opacity, { timeout: 4000 }).toBe('1')
  // …moving away fades it back out.
  await page.mouse.move(4, 120)
  await expect.poll(opacity, { timeout: 4000 }).toBe('0')
})

test('clicking a no-option action dispatches toolbar_action immediately', async () => {
  await trackActions()
  await openFourD()
  // Raise the signal window (its title) so its rail isn't covered by the
  // overlapping navigator window before clicking.
  await page.getByTestId('subwindow-title').filter({ hasText: 'Diffraction' }).click()
  await reveal('Diffraction')   // toolbar reveals on window hover
  // Virtual Image in openFourD has no parameters → runs immediately, no flyout.
  await page.getByTestId('action-btn-Virtual Image').last().click()
  expect(await sentActions()).toContainEqual({
    action: 'toolbar_action',
    payload: { name: 'Virtual Image', params: {} },
    windowId: 1,
  })
})

test('an action WITH parameters opens a flyout and Run dispatches the params', async () => {
  await trackActions()
  // Single window with a parametered Virtual Image action.
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Virtual Image', icon: '', side: 'left', toggle: false,
      parameters: { type: { name: 'Detector', type: 'enum', default: 'disk',
        options: ['disk', 'annular', 'rectangle'] } },
      subfunctions: [],
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>sig</body></html>', title: 'Diffraction', is_navigator: false })

  // Clicking the action opens its flyout (it has options).
  await reveal()
  await page.getByTestId('action-btn-Virtual Image').click()
  await expect(page.getByTestId('action-flyout')).toBeVisible()
  // Choose a non-default option, then Run.
  await page.getByTestId('param-type').selectOption('annular')
  await page.getByTestId('action-run').click()
  expect(await sentActions()).toContainEqual({
    action: 'toolbar_action',
    payload: { name: 'Virtual Image', params: { type: 'annular' } },
    windowId: 1,
  })
  // Flyout closes after Run.
  await expect(page.getByTestId('action-flyout')).toHaveCount(0)
})

test('the param popout opens BELOW the floating toolbar (does not cover plot)', async () => {
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Virtual Image', icon: '', side: 'left', toggle: false, subfunctions: [],
      parameters: { type: { name: 'Detector', type: 'enum', default: 'disk', options: ['disk', 'annular'] } },
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  await reveal()
  const btn = await page.getByTestId('action-btn-Virtual Image').boundingBox()
  await page.getByTestId('action-btn-Virtual Image').click()
  const pop = await page.getByTestId('action-flyout').boundingBox()
  // The popout sits BELOW the button (caret pointing up at it) → off the plot.
  expect(pop!.y).toBeGreaterThanOrEqual(btn!.y + btn!.height - 1)
})

test('the param caret stays open while dragging the window (and moves with it)', async () => {
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Virtual Image', icon: '', side: 'left', toggle: false, subfunctions: [],
      parameters: { type: { name: 'Detector', type: 'enum', default: 'disk', options: ['disk', 'annular'] } },
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  await reveal()
  await page.getByTestId('action-btn-Virtual Image').click()
  await expect(page.getByTestId('action-flyout')).toBeVisible()
  const before = (await page.getByTestId('action-flyout').boundingBox())!

  // Drag the window by its title bar — the title-bar mousedown must NOT close the caret.
  const bb = (await page.getByTestId('subwindow-titlebar').boundingBox())!
  await page.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2)
  await page.mouse.down()
  for (let s = 1; s <= 6; s++) {
    await page.mouse.move(bb.x + bb.width / 2 + 16 * s, bb.y + bb.height / 2 + 11 * s)
    await page.waitForTimeout(15)
  }
  await page.mouse.up()

  // Still open, and it moved with the window.
  await expect(page.getByTestId('action-flyout')).toBeVisible()
  const after = (await page.getByTestId('action-flyout').boundingBox())!
  expect(Math.round(after.x - before.x), 'caret did not move with the window').toBeGreaterThan(50)
})

test('a multi-param popout is WIDER than tall (params sit side-by-side)', async () => {
  // Find-Vectors-like: five params must lay out horizontally, not stack tall
  // (a tall popout used to run off the bottom of the MDI).
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Multi Param Action', icon: '', side: 'left', toggle: false, subfunctions: [],
      parameters: {
        sigma: { name: 'σ', type: 'float', default: 1, min: 0, max: 5, step: 0.1 },
        kernel_radius: { name: 'Radius', type: 'int', default: 5, min: 1, max: 30 },
        threshold: { name: 'Threshold', type: 'float', default: 0.5, min: 0, max: 1, step: 0.05 },
        min_distance: { name: 'Min Dist', type: 'int', default: 5, min: 1, max: 30 },
        subpixel: { name: 'Subpixel', type: 'bool', default: true },
      },
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  await reveal()
  await page.getByTestId('action-btn-Multi Param Action').click()
  const pop = (await page.getByTestId('action-flyout').boundingBox())!
  expect(pop.width).toBeGreaterThan(pop.height)        // wider, not taller
  expect(pop.height).toBeLessThan(160)                 // ≤ 2 rows — fits the MDI
  expect(pop.width).toBeLessThanOrEqual(472)           // capped so it's not super long
  // Numeric params with a range render as sliders; bool as a checkbox.
  expect(await page.getByTestId('param-sigma').getAttribute('type')).toBe('range')
  expect(await page.getByTestId('param-subpixel').getAttribute('type')).toBe('checkbox')
})

test('Find Diffraction Vectors opens the staged wizard (live preview + Compute)', async () => {
  await trackActions()
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Find Diffraction Vectors', icon: '', side: 'left', toggle: false,
      subfunctions: [],
      parameters: { sigma: { name: 'σ', type: 'float', default: 1, min: 0, max: 5, step: 0.1 } },
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  await reveal()
  await page.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()

  // Opening the wizard starts the live preview (dispatches fv_preview).
  await expect.poll(async () => (await sentActions()).map((s: any) => s.action))
    .toContain('fv_preview')

  // Nudge the threshold slider → debounced fv_tune (native value setter so
  // React's onChange fires).
  await page.getByTestId('fv-threshold').evaluate((el: HTMLInputElement) => {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value')!.set!
    setter.call(el, '0.3')
    el.dispatchEvent(new Event('input', { bubbles: true }))
  })
  // Compute → fv_run.
  await page.getByTestId('fv-compute').click()

  await expect.poll(async () => {
    const names = (await sentActions()).map((s: any) => s.action)
    return ['fv_preview', 'fv_tune', 'fv_run'].filter(a => names.includes(a))
  }).toEqual(['fv_preview', 'fv_tune', 'fv_run'])
})

test('Vector Orientation Mapping opens the staged wizard and drives Generate→Compute', async () => {
  await trackActions()
  await app.evaluate(({ ipcMain }) => {
    ipcMain.removeHandler('spyde:pick-file')
    ipcMain.handle('spyde:pick-file', async () => '/tmp/Ag.cif')
  })
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    // No params/subfunctions — the wizard opens by name (WIZARD_ACTIONS).
    toolbar_actions: [{
      name: 'Vector Orientation Mapping', icon: '', side: 'left', toggle: true,
      subfunctions: [], parameters: {},
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Vectors', is_navigator: false })

  await reveal()
  await page.getByTestId('action-btn-Vector Orientation Mapping').click()
  await expect(page.getByTestId('vector-orientation-wizard')).toBeVisible()
  await expect(page.getByTestId('vom-tab-Run')).toBeDisabled()

  await page.getByTestId('vom-pick-cif').click()
  await expect(page.getByTestId('vom-pick-cif')).toHaveText('Ag.cif')
  await page.getByTestId('vom-tab-Library').click()
  await page.getByTestId('vom-generate').click()
  await expect(page.getByTestId('vom-tab-Run')).toBeEnabled()
  await page.getByTestId('vom-tab-Run').click()
  await page.getByTestId('vom-compute').click()

  await expect.poll(async () => {
    const names = (await sentActions()).map((s: any) => s.action)
    return ['vom_generate_library', 'vom_run'].filter(a => names.includes(a))
  }).toEqual(['vom_generate_library', 'vom_run'])
})

test('Center Zero Beam opens the two-tab wizard and drives auto + manual', async () => {
  await trackActions()
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Center Zero Beam', icon: '', side: 'left', toggle: true,
      subfunctions: [], parameters: {},
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  await reveal()
  await page.getByTestId('action-btn-Center Zero Beam').click()
  await expect(page.getByTestId('center-zero-beam-wizard')).toBeVisible()

  // Automatic → Center dispatches czb_auto.
  await page.getByTestId('czb-center').click()
  // Manual tab → czb_manual_start (crosshair); Apply → czb_manual.
  await page.getByTestId('czb-tab-Manual').click()
  await page.getByTestId('czb-apply').click()

  await expect.poll(async () => {
    const names = (await sentActions()).map((s: any) => s.action)
    return ['czb_auto', 'czb_manual_start', 'czb_manual'].filter(a => names.includes(a))
  }).toEqual(['czb_auto', 'czb_manual_start', 'czb_manual'])
})

test('a caret with tabbed params shows only the active tab (Orientation-style)', async () => {
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Tabbed Action', icon: '', side: 'left', toggle: false, subfunctions: [],
      parameters: {
        cif_path: { name: 'Crystal', type: 'file', extensions: ['.cif'], default: '', tab: 'Library' },
        resolution: { name: 'Resolution', type: 'float', default: 1.0, tab: 'Library' },
        n_best: { name: 'N Best', type: 'int', default: 5, tab: 'Matching' },
        gamma: { name: 'Gamma', type: 'float', default: 0.5, tab: 'Matching' },
      },
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  await reveal()
  await page.getByTestId('action-btn-Tabbed Action').click()
  await expect(page.getByTestId('param-tab-Library')).toBeVisible()
  await expect(page.getByTestId('param-tab-Matching')).toBeVisible()
  // First tab (Library) active → its params show, Matching's don't.
  await expect(page.getByTestId('param-row-cif_path')).toBeVisible()
  await expect(page.getByTestId('param-row-n_best')).toHaveCount(0)
  // Switch tab → Matching params show, Library's don't.
  await page.getByTestId('param-tab-Matching').click()
  await expect(page.getByTestId('param-row-n_best')).toBeVisible()
  await expect(page.getByTestId('param-row-cif_path')).toHaveCount(0)
})

test('a file param renders a picker button', async () => {
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'File Param Action', icon: '', side: 'left', toggle: false, subfunctions: [],
      parameters: { cif_path: { name: 'Crystal (.cif)', type: 'file', extensions: ['.cif'], default: '' } },
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  await reveal()
  await page.getByTestId('action-btn-File Param Action').click()
  const picker = page.getByTestId('param-cif_path')
  await expect(picker).toBeVisible()
  expect(await picker.evaluate((el) => el.tagName)).toBe('BUTTON')
  await expect(picker).toHaveText('Choose…')
})

test('Orientation Mapping opens the staged wizard and drives the staged actions', async () => {
  await trackActions()
  // window.electron is immutable, so mock the native picker at the main process.
  await app.evaluate(({ ipcMain }) => {
    ipcMain.removeHandler('spyde:pick-file')
    ipcMain.handle('spyde:pick-file', async () => '/tmp/Ag.cif')
  })
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Orientation Mapping', icon: '', side: 'left', toggle: false,
      subfunctions: [], parameters: { gamma: { name: 'Gamma', type: 'float', default: 0.5 } },
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  await reveal()
  await page.getByTestId('action-btn-Orientation Mapping').click()
  await expect(page.getByTestId('orientation-wizard')).toBeVisible()
  // The 4 staged tabs; Refine/Run locked until the library is generated.
  await expect(page.getByTestId('om-tab-Load')).toBeVisible()
  await expect(page.getByTestId('om-tab-Refine')).toBeDisabled()

  // 1 Load → pick a .cif (mocked). Wait for the async picker to resolve (the
  // button label becomes the filename) before generating.
  await page.getByTestId('om-pick-cif').click()
  await expect(page.getByTestId('om-cif-list')).toContainText('Ag.cif')
  // 2 Library → Generate Library → dispatches om_generate_library + unlocks Refine.
  await page.getByTestId('om-tab-Library').click()
  await page.getByTestId('om-generate').click()
  await expect(page.getByTestId('om-tab-Refine')).toBeEnabled()
  // 3 Refine → nudging the gamma slider dispatches om_refine (debounced). Use the
  // native value setter so React's onChange fires.
  await page.getByTestId('om-tab-Refine').click()
  await page.getByTestId('om-gamma').evaluate((el: HTMLInputElement) => {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value')!.set!
    setter.call(el, '0.8')
    el.dispatchEvent(new Event('input', { bubbles: true }))
  })
  // 4 Run → Compute Map → dispatches om_run.
  await page.getByTestId('om-tab-Run').click()
  await page.getByTestId('om-compute').click()

  await expect.poll(async () => {
    const names = (await sentActions()).map((s: any) => s.action)
    return ['om_generate_library', 'om_refine', 'om_run'].filter(a => names.includes(a))
  }).toEqual(['om_generate_library', 'om_refine', 'om_run'])
})

test('CIF picker remembers recents and re-selects them across reopen', async () => {
  await page.evaluate(() => localStorage.removeItem('spyde:cif-recents'))
  await app.evaluate(({ ipcMain }) => {
    ipcMain.removeHandler('spyde:pick-file')
    ipcMain.handle('spyde:pick-file', async () => '/tmp/Quartz.cif')
  })
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Orientation Mapping', icon: '', side: 'left', toggle: false,
      subfunctions: [], parameters: { gamma: { name: 'Gamma', type: 'float', default: 0.5 } },
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  // Pick a .cif → added to the phase list AND remembered.
  await reveal()
  await page.getByTestId('action-btn-Orientation Mapping').click()
  await expect(page.getByTestId('orientation-wizard')).toBeVisible()
  await page.getByTestId('om-pick-cif').click()
  await expect(page.getByTestId('om-cif-list')).toContainText('Quartz.cif')

  // Close + reopen → the wizard persists its loaded phases, so Quartz is still in
  // the list (and correctly NOT offered as a "recent" chip while it's loaded).
  await page.getByTestId('om-close').click()
  await page.getByTestId('action-btn-Orientation Mapping').click()
  await expect(page.getByTestId('orientation-wizard')).toBeVisible()
  await expect(page.getByTestId('om-cif-list')).toContainText('Quartz.cif')

  // Remove it → now it's remembered but not loaded, so the quick-select "Recent"
  // chip appears; clicking it re-selects the crystal without a file dialog.
  await page.getByTestId('om-cif-remove-Quartz.cif').click()
  const chip = page.getByTestId('cif-recent-Quartz.cif')
  await expect(chip).toBeVisible()
  await chip.click()
  await expect(page.getByTestId('om-cif-list')).toContainText('Quartz.cif')
})

test('an active action highlights, and clicking it deselects to hide output', async () => {
  await trackActions()
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Virtual Image', icon: '', side: 'left', toggle: false, subfunctions: [],
      parameters: { type: { name: 'Detector', type: 'enum', default: 'disk', options: ['disk', 'annular'] } },
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  // Backend reports the action produced live output → the button goes active.
  await inject({ type: 'action_active', window_id: 1, name: 'Virtual Image', active: true })
  const btn = page.getByTestId('action-btn-Virtual Image')
  await expect.poll(() => btn.evaluate(el => getComputedStyle(el).backgroundColor))
    .toContain('137, 180, 250')   // accent #89b4fa

  // Clicking an active action deselects it → set_action_active(false) (not a re-run).
  await btn.click()
  expect(await sentActions()).toContainEqual({
    action: 'set_action_active',
    payload: { name: 'Virtual Image', active: false },
    windowId: 1,
  })
  // It did NOT open the param popout.
  await expect(page.getByTestId('action-flyout')).toHaveCount(0)
})

test('display_condition shows/hides a param row based on another param', async () => {
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Center', icon: '', side: 'left', toggle: false, subfunctions: [],
      parameters: {
        method: { name: 'Method', type: 'enum', default: 'blur', options: ['blur', 'cross_correlate'] },
        sigma: { name: 'Sigma', type: 'float', default: 2.0,
          display_condition: { parameter: 'method', value: 'blur' } },
        radius: { name: 'Radius', type: 'int', default: 5,
          display_condition: { parameter: 'method', value: 'cross_correlate' } },
      },
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  await reveal()
  await page.getByTestId('action-btn-Center').click()
  // Default method=blur → sigma visible, radius hidden.
  await expect(page.getByTestId('param-row-sigma')).toBeVisible()
  await expect(page.getByTestId('param-row-radius')).toHaveCount(0)
  // Switch method → cross_correlate → radius shows, sigma hides (live).
  await page.getByTestId('param-method').selectOption('cross_correlate')
  await expect(page.getByTestId('param-row-radius')).toBeVisible()
  await expect(page.getByTestId('param-row-sigma')).toHaveCount(0)
})

test('a subfunction action opens a SECOND toolbar whose buttons dispatch', async () => {
  await trackActions()
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Vector Virtual Imaging', icon: '', side: 'left', toggle: false, parameters: {},
      subfunctions: [{ name: 'add_vector_virtual_image', icon: '',
        label: 'Add Vector Virtual Image', toggle: false, parameters: {} }],
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Vectors', is_navigator: false })

  // Clicking the action pops up a SECOND floating toolbar (not a caret popout).
  await reveal()
  await page.getByTestId('action-btn-Vector Virtual Imaging').click()
  await expect(page.getByTestId('sub-toolbar')).toBeVisible()
  await page.getByTestId('subaction-add_vector_virtual_image').click()
  expect(await sentActions()).toContainEqual({
    action: 'toolbar_action',
    payload: { name: 'add_vector_virtual_image', params: {} },
    windowId: 1,
  })
})

test('Virtual Imaging sub-toolbar: + adds VI icons, each with its own param caret', async () => {
  await trackActions()
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Virtual Imaging', icon: '', side: 'left', toggle: false, parameters: {},
      subfunctions: [{
        name: 'add_virtual_image', icon: '', label: 'Add Virtual Image', toggle: false,
        parameters: {
          type: { name: 'Detector Type', type: 'enum', default: 'disk', options: ['annular', 'disk', 'rectangle'] },
          calculation: { name: 'Calculation', type: 'enum', default: 'mean', options: ['mean', 'sum'] },
        },
      }],
    }],
  })
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })

  // The sub-toolbar (below the main bar) has a "＋" add button.
  await reveal()
  await page.getByTestId('action-btn-Virtual Imaging').click()
  const subBar = page.getByTestId('sub-toolbar')
  await expect(subBar).toBeVisible()
  await expect(page.getByTestId('subaction-add_virtual_image')).toHaveText('＋')

  // Backend reports two VIs (different detector shapes/colours) → shape icons appear.
  await inject({ type: 'sub_item', window_id: 1, action: 'Virtual Imaging',
    name: 'Virtual Image 1 (red)', color: 'red', vtype: 'disk', calculation: 'mean', active: true })
  await inject({ type: 'sub_item', window_id: 1, action: 'Virtual Imaging',
    name: 'Virtual Image 2 (green)', color: 'green', vtype: 'annular', calculation: 'sum', active: true })
  await expect(page.getByTestId('vi-icon-Virtual Image 1 (red)')).toBeVisible()
  await expect(page.getByTestId('vi-icon-Virtual Image 2 (green)')).toBeVisible()
  // The sub-toolbar sits BELOW the main toolbar.
  const mainBar = page.getByTestId('floating-toolbar')
  expect((await subBar.boundingBox())!.y)
    .toBeGreaterThan((await mainBar.boundingBox())!.y)

  // Clicking a VI icon opens ITS own parameter caret; editing dispatches update_vi.
  await page.getByTestId('vi-icon-Virtual Image 1 (red)').click()
  await expect(page.getByTestId('vi-caret-Virtual Image 1 (red)')).toBeVisible()
  await page.getByTestId('param-vi-calculation').selectOption('sum')
  expect(await sentActions()).toContainEqual({
    action: 'update_vi',
    payload: { name: 'Virtual Image 1 (red)', params: { calculation: 'sum' } },
    windowId: 1,
  })

  // Remove (from the caret) dispatches set_action_active(false).
  await page.getByTestId('vi-remove-Virtual Image 1 (red)').click()
  expect(await sentActions()).toContainEqual({
    action: 'set_action_active',
    payload: { name: 'Virtual Image 1 (red)', active: false },
    windowId: 1,
  })
  await inject({ type: 'sub_item', window_id: 1, action: 'Virtual Imaging',
    name: 'Virtual Image 1 (red)', color: 'red', active: false })
  await expect(page.getByTestId('vi-icon-Virtual Image 1 (red)')).toHaveCount(0)
  await expect(page.getByTestId('vi-icon-Virtual Image 2 (green)')).toBeVisible()
})

// ── Sidebar (the reported "no sidebar" bug) ───────────────────────────────────

test('sidebar shows colormap + draggable histogram once a signal plot is active', async () => {
  await openFourD()
  await inject({ type: 'histogram', window_id: 1, counts: [1, 5, 3, 1], edges: [0, 25, 50, 75, 100], vmin: 10, vmax: 90 })
  await expect(page.getByTestId('colormap-select')).toBeVisible()
  await expect(page.getByTestId('histogram')).toBeVisible()
  // Draggable handles replace the old min/max inputs + Apply button.
  await expect(page.getByTestId('hist-min-handle')).toBeVisible()
  await expect(page.getByTestId('hist-max-handle')).toBeVisible()
})

test('changing colormap sends set_colormap for the active window', async () => {
  await trackActions()
  await openFourD()
  await page.getByTestId('colormap-select').selectOption('viridis')
  expect(await sentActions()).toContainEqual({
    action: 'set_colormap', payload: { name: 'viridis' }, windowId: 1,
  })
})

test('dragging the histogram max handle sends set_clim', async () => {
  await trackActions()
  await openFourD()
  await inject({ type: 'histogram', window_id: 1, counts: [1, 5, 3, 1], edges: [0, 25, 50, 75, 100], vmin: 10, vmax: 90 })
  const handle = page.getByTestId('hist-max-handle')
  const box = (await handle.boundingBox())!
  // Drag the max handle left (lowers vmax).
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.down()
  await page.mouse.move(box.x - 60, box.y + box.height / 2, { steps: 6 })
  await page.mouse.up()
  const sent = await sentActions()
  const climCalls = sent.filter((s: any) => s.action === 'set_clim' && s.windowId === 1)
  expect(climCalls.length, 'no set_clim sent on drag').toBeGreaterThan(0)
  // vmax must have decreased from the starting 90.
  const last = climCalls[climCalls.length - 1]
  expect((last.payload as any).vmax).toBeLessThan(90)
})

// ── Histogram + metadata (reported missing) ───────────────────────────────────

test('sidebar renders a histogram for the active window', async () => {
  await openFourD()
  await inject({
    type: 'histogram', window_id: 1,
    counts: [1, 5, 12, 8, 3, 1], edges: [0, 1, 2, 3, 4, 5, 6],
    vmin: 1, vmax: 5,
  })
  await expect(page.getByTestId('histogram')).toBeVisible()
})

test('display range shows as min/max labels on the histogram (no Scale section)', async () => {
  await openFourD()
  await inject({
    type: 'histogram', window_id: 1,
    counts: [1, 5, 12, 8, 3, 1], edges: [0, 1, 2, 3, 4, 5, 6], vmin: 1, vmax: 5,
  })
  await expect(page.getByTestId('clim-min')).toHaveText('1.00')
  await expect(page.getByTestId('clim-max')).toHaveText('5.00')
  // The standalone Scale section is gone.
  await expect(page.getByTestId('scale-section')).toHaveCount(0)
})

test('sidebar renders signal metadata', async () => {
  await openFourD()
  await inject({
    type: 'metadata', window_ids: [1],
    metadata: {
      'Instrument Metadata': { 'Acc. Volt.': '200 kV', 'Mag': '-- x' },
      'Root Experiment Details': { 'Name': 'TestSignal' },
    },
  })
  const panel = page.getByTestId('metadata-panel')
  await expect(panel).toBeVisible()
  await expect(panel).toContainText('Acc. Volt.')
  await expect(panel).toContainText('200 kV')
  await expect(panel).toContainText('TestSignal')
})

// ── Navigator selector toggle (reported: should show in dock, not show both) ──

test('navigator selector toggle appears in the dock and dispatches mode', async () => {
  await trackActions()
  await inject({ type: 'selector_info', window_id: 0, mode: 'crosshair', title: 'Navigator' })
  await expect(page.getByTestId('selector-control')).toBeVisible()
  await expect(page.getByTestId('selector-crosshair')).toBeVisible()
  await expect(page.getByTestId('selector-integrate')).toBeVisible()

  await page.getByTestId('selector-integrate').click()
  expect(await sentActions()).toContainEqual({
    action: 'set_selector_mode', payload: { integrate: true }, windowId: 0,
  })
})

// ── Signal-tree switcher (reported: no UI yet) ────────────────────────────────

test('signal-tree switcher renders nodes and switches on click', async () => {
  await trackActions()
  // A signal window must be active for the dock to show its tree.
  await inject({ type: 'figure', window_id: 1, fig_id: 'st',
    html: '<html><body>sig</body></html>', title: 'Signal', is_navigator: false })
  await inject({
    type: 'signal_tree', window_id: 1,
    tree: {
      name: 'root', signal_id: 100,
      children: [{ name: 'Binned', signal_id: 200, children: [] }],
    },
  })
  await expect(page.getByTestId('signal-tree')).toBeVisible()
  await expect(page.getByTestId('tree-node-root')).toBeVisible()
  await expect(page.getByTestId('tree-node-Binned')).toBeVisible()

  await page.getByTestId('tree-node-Binned').click()
  expect(await sentActions()).toContainEqual({
    action: 'select_signal_node', payload: { signal_id: 200 }, windowId: 1,
  })
})

// ── Dock section order (Histogram, Colormap, Signal type, Metadata, Scale, Selector) ──

test('dock sections are in the required order', async () => {
  await openFourD()
  // Give the active (signal) window metadata + histogram + a signal tree so all
  // sections are present.
  await inject({ type: 'histogram', window_id: 1, counts: [1, 2, 3], edges: [0, 1, 2, 3], vmin: 0, vmax: 3 })
  await inject({ type: 'metadata', window_ids: [1], metadata: { Group: { K: 'v' } } })
  await inject({
    type: 'signal_tree', window_id: 1,
    tree: { name: 'root', signal_id: 1, children: [] },
  })
  await inject({ type: 'selector_info', window_id: 0, mode: 'crosshair', title: 'Navigator' })
  await inject({
    type: 'axes_info', window_ids: [1],
    axes: [{ index: 0, name: 'x', size: 8, scale: 1, offset: 0, units: '', navigate: false }],
  })

  const order = await page.getByTestId('plot-control-dock').evaluate((dock) => {
    const ids = ['histogram-section', 'colormap-select', 'signal-tree',
      'metadata-panel', 'axes-section', 'selector-control']
    const positions = ids.map(id => {
      const el = dock.querySelector(`[data-testid="${id}"]`)
      return el ? (el as HTMLElement).getBoundingClientRect().top : Infinity
    })
    return positions
  })
  // Each present section must appear strictly below the previous one.
  for (let i = 1; i < order.length; i++) {
    if (order[i] === Infinity || order[i - 1] === Infinity) continue
    expect(order[i], `section ${i} out of order`).toBeGreaterThan(order[i - 1])
  }
})

// ── Status ────────────────────────────────────────────────────────────────────

test('status message updates the status bar', async () => {
  await inject({ type: 'status', text: 'Loaded: test.hspy' })
  await expect(page.getByTestId('status-text')).toHaveText('Loaded: test.hspy')
})
