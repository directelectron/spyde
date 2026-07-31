/**
 * signal_type.spec.ts — the right-sidebar Signal-type selector (Qt parity).
 *
 * `signal_type_info` populates the dropdown with the current type + options;
 * changing it dispatches `set_signal_type` for the active window. Renderer-only.
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

test.beforeEach(async () => {
  await page.reload()
  await page.waitForSelector('[data-testid="mdi-area"]')
})

async function inject(msg: Record<string, unknown>) {
  await page.evaluate((m) => { (window as any)._spyde_test_inject?.(m) }, msg)
}

async function trackActions() {
  await app.evaluate(({ ipcMain }) => {
    ;(globalThis as any).__sent = []
    ipcMain.removeAllListeners('spyde:action')
    ipcMain.on('spyde:action', (_e, action, payload, windowId) => {
      ;(globalThis as any).__sent.push({ action, payload, windowId })
    })
  })
}
const sentActions = () => app.evaluate(() => (globalThis as any).__sent)

// A signal window must be active for the dock to show its controls.
async function openSignal() {
  await inject({ type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false })
}

test('signal_type_info populates the dropdown with the current type', async () => {
  await openSignal()
  await inject({
    type: 'signal_type_info', window_ids: [1],
    current: 'electron_diffraction',
    options: ['', 'electron_diffraction', 'EELS', 'EDS_TEM'],
  })
  const sel = page.getByTestId('signal-type-select')
  await expect(sel).toBeVisible()
  // Themed Dropdown: the trigger button carries the value as data-value.
  expect(await sel.getAttribute('data-value')).toBe('electron_diffraction')
  // The generic option renders with a friendly label (open the menu to see it).
  await sel.click()
  await expect(page.getByTestId('signal-type-select-opt-')).toHaveText('Generic (none)')
  await page.keyboard.press('Escape')
})

test('changing the type dispatches set_signal_type for the active window', async () => {
  await trackActions()
  await openSignal()
  await inject({
    type: 'signal_type_info', window_ids: [1],
    current: '', options: ['', 'electron_diffraction', 'EELS'],
  })
  await page.getByTestId('signal-type-select').click()
  await page.getByTestId('signal-type-select-opt-electron_diffraction').click()
  expect(await sentActions()).toContainEqual({
    action: 'set_signal_type',
    payload: { signal_type: 'electron_diffraction' },
    windowId: 1,
  })
})

test('signal-type shares a row with colormap, both above metadata', async () => {
  await openSignal()
  await inject({ type: 'signal_type_info', window_ids: [1], current: '', options: ['', 'EELS'] })
  await inject({ type: 'metadata', window_ids: [1], metadata: { Group: { K: 'v' } } })

  const box = (id: string) => page.getByTestId(id).boundingBox()
  const cmap = (await box('colormap-select'))!
  const sig = (await box('signal-type-select'))!
  const meta = (await box('metadata-panel'))!

  // Side by side (they are set once in a while and cost two stacked sections),
  // colormap on the left.
  expect(Math.round(cmap.y)).toBe(Math.round(sig.y))
  expect(cmap.x + cmap.width).toBeLessThanOrEqual(sig.x + 1)
  // …and the pair still sits above the metadata panel.
  expect(sig.y + sig.height).toBeLessThan(meta.y)
})
