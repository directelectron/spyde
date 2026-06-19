/**
 * composition.spec.ts — sample composition (periodic-table picker → HyperSpy
 * metadata) + the COD "easy CIF" structure picker.
 *
 * Drives the UI via the test-inject hook and observes outgoing actions on the
 * ipcMain channel (set_composition / cod_search / cod_pick) — the backend
 * round-trip is unit-tested separately in test_composition.py.
 *
 * Self-contained (Node 23 + Playwright break on cross-file .ts imports).
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
  await page.evaluate((m) => { (window as Window & { _spyde_test_inject?: (m: unknown) => void })._spyde_test_inject?.(m) }, msg)
}
async function trackActions() {
  await app.evaluate(({ ipcMain }) => {
    ;(globalThis as unknown as { __sent: unknown[] }).__sent = []
    ipcMain.removeAllListeners('spyde:action')
    ipcMain.on('spyde:action', (_e, action, payload, windowId) => {
      ;(globalThis as unknown as { __sent: unknown[] }).__sent.push({ action, payload, windowId })
    })
  })
}
const sent = () => app.evaluate(() => (globalThis as unknown as { __sent: unknown[] }).__sent)

// Two testids share the same flex-row parent — a layout-timing-free way to
// assert "side by side in a row" (boundingBox geometry can flake mid-layout).
const sameRow = (a: string, b: string) => page.evaluate(([x, y]) => {
  const f = document.querySelector(`[data-testid="${x}"]`)
  const s = document.querySelector(`[data-testid="${y}"]`)
  return !!f && !!s && f.parentElement === s.parentElement
}, [a, b])

async function aSignalWindow() {
  await inject({
    type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false,
  })
  await expect(page.getByTestId('plot-control-dock')).toBeVisible()
}

test('periodic-table picker writes the composition (set_composition)', async () => {
  await trackActions()
  await aSignalWindow()

  // The dock shows a Composition section; opening it reveals the periodic table.
  await expect(page.getByTestId('composition-section')).toBeVisible()
  await page.getByTestId('composition-edit').click()
  await expect(page.getByTestId('periodic-table')).toBeVisible()

  // Pick Fe + Ni, give Fe 70 %.
  await page.getByTestId('ptable-el-Fe').click()
  await page.getByTestId('ptable-el-Ni').click()
  await expect(page.getByTestId('ptable-selected')).toContainText('Fe')
  await expect(page.getByTestId('ptable-selected')).toContainText('Ni')
  await page.getByTestId('ptable-pct-Fe').fill('70')
  await page.screenshot({ path: join(__dirname, '..', 'periodic_table.png') })

  await page.getByTestId('ptable-apply').click()
  await expect(page.getByTestId('periodic-table')).toBeHidden()

  // It dispatched set_composition with the chosen elements + percentage.
  const calls = (await sent()) as Array<{ action: string; payload: Record<string, unknown> }>
  const setc = calls.find(c => c.action === 'set_composition')
  expect(setc).toBeTruthy()
  expect(setc!.payload.elements).toEqual(['Fe', 'Ni'])
  expect((setc!.payload.percentages as Record<string, number>).Fe).toBe(70)
})

test('dock shows composition chips from the backend echo', async () => {
  await aSignalWindow()
  // Backend echo (set in metadata.Sample) → chips with element + atomic %.
  await inject({ type: 'composition', window_ids: [1], elements: ['Si', 'O'], percentages: { Si: 33, O: 67 } })
  await expect(page.getByTestId('composition-chip-Si')).toContainText('Si')
  await expect(page.getByTestId('composition-chip-Si')).toContainText('33%')
  await expect(page.getByTestId('composition-chip-O')).toContainText('67%')
})

test('COD picker lists structures (a/b/c/α/β/γ) and picks one', async () => {
  await trackActions()
  // A DP window with the Orientation Mapping action so the wizard can open.
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Orientation Mapping', icon: '', side: 'left', toggle: false, subfunctions: [],
      parameters: { gamma: { name: 'Gamma', type: 'float', default: 1.0 } },
    }],
  })
  await inject({
    type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false,
  })

  // Open the wizard (toolbar reveals on hover).
  await page.getByTestId('subwindow').first().getByTestId('subwindow-titlebar').hover()
  await page.getByTestId('action-btn-Orientation Mapping').click()
  await expect(page.getByTestId('orientation-wizard')).toBeVisible()

  // "From file" and "Search" sit side by side in one row.
  await expect(page.getByTestId('om-pick-cif')).toBeVisible()
  await expect(page.getByTestId('cod-search')).toBeVisible()
  expect(await sameRow('om-pick-cif', 'cod-search')).toBeTruthy()
  await page.screenshot({ path: join(__dirname, '..', 'cif_row.png') })

  // Search COD by composition → dispatches cod_search + opens a POPOUT.
  await page.getByTestId('cod-search').click()
  const c1 = (await sent()) as Array<{ action: string }>
  expect(c1.some(c => c.action === 'cod_search')).toBeTruthy()
  await expect(page.getByTestId('cod-popout')).toBeVisible()

  // Backend (mocked here) returns candidate structures with cell parameters,
  // shown in the popout (not expanding the wizard downward).
  await inject({
    type: 'cod_results', window_id: 1, elements: ['Si', 'O'],
    results: [{
      id: '1011200', formula: 'O2 Si', phase: 'Quartz', sg: 'P 31 2 1',
      a: 4.913, b: 4.913, c: 5.405, alpha: 90, beta: 90, gamma: 120, volume: 113,
    }],
  })
  await expect(page.getByTestId('cod-list')).toBeVisible()
  await expect(page.getByTestId('cod-row-1011200')).toContainText('Quartz')
  await expect(page.getByTestId('cod-row-1011200')).toContainText('4.913')
  await page.screenshot({ path: join(__dirname, '..', 'composition_cod.png') })

  // Pick it → dispatches cod_pick, the popout closes, and the returned .cif path
  // is added as a phase.
  await page.getByTestId('cod-row-1011200').click()
  await expect(page.getByTestId('cod-popout')).toBeHidden()
  const c2 = (await sent()) as Array<{ action: string; payload: Record<string, unknown> }>
  expect(c2.find(c => c.action === 'cod_pick')?.payload.cod_id).toBe('1011200')

  await inject({ type: 'cod_cif_ready', window_id: 1, cod_id: '1011200', path: '/tmp/cod_1011200.cif', label: 'Quartz' })
  await expect(page.getByTestId('om-cif-list')).toContainText('cod_1011200.cif')
})

test('the SAME file+search row + COD popout work in the VECTOR OM wizard', async () => {
  await trackActions()
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{
      name: 'Vector Orientation Mapping', icon: '', side: 'left', toggle: false, subfunctions: [],
      parameters: { strain_cap: { name: 'Strain cap', type: 'float', default: 0.05 } },
    }],
  })
  await inject({
    type: 'figure', window_id: 1, fig_id: 'vec',
    html: '<html><body>v</body></html>', title: 'Vectors', is_navigator: false,
  })

  await page.getByTestId('subwindow').first().getByTestId('subwindow-titlebar').hover()
  await page.getByTestId('action-btn-Vector Orientation Mapping').click()
  await expect(page.getByTestId('vector-orientation-wizard')).toBeVisible()

  // File + Search share a row here too.
  await expect(page.getByTestId('vom-pick-cif')).toBeVisible()
  await expect(page.getByTestId('cod-search')).toBeVisible()
  expect(await sameRow('vom-pick-cif', 'cod-search')).toBeTruthy()

  // Search → popout list → pick downloads + loads the structure.
  await page.getByTestId('cod-search').click()
  await expect(page.getByTestId('cod-popout')).toBeVisible()
  await inject({
    type: 'cod_results', window_id: 1, elements: ['Ag'],
    results: [{ id: '1100136', formula: 'Ag', phase: 'Silver', sg: 'F m -3 m',
      a: 4.0855, b: 4.0855, c: 4.0855, alpha: 90, beta: 90, gamma: 90, volume: 68 }],
  })
  await expect(page.getByTestId('cod-row-1100136')).toContainText('Silver')
  await page.getByTestId('cod-row-1100136').click()
  await expect(page.getByTestId('cod-popout')).toBeHidden()
  const calls = (await sent()) as Array<{ action: string; payload: Record<string, unknown> }>
  expect(calls.find(c => c.action === 'cod_pick')?.payload.cod_id).toBe('1100136')

  // The vector wizard is single-phase → the picked .cif becomes the chosen file.
  await inject({ type: 'cod_cif_ready', window_id: 1, cod_id: '1100136', path: '/tmp/cod_1100136.cif', label: 'Silver' })
  await expect(page.getByTestId('vom-pick-cif')).toHaveText('cod_1100136.cif')
})
