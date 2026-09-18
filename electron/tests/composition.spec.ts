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

  // Every element click writes through, so take the LAST one — the composition
  // as committed. (Clicking with a phase selected also sends `set_phase` as you
  // go: a phase is a subset of this list.)
  const calls = (await sent()) as Array<{ action: string; payload: Record<string, unknown> }>
  const setc = calls.filter(c => c.action === 'set_composition').pop()
  expect(setc).toBeTruthy()
  expect(setc!.payload.elements).toEqual(['Fe', 'Ni'])
  expect((setc!.payload.percentages as Record<string, number>).Fe).toBe(70)
})

test('the f-block rows are full-height, not squashed into the spacer', async () => {
  await aSignalWindow()
  await page.getByTestId('composition-edit').click()
  await expect(page.getByTestId('periodic-table')).toBeVisible()

  // The gap between the main table and the detached f-block is a REAL grid row,
  // and the lanthanides used to be placed on it — so La…Lu rendered 8 px tall,
  // unreadable and barely clickable. Compare against a d-block cell rather than
  // a magic number, so this stays true if the cell size ever changes.
  const fe = (await page.getByTestId('ptable-el-Fe').boundingBox())!
  for (const sym of ['La', 'Lu', 'Ac', 'Lr']) {
    const box = (await page.getByTestId(`ptable-el-${sym}`).boundingBox())!
    expect(box.height, `${sym} is not a full-height cell`).toBeCloseTo(fe.height, 0)
  }
  // …and the two f-block rows are still separate rows, below the main table.
  const la = (await page.getByTestId('ptable-el-La').boundingBox())!
  const ac = (await page.getByTestId('ptable-el-Ac').boundingBox())!
  const ra = (await page.getByTestId('ptable-el-Ra').boundingBox())!
  expect(la.y).toBeGreaterThan(ra.y + ra.height)
  expect(ac.y).toBeGreaterThan(la.y + la.height * 0.9)

  await page.screenshot({ path: join(__dirname, '..', 'periodic_table_fblock.png') })
  await page.getByTestId('ptable-close').click()
})

test('dock shows composition chips from the backend echo', async () => {
  await aSignalWindow()
  // Backend echo (set in metadata.Sample) → chips with element + atomic %.
  await inject({ type: 'composition', window_ids: [1], elements: ['Si', 'O'], percentages: { Si: 33, O: 67 } })
  await expect(page.getByTestId('composition-chip-Si')).toContainText('Si')
  await expect(page.getByTestId('composition-chip-Si')).toContainText('33%')
  await expect(page.getByTestId('composition-chip-O')).toContainText('67%')
})

test('the COD search is scoped to ONE phase, and its pick is recorded there', async () => {
  // COD matches the elements EXACTLY, so the query has to be one phase's
  // elements. A Cu/Nb sample asked as a single composition asks for a Cu-Nb
  // compound and gets nothing back; asked a phase at a time it finds both.
  await trackActions()
  await aSignalWindow()

  await inject({
    type: 'composition', window_ids: [1], elements: ['Cu', 'Nb'],
    percentages: {},
    phases: [{ elements: ['Cu'], percentages: {}, cif_path: null, label: null, cod_id: null },
             { elements: ['Nb'], percentages: {}, cif_path: null, label: null, cod_id: null }],
  })

  // The dock shows one group per phase, with & between them.
  await expect(page.getByTestId('composition-phase-0')).toContainText('Cu')
  await expect(page.getByTestId('composition-phase-1')).toContainText('Nb')
  await expect(page.getByTestId('composition-section')).toContainText('&')

  await page.getByTestId('composition-edit').click()
  await expect(page.getByTestId('periodic-table')).toBeVisible()

  // Searching row 1 asks for THAT row's elements, and says so.
  await page.getByTestId('phase-btn-1').click()
  await page.getByTestId('phase-1-cod').click()
  const calls = (await sent()) as Array<{ action: string; payload: Record<string, unknown> }>
  expect(calls.find(c => c.action === 'cod_search')?.payload.phase).toBe(1)

  // Results arrive tagged with the phase and open INLINE in its row — the
  // question "which phase am I filling?" should not need remembering.
  await inject({
    type: 'cod_results', window_id: 1, phase: 1, elements: ['Nb'],
    results: [{ id: '1100136', formula: 'Nb', phase: '', sg: 'I m -3 m',
      a: 3.301, b: 3.301, c: 3.301, alpha: 90, beta: 90, gamma: 90, volume: 36 }],
  })
  const row = page.getByTestId('phase-1-cod-results')
  await expect(row).toBeVisible()
  await expect(row).toContainText('I m -3 m')
  await expect(row).toContainText('3.301')

  await page.getByTestId('cod-row-1100136').click()
  const after = (await sent()) as Array<{ action: string; payload: Record<string, unknown> }>
  const pick = after.find(c => c.action === 'cod_pick')
  expect(pick?.payload.cod_id).toBe('1100136')
  // The download is BOUND to its phase — that is what lets the sample remember
  // its own structures instead of the wizard keeping a private list.
  expect(pick?.payload.phase).toBe(1)
})

test('a phase carries its structure, and the dock shows it beside the chips', async () => {
  await aSignalWindow()
  await inject({
    type: 'composition', window_ids: [1], elements: ['Cu', 'Nb'], percentages: {},
    phases: [
      { elements: ['Cu'], percentages: {}, cif_path: '/tmp/Cu.cif',
        label: 'Cu Fm-3m', cod_id: '9008468' },
      { elements: ['Nb'], percentages: {}, cif_path: null, label: null, cod_id: null },
    ],
  })
  // Previously a picked .cif was visible nowhere outside the wizard.
  await expect(page.getByTestId('composition-structure-0')).toContainText('Cu Fm-3m')
  await expect(page.getByTestId('composition-phase-1')).toContainText('Nb')
})

test('both orientation wizards read the sample phases, not a private list', async () => {
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [
      { name: 'Orientation Mapping', icon: '', side: 'left', toggle: false,
        subfunctions: [], parameters: {} },
      { name: 'Vector Orientation Mapping', icon: '', side: 'left', toggle: false,
        subfunctions: [], parameters: {} },
    ],
  })
  await aSignalWindow()
  await inject({
    type: 'composition', window_ids: [1], elements: ['Ag'], percentages: {},
    phases: [{ elements: ['Ag'], percentages: {}, cif_path: '/tmp/Ag.cif',
               label: 'Ag Fm-3m', cod_id: null }],
  })
  for (const [action, prefix] of [
    ['Orientation Mapping', 'om'],
    ['Vector Orientation Mapping', 'vom'],
  ] as Array<[string, string]>) {
    await page.getByTestId('subwindow').first().getByTestId('subwindow-titlebar').hover()
    await page.getByTestId(`action-btn-${action}`).click()
    await expect(page.getByTestId(`${prefix}-cif-list`)).toContainText('Ag Fm-3m')
    await page.getByTestId(`${prefix}-close`).click()
  }
})
