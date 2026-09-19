/**
 * composition.spec.ts — the sample's phases: the dock section, the popout that
 * edits them, the COD structure search, and the wizards that read them.
 *
 * A sample is a list of phases, addressed by id; the periodic table edits the
 * SELECTED one. Drives the UI via the test-inject hook and observes outgoing
 * actions on the ipcMain channel — the backend is mocked, so each test injects
 * the `composition` echo the real handler would send. The backend itself is
 * covered by test_composition.py, and the round trip by
 * phases_composition.spec.ts.
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

type Sent = { action: string; payload: Record<string, unknown> }
type EchoPhase = {
  id: string; elements?: string[]; percentages?: Record<string, number>
  trace?: string[]; cif_path?: string | null; label?: string | null
}

async function inject(message: Record<string, unknown>) {
  await page.evaluate((injected) => {
    (window as Window & { _spyde_test_inject?: (m: unknown) => void })._spyde_test_inject?.(injected)
  }, message)
}
async function trackActions() {
  await app.evaluate(({ ipcMain }) => {
    ;(globalThis as unknown as { __sent: unknown[] }).__sent = []
    ipcMain.removeAllListeners('spyde:action')
    ipcMain.on('spyde:action', (_event, action, payload, windowId) => {
      ;(globalThis as unknown as { __sent: unknown[] }).__sent.push({ action, payload, windowId })
    })
  })
}
const sent = async () =>
  (await app.evaluate(() => (globalThis as unknown as { __sent: unknown[] }).__sent)) as Sent[]
/** The payloads sent for *action* so far. Poll it: the send crosses a process. */
const payloadsOf = async (action: string) =>
  (await sent()).filter(call => call.action === action).map(call => call.payload)
/** The phase id the popout sent with its first *action*, once it has. */
async function phaseIdOf(action: string): Promise<string> {
  await expect.poll(async () => (await payloadsOf(action)).length).toBeGreaterThan(0)
  return (await payloadsOf(action))[0].phase as string
}

/** What the backend sends after an edit: the phases, and their union. */
const phasesEcho = (phases: EchoPhase[]) => inject({
  type: 'composition', window_ids: [1],
  elements: [...new Set(phases.flatMap(phase => phase.elements ?? []))],
  phases: phases.map(phase => ({
    id: phase.id, elements: phase.elements ?? [], percentages: phase.percentages ?? {},
    trace: phase.trace ?? [], cif_path: phase.cif_path ?? null, label: phase.label ?? null,
    cod_id: null, structure_elements: null,
  })),
})

async function aSignalWindow() {
  await inject({
    type: 'figure', window_id: 1, fig_id: 'sig',
    html: '<html><body>s</body></html>', title: 'Diffraction', is_navigator: false,
  })
  await expect(page.getByTestId('plot-control-dock')).toBeVisible()
}

test('with no phases, the first element click builds Phase 1', async () => {
  await trackActions()
  await aSignalWindow()
  await expect(page.getByTestId('composition-empty')).toBeVisible()
  await page.getByTestId('composition-edit').click()
  await expect(page.getByTestId('periodic-table')).toBeVisible()

  // Phase 1 is selected before it exists, and there is nothing to add yet.
  await expect(page.getByTestId('phase-btn-0')).toHaveAttribute('data-active', 'true')
  await expect(page.getByTestId('ptable-add-phase')).toHaveCount(0)

  await page.getByTestId('ptable-el-Fe').click()
  await page.getByTestId('ptable-el-O').click()
  const id = await phaseIdOf('toggle_phase_element')
  // Both clicks name the same new phase, so the second cannot make another.
  await expect.poll(() => payloadsOf('toggle_phase_element'))
    .toEqual([{ phase: id, element: 'Fe' }, { phase: id, element: 'O' }])
  await phasesEcho([{ id, elements: ['Fe', 'O'] }])
  await expect(page.getByTestId('ptable-el-Fe')).toHaveAttribute('data-in-phase', 'true')
  await expect(page.getByTestId('ptable-selected')).toContainText('Fe')
  await expect(page.getByTestId('ptable-add-phase')).toBeVisible()
  await page.screenshot({ path: join(__dirname, '..', 'periodic_table.png') })
})

test('a percentage is saved when the field is left, as typed', async () => {
  await trackActions()
  await aSignalWindow()
  await phasesEcho([{ id: 'steel', elements: ['Fe', 'Ni'] }])
  await page.getByTestId('composition-edit').click()

  // "12." must survive being typed on the way to 12.5, and only the edited
  // element is sent, so two quick saves cannot overwrite each other.
  await page.getByTestId('phase-0-pct-Fe').fill('12.5')
  await page.getByTestId('phase-0-pct-Fe').press('Enter')
  await expect.poll(() => payloadsOf('set_phase_percentages'))
    .toEqual([{ phase: 'steel', percentages: { Fe: 12.5 } }])

  await page.getByTestId('ptable-done').click()
  await expect(page.getByTestId('periodic-table')).toBeHidden()
})

test('a percentage typed in one phase stays in that phase', async () => {
  await trackActions()
  await aSignalWindow()
  await phasesEcho([{ id: 'zirconia', elements: ['Zr', 'O'] }, { id: 'alpha', elements: ['Zr'] }])
  await page.getByTestId('composition-edit').click()

  await page.getByTestId('phase-0-pct-Zr').fill('33.3')
  await page.getByTestId('phase-btn-1').click()
  await expect(page.getByTestId('phase-1-pct-Zr')).toHaveValue('')
  await expect.poll(() => payloadsOf('set_phase_percentages'))
    .toEqual([{ phase: 'zirconia', percentages: { Zr: 33.3 } }])
})

test('an element another phase has can be added to this one', async () => {
  await trackActions()
  await aSignalWindow()
  await phasesEcho([{ id: 'zirconia', elements: ['Zr', 'O'] }, { id: 'alpha' }])
  await page.getByTestId('composition-edit').click()
  await page.getByTestId('phase-btn-1').click()

  await expect(page.getByTestId('ptable-el-Zr')).not.toHaveAttribute('data-in-phase', 'true')
  await expect(page.getByTestId('ptable-el-Zr')).toHaveAttribute('title', /in another phase/)
  await page.getByTestId('ptable-el-Zr').click()
  await expect.poll(() => payloadsOf('toggle_phase_element'))
    .toEqual([{ phase: 'alpha', element: 'Zr' }])
})

test('a trace element is marked, and left out of the COD search', async () => {
  // Fe with trace O: the O matters to EELS and EDS, the structure is still Fe's.
  await trackActions()
  await aSignalWindow()
  await phasesEcho([{ id: 'iron', elements: ['Fe', 'O'] }])
  await page.getByTestId('composition-edit').click()
  await expect(page.getByTestId('phase-0-cod')).toHaveAttribute('title', /Fe-O structures/)

  await page.getByTestId('phase-0-trace-O').click()
  await expect.poll(() => payloadsOf('set_phase_trace'))
    .toEqual([{ phase: 'iron', element: 'O', trace: true }])
  await phasesEcho([{ id: 'iron', elements: ['Fe', 'O'], trace: ['O'] }])
  await expect(page.getByTestId('phase-0-trace-O')).toHaveAttribute('data-on', 'true')
  await expect(page.getByTestId('phase-0-cod')).toHaveAttribute('title', /Search COD for Fe structures/)
  await expect(page.getByTestId('composition-chip-0-O')).toHaveAttribute('data-trace', 'true')
})

test('+ Phase adds a phase and selects it', async () => {
  await trackActions()
  await aSignalWindow()
  await phasesEcho([{ id: 'copper', elements: ['Cu'] }])
  await page.getByTestId('composition-edit').click()
  await page.getByTestId('ptable-add-phase').click()
  await expect(page.getByTestId('phase-btn-1')).toHaveAttribute('data-active', 'true')
  const id = await phaseIdOf('add_phase')

  await phasesEcho([{ id: 'copper', elements: ['Cu'] }, { id }])
  await page.getByTestId('ptable-el-Nb').click()
  await expect.poll(() => payloadsOf('toggle_phase_element'))
    .toEqual([{ phase: id, element: 'Nb' }])
})

test('the f-block rows are full-height, not squashed into the spacer', async () => {
  await aSignalWindow()
  await page.getByTestId('composition-edit').click()
  await expect(page.getByTestId('periodic-table')).toBeVisible()

  // The gap between the main table and the detached f-block is a REAL grid row,
  // and the lanthanides used to be placed on it — so La…Lu rendered 8 px tall,
  // unreadable and barely clickable. Compare against a d-block cell rather than
  // a magic number, so this stays true if the cell size ever changes.
  const iron = (await page.getByTestId('ptable-el-Fe').boundingBox())!
  for (const symbol of ['La', 'Lu', 'Ac', 'Lr']) {
    const box = (await page.getByTestId(`ptable-el-${symbol}`).boundingBox())!
    expect(box.height, `${symbol} is not a full-height cell`).toBeCloseTo(iron.height, 0)
  }
  // …and the two f-block rows are still separate rows, below the main table.
  const lanthanum = (await page.getByTestId('ptable-el-La').boundingBox())!
  const actinium = (await page.getByTestId('ptable-el-Ac').boundingBox())!
  const radium = (await page.getByTestId('ptable-el-Ra').boundingBox())!
  expect(lanthanum.y).toBeGreaterThan(radium.y + radium.height)
  expect(actinium.y).toBeGreaterThan(lanthanum.y + lanthanum.height * 0.9)

  await page.screenshot({ path: join(__dirname, '..', 'periodic_table_fblock.png') })
  await page.getByTestId('ptable-close').click()
})

test('the dock shows each phase, its percentages and its structure', async () => {
  await aSignalWindow()
  await phasesEcho([
    { id: 'silica', elements: ['Si', 'O'], percentages: { Si: 33, O: 67 } },
    { id: 'niobium', elements: ['Nb'], cif_path: '/tmp/Nb.cif', label: 'Nb Im-3m' },
  ])
  await expect(page.getByTestId('composition-chip-0-Si')).toContainText('33%')
  await expect(page.getByTestId('composition-chip-0-O')).toContainText('67%')
  await expect(page.getByTestId('composition-phase-1')).toContainText('Nb')
  await expect(page.getByTestId('composition-structure-1')).toContainText('Nb Im-3m')
  await expect(page.getByTestId('composition-section')).toContainText('&')
})

test('the COD search is scoped to ONE phase, and its pick is recorded there', async () => {
  // COD matches the elements EXACTLY, so the query has to be one phase's
  // elements. A Cu/Nb sample asked as a single composition asks for a Cu-Nb
  // compound and gets nothing back; asked a phase at a time it finds both.
  await trackActions()
  await aSignalWindow()
  await phasesEcho([{ id: 'copper', elements: ['Cu'] }, { id: 'niobium', elements: ['Nb'] }])
  await page.getByTestId('composition-edit').click()

  await page.getByTestId('phase-btn-1').click()
  await expect(page.getByTestId('phase-1-cod')).toHaveAttribute('title', /Nb structures/)
  await page.getByTestId('phase-1-cod').click()
  await expect.poll(() => payloadsOf('cod_search')).toEqual([{ phase: 'niobium' }])

  // A reply for another phase is not shown here.
  await inject({ type: 'cod_results', window_id: 1, phase: 'copper', elements: ['Cu'],
    results: [{ id: '9008468', formula: 'Cu', phase: '', sg: 'F m -3 m',
      a: 3.615, b: 3.615, c: 3.615, alpha: 90, beta: 90, gamma: 90, volume: 47 }] })
  await expect(page.getByTestId('cod-row-9008468')).toHaveCount(0)

  await inject({ type: 'cod_results', window_id: 1, phase: 'niobium', elements: ['Nb'],
    results: [{ id: '1100136', formula: 'Nb', phase: '', sg: 'I m -3 m',
      a: 3.301, b: 3.301, c: 3.301, alpha: 90, beta: 90, gamma: 90, volume: 36 }] })
  const row = page.getByTestId('phase-1-cod-results')
  await expect(row).toContainText('I m -3 m')
  await expect(row).toContainText('3.301')

  await page.getByTestId('cod-row-1100136').click()
  await expect.poll(() => payloadsOf('cod_pick'))
    .toEqual([{ phase: 'niobium', cod_id: '1100136', label: 'Nb I m -3 m' }])
})

test('every indexing wizard reads the sample phases, not a private list', async () => {
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: ['Orientation Mapping', 'Vector Orientation Mapping', 'EBSD Indexing']
      .map(name => ({ name, icon: '', side: 'left', toggle: false, subfunctions: [], parameters: {} })),
  })
  await aSignalWindow()
  await phasesEcho([{ id: 'silver', elements: ['Ag'], cif_path: '/tmp/Ag.cif', label: 'Ag Fm-3m' }])
  for (const [action, prefix] of [
    ['Orientation Mapping', 'om'],
    ['Vector Orientation Mapping', 'vom'],
    ['EBSD Indexing', 'ebsd'],
  ] as Array<[string, string]>) {
    await page.getByTestId('subwindow').first().getByTestId('subwindow-titlebar').hover()
    await page.getByTestId(`action-btn-${action}`).click()
    await expect(page.getByTestId(`${prefix}-cif-list`)).toContainText('Ag Fm-3m')
    await page.getByTestId(`${prefix}-close`).click()
  }
})

test('EBSD indexes against the chosen phase, and rebuilds when it changes', async () => {
  await trackActions()
  await inject({
    type: 'toolbar_config', window_id: 1, plot_id: 1,
    toolbar_actions: [{ name: 'EBSD Indexing', icon: '', side: 'left', toggle: false,
                        subfunctions: [], parameters: {} }],
  })
  await aSignalWindow()
  await page.getByTestId('subwindow').first().getByTestId('subwindow-titlebar').hover()
  await page.getByTestId('action-btn-EBSD Indexing').click()

  // No structure anywhere: the generic cubic band set, by space group.
  await expect(page.getByTestId('ebsd-spacegroup')).toBeVisible()
  await expect(page.getByTestId('ebsd-phase')).toHaveCount(0)

  await phasesEcho([
    { id: 'ferrite', elements: ['Fe'], cif_path: '/tmp/ferrite.cif', label: 'ferrite' },
    { id: 'austenite', elements: ['Fe'], cif_path: '/tmp/austenite.cif', label: 'austenite' },
  ])
  await expect(page.getByTestId('ebsd-spacegroup')).toHaveCount(0)
  // Two structures, so say which one — the first until chosen.
  await page.getByTestId('ebsd-phase').click()
  await page.getByTestId('ebsd-phase-opt-/tmp/austenite.cif').click()
  await page.getByTestId('ebsd-tab-Library').click()
  await page.getByTestId('ebsd-build').click()
  await expect.poll(async () => (await payloadsOf('ebsd_build_dictionary'))[0]?.cif_path)
    .toBe('/tmp/austenite.cif')
  await expect(page.getByTestId('ebsd-tab-Run')).toBeEnabled()

  // The phase it was built from goes away: the dictionary no longer applies.
  await phasesEcho([
    { id: 'ferrite', elements: ['Fe'], cif_path: '/tmp/ferrite.cif', label: 'ferrite' },
  ])
  await expect(page.getByTestId('ebsd-tab-Run')).toBeDisabled()
  await expect(page.getByTestId('ebsd-status')).toContainText('build the dictionary again')
})
