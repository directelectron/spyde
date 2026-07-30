/**
 * data_table.spec.ts — the bottom table dock + the generic DataTable.
 *
 * Drives everything through injected `particles_table` messages
 * (`window._spyde_test_inject`, the same hook app_log.spec.ts uses), so the spec
 * needs NO Python particle backend — that side is a separate workstream.
 *
 * What it pins:
 *  - visibility wiring: StatusBar toggle, View menu item (with its ✓), close ×
 *  - the empty state when no data has arrived (the degrade-gracefully contract)
 *  - VIRTUALISATION: 5000 rows, only a windowed slice in the DOM
 *  - sort cycle asc → desc → none, single + ctrl multi selection
 *  - the Events tab's fixed columns + per-kind colour swatch
 *  - top-edge resize, and the 50%-of-window cap that protects the MDI area
 *
 * Screenshots land in electron/data_table_shots/ and are meant to be LOOKED at.
 */
import { test, expect, Page, ElectronApplication } from '@playwright/test'
import { join } from 'path'
const { launchApp } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'data_table_shots')

let app: ElectronApplication
let page: Page
let assertNoJsErrors: () => void

test.beforeAll(async () => {
  const ctx = await launchApp()
  app = ctx.app
  page = ctx.page
  assertNoJsErrors = ctx.assertNoJsErrors
})
test.afterAll(async () => { await app?.close() })

async function inject(msg: Record<string, unknown>) {
  await page.evaluate((m) => {
    ;(window as unknown as { _spyde_test_inject?: (x: unknown) => void })._spyde_test_inject?.(m)
  }, msg)
}

/** A realistic particle table: backend-shaped column descriptors + N rows keyed
 *  by `spyde.signals.particles.COLUMNS` / MEASURED_COLUMNS.
 *
 *  `wide` repeats the measured block so the fixed widths overflow the dock —
 *  that is the horizontal-scroll + sticky-header case. */
async function injectTable(n: number, opts: { partial?: boolean; wide?: boolean } = {}) {
  await page.evaluate(({ count, partial, wide }) => {
    const columns: Record<string, unknown>[] = [
      { key: 'id', label: '#', width: 54, numeric: true },
      { key: 't', label: 'frame', width: 60, numeric: true },
      { key: 'track_id', label: 'track', width: 72, numeric: true, kind: 'swatch' },
      { key: 'y', label: 'y', width: 76, numeric: true, precision: 2, units: 'nm' },
      { key: 'x', label: 'x', width: 76, numeric: true, precision: 2, units: 'nm' },
      { key: 'area', label: 'area', width: 88, numeric: true, precision: 1, units: 'nm²' },
      { key: 'equiv_diameter', label: 'diameter', width: 84, numeric: true, precision: 2, units: 'nm' },
      { key: 'major_axis', label: 'major', width: 80, numeric: true, precision: 2, units: 'nm' },
      { key: 'minor_axis', label: 'minor', width: 80, numeric: true, precision: 2, units: 'nm' },
      { key: 'perimeter', label: 'perimeter', width: 88, numeric: true, precision: 1, units: 'nm' },
      { key: 'circularity', label: 'circularity', width: 92, numeric: true, precision: 3 },
      { key: 'eccentricity', label: 'eccentricity', width: 96, numeric: true, precision: 3 },
      { key: 'solidity', label: 'solidity', width: 82, numeric: true, precision: 3 },
      { key: 'intensity_mean', label: 'mean I', width: 80, numeric: true, precision: 1 },
      { key: 'intensity_std', label: 'std I', width: 76, numeric: true, precision: 1 },
      { key: 'background', label: 'bkgd', width: 72, numeric: true, precision: 1 },
      // Width-less: absorbs whatever space the fixed columns leave over.
      { key: 'intensity_max', label: 'max I', numeric: true, precision: 1 },
    ]
    if (wide) {
      for (const key of ['area', 'perimeter', 'circularity', 'solidity',
                         'intensity_mean', 'intensity_std', 'background', 'major_axis']) {
        columns.push({ key, label: `${key} (2)`, width: 110, numeric: true, precision: 2 })
      }
    }
    const rows: Record<string, unknown>[] = []
    for (let i = 0; i < count; i++) {
      // Deterministic pseudo-random so the screenshot is stable run to run.
      const r = (k: number) => ((Math.sin(i * 12.9898 + k * 78.233) * 43758.5453) % 1 + 1) % 1
      const d = 4 + r(1) * 26
      rows.push({
        id: i,
        t: Math.floor(i / 24),
        track_id: i % 37,
        y: r(2) * 512,
        x: r(3) * 512,
        area: Math.PI * (d / 2) ** 2,
        equiv_diameter: d,
        major_axis: d * (1.0 + r(8) * 0.6),
        minor_axis: d * (0.6 + r(9) * 0.3),
        perimeter: Math.PI * d * (1 + r(10) * 0.15),
        circularity: 0.55 + r(4) * 0.45,
        eccentricity: r(5) * 0.9,
        solidity: 0.8 + r(11) * 0.2,
        intensity_mean: 1200 + r(6) * 2400,
        intensity_std: 60 + r(12) * 340,
        background: 300 + r(13) * 500,
        intensity_max: 3000 + r(7) * 5000,
      })
    }
    const events: Record<string, unknown>[] = []
    const kinds = ['birth', 'death', 'merge', 'split']
    for (let i = 0; i < 40; i++) {
      const kind = kinds[i % 4]
      events.push({
        frame: 3 + i * 2,
        kind,
        tracks: kind === 'merge' || kind === 'split' ? [i % 37, (i + 5) % 37] : [i % 37],
        particles: kind === 'merge' || kind === 'split' ? [i * 7, i * 7 + 3] : [i * 7],
      })
    }
    ;(window as unknown as { _spyde_test_inject?: (x: unknown) => void })._spyde_test_inject?.({
      type: 'particles_table',
      window_id: null,
      title: 'Particles — au_nanoparticles (in-situ)',
      units: 'nm',
      partial,
      columns,
      rows,
      events,
    })
  }, { count: n, partial: Boolean(opts.partial), wide: Boolean(opts.wide) })
  await expect(page.locator('[data-testid="bottom-dock-count"]')).toHaveText(String(n))
}

const dock = () => page.locator('[data-testid="bottom-dock"]')
const rows = () => page.locator('[data-testid="particle-table-row"]')

test('the dock is hidden until a toggle opens it, and shows a clear empty state', async () => {
  await expect(dock()).toHaveCount(0)

  await page.click('[data-testid="toggle-table-dock"]')
  await expect(dock()).toBeVisible()
  // No backend data has arrived → the degrade-gracefully message, not a blank.
  await expect(page.locator('[data-testid="particle-table-empty"]'))
    .toContainText('Segment Particles')
  await expect(page.locator('[data-testid="bottom-dock-count"]')).toHaveText('0')

  await page.screenshot({ path: join(SHOTS, '01-empty.png') })
})

test('5000 rows render as a windowed slice, not 5000 DOM nodes', async () => {
  await injectTable(5000)

  // Virtualisation: a ~230 px body at 24 px/row is ~10 visible rows; with the
  // default overscan of 8 either side the slice is well under 50.
  const rendered = await rows().count()
  expect(rendered).toBeGreaterThan(4)
  expect(rendered).toBeLessThan(50)

  // The first row is row 0 and the spacers give the body the full scroll height.
  await expect(rows().first()).toHaveAttribute('data-row-index', '0')
  const scrollH = await page.locator('[data-testid="particle-table-body"]')
    .evaluate((el) => el.scrollHeight)
  expect(scrollH).toBeGreaterThan(5000 * 24 * 0.9)

  await page.screenshot({ path: join(SHOTS, '02-table.png') })
})

test('scrolling swaps the rendered window without growing the DOM', async () => {
  await injectTable(5000)
  const body = page.locator('[data-testid="particle-table-body"]')
  await body.evaluate((el) => { el.scrollTop = 24 * 2000 })
  // The window follows scrollTop: the first rendered row is near index 2000.
  await expect
    .poll(async () => Number(await rows().first().getAttribute('data-row-index')))
    .toBeGreaterThan(1900)
  expect(await rows().count()).toBeLessThan(50)

  await body.evaluate((el) => { el.scrollTop = 0 })
  await expect(rows().first()).toHaveAttribute('data-row-index', '0')
})

test('a sortable header cycles asc → desc → unsorted', async () => {
  await injectTable(5000)
  const th = page.locator('[data-testid="particle-table-th-area"]')
  // Normalise: sort state survives a re-injection (only a tab switch remounts
  // the table), so click round the cycle rather than assuming it starts clean.
  for (let i = 0; i < 3 && (await th.getAttribute('data-sort')) !== 'none'; i++) await th.click()
  await expect(th).toHaveAttribute('data-sort', 'none')

  const firstAreaText = async () =>
    (await rows().first().locator('[data-col="area"]').innerText()).trim()
  const areaNum = async () => parseFloat((await firstAreaText()).replace(/[^\d.]/g, ''))

  await th.click()
  await expect(th).toHaveAttribute('data-sort', 'asc')
  const asc = await areaNum()

  await th.click()
  await expect(th).toHaveAttribute('data-sort', 'desc')
  const desc = await areaNum()
  expect(desc).toBeGreaterThan(asc)
  // Shot in the SORTED state — the accent-coloured header + ▾ caret + the
  // largest areas on top are what the eye has to confirm.
  await page.screenshot({ path: join(SHOTS, '03-sorted.png') })

  await th.click()
  await expect(th).toHaveAttribute('data-sort', 'none')
  // Unsorted restores the original order (row 0 first).
  await expect(rows().first()).toHaveAttribute('data-row-index', '0')
})

test('rows select singly, and ctrl-click extends the selection', async () => {
  await injectTable(5000)
  await rows().nth(1).click()
  await expect(page.locator('[data-testid="particle-table-row"][data-selected="true"]'))
    .toHaveCount(1)

  await rows().nth(3).click({ modifiers: ['ControlOrMeta'] })
  await rows().nth(5).click({ modifiers: ['ControlOrMeta'] })
  await expect(page.locator('[data-testid="particle-table-row"][data-selected="true"]'))
    .toHaveCount(3)

  // A plain click collapses back to one.
  await rows().nth(7).click()
  await expect(page.locator('[data-testid="particle-table-row"][data-selected="true"]'))
    .toHaveCount(1)

  await page.screenshot({ path: join(SHOTS, '04-selection.png') })
})

test('the filter box narrows the rows and the count shows both totals', async () => {
  await injectTable(5000)
  await page.fill('[data-testid="bottom-dock-search"]', '0.9')
  await expect(page.locator('[data-testid="bottom-dock-count"]')).toContainText('/ 5000')
  const shown = Number((await page.locator('[data-testid="bottom-dock-count"]').innerText())
    .split('/')[0].trim())
  expect(shown).toBeGreaterThan(0)
  expect(shown).toBeLessThan(5000)
  await page.fill('[data-testid="bottom-dock-search"]', '')
  await expect(page.locator('[data-testid="bottom-dock-count"]')).toHaveText('5000')
})

test('a column set wider than the dock scrolls sideways under a pinned header', async () => {
  await injectTable(400, { wide: true })
  const body = page.locator('[data-testid="particle-table-body"]')
  const { sw, cw } = await body.evaluate((el) => ({ sw: el.scrollWidth, cw: el.clientWidth }))
  expect(sw).toBeGreaterThan(cw)

  // Scroll right: the header must travel WITH the cells (a sticky header pins
  // vertically only) and stay pinned to the top after a vertical scroll.
  await body.evaluate((el) => { el.scrollLeft = 600; el.scrollTop = 24 * 100 })
  const { headTop, bodyTop } = await page.evaluate(() => {
    const b = document.querySelector('[data-testid="particle-table-body"]')!.getBoundingClientRect()
    const h = document.querySelector('[data-testid="particle-table-head"]')!.getBoundingClientRect()
    return { headTop: h.top, bodyTop: b.top }
  })
  expect(Math.abs(headTop - bodyTop)).toBeLessThan(2)
  await page.screenshot({ path: join(SHOTS, '05-wide-scrolled.png') })

  await body.evaluate((el) => { el.scrollLeft = 0; el.scrollTop = 0 })
})

test('the Events tab shows the fixed event columns with a colour per kind', async () => {
  await injectTable(5000)
  await page.click('[data-testid="bottom-dock-tab-events"]')
  await expect(dock()).toHaveAttribute('data-tab', 'events')
  await expect(page.locator('[data-testid="bottom-dock-count"]')).toHaveText('40')
  await expect(page.locator('[data-testid="particle-table-th-kind"]')).toContainText('event')
  // The swatch is what makes birth/death/merge/split readable at a glance.
  expect(await page.locator('[data-testid="particle-table-swatch"]').count())
    .toBeGreaterThan(4)
  await expect(rows().first().locator('[data-col="kind"]')).toContainText('birth')

  await page.screenshot({ path: join(SHOTS, '05-events.png') })

  await page.click('[data-testid="bottom-dock-tab-table"]')
  await expect(dock()).toHaveAttribute('data-tab', 'table')
})

test('the streaming pill appears while a batch is still filling the table', async () => {
  await injectTable(200, { partial: true })
  await expect(page.locator('[data-testid="bottom-dock-streaming"]')).toBeVisible()
  await injectTable(5000)
  await expect(page.locator('[data-testid="bottom-dock-streaming"]')).toHaveCount(0)
})

test('the top edge resizes the dock and never exceeds half the window', async () => {
  const before = Number(await dock().getAttribute('data-height'))
  const box = await dock().boundingBox()
  if (!box) throw new Error('dock has no bounding box')

  // Drag the top edge UP by 120 px → the dock grows.
  await page.mouse.move(box.x + box.width / 2, box.y + 1)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width / 2, box.y - 120, { steps: 8 })
  await page.mouse.up()
  const grown = Number(await dock().getAttribute('data-height'))
  expect(grown).toBeGreaterThan(before)

  await page.screenshot({ path: join(SHOTS, '06-resized.png') })

  // Now drag far past the ceiling: the dock must cap at 50% of the window and
  // the MDI area must still have real height (the flexShrink:0 starvation trap).
  const box2 = await dock().boundingBox()
  if (!box2) throw new Error('dock has no bounding box')
  await page.mouse.move(box2.x + box2.width / 2, box2.y + 1)
  await page.mouse.down()
  await page.mouse.move(box2.x + box2.width / 2, 0, { steps: 10 })
  await page.mouse.up()

  const { height, inner, mdi } = await page.evaluate(() => ({
    height: Number(document.querySelector('[data-testid="bottom-dock"]')
      ?.getAttribute('data-height')),
    inner: window.innerHeight,
    mdi: document.querySelector('[data-testid="mdi-area"]')?.getBoundingClientRect().height ?? 0,
  }))
  expect(height).toBeLessThanOrEqual(Math.round(inner * 0.5) + 1)
  expect(mdi).toBeGreaterThan(100)

  await page.screenshot({ path: join(SHOTS, '07-capped.png') })
})

test('the View menu toggles the dock and shows its checked state', async () => {
  await page.click('[data-testid="menu-view"]')
  const item = page.locator('[data-testid="menu-item-table-dock"]')
  await expect(item).toContainText('✓')            // open right now
  // Shot WITH the menu open — the ✓ is the thing being verified.
  await page.screenshot({ path: join(SHOTS, '08-view-menu.png') })
  await item.click()
  await expect(dock()).toHaveCount(0)

  await page.click('[data-testid="menu-view"]')
  await expect(page.locator('[data-testid="menu-item-table-dock"]')).not.toContainText('✓')
  await page.screenshot({ path: join(SHOTS, '09-view-menu-unchecked.png') })
  await page.locator('[data-testid="menu-item-table-dock"]').click()
  await expect(dock()).toBeVisible()
})

test('the close button hides the dock; reopening keeps the loaded table', async () => {
  await injectTable(5000)
  await page.click('[data-testid="bottom-dock-close"]')
  await expect(dock()).toHaveCount(0)

  await page.click('[data-testid="toggle-table-dock"]')
  await expect(dock()).toBeVisible()
  // BottomDock stays MOUNTED while hidden (it returns null after its hooks), so
  // the fetched table survives a hide/show — no re-query round trip.
  await expect(page.locator('[data-testid="bottom-dock-count"]')).toHaveText('5000')

  assertNoJsErrors()
})

