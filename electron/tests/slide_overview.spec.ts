/**
 * slide_overview.spec.ts — the Slide Overview grid (Present mode), e2e.
 *
 * Real Dask + bundled-synthetic Si-grains (so a figure slide has a live window to
 * snapshot). Builds a 3-slide deck with DISTINCT markdown titles, enters Present
 * mode, opens the overview grid (the `▦` header button / `O` key), and drives it
 * the way a presenter would:
 *   • assert 3 thumbnails with the right slide numbers + titles,
 *   • click thumbnail 3 → Present jumps to slide 3 (counter 3 / 3) and the grid
 *     closes,
 *   • re-open the overview and REORDER: drag slide 3 before slide 1 via the
 *     `report_move_slide` verb → assert the deck's slide order changed (the grid
 *     re-renders from the new report_state; thumbnail 1 is now the old slide 3).
 *
 * Screenshots the overview grid to overview_shots/ — a grid of slide thumbnails
 * with numbers + titles; each shot is Read by the author (a blank grid is a
 * failure even when selectors pass).
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'overview_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)   // navigator + signal
  await page.waitForTimeout(2500)                 // let the DP paint
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally {
    await ctx?.app?.close()
  }
})

/** The active slide's counter text ("n / N"). */
async function counterText(page: any): Promise<string> {
  return (await page.getByTestId('present-counter').textContent())?.trim() ?? ''
}

/** The ordered list of thumbnail LABELS in the overview grid. */
async function thumbLabels(page: any): Promise<string[]> {
  return await page.$$eval('[data-testid^="slide-thumb-label-"]',
    (els: Element[]) => els
      .sort((a, b) => {
        const ai = Number((a.getAttribute('data-testid') || '').replace('slide-thumb-label-', ''))
        const bi = Number((b.getAttribute('data-testid') || '').replace('slide-thumb-label-', ''))
        return ai - bi
      })
      .map(e => (e.textContent || '').trim()))
}

test('1) build a 3-slide deck with distinct titles', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()

  await backendAction(page, 'report_new', {})
  await backendAction(page, 'report_set_title', { title: 'Overview Demo Deck' })
  // Slide 1 — a title slide "Alpha".
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '# Alpha\n\nThe first slide.',
    html: '<h1>Alpha</h1><p>The first slide.</p>',
    slide_kind: 'title',
    notes: 'Speaker notes for Alpha.',
  })
  // Slide 2 — "Bravo".
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '## Bravo\n\nThe second slide.',
    html: '<h2>Bravo</h2><p>The second slide.</p>',
    slide_break: true,
  })
  // Slide 3 — "Charlie".
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '## Charlie\n\nThe third slide.',
    html: '<h2>Charlie</h2><p>The third slide.</p>',
    slide_break: true,
  })
  await page.waitForTimeout(800)
  await page.screenshot({ path: join(SHOTS, '01-deck-built.png') })
  ctx.assertNoJsErrors()
})

test('2) enter Present mode + open the overview grid → 3 thumbnails', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await expect(page.getByTestId('present-mode')).toBeVisible({ timeout: 10_000 })
  expect(await counterText(page)).toBe('1 / 3')

  // Open the overview via the header grid button.
  await page.getByTestId('present-overview-toggle').click()
  await expect(page.getByTestId('slide-overview')).toBeVisible({ timeout: 5_000 })

  // Three thumbnails with the right numbers + titles.
  await expect(page.locator('[data-testid^="slide-thumb-"][data-slide-index]')).toHaveCount(3)
  expect(await thumbLabels(page)).toEqual(['Alpha', 'Bravo', 'Charlie'])
  // Slide numbers 1,2,3.
  await expect(page.getByTestId('slide-thumb-num-0')).toHaveText('1')
  await expect(page.getByTestId('slide-thumb-num-1')).toHaveText('2')
  await expect(page.getByTestId('slide-thumb-num-2')).toHaveText('3')
  // Slide 1 is a title slide (T badge) + has notes (📝 badge).
  await expect(page.getByTestId('slide-thumb-title-0')).toBeVisible()
  await expect(page.getByTestId('slide-thumb-notes-0')).toBeVisible()

  await page.waitForTimeout(400)
  await page.screenshot({ path: join(SHOTS, '02-overview-grid.png') })
  ctx.assertNoJsErrors()
})

test('3) click thumbnail 3 → Present jumps to slide 3 and the grid closes', async () => {
  const { page } = ctx
  await page.getByTestId('slide-thumb-2').click()
  // Grid closed.
  await expect(page.getByTestId('slide-overview')).toHaveCount(0, { timeout: 5_000 })
  // Present jumped to slide 3.
  expect(await counterText(page)).toBe('3 / 3')
  const active = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(active.locator('h2')).toHaveText(/Charlie/)
  await page.screenshot({ path: join(SHOTS, '03-jumped-to-slide3.png') })
  ctx.assertNoJsErrors()
})

test('4) re-open the overview and reorder slide 3 → position 1', async () => {
  const { page } = ctx
  // Re-open the overview (via the `O` key this time).
  await page.keyboard.press('o')
  await expect(page.getByTestId('slide-overview')).toBeVisible({ timeout: 5_000 })
  expect(await thumbLabels(page)).toEqual(['Alpha', 'Bravo', 'Charlie'])

  // Drag slide 3 (Charlie, index 2) onto slide 1 (index 0). HTML5 DnD via a
  // synthetic dragstart/dragover/drop sequence (Playwright's dragTo is flaky on
  // HTML5 DnD, so we dispatch the events directly like the other report specs).
  await page.evaluate(() => {
    const from = document.querySelector('[data-testid="slide-thumb-2"]') as HTMLElement
    const to = document.querySelector('[data-testid="slide-thumb-0"]') as HTMLElement
    const dt = new DataTransfer()
    const fire = (el: HTMLElement, type: string) => {
      const r = el.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      el.dispatchEvent(ev)
    }
    fire(from, 'dragstart')
    fire(to, 'dragover')
    fire(to, 'drop')
    fire(from, 'dragend')
  })

  // The report_state re-emits and the grid re-renders with the new order:
  // Charlie is now slide 1.
  await expect.poll(async () => (await thumbLabels(page)).join(','), {
    timeout: 8_000, message: 'slide order did not change after report_move_slide',
  }).toBe('Charlie,Alpha,Bravo')

  await page.waitForTimeout(400)
  await page.screenshot({ path: join(SHOTS, '04-reordered.png') })
  ctx.assertNoJsErrors()
})

test('5) close the overview + exit Present mode', async () => {
  const { page } = ctx
  await page.getByTestId('slide-overview-close').click()
  await expect(page.getByTestId('slide-overview')).toHaveCount(0, { timeout: 5_000 })
  // Present mode still up.
  await expect(page.getByTestId('present-mode')).toBeVisible()
  // The reordered deck: slide 1 is now Charlie.
  const active = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(active.locator('h2')).toHaveText(/Charlie/, { timeout: 5_000 })
  // ESC exits Present mode entirely.
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('present-mode')).toHaveCount(0, { timeout: 5_000 })
  ctx.assertNoJsErrors()
})

test('6) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[overview] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
