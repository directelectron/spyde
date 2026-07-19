/**
 * presenter_view.spec.ts — Report Builder presenter view + speaker notes, e2e.
 *
 * Real Dask + bundled-synthetic Si-grains. Builds a 2-slide deck, sets SPEAKER
 * NOTES on slide 1 (via report_set_slide_notes), enters Present mode, then toggles
 * the PRESENTER VIEW (S key / the header button) and asserts the presenter
 * dashboard shows: the notes text, a running timer, the NEXT-slide preview, and
 * the slide position — and that arrowing advances BOTH views in sync. Also asserts
 * the clean AUDIENCE view (presenter OFF) does NOT show the notes.
 *
 * Screenshots each stage to presenter_shots/ and the author Reads them (a blank
 * dashboard is a failure even when selectors pass).
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'presenter_shots')

const NOTES_1 = 'REMEMBER: mention the beamstop, then dwell 5 seconds before advancing.'

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)   // navigator + signal
  await page.waitForTimeout(2000)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally {
    await ctx?.app?.close()
  }
})

test('1) build a 2-slide deck and set speaker notes on slide 1', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()

  await backendAction(page, 'report_new', {})
  await backendAction(page, 'report_set_title', { title: 'Presenter View Demo' })
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '# Slide One\n\nThe opening slide of the talk.',
    html: '<h1>Slide One</h1><p>The opening slide of the talk.</p>',
  })
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown',
    source: '## Slide Two\n\n- second point\n- third point',
    html: '<h2>Slide Two</h2><ul><li>second point</li><li>third point</li></ul>',
    slide_break: true,
  })

  // Wait for the first markdown cell to render in the sidebar, then resolve its
  // id from the rendered-body testid (report-cell-rendered-<id>) — a stable,
  // unambiguous handle to the cell id.
  const rendered = page.locator('[data-testid^="report-cell-rendered-"]').first()
  await expect(rendered).toBeVisible({ timeout: 10_000 })
  const firstId: string = await rendered.evaluate((el) =>
    (el.getAttribute('data-testid') || '').replace('report-cell-rendered-', ''))
  expect(firstId).toBeTruthy()
  await backendAction(page, 'report_set_slide_notes', { cell_id: firstId, notes: NOTES_1 })
  await page.waitForTimeout(600)

  await page.screenshot({ path: join(SHOTS, '01-deck-with-notes.png') })
  ctx.assertNoJsErrors()
})

test('2) enter Present mode — audience slide shows NO notes', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await expect(page.getByTestId('present-mode')).toBeVisible({ timeout: 10_000 })
  const active = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(active.locator('h1')).toHaveText(/Slide One/)
  // The audience view (presenter OFF) must NOT contain the notes text anywhere.
  const audienceText = (await page.getByTestId('present-mode').innerText()) || ''
  expect(audienceText).not.toContain('REMEMBER')
  // The presenter dashboard is not shown yet.
  await expect(page.getByTestId('presenter-view')).toHaveCount(0)
  await page.screenshot({ path: join(SHOTS, '02-audience-slide1.png') })
  ctx.assertNoJsErrors()
})

test('3) toggle presenter view (S) — dashboard shows notes + timer + next', async () => {
  const { page } = ctx
  await page.keyboard.press('s')
  await expect(page.getByTestId('presenter-view')).toBeVisible({ timeout: 5_000 })

  // The current slide's SPEAKER NOTES render (the big readable panel).
  const notesBody = page.getByTestId('presenter-notes-body')
  await expect(notesBody).toBeVisible()
  await expect(notesBody).toContainText('REMEMBER: mention the beamstop')

  // A running TIMER (mm:ss) is present.
  const timer = page.getByTestId('presenter-timer')
  await expect(timer).toBeVisible()
  await expect(timer).toHaveText(/^\d{2}:\d{2}$/)

  // The NEXT slide preview shows Slide Two.
  const nextPreview = page.getByTestId('presenter-next-preview')
  await expect(nextPreview).toBeVisible()
  await expect(nextPreview).toContainText('Slide Two')

  // The current-slide preview shows Slide One.
  await expect(page.getByTestId('presenter-current')).toContainText('Slide One')

  // Slide position n / N.
  await expect(page.getByTestId('presenter-counter')).toHaveText('1 / 2')

  await page.waitForTimeout(1200)   // let the timer tick past 00:00
  await page.screenshot({ path: join(SHOTS, '03-presenter-dashboard.png') })
  ctx.assertNoJsErrors()
})

test('4) the timer is actually ticking', async () => {
  const { page } = ctx
  const t0 = (await page.getByTestId('presenter-timer').textContent())?.trim() ?? ''
  await page.waitForTimeout(2200)
  const t1 = (await page.getByTestId('presenter-timer').textContent())?.trim() ?? ''
  expect(t1).not.toBe('00:00')
  expect(t1).not.toBe(t0)     // it advanced
  ctx.assertNoJsErrors()
})

test('5) arrowing advances BOTH the presenter dashboard and the audience slide', async () => {
  const { page } = ctx
  await page.keyboard.press('ArrowRight')
  // Presenter dashboard now on slide 2.
  await expect(page.getByTestId('presenter-counter')).toHaveText('2 / 2')
  await expect(page.getByTestId('presenter-current')).toContainText('Slide Two')
  // Slide 2 has no notes.
  await expect(page.getByTestId('presenter-notes-empty')).toBeVisible()
  // The NEXT preview shows "End of deck" (last slide).
  await expect(page.getByTestId('presenter-next-preview')).toContainText('End of deck')

  await page.screenshot({ path: join(SHOTS, '04-presenter-slide2.png') })

  // Toggle presenter OFF (S) → the audience slide is ALSO on slide 2 (in sync).
  await page.keyboard.press('s')
  await expect(page.getByTestId('presenter-view')).toHaveCount(0)
  await expect(page.getByTestId('present-counter')).toHaveText('2 / 2')
  const active = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(active.locator('h2')).toHaveText(/Slide Two/)
  await page.screenshot({ path: join(SHOTS, '05-audience-synced-slide2.png') })
  ctx.assertNoJsErrors()
})

test('6) header presenter-toggle button also works; ESC exits', async () => {
  const { page } = ctx
  await page.getByTestId('present-presenter-toggle').click()
  await expect(page.getByTestId('presenter-view')).toBeVisible({ timeout: 5_000 })
  // Back on slide 2, no notes.
  await expect(page.getByTestId('presenter-counter')).toHaveText('2 / 2')
  await page.screenshot({ path: join(SHOTS, '06-presenter-via-button.png') })

  await page.keyboard.press('Escape')
  await expect(page.getByTestId('present-mode')).toHaveCount(0, { timeout: 5_000 })
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  ctx.assertNoJsErrors()
})

test('7) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[presenter] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
