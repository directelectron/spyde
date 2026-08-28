/**
 * report_delete_undo.spec.ts — deleting a cell must take TWO clicks, and must
 * be undoable.
 *
 * The bug this exists for: `CellChrome` is shown on `hover || showEditor`, so
 * opening a figure's edit toolbar MAKES its delete ✕ appear — top-right, where
 * the editor's own Close × also sits. One click destroyed the slide, and the
 * report system had no undo of any kind, so the content was simply gone.
 *
 * Two independent guarantees, tested separately because they fail separately:
 *   1. the first click ARMS (label → "Delete?"), only the second deletes
 *   2. a completed delete can be undone, restoring the cell in place
 *
 * Screenshots to report_delete_undo_shots/ — each Read by the author.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_delete_undo_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2000)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

/** The rendered text of every markdown cell, in order. */
async function texts(page: any): Promise<string[]> {
  return await page.locator('[data-testid^="report-cell-rendered-"]')
    .allInnerTexts().then((t: string[]) => t.map(s => s.trim()))
}

/**
 * Add a cell and commit `body` into it.
 *
 * Every step waits for the state the NEXT one reaches for, because each step
 * addresses "the last cell" and is therefore wrong the moment it runs early:
 * dblclick before the new cell mounts opens the PREVIOUS cell's editor, and
 * returning before the commit renders means the following call races it. That
 * happened to hold together on Electron 34 and stopped doing so on 44 — the
 * cells came back as ['Beta', empty, empty]. The app was never at fault; the
 * assumption that a click had landed by the next line was.
 */
async function addText(page: any, body: string) {
  const before = await page.locator('[data-testid^="report-cell-rendered-"]').count()
  await page.getByTestId('report-add-text').click()
  await expect
    .poll(async () => page.locator('[data-testid^="report-cell-rendered-"]').count())
    .toBeGreaterThan(before)

  await page.locator('[data-testid^="report-cell-rendered-"]').last().dblclick()
  const ta = page.locator('[data-testid^="report-cell-textarea-"]').last()
  await expect(ta).toBeVisible()
  await ta.fill(body)
  await ta.press('Control+Enter')
  await expect(page.locator('[data-testid^="report-cell-rendered-"]').last())
    .toHaveText(body, { timeout: 10_000 })
}

test('1) a report with three text cells', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await backendAction(page, 'report_new', {})
  await expect(page.getByTestId('report-body')).toBeVisible()
  for (const s of ['Alpha', 'Beta', 'Gamma']) await addText(page, s)
  expect(await texts(page)).toEqual(['Alpha', 'Beta', 'Gamma'])
  await page.screenshot({ path: join(SHOTS, '01-three-cells.png') })
})

test('2) ONE click on ✕ arms it and deletes nothing', async () => {
  const { page } = ctx
  const cell = page.locator('[data-testid^="report-cell-rendered-"]').nth(1)
  await cell.hover()
  const del = page.locator('[data-testid^="report-cell-delete-"]').first()
  await expect(del).toBeVisible()

  await del.click()
  // Armed: the label changes, which is the signal a recolour alone would not
  // give on a small glyph in a dark UI.
  await expect(del).toHaveAttribute('data-armed', 'true')
  await expect(del).toHaveText('Delete?')
  expect(await texts(page), 'the first click must not delete').toEqual(
    ['Alpha', 'Beta', 'Gamma'])
  await page.screenshot({ path: join(SHOTS, '02-armed.png') })
})

test('3) moving the pointer away disarms it', async () => {
  const { page } = ctx
  // Leaving the chrome cancels — an armed button must not survive to ambush the
  // next time the user reaches for Copy or Duplicate.
  await page.mouse.move(5, 5)
  await page.waitForTimeout(300)
  const cell = page.locator('[data-testid^="report-cell-rendered-"]').nth(1)
  await cell.hover()
  const del = page.locator('[data-testid^="report-cell-delete-"]').first()
  await expect(del).toHaveAttribute('data-armed', 'false')
  expect(await texts(page)).toEqual(['Alpha', 'Beta', 'Gamma'])
})

test('4) the SECOND click deletes, and Undo brings it back in place', async () => {
  const { page } = ctx
  const cell = page.locator('[data-testid^="report-cell-rendered-"]').nth(1)
  await cell.hover()
  const del = page.locator('[data-testid^="report-cell-delete-"]').first()
  await del.click()                                   // arm
  await expect(del).toHaveAttribute('data-armed', 'true')
  await del.click()                                   // confirm
  await expect
    .poll(async () => await texts(page), { timeout: 10_000 })
    .toEqual(['Alpha', 'Gamma'])
  await page.screenshot({ path: join(SHOTS, '03-deleted.png') })

  // The Undo button appears only once there is something to undo — its arrival
  // is itself the feedback that the delete landed.
  const undo = page.getByTestId('report-undo')
  await expect(undo).toBeVisible({ timeout: 10_000 })
  await page.screenshot({ path: join(SHOTS, '04-undo-offered.png') })

  await undo.click()
  await expect
    .poll(async () => await texts(page), { timeout: 10_000 })
    .toEqual(['Alpha', 'Beta', 'Gamma'])              // back IN ITS SLOT, not at the end
  await expect(undo).toBeHidden({ timeout: 10_000 })  // stack now empty
  await page.screenshot({ path: join(SHOTS, '05-undone.png') })
  ctx.assertNoJsErrors()
})
