/**
 * multiangle_dialog_e2e.spec.ts — the loader dialog driving the REAL backend.
 *
 * The dialog's own spec runs against injected `maped_state` snapshots, which
 * proves the UI renders them but not that any of it is reachable. This one
 * walks the whole thing against live Python: write real MRC members to disk,
 * add them through the dialog, supply the scan grid MRC cannot record, set the
 * tilts, solve both alignments, and commit — then check a real signal tree
 * opened on the other side.
 *
 * Screenshots land in electron/multiangle_dialog_shots/ because a green
 * assertion here says nothing about whether any of it drew (CLAUDE.md).
 */
import { test, expect, Page } from '@playwright/test'
import { join } from 'path'
import { tmpdir } from 'os'

const { launchApp } = require('./_harness.cjs')

let ctx: any
let page: Page

const SHOTS = join(__dirname, '..', 'multiangle_dialog_shots')
// Matches _write_test_multiangle_files: three members at 1.0°, two at 0.5°.
const MEMBER_DIRECTORY = join(tmpdir(), 'spyde-multiangle-test')
const TILTS = [1.0, 1.0, 1.0, 0.5, 0.5]
const AZIMUTHS = [0.0, 120.0, 240.0, 17.0, 197.0]
const SCAN = 28

const paths = () =>
  TILTS.map((_t, index) =>
    join(MEMBER_DIRECTORY, `angle${String(index).padStart(2, '0')}.mrc`))

const send = (action: string, payload: Record<string, unknown> = {}) =>
  page.evaluate(([a, p]) => window.electron.action(a as string, p as object),
    [action, payload] as const)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  page = ctx.page
  await page.waitForTimeout(1500)
  await send('write_test_multiangle_files', { nav: SCAN, sig: 32 })
  await page.waitForTimeout(2500)
})

test.afterAll(async () => { await ctx?.app?.close() })

test('the loader walks load → real space → reciprocal → open', async () => {
  // Two real solves over five members on a live backend.
  test.setTimeout(300_000)

  // Opened the way a user does — the dialog's visibility is renderer state,
  // and `maped_open_loader` only initialises the backend half.
  await page.getByTestId('menu-file').click()
  await page.getByTestId('menu-load-multiangle').click()
  const dialog = page.getByTestId('multiangle-loader')
  await expect(dialog).toBeVisible({ timeout: 20_000 })

  // ── Load: MRC records no scan grid, so every member must error first ─────
  await send('maped_add_files', { paths: paths() })
  await page.waitForTimeout(4000)
  await dialog.screenshot({ path: join(SHOTS, '01-added-no-scan-grid.png') })

  // The scan-grid fallback is the whole point of this step: supply it and the
  // same files re-probe into real members.
  await send('maped_set_scan_shape', { scan_shape: [SCAN, SCAN] })
  await page.waitForTimeout(6000)
  await dialog.screenshot({ path: join(SHOTS, '02-scan-grid-supplied.png') })

  // ── Tilts: MRC carries no angle metadata either, so the shells are ours ──
  for (let index = 0; index < TILTS.length; index += 1) {
    await send('maped_set_member',
      { index, tilt: TILTS[index], azimuth: AZIMUTHS[index] })
    await page.waitForTimeout(400)
  }
  await page.waitForTimeout(1500)
  await expect(dialog).toContainText('2 shells', { timeout: 20_000 })
  // The tableau is what you look at to DECIDE whether to run, so it has to
  // carry images before anything has been solved.
  await expect(dialog.locator('img').first())
    .toBeVisible({ timeout: 120_000 })
  await page.waitForTimeout(4000)
  await dialog.screenshot({ path: join(SHOTS, '03-shells.png') })
  const tiles = await dialog.locator('img').count()
  expect(tiles, 'no thumbnails on the tableau before solving').toBeGreaterThan(4)

  // ── Align real space ────────────────────────────────────────────────────
  await dialog.getByText('Align real space').click()
  await dialog.screenshot({ path: join(SHOTS, '04a-real-tab.png') })
  await send('maped_align_real', { params: { max_shift: 32 } })
  await page.waitForFunction(
    () => !document.body.innerText.includes('Aligning real space'),
    { timeout: 120_000 })
  await page.waitForTimeout(3000)
  await dialog.screenshot({ path: join(SHOTS, '04-real-solved.png') })
  await page.screenshot({ path: join(SHOTS, '04b-aligned-window.png') })

  // A solve that silently returned all zeros reads EXACTLY like a perfect
  // one — same "solved", same 0.00 px residual. The offsets are what separate
  // them, and the planted ones are not zero.
  const realOffsets = await dialog.innerText()
  expect(realOffsets).toMatch(/-?\d+\s*,\s*-?\d+/)
  expect(realOffsets.replace(/\s/g, '')).not.toMatch(/^(.*?)(0,0){5}/)

  // ── Align reciprocal space ──────────────────────────────────────────────
  await dialog.getByText('Align reciprocal space').click()
  await dialog.screenshot({ path: join(SHOTS, '05a-reciprocal-tab.png') })
  await send('maped_align_reciprocal', { method: 'corners', params: {} })
  // Waited on the real completion signal, not a sleep: Open is disabled until
  // the backend reports can_commit, so a fixed wait that is fractionally short
  // leaves a disabled button and a test that hangs rather than fails.
  const openButton = dialog.getByTestId('maped-open')
  await expect(openButton).toBeEnabled({ timeout: 120_000 })
  await dialog.screenshot({ path: join(SHOTS, '05-reciprocal-solved.png') })

  // ── Commit: CLICKED, not dispatched. Closing the dialog is the button's
  // doing, so sending the action behind its back tests a path no user takes.
  await openButton.click()
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 4,
    { timeout: 120_000 })
  await page.waitForTimeout(3000)
  await page.screenshot({ path: join(SHOTS, '06-committed.png') })

  await expect(page.getByTestId('multiangle-loader')).toHaveCount(0)
  await expect(page.getByTestId('signal-tree')).toContainText('Aligned Stack')
  await expect(page.getByTestId('signal-tree')).toContainText('Summed')

  // The composed scan must be SMALLER than a member's: the crop is the offset
  // spread, so a full-size scan means every offset was zero and the alignment
  // did nothing — which every other assertion here would still pass.
  const pills = await page.getByTestId('signal-tree').locator('..').innerText()
  const nav = pills.match(/nav\s+(\d+)\s*[×x]\s*(\d+)/)
  expect(nav, `no nav pill found in: ${pills.slice(0, 400)}`).not.toBeNull()
  expect(Number(nav![1])).toBeLessThan(SCAN)
})
