/**
 * strain_lazy.spec.ts — Strain Mapping end-to-end on a found-vectors result.
 *   load_test_vectors → click the "Strain Mapping" toolbar action on the vectors
 *   signal window → the Strain CARET (wizard) opens and runs the live field → a
 *   Strain map window opens with the εxx/εyy/εxy/ω component toggle. The caret
 *   carries Method (Region/CIF), Match radius, and Submit; Submit commits a new
 *   signal tree.
 *
 * Runs with SPYDE_NO_DASK=1 (threaded scheduler): this dev box's per-worker
 * process spawn for a real LocalCluster is extremely slow (Windows re-imports
 * the full hyperspy/pyxem/torch stack per worker), which blows out even a tiny
 * 6x6x32x32 Find Vectors compute to 60-95s. Strain's own logic doesn't care
 * whether dask is threaded or distributed, so this keeps the test fast.
 *
 * Uses the shared _harness.cjs launcher (signal-based waits, not ad-hoc string
 * matching) — see CLAUDE.md "Verify by RUNNING THE APP".
 *
 * The strain physics + the selection-overlay logic are covered headless; this
 * verifies the UI wiring: action → caret → strain window + component toggle + Submit.
 */
import { test, expect } from '@playwright/test'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  const { page } = ctx
  // launchApp's backend-ready signal can land slightly before the Python
  // stdin reader loop is actually pumping messages (best-effort log match,
  // not a hard synchronization point) — a settle wait here avoids the action
  // being silently dropped, same pattern other specs in this suite use.
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_vectors')
  await waitForSubwindowCount(page, 4, 60_000)
  await page.waitForTimeout(2500)
})

test.afterAll(async () => { await ctx?.app?.close() })
test.setTimeout(120_000)

test('Strain Mapping: caret opens, runs the field, and Submit commits a new tree', async () => {
  const { page } = ctx
  // The vectors SIGNAL window carries the Strain Mapping action button.
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Strain Mapping') }).first()
  await vsig.getByTestId('subwindow-titlebar').click()    // raise
  await vsig.getByTestId('subwindow-titlebar').hover()    // reveal toolbar
  const btn = vsig.getByTestId('action-btn-Strain Mapping')
  await expect(btn).toBeVisible({ timeout: 15_000 })

  const before = await page.getByTestId('subwindow').count()
  await btn.click()

  // The Strain caret (wizard) opens with Method + Match radius + Submit.
  await expect(page.getByTestId('strain-wizard')).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('strain-method')).toBeVisible()
  await expect(page.getByTestId('strain-match-radius')).toBeVisible()
  await expect(page.getByTestId('strain-submit')).toBeVisible()

  // Opening the caret runs the live field → a Strain map window opens with the
  // εxx/εyy/εxy/ω component toggle (component swap dispatches strain_set_component).
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 60_000, message: 'strain map window never opened',
  }).toBeGreaterThan(before)
  const swin = page.getByTestId('subwindow').filter({ has: page.getByTestId(/^strain-toggle-/) }).first()
  await expect(swin.getByTestId(/^strain-comp-exx-/)).toBeVisible({ timeout: 15_000 })
  for (const c of ['eyy', 'exy', 'omega']) {
    await expect(swin.getByTestId(new RegExp(`^strain-comp-${c}-`))).toBeVisible()
  }
  // The freshly-opened Strain window must be focused/topmost — otherwise an
  // overlapping earlier window's iframe covers its component-toggle buttons
  // and this click silently fails to land.
  await swin.getByTestId(/^strain-comp-eyy-/).click({ timeout: 10_000 })

  // Submit freezes the field as a NEW committed signal tree → one more window.
  const beforeCommit = await page.getByTestId('subwindow').count()
  await page.getByTestId('strain-submit').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 30_000, message: 'Submit did not open a committed strain window',
  }).toBeGreaterThan(beforeCommit)

  ctx.assertNoJsErrors()
})
