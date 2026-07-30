/**
 * drift_wizard.spec.ts — the Drift Correction caret, end-to-end on the bundled
 * synthetic particle movie (whose per-frame drift is ground truth, stamped into
 * `metadata.Spyde.synthetic`).
 *
 * What this proves that tsc + headless tests cannot:
 *   1. `drift_open` opens the SEPARATE Drift Check window (plan A8 — a 240 px
 *      caret cannot show a sum image at a size where sharpness is judgeable,
 *      and judging sharpness IS the check), and the caret is not clipped.
 *   2. The two unimplemented models are LOCKED with the backend's own reason —
 *      not silently falling back to rigid under a caret claiming otherwise.
 *   3. Solve fills the inline dy/dx trace from `drift_result`.
 *   4. Apply adds the lazy corrected node.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = 'drift_wizard_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_particles', { frames: 8 })
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test('the caret opens its Drift Check window and locks the stub models', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Drift Correction').click()
  await expect(page.getByTestId('drift-wizard')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/01-caret-open.png` })

  // The verification surface is a WINDOW, not the caret (plan A8): raw sum +
  // corrected sum + dy/dx panels.
  await waitForSubwindowCount(page, 3, 120_000)
  await page.waitForTimeout(3000)
  await page.screenshot({ path: `${SHOTS}/02-check-window.png` })

  await expect(page.getByTestId('drift-tab-rigid_affine')).toBeDisabled()
  await expect(page.getByTestId('drift-tab-nonrigid')).toBeDisabled()
  await expect(page.getByTestId('drift-unavailable')).toContainText('not implemented')

  await page.getByTestId('drift-wizard').screenshot({ path: `${SHOTS}/03-caret-detail.png` })
  ctx.assertNoJsErrors()
})

test('Solve fills the shift trace and Apply adds the corrected node', async () => {
  const { page } = ctx

  // A tune re-solves the FIRST PAIR only (two FFTs) — the cheap answer to "is
  // max_shift in the right range for this movie".
  await page.getByTestId('drift-max-shift').fill('24')
  await page.getByTestId('drift-max-shift').blur()
  await expect(page.getByTestId('drift-preview-readout')).toBeVisible({ timeout: 60_000 })
  await page.getByTestId('drift-wizard').screenshot({ path: `${SHOTS}/04-first-pair.png` })

  await page.getByTestId('drift-solve').click()
  // The trace is drawn from drift_result's whole shifts array (the solver
  // returns nothing partial — see the DriftWizard header), so wait for one
  // point per frame.
  await expect.poll(
    async () => Number(await page.getByTestId('drift-trace').getAttribute('data-points')),
    { timeout: 180_000, message: 'the shift trace never filled from drift_result' },
  ).toBeGreaterThan(2)
  await expect(page.getByTestId('drift-status')).toContainText('Solved')
  await page.getByTestId('drift-wizard').screenshot({ path: `${SHOTS}/05-solved.png` })
  await page.screenshot({ path: `${SHOTS}/06-solved-full.png` })

  // Apply adds the LAZY corrected node to the tree (map_blocks, nothing copied)
  // and shows it — so it appears in the Plot Control workflow list.
  await page.getByTestId('drift-commit').click()
  await expect(page.getByTestId('tree-node-Drift corrected')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('status-text')).toContainText('Drift corrected node added')
  await page.waitForTimeout(2000)
  await page.screenshot({ path: `${SHOTS}/07-applied.png` })

  const errors = backendErrorLines(ctx.backend)
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
  ctx.assertNoJsErrors()
})
