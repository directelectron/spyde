/**
 * drift_wizard.spec.ts — the Drift Correction caret, end-to-end on the bundled
 * synthetic particle movie (whose per-frame drift is ground truth, stamped into
 * `metadata.Spyde.synthetic`).
 *
 * What this proves that tsc + headless tests cannot:
 *   1. The caret's default face is SMALL (plan §0.9a) — two toggles, one
 *      readout, one button, and a collapsed Advanced. The count is asserted,
 *      not eyeballed, because "too many options" is exactly what regressed.
 *   2. The discovery loop is real: a draggable box on the movie, a live
 *      drift-corrected sum of just that box in the Drift Check window, and a
 *      sharpness number that MOVES when the box moves.
 *   3. The dy/dx curve is its OWN window, opened by the solve and filled while
 *      it runs — not caret furniture.
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

/** Every interactive control the caret shows by default. If this list grows,
 *  §0.9a has been walked back and the review comment applies again. */
const FACE = ['drift-use-roi', 'drift-reject', 'drift-solve', 'drift-advanced-toggle']

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  // Enough frames that the solve takes about a second — a 12-frame movie
  // finishes before a screenshot can catch the dy/dx window mid-fill, which is
  // the thing this spec has to see.
  await backendAction(page, 'load_test_data_particles', { frames: 40 })
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test('the caret opens small: 2 toggles, 1 button, Advanced collapsed', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Drift Correction').click()
  await expect(page.getByTestId('drift-wizard')).toBeVisible()

  // The verification surface is a WINDOW, not the caret (plan A8): whole-movie
  // raw/corrected sums on top, the ROI discovery pair beneath.
  await waitForSubwindowCount(page, 3, 120_000)
  // …and the discovery preview lands on its own, with no click at all.
  await expect.poll(
    async () => await page.getByTestId('drift-roi-readout').getAttribute('data-gain'),
    { timeout: 120_000, message: 'the discovery preview never reported a gain' },
  ).toBeTruthy()
  await page.waitForTimeout(1500)
  await page.screenshot({ path: `${SHOTS}/01-caret-open.png` })
  await page.getByTestId('drift-wizard').screenshot({ path: `${SHOTS}/02-caret-face.png` })

  for (const id of FACE) await expect(page.getByTestId(id)).toBeVisible()
  await expect(page.getByTestId('drift-advanced')).toHaveCount(0)
  // Everything algorithmic is behind the disclosure.
  for (const id of ['drift-reference', 'drift-upsample', 'drift-max-shift',
    'drift-order', 'drift-tab-rigid', 'drift-apodize']) {
    await expect(page.getByTestId(id)).toHaveCount(0)
  }
  // Count what a user actually sees: buttons + inputs inside the caret.
  const shown = await page.getByTestId('drift-wizard')
    .locator('button, input, [data-testid$="-trigger"]').count()
  expect(shown, 'controls visible on the default face').toBeLessThanOrEqual(5)

  await page.getByTestId('drift-solve').click({ trial: true })   // enabled
  ctx.assertNoJsErrors()
})

test('Advanced holds the algorithm; non-rigid is selectable, affine stays locked', async () => {
  const { page } = ctx
  await page.getByTestId('drift-advanced-toggle').click()
  await expect(page.getByTestId('drift-advanced')).toBeVisible()
  await expect(page.getByTestId('drift-tab-rigid_affine')).toBeDisabled()
  // Non-rigid is IMPLEMENTED (spyde.drift.nonrigid). This assertion is the one
  // that catches the trap that actually happened: the backend's _UNAVAILABLE
  // list is duplicated in the renderer, so implementing the solve while leaving
  // the tab locked here made a finished feature unreachable — with every
  // headless test still green, because none of them can see a disabled tab.
  await expect(page.getByTestId('drift-tab-nonrigid')).toBeEnabled()
  await page.getByTestId('drift-wizard').screenshot({ path: `${SHOTS}/03-advanced.png` })

  await page.getByTestId('drift-max-shift').fill('24')
  await page.getByTestId('drift-max-shift').blur()
  await page.getByTestId('drift-advanced-toggle').click()
  await expect(page.getByTestId('drift-advanced')).toHaveCount(0)
  ctx.assertNoJsErrors()
})

test('selecting Non-rigid reveals its field controls and the choice sticks', async () => {
  const { page } = ctx
  await page.getByTestId('drift-advanced-toggle').click()
  await expect(page.getByTestId('drift-advanced')).toBeVisible()

  // Under RIGID the non-rigid controls must not be on screen at all — they mean
  // nothing to a rigid solve, and §0.9a is that the caret shows the task.
  await expect(page.getByTestId('drift-nonrigid-model')).toHaveCount(0)

  await page.getByTestId('drift-tab-nonrigid').click()
  await expect(page.getByTestId('drift-nonrigid-model')).toBeVisible()
  await expect(page.getByTestId('drift-nonrigid-steps')).toBeVisible()
  await page.getByTestId('drift-wizard')
    .screenshot({ path: `${SHOTS}/10-nonrigid-selected.png` })

  // The tab follows what the BACKEND selected, not what was clicked — so this
  // failing means the backend refused it and silently stayed on rigid.
  await expect(page.getByTestId('drift-tab-nonrigid')).toHaveAttribute('aria-selected', 'true')

  // Both parameterisations reach the backend. NB `Select` renders the themed
  // Dropdown, not a native <select>, so `selectOption`/`inputValue` do nothing
  // here — click the trigger, then the option, and read `data-value`.
  await expect(page.getByTestId('drift-nonrigid-model'))
    .toHaveAttribute('data-value', 'scan_knot')
  await page.getByTestId('drift-nonrigid-model').click()
  await page.getByTestId('drift-nonrigid-model-opt-dense').click()
  await expect(page.getByTestId('drift-nonrigid-model'))
    .toHaveAttribute('data-value', 'dense')
  await page.getByTestId('drift-wizard')
    .screenshot({ path: `${SHOTS}/11-nonrigid-dense.png` })
  ctx.assertNoJsErrors()
})

test('a Non-rigid solve actually runs the non-rigid fit, not a quiet fallback', async () => {
  const { page, backend } = ctx
  // The caret showing "Non-rigid" proves nothing about what the SOLVER did:
  // drift_run falls back to rigid whenever the fit cannot run (no torch, OOM,
  // a device that will not take the graph) and that fallback is deliberate.
  // So assert on the backend's own report of the KIND it produced.
  await page.getByTestId('drift-solve').click()
  await backend.waitForLog('non-rigid fit:', 120_000)
  const done = await backend.waitForLog('non-rigid fit done:', 300_000)
  expect(String(done), 'the solve fell back to rigid').toContain('kind=dense')
  await page.screenshot({ path: `${SHOTS}/12-nonrigid-solved.png` })
  expect(backendErrorLines(backend), 'backend errors during the non-rigid solve')
    .toEqual([])
  ctx.assertNoJsErrors()
})

test('dragging the ROI re-solves the preview and moves the sharpness number', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  const gain = () => page.getByTestId('drift-roi-readout').getAttribute('data-gain')
  const before = await gain()
  expect(before, 'no gain before the drag').toBeTruthy()

  // Drag the box's centre a long way — the widget lives inside the signal
  // window's figure iframe, so this is a real pointer drag on real pixels.
  const box = await sig.locator('iframe').first().boundingBox()
  expect(box).toBeTruthy()
  const cx = box!.x + box!.width / 2, cy = box!.y + box!.height / 2
  await page.mouse.move(cx, cy)
  await page.mouse.down()
  for (let i = 1; i <= 8; i++) {
    await page.mouse.move(cx - i * 6, cy - i * 4)
    await page.waitForTimeout(30)
  }
  await page.screenshot({ path: `${SHOTS}/04-roi-mid-drag.png` })
  await page.mouse.up()

  await expect.poll(async () => await gain(),
    { timeout: 90_000, message: 'the preview never re-solved after the drag' },
  ).not.toBe(before)
  await page.waitForTimeout(1500)
  await page.screenshot({ path: `${SHOTS}/05-roi-settled.png` })
  ctx.assertNoJsErrors()
})

test('Correct Drift opens the dy/dx window and fills it, then Apply lands the node', async () => {
  const { page } = ctx
  await page.getByTestId('drift-solve').click()

  // The curve is its OWN window (plan §0.9a), opened by the solve and filled
  // from the on_shift stream — so it has points BEFORE the solve finishes.
  await waitForSubwindowCount(page, 4, 120_000)
  await expect(page.getByTestId('drift-progress')).toBeVisible({ timeout: 60_000 })
  await page.screenshot({ path: `${SHOTS}/06-trace-filling.png` })

  await expect(page.getByTestId('drift-result')).toBeVisible({ timeout: 180_000 })
  await expect(page.getByTestId('drift-result')).toContainText('px drift')
  await expect(page.getByTestId('drift-status')).toContainText('Solved')
  await page.waitForTimeout(2000)
  await page.getByTestId('drift-wizard').screenshot({ path: `${SHOTS}/07-solved-caret.png` })
  await page.screenshot({ path: `${SHOTS}/08-solved-full.png` })

  // Apply adds the LAZY corrected node to the tree (map_blocks, nothing copied)
  // and shows it — so it appears in the Plot Control workflow list.
  await page.getByTestId('drift-commit').click()
  await expect(page.getByTestId('tree-node-Drift corrected')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('status-text')).toContainText('Drift corrected node added')
  await page.waitForTimeout(2000)
  await page.screenshot({ path: `${SHOTS}/09-applied.png` })

  const errors = backendErrorLines(ctx.backend)
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
  ctx.assertNoJsErrors()
})
