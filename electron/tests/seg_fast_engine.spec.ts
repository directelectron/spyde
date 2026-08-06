/**
 * seg_fast_engine.spec.ts — the FAST engine, driven through the real caret.
 *
 * `FastScribbleClassifier` replaced the 34-channel stack + 64-wide MLP so a whole
 * 4096² frame can be segmented at interactive rates. Everything downstream
 * (`split_instances`, `measure_frame`, the batch fan-out) is unchanged, so what
 * needs proving in the running app is the seam:
 *
 *   1. Training through the real Train button dispatches the FAST engine
 *      (`fit` logs "fast engine trained"), rather than silently falling back.
 *   2. It finds particles and DRAWS them.
 *   3. A committed run reaches the particle tree, i.e. the batch path can load a
 *      head this engine saved (the `_engine` stamp in the .npz).
 *
 * Screenshots land in `seg_fast_shots/` and are the actual verification; a green
 * assertion here without a look at the pixels is not.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
import { openCaret, trainFromGroundTruth, windowIdOf } from './_seg'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = 'seg_fast_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(600_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await ctx.page.waitForTimeout(1500)
})

test.afterAll(async () => { await ctx?.close?.() })

test('the fast engine trains and segments through the caret', async () => {
  const { page } = ctx
  // 256² stays under the preview pixel budget so the preview segments the WHOLE
  // frame, and `particle_movie`'s particles sit at their original 16..102 px
  // coordinates, so they are inside it (see seg_overlay.spec.ts for why that
  // matters and how a larger frame silently moves them out of the crop).
  await backendAction(page, 'load_test_data_particles', { size: [256, 256] })
  await waitForSubwindowCount(page, 2, 180_000)
  // The window appears before the data paints — the navigator is still being
  // computed at this point, and clicking the toolbar through that leaves the
  // caret unopened. seg_overlay.spec.ts settles for the same reason.
  await page.waitForTimeout(6000)
  const sig = sigWindow(page)
  await page.screenshot({ path: `${SHOTS}/01-loaded.png` })

  await openCaret(page, sig)
  await page.screenshot({ path: `${SHOTS}/02-caret.png` })

  const windowId = await windowIdOf(sig)
  await trainFromGroundTruth(page, windowId)
  await page.screenshot({ path: `${SHOTS}/03-trained-and-previewed.png` })

  // Proof the engine actually dispatched: `fit` logs "fast engine trained: ..."
  // and the harness tees `logging` to stderr at INFO.
  // logBuffer is an ARRAY of lines, so join it — `toContain` on an array is an
  // exact element match and would never fire on a substring.
  const log = (ctx.backend.logBuffer as string[]).join('\n')
  expect(log, 'the fast engine did not run — something fell back to the old one')
    .toContain('fast engine trained')

  // Mask-only preview: coverage is the result, and count is -1 by design.
  const stats = page.getByTestId('seg-preview-stats')
  const coverage = Number(await stats.getAttribute('data-coverage'))
  const count = Number(await stats.getAttribute('data-count'))
  expect(count, 'the live preview should not be counting instances').toBe(-1)
  expect(coverage, 'the mask is empty').toBeGreaterThan(0)
  expect(coverage, 'the mask covers the whole frame — it learnt "everything"')
    .toBeLessThan(0.35)
  const ms = Number(await stats.getAttribute('data-elapsed') ?? 0)
  console.log(`[spec] mask preview: coverage ${(coverage * 100).toFixed(1)}%` +
    (ms ? `, ${ms} ms` : ''))

  expect(backendErrorLines(ctx.backend)).toEqual([])
})

test('a run commits to a particle tree', async () => {
  const { page } = ctx
  await page.getByTestId('seg-run').click()
  // The batch opens its window early with a placeholder and attaches
  // `tree.particles` only at _finalize, so wait for the real completion signal
  // rather than a fixed sleep (CLAUDE.md's find-vectors timing trap, same shape).
  // The window count is NOT the completion signal: the batch opens its result
  // window immediately with a "Calculating…" placeholder and attaches
  // `tree.particles` only at _finalize. Waiting on the count alone screenshots a
  // placeholder and calls it a pass — the find-vectors timing trap in CLAUDE.md,
  // same shape. Wait for the placeholder to go away.
  await waitForSubwindowCount(page, 4, 480_000)
  // `_finalize` emits exactly this once the batch has landed. Do NOT assert on
  // the "Calculating…" placeholder going away — it is drawn inside the figure,
  // not in the main-frame DOM, so `getByText` matches nothing and toBeHidden
  // passes on a run that never finished. (It did, and screenshotted a
  // placeholder.)
  // `emit_status` is the PLOTAPP line protocol, consumed by the main process —
  // it never reaches the harness log buffer, so waiting on logBuffer for it
  // times out even on a healthy run (CLAUDE.md). The status bar IS the DOM.
  await expect
    .poll(() => page.locator('body').innerText(),
      { timeout: 480_000, message: 'the batch never reported completion' })
    .toMatch(/Found \d+ particles in \d+ frames/)
  await page.waitForTimeout(2500)
  await page.screenshot({ path: `${SHOTS}/04-after-run.png` })
  expect(backendErrorLines(ctx.backend)).toEqual([])
})
