/**
 * _probe_fv_stall.spec.ts — DIAGNOSTIC probe for the "find-vectors batch never
 * completes without UI interaction" stall. Not part of the regular suite
 * (underscore prefix; run explicitly):
 *
 *   npx playwright test tests/_probe_fv_stall.spec.ts --project=electron
 *
 * Runs the batch and then does NOTHING except dump dask scheduler/worker state
 * every 60 s ([dask-state] lines land in the harness logBuffer at WARNING).
 * Read the tail of the output to see, at each dump, whether the chunk tasks
 * are executing (call stacks shown), queued, or absent — that localizes the
 * stall to worker-side execution vs scheduling vs submission.
 */
import { test, expect } from '@playwright/test'
const { launchApp, backendAction } = require('./_harness.cjs')

// Diagnostic only — several minutes of deliberate waiting. Opt in with
// SPYDE_PROBE=1; in a normal suite run it shows up as one skipped test.
test.skip(!process.env.SPYDE_PROBE, 'diagnostic probe — run with SPYDE_PROBE=1')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  await backendAction(page, 'load_test_data_si_grains')
  await expect.poll(() => page.getByTestId('subwindow').count(),
    { timeout: 120_000 }).toBeGreaterThanOrEqual(2)
})

test.afterAll(async () => {
  const buf = ctx?.backend?.logBuffer ?? []
  const interesting = buf.filter((l: string) =>
    /\[dask-state\]|\[fv-batch\]|\[fv-run\]|\[dask\]|\[timers\]|poke|dispatcher|loops:|keepalive|patched|plugin|Found|error|Error|Traceback/.test(l))
  console.log(`[probe] ${interesting.length} interesting log lines:`)
  interesting.forEach((l: string) => console.log('[BE]', l.trim()))
  await ctx?.app?.close()
})

test.setTimeout(600_000)

test('probe: batch health with zero interaction', async () => {
  const { page } = ctx
  // By what it HAS, not text negatives (hasNotText is case-insensitive and
  // the FV wizard hint contains "navigator").
  const sig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Find Diffraction Vectors') }).first()
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  const t0 = Date.now()
  await page.getByTestId('fv-compute').click()
  console.log('[probe] fv_run clicked at t=0')

  // STAGED-POKE EXPERIMENT (run with SPYDE_DASK_KEEPALIVE=0): fire each
  // client-call type individually and read from the [fv-batch] timestamps
  // which one unsticks task delivery. Known: full dump at t=X → compute done
  // t=X+4s; run_on_scheduler(noop)@1Hz alone did NOT unstick.
  const pokes: Array<[number, string]> = [
    [15_000, 'scheduler'], [30_000, 'info'], [45_000, 'call_stack'],
  ]
  let attached = false
  let next = 0
  for (let elapsed = 0; elapsed < 120_000; elapsed += 5_000) {
    await page.waitForTimeout(5_000)
    const t = ((Date.now() - t0) / 1000).toFixed(0)
    if (next < pokes.length && elapsed + 5_000 >= pokes[next][0]) {
      console.log(`[probe] t=${t}s firing poke only=${pokes[next][1]}`)
      await backendAction(page, 'dump_dask_state', { only: pokes[next][1] })
      next += 1
    }
    const n = await page.getByTestId('action-btn-Strain Mapping').count()
    console.log(`[probe] t=${t}s strain-btn-count=${n}`)
    if (n > 0) { attached = true; console.log(`[probe] ATTACHED at t=${t}s`); break }
  }
  console.log('[probe] final attached =', attached)
  expect(attached, 'batch never completed even after all poke types').toBe(true)
})
