/**
 * shutdown.spec.ts — the app must actually CLOSE, and in bounded time.
 *
 * Nothing else in the suite asserts this. Every other spec closes its app in
 * `afterAll`, where a slow or hung shutdown is invisible: Playwright attributes
 * it to the WORKER, not to any test, and surfaces it as
 * "Worker teardown timeout of 120000ms exceeded" plus a bare
 * "1 error was not a part of any test" — which fails the shard while every test
 * in it reports green. That is exactly how it presented in CI, and it cost a
 * long time to trace back to shutdown at all.
 *
 * A hung shutdown is also a real user-facing bug, not just a CI nuisance: it is
 * the backend refusing to exit when the window closes, leaving an orphaned
 * process holding the dataset.
 *
 * The cases run cheapest-first so a failure names the narrowest scenario that
 * breaks.
 */
import { test, expect } from '@playwright/test'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

/** Generous but far below Playwright's own 120 s worker-teardown limit — the
 *  point is to fail as a TEST, with a name, rather than as an unattributed
 *  worker error. */
const CLOSE_BUDGET_MS = 60_000

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

async function timeClose(ctx: any, label: string): Promise<number> {
  const t0 = Date.now()
  await ctx.app.close()
  const ms = Date.now() - t0
  console.log(`SHUTDOWN ${label} = ${ms} ms`)
  return ms
}

test('an idle session closes promptly', async () => {
  const ctx = await launchApp({ dask: true })
  // Let the cluster finish coming up. Closing DURING startup is a separate,
  // pre-existing hang — see the `fixme` at the bottom of this file.
  await ctx.page.waitForTimeout(12_000)
  expect(await timeClose(ctx, 'idle'),
    'an idle backend did not exit').toBeLessThan(CLOSE_BUDGET_MS)
})

test('a session closes after a navigator fill has finished', async () => {
  const ctx = await launchApp({ dask: true })
  await ctx.page.waitForTimeout(2000)
  await backendAction(ctx.page, 'load_test_data_movie')
  await waitForSubwindowCount(ctx.page, 2, 120_000)
  await ctx.page.waitForTimeout(10_000)          // let the fill complete
  expect(await timeClose(ctx, 'after-fill'),
    'a settled session with data loaded did not exit').toBeLessThan(CLOSE_BUDGET_MS)
})

// ── the two cases that used to HANG ────────────────────────────────────────
// Both of these wedged the backend forever until `DaskManager.shutdown` stopped
// falling back to an unbounded `cluster.close()`. They are the expensive cases:
// a cluster whose workers are mid-task, or still spawning, cannot close
// promptly, so these take ~15 s where idle takes ~2.6 s. That is the price of
// bounding it, and it is bounded — which is the whole point.
//
// Keep them: this is the only place that exercises shutdown against a cluster
// that is genuinely busy, and it is the shape a regression here would take.

test('closing WHILE a navigator fill is still running', async () => {
  const ctx = await launchApp({ dask: true })
  await ctx.page.waitForTimeout(2000)
  await backendAction(ctx.page, 'load_test_data_movie')
  await waitForSubwindowCount(ctx.page, 2, 120_000)
  // Deliberately do NOT wait for the fill. Workers that are mid-task cannot
  // close politely, so this is the path that used to wedge: the bounded
  // `cluster.close(timeout=2)` raises, and the old code retried it WITHOUT a
  // timeout. ~15 s here against ~2.6 s idle is the cost of giving up on a busy
  // cluster and letting `process_guard` reap it.
  expect(await timeClose(ctx, 'mid-fill'),
    'closing mid-fill hung: cluster.close() is unbounded again')
    .toBeLessThan(CLOSE_BUDGET_MS)
})

test('closing DURING cluster startup', async () => {
  const ctx = await launchApp({ dask: true })
  await ctx.page.waitForTimeout(1500)            // mid-startup, on purpose
  expect(await timeClose(ctx, 'during-startup')).toBeLessThan(CLOSE_BUDGET_MS)
})
