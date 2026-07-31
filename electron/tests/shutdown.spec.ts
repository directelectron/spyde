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

// ── KNOWN PRE-EXISTING HANGS ────────────────────────────────────────────────
// Both reproduce on clean `main` (verified at 1db9ffa by checking the branch
// out and running these same tests), so they are NOT owned by any feature
// branch. They are `fixme` rather than deleted because each is a real
// user-facing defect — the backend refusing to exit leaves an orphaned process
// holding the dataset — and this file is the only place that records how to
// reproduce them.
//
// They also explain a CI failure mode that is otherwise very hard to trace:
// a shard goes red with every test green, reporting only
// "Worker teardown timeout of 120000ms exceeded" and
// "1 error was not a part of any test". Any spec that happens to close its app
// while a fill is in flight can trigger it, which is why the failing test name
// moves around between runs.

test.fixme('closing WHILE a navigator fill runs hangs (pre-existing)', async () => {
  const ctx = await launchApp({ dask: true })
  await ctx.page.waitForTimeout(2000)
  await backendAction(ctx.page, 'load_test_data_movie')
  await waitForSubwindowCount(ctx.page, 2, 120_000)
  // Deliberately do NOT wait for the fill: `BaseSignalTree.close()` cancels the
  // in-flight dispatch, but something downstream still does not unwind.
  // Contrast the two passing cases above — idle and after-fill both close in
  // ~2.6 s, so it is specifically an IN-FLIGHT compute that wedges shutdown.
  expect(await timeClose(ctx, 'mid-fill'),
    'closing mid-fill hung: the in-flight compute was not cancelled')
    .toBeLessThan(CLOSE_BUDGET_MS)
})

test.fixme('closing DURING cluster startup hangs (pre-existing)', async () => {
  const ctx = await launchApp({ dask: true })
  await ctx.page.waitForTimeout(1500)            // mid-startup, on purpose
  expect(await timeClose(ctx, 'during-startup')).toBeLessThan(CLOSE_BUDGET_MS)
})
