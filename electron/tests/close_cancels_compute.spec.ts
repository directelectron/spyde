/**
 * close_cancels_compute.spec.ts — closing a tree mid-compute must tear down
 * CLEANLY (no renderer crash) while the cancellation registry stops the
 * in-flight Dask work.
 *
 * The mechanism-level assertion (a registered future actually transitions to
 * "cancelled" on a real scheduler) is covered by the plain-Python real-cluster
 * repro (spyde/tests/repro_close_cancels.py — a LocalCluster(processes=True)
 * can't run under the Playwright/agent sandbox). THIS spec covers the app
 * integration: driving Find Vectors to compute and then closing the whole tree
 * (navigator X) mid-flight must not throw in the renderer and must remove every
 * window — the racy teardown path SignalTree.close -> _cancel_all_compute +
 * per-plot close all runs here.
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount,
} = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true })
  const { page } = ctx
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test.setTimeout(180_000)

test('closing the navigator mid-Find-Vectors tears down cleanly', async () => {
  const { page } = ctx

  const sig = page.getByTestId('subwindow').filter({ hasNotText: 'Navigator' }).first()
  await sig.getByTestId('subwindow-title').click()

  // Start the full-scan compute via the wizard.
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await page.getByTestId('fv-compute').click()

  // Immediately close the SOURCE tree (navigator X) — racing the batch. The
  // source navigator+signal windows must disappear; the find-vectors result
  // window opens EARLY on a SEPARATE tree (count-map placeholder) so it may
  // legitimately remain. Whether the compute is mid-flight or just finished,
  // the teardown must be clean: _cancel_all_compute flips the stopped_flag /
  // cancels futures, then the source windows close. A renderer exception here
  // (stale figure, landing result on a torn-down tree) is the class of bug the
  // cancellation path could introduce.
  const navWindow = page.getByTestId('subwindow').filter({ hasText: 'Navigator' }).first()
  await navWindow.getByTestId('subwindow-titlebar').hover()
  await navWindow.getByTestId('close-btn').click()

  // The source tree's Navigator + Signal windows are gone.
  await expect.poll(() => page.getByTestId('subwindow')
    .filter({ hasText: 'Navigator' }).count(), {
    timeout: 30_000, message: 'closing the navigator did not remove it',
  }).toBe(0)
  await expect(page.getByTestId('subwindow').filter({ hasText: 'Signal' }))
    .toHaveCount(0)

  // Give any late compute callback a beat to try (and no-op / cancel) so a
  // crash from a landing result on a torn-down source tree would surface.
  await page.waitForTimeout(4_000)

  // The app must still be alive and responsive after the racy teardown.
  await backendAction(page, 'load_test_data_si_grains')
  await expect.poll(() => page.getByTestId('subwindow')
    .filter({ hasText: 'Navigator' }).count(), {
    timeout: 120_000, message: 'app unresponsive after racy teardown',
  }).toBe(1)

  ctx.assertNoJsErrors()
})
