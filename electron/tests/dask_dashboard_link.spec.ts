/**
 * dask_dashboard_link.spec.ts — the "Open full Dask dashboard" affordance must
 * actually be there once a cluster is up.
 *
 * TWO different backend messages dispatch the renderer's READY action: `ready`
 * (the stdin loop is up — carries NO dashboard field) and `dask_ready` (the
 * cluster is up — carries the URL). The reducer used `action.dashboardUrl ??
 * null`, so ANY `ready` arriving after `dask_ready` erased the URL. That
 * disabled the File-menu item (`disabled: !state.dashboardUrl`) and hid the
 * button in DaskMonitor entirely — the dashboard simply could not be opened.
 *
 * Needs a REAL cluster (dask: true), because the whole point is the second
 * message.
 *
 * Run:
 *   npx playwright test tests/dask_dashboard_link.spec.ts --project=electron \
 *     --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const { launchApp } = require('./_harness.cjs')

test.setTimeout(300_000)

test('the Dask dashboard link survives every ready message', async () => {
  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  try {
    // Let BOTH messages land in whatever order they arrive — the bug was that
    // one order silently destroyed the URL the other had just delivered.
    await ctx.page.waitForTimeout(4000)

    await ctx.page.getByTestId('dask-monitor').click()
    await expect(
      ctx.page.getByRole('button', { name: /Open full Dask dashboard/ }),
      'no dashboard button — the URL was lost between ready and dask_ready',
    ).toBeVisible({ timeout: 20_000 })

    ctx.assertNoJsErrors()
  } finally {
    await ctx.app?.close()
  }
})
