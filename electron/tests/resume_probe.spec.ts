/**
 * resume_probe.spec.ts — PHASE 1 PROBE for SESSION_RESYNC_PLAN.md.
 *
 * Not a regression test. This MEASURES what survives a renderer loss and what
 * does not, so the plan stops guessing. It prints a findings block; the
 * assertions only pin things confident enough to regress on.
 *
 * A laptop lid-close cannot be driven from CI, so this uses the faithful proxy:
 * reload the renderer with data loaded. That recreates the renderer's JS context
 * and destroys its in-memory React state exactly as a renderer-process death
 * would, while leaving the Python backend and its Dask cluster untouched — which
 * is precisely the situation §1 of the plan reasons about.
 *
 * What it answers:
 *   1. Do the plot windows survive a renderer loss?           (expected: NO)
 *   2. Does the Python backend survive it?                    (expected: YES)
 *   3. Does the Dask dashboard link survive?                  (expected: NO — same React state)
 *   4. Do the figure HTML files still exist afterwards?       (plan §3.3 Q1 — the cost driver)
 *   5. Can the app still load data afterwards?                (the "it still works" claim)
 *
 * Run:
 *   npx playwright test tests/resume_probe.spec.ts --project=electron \
 *     --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { existsSync, mkdirSync } from 'fs'
import { tmpdir } from 'os'
const { launchApp, backendAction, backendErrorLines } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'resume_probe_shots')

test.setTimeout(420_000)

test('what survives a renderer loss', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  const findings: string[] = []

  try {
    await page.waitForTimeout(2000)

    // ── Establish a populated workspace ────────────────────────────────────
    await backendAction(page, 'load_test_data_si_grains', {})
    await expect(page.getByTestId('subwindow').first())
      .toBeVisible({ timeout: 90_000 })
    await page.waitForTimeout(4000)

    const windowsBefore = await page.getByTestId('subwindow').count()
    const framesBefore = await page.locator('iframe').count()
    // Every figure iframe's src — the HTML files on disk (plan §3.3 Q1).
    const figureSrcs: string[] = await page.evaluate(() =>
      Array.from(document.querySelectorAll('iframe'))
        .map(f => (f as HTMLIFrameElement).src)
        // Figures are served over the custom spyde-fig:// protocol, which
        // resolves to a real HTML file in the OS tmpdir (main/index.ts
        // resolveFigPath). Whether those files outlive a renderer loss is the
        // cost driver for the plan's Phase 2.
        .filter(s => s.startsWith('spyde-fig://')))
    await page.getByTestId('dask-monitor').click()
    const dashBefore = await page.getByRole(
      'button', { name: /Open full Dask dashboard/ }).isVisible().catch(() => false)
    await page.keyboard.press('Escape')
    await page.screenshot({ path: join(SHOTS, '01-populated.png') })

    findings.push(`BEFORE  windows=${windowsBefore} iframes=${framesBefore} ` +
                  `dashboardLink=${dashBefore} figureFiles=${figureSrcs.length}`)

    // The backend's identity, so we can prove it is the SAME process after.
    const backendLinesBefore = (ctx.backend.logBuffer || []).length

    // ── The renderer loss ──────────────────────────────────────────────────
    await page.reload({ waitUntil: 'domcontentloaded' })
    await page.waitForTimeout(6000)
    await page.screenshot({ path: join(SHOTS, '02-after-renderer-loss.png') })

    const windowsAfter = await page.getByTestId('subwindow').count()
    const framesAfter = await page.locator('iframe').count()

    // Did the backend die? It is never respawned, so if it had, its log would
    // stop and every action below would no-op.
    const backendExited = (ctx.backend.logBuffer || [])
      .some((l: string) => /SpyDE exited with code/.test(l))

    // Do the figure HTML files still exist on disk? THE cost driver for the
    // plan's Phase 2 — if they survive, a resync re-sends the same paths.
    const filesOnDisk = figureSrcs.map(src => {
      try {
        const name = decodeURIComponent(new URL(src).pathname).replace(/^\//, '')
        return existsSync(join(tmpdir(), name))
      } catch { return false }
    })
    const survivingFiles = filesOnDisk.filter(Boolean).length

    let dashAfter = false
    try {
      await page.getByTestId('dask-monitor').click({ timeout: 5000 })
      dashAfter = await page.getByRole(
        'button', { name: /Open full Dask dashboard/ })
        .isVisible().catch(() => false)
      await page.keyboard.press('Escape')
    } catch { /* the monitor itself may be gone */ }

    findings.push(`AFTER   windows=${windowsAfter} iframes=${framesAfter} ` +
                  `dashboardLink=${dashAfter} ` +
                  `figureFilesStillOnDisk=${survivingFiles}/${figureSrcs.length}`)
    findings.push(`BACKEND exited=${backendExited} ` +
                  `logGrew=${(ctx.backend.logBuffer || []).length > backendLinesBefore}`)

    // ── Can it still be used? ("it still works, you just re-load the data") ──
    await backendAction(page, 'load_test_data_si_grains', {})
    let reloadWorks = false
    try {
      await expect(page.getByTestId('subwindow').first())
        .toBeVisible({ timeout: 90_000 })
      reloadWorks = true
    } catch { /* recorded below */ }
    await page.waitForTimeout(3000)
    const windowsAfterReload = await page.getByTestId('subwindow').count()
    await page.screenshot({ path: join(SHOTS, '03-after-reloading-data.png') })

    findings.push(`RELOAD  worked=${reloadWorks} windows=${windowsAfterReload}`)

    console.log('\n──────── PROBE FINDINGS ────────\n' +
                findings.join('\n') +
                '\n────────────────────────────────\n')

    // ── The only assertions: the two claims the plan is built on ───────────
    expect(backendExited,
           'the backend must NOT have died — the plan assumes it survives')
      .toBe(false)
    expect(reloadWorks,
           'the app must still be usable after a renderer loss').toBe(true)
    // Recorded, not asserted: window survival. If windows DO come back, the
    // plan's premise is wrong and that is the most valuable thing this can say.
    if (windowsAfter > 0) {
      console.log('!! WINDOWS SURVIVED — SESSION_RESYNC_PLAN §1 premise is WRONG')
    }

    const errs = backendErrorLines(ctx.backend)
    console.log('BACKEND ERRORS:', JSON.stringify(errs.slice(0, 5)))
  } finally {
    await ctx.app?.close()
  }
})
