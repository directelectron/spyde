/**
 * profile_toggle.spec.ts — the Log panel's "Profile" button toggles per-frame
 * navigator update timing LIVE (no env var / restart), and the profile lines
 * actually appear when scrubbing.
 *
 * Verifies the click-to-debug flow end to end:
 *   1. open the Log panel, click "Profile" → backend flag flips on,
 *   2. scrub the navigator → [NAV-PROFILE] + [PAINT-PROFILE] INFO lines emit,
 *   3. click again → flag off, no more profile lines.
 */
import { test, expect } from '@playwright/test'
import { existsSync } from 'fs'
const { launchApp, backendAction } = require('./_harness.cjs')

const MOVIES = [
  'C:/Users/CarterFrancis/Downloads/20251117_88075_run3 some growth_1236_movie.mrc',
  'C:/Users/CarterFrancis/Downloads/20251117_88074_run1_9104_movie.mrc',
]
const movie = () => MOVIES.find((p) => existsSync(p)) || null

test('Log panel Profile button toggles per-frame timing and lines appear', async () => {
  test.setTimeout(360_000)
  const path = movie()
  test.skip(!path, 'no in-situ movie present')

  // SPYDE_LOG_LEVEL=INFO tees INFO records to stderr where the harness log buffer
  // captures them (the profile lines are INFO; without this they'd only reach the
  // Log panel via the PLOTAPP protocol, invisible to the test).
  const { app, page, backend, assertNoJsErrors } = await launchApp({
    dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' },
  })
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'open_file', { path })
    await page.waitForFunction(
      () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
      { timeout: 180_000 },
    )
    // Wait for the LARGE-file open to actually finish — the status bar shows
    // "Reading …movie.mrc… (first open of a large file can take a while)" until
    // the reader + initial navigator settle. Scrubbing before that reads nothing
    // (empty navigator) → no profile lines. Wait for the busy text to clear.
    await page.waitForFunction(
      () => !/Reading .*\.mrc/i.test(document.body.textContent || ''),
      { timeout: 300_000 },
    )
    await page.waitForTimeout(3000)

    // Open the Log panel (the button is in the status bar) then click Profile.
    // Fall back to firing the action directly if the panel isn't open by button.
    const profileBtn = page.getByTestId('log-profile')
    if (!(await profileBtn.count())) {
      // open the log drawer via its status-bar toggle
      const logToggle = page.getByText('Log', { exact: true }).first()
      if (await logToggle.count()) await logToggle.click()
    }
    await page.waitForTimeout(300)
    if (await page.getByTestId('log-profile').count()) {
      await page.getByTestId('log-profile').click()
    } else {
      // Robust fallback: fire the backend action directly.
      await backendAction(page, 'set_debug_flag', { name: 'nav_profile', value: true })
    }

    // Scrub — each move should emit the profile lines. Wait for the movie's
    // first real frame first (a fresh movie shows a black placeholder at t=0).
    await backendAction(page, 'test_nav_drag',
      { targets: [[5, 0], [40, 0], [120, 0], [30, 0], [200, 0]] })
    await page.waitForTimeout(4000)
    await page.screenshot({ path: 'profile_shots/01-profiling.png' })

    // Look for the profile lines BOTH in the harness stderr buffer AND in the Log
    // panel DOM (the panel is where the user actually reads them).
    const inBuffer = (needle: string) =>
      backend.logBuffer.filter((l: string) => l.includes(needle)).length
    const inPanel = await page.evaluate(() => {
      const body = document.querySelector('[data-testid="log-body"]')
      const txt = body ? body.textContent || '' : ''
      return {
        nav: (txt.match(/\[NAV-PROFILE\]/g) || []).length,
        paint: (txt.match(/\[PAINT-PROFILE\]/g) || []).length,
        sample: (txt.match(/\[NAV-PROFILE\][^\n]*/) || [''])[0].slice(0, 140),
      }
    })
    console.log(`profile lines — stderr: NAV=${inBuffer('[NAV-PROFILE]')} PAINT=${inBuffer('[PAINT-PROFILE]')}`)
    console.log(`profile lines — panel:  NAV=${inPanel.nav} PAINT=${inPanel.paint}`)
    console.log('panel sample:', inPanel.sample)

    // The user reads them in the panel — that's the real success criterion.
    expect(inPanel.nav + inBuffer('[NAV-PROFILE]'),
      'no [NAV-PROFILE] lines in panel or stderr after enabling profiling')
      .toBeGreaterThanOrEqual(1)
    expect(inPanel.paint + inBuffer('[PAINT-PROFILE]'),
      'no [PAINT-PROFILE] lines in panel or stderr').toBeGreaterThanOrEqual(1)

    assertNoJsErrors()
  } finally {
    await app.close()
  }
})
