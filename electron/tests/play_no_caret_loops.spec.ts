/**
 * play_no_caret_loops.spec.ts — the Play button is a plain toggle now.
 *
 * The Loop parameter (and therefore the caret/params popout with its Run
 * button) was removed from the Play toolbar action (spyde/toolbars.yaml).
 * Clicking Play must:
 *   1. NOT open a caret/params popout (no "Run" panel) — it toggles directly,
 *   2. start playback that LOOPS (a 6-frame movie at ~20 fps that auto-stopped
 *      in ~0.3 s with the old loop=false default now keeps playing), and
 *   3. toggle back off on a second click.
 *
 * Synthetic bundled movie (no file/download). PLOTAPP messages don't reach
 * Playwright stdout, so state is read from the DOM (button lit state + the
 * absence of a params popout) per CLAUDE.md.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = 'play_caret_shots'
mkdirSync(SHOTS, { recursive: true })

function navWindow(page: any) {
  return page.getByTestId('subwindow').filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) }).first()
}

const ACTIVE_BG = 'rgb(137, 180, 250)'   // #89b4fa — a lit/active toolbar button

test('Play has no caret and always loops', async () => {
  test.setTimeout(180_000)
  const { app, page, assertNoJsErrors } = await launchApp({ dask: true })
  let shotN = 0
  const shot = (name: string) =>
    page.screenshot({ path: `${SHOTS}/${String(++shotN).padStart(2, '0')}-${name}.png` })

  try {
    await page.waitForTimeout(1200)
    await backendAction(page, 'load_test_data_movie', { size: 512, frames: 6 })
    await waitForSubwindowCount(page, 2, 60_000)
    await page.waitForTimeout(1500)

    const nav = navWindow(page)
    const navTitlebar = nav.getByTestId('subwindow-titlebar')
    const playBtn = nav.getByTestId('action-btn-Play')

    // Focus-raise the navigator (its floating toolbar shares the window's
    // z-level; without raising, the mdi-area intercepts the click — CLAUDE.md
    // "toolbar below window SAME z — focus-raise before clicking").
    async function clickPlay() {
      await navTitlebar.click()
      await nav.hover()
      await expect(playBtn, 'Play must be visible before clicking')
        .toBeVisible({ timeout: 10_000 })
      await playBtn.click()
    }

    await navTitlebar.click()
    await nav.hover()
    await expect(playBtn, 'Play must appear on the in-situ movie navigator')
      .toBeVisible({ timeout: 10_000 })

    // ── 1. Clicking Play must NOT open a caret/params popout. ────────────────
    // The params popout carries the Run button (data-testid="action-run").
    await clickPlay()
    await page.waitForTimeout(500)
    await shot('after-first-play-click')
    await expect(page.getByTestId('action-run'),
      'Play must not open a params popout with a Run button (caret removed)')
      .toHaveCount(0)

    // ── 2. Play started and LIT UP (playing). ────────────────────────────────
    await expect.poll(async () => {
      await nav.hover()
      return playBtn.evaluate((el: HTMLElement) => getComputedStyle(el).backgroundColor)
    }, { timeout: 8_000, message: 'Play should light up once playing' })
      .toBe(ACTIVE_BG)

    // ── 3. It LOOPS: a 6-frame movie at ~20 fps wraps in ~0.3 s. With the old
    // loop=false default it would auto-stop and un-light within ~1 s. Wait well
    // past that and confirm it is STILL playing (still lit). ──────────────────
    await page.waitForTimeout(3_000)
    await nav.hover()
    const stillLit = await playBtn.evaluate((el: HTMLElement) =>
      getComputedStyle(el).backgroundColor)
    await shot('still-playing-after-3s')
    expect(stillLit,
      'Play should STILL be playing after 3 s (it loops; did not auto-stop)')
      .toBe(ACTIVE_BG)

    // ── 4. Second click toggles playback OFF. ────────────────────────────────
    await clickPlay()
    await expect.poll(async () => {
      await nav.hover()
      return playBtn.evaluate((el: HTMLElement) => getComputedStyle(el).backgroundColor)
    }, { timeout: 8_000, message: 'second Play click should pause (un-light)' })
      .not.toBe(ACTIVE_BG)

    assertNoJsErrors()
  } finally {
    await app.close()
  }
})
