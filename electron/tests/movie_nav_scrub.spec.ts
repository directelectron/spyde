/**
 * movie_nav_scrub.spec.ts — the in-situ MOVIE navigator on the unified cached
 * read path (Phase 2).
 *
 * Loads a real Direct-Electron in-situ movie (a nav-dim-1 stack of large image
 * frames over a time axis) and scrubs the 1-D time navigator across many frames,
 * asserting the displayed image ACTUALLY CHANGES per move — i.e. the synchronous
 * cached read (get_index, no distributed client, on the serial _NavDispatcher)
 * paints a fresh frame each scrub, not a frozen/stale one.
 *
 * Uses the shared harness (real Dask) + the test_nav_drag backend action, which
 * reports per-move whether the signal plot's painted data changed. Result is
 * captured via backend.waitForMessage (robust against the stdout race that the
 * old hand-rolled grabber in nav_drag_distributed.spec.ts hit).
 *
 * The movie path is skipped (not failed) if no real movie file is present on the
 * box, so CI without the data still passes.
 */
import { test, expect } from '@playwright/test'
import { existsSync } from 'fs'
const { launchApp, backendAction } = require('./_harness.cjs')

// Real in-situ movies on the dev box (first that exists wins). Large 2-D frames
// over a time axis — the case Phase 1/2 target.
const MOVIE_CANDIDATES = [
  'C:/Users/CarterFrancis/Downloads/20251117_88075_run3 some growth_1236_movie.mrc',
  'C:/Users/CarterFrancis/Downloads/20251117_88074_run1_9104_movie.mrc',
  'C:/Users/CarterFrancis/Downloads/20241002_07954_movie.mrc',
]

function firstMovie(): string | null {
  for (const p of MOVIE_CANDIDATES) if (existsSync(p)) return p
  return null
}

test('scrubbing the in-situ movie time navigator paints a fresh frame each move', async () => {
  // A cold open of a multi-GB movie + Dask startup + scrub can take minutes.
  test.setTimeout(480_000)
  const moviePath = firstMovie()
  test.skip(!moviePath, 'no real in-situ movie file present on this machine')

  // INFO level so the backend's "[REDRAW] test_nav_drag: N/N moves changed the
  // DP" verdict tees to stderr (which the harness log buffer captures) — the
  // nav_drag_result PLOTAPP message rides the stdout line protocol the main
  // process consumes, so it is NOT reliably visible to the test.
  const { app, page, backend, assertNoJsErrors } = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'INFO' },
  })
  try {
    // Load the real movie by path (the user's open-file path). A large first
    // open is a cold header read — give it room.
    await page.waitForTimeout(1500)
    await backendAction(page, 'open_file', { path: moviePath })
    // Navigator + signal windows appear once the movie opens.
    await page.waitForFunction(
      () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
      { timeout: 180_000 },
    )
    await page.waitForTimeout(3000)   // initial nav render + first frame
    await page.screenshot({ path: 'movie_shots/00-movie-loaded.png' })

    // Scrub the 1-D time navigator across frames. For a 1-D navigator the
    // crosshair x = time index; y is ignored. Spread across the stack so moves
    // cross the 1-frame-per-chunk boundaries (each is a fresh disk read).
    const targets: number[][] = [
      [1, 0], [30, 0], [100, 0], [250, 0], [500, 0], [120, 0], [10, 0],
    ]

    // The backend logs its verdict ("[REDRAW] test_nav_drag: N/N moves changed
    // the DP") to stderr — wait for that line (robust) rather than the stdout
    // nav_drag_result message.
    const verdictP = backend.waitForLog('test_nav_drag:', 180_000)
    await backendAction(page, 'test_nav_drag', { targets })
    const verdictLine = await verdictP
    await page.screenshot({ path: 'movie_shots/01-movie-scrub.png' })

    console.log('\n===== MOVIE NAV_DRAG VERDICT =====\n' + verdictLine)
    const m = verdictLine.match(/test_nav_drag:\s*(\d+)\/(\d+)\s+moves changed/)
    expect(m, `could not parse nav_drag verdict from: ${verdictLine}`).toBeTruthy()
    const changed = Number(m![1])
    const total = Number(m![2])
    // Every scrub move must repaint a fresh frame (allow 1 slack for a possible
    // duplicate at a clamped edge). This is THE Phase-2 assertion: the movie
    // navigator paints a fresh frame per scrub on the unified cached read path.
    expect(
      changed,
      `movie frame updated only ${changed}/${total} scrub moves — frozen/stale`,
    ).toBeGreaterThanOrEqual(total - 1)

    assertNoJsErrors()
  } finally {
    await app.close()
  }
})
