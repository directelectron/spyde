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

/** Mean-abs red-channel diff between two same-size PNG buffers; `differs` when
 *  the summed diff exceeds `thr` (the movie band moving clears any threshold). */
async function pixelsDiffer(page: any, a: Buffer, b: Buffer, thr: number) {
  return await page.evaluate(async ({ ba, bb, t }: any) => {
    const load = (s: string) => new Promise<HTMLImageElement>((res, rej) => {
      const img = new Image(); img.onload = () => res(img); img.onerror = rej
      img.src = 'data:image/png;base64,' + s
    })
    const ia = await load(ba), ib = await load(bb)
    if (ia.width !== ib.width || ia.height !== ib.height) return { differs: true, sum: -1 }
    const cv = document.createElement('canvas'); cv.width = ia.width; cv.height = ia.height
    const ctx = cv.getContext('2d')!
    ctx.drawImage(ia, 0, 0); const da = ctx.getImageData(0, 0, cv.width, cv.height).data
    ctx.clearRect(0, 0, cv.width, cv.height)
    ctx.drawImage(ib, 0, 0); const db = ctx.getImageData(0, 0, cv.width, cv.height).data
    let sum = 0; for (let i = 0; i < da.length; i += 4) sum += Math.abs(da[i] - db[i])
    return { differs: sum > t, sum }
  }, { ba: a.toString('base64'), bb: b.toString('base64'), t: thr })
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
    // DEBUG so the "[REDRAW] test_nav_drag" verdict AND the per-frame
    // "[plot-paint] SIG hash=" lines (used to count distinct played frames) both
    // tee to stderr where the harness log buffer captures them.
    env: { SPYDE_LOG_LEVEL: 'DEBUG' },
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
    // Wait for the large-file read to actually FINISH (status bar shows
    // "Reading …mrc…" until the reader settles) — scrubbing before that races
    // the cold open and the nav_drag verdict times out (same guard as
    // viewport_detail / webgpu_image specs).
    await page.waitForFunction(
      () => !/Reading .*\.mrc/i.test(document.body.textContent || ''),
      { timeout: 300_000 },
    )
    await page.waitForTimeout(3000)   // initial nav render + first frame
    await page.screenshot({ path: 'movie_shots/00-movie-loaded.png' })

    // Scrub the 1-D time navigator across frames. For a 1-D navigator the
    // crosshair x = time index; y is ignored. Spread across the stack so moves
    // cross the 1-frame-per-chunk boundaries (each is a fresh disk read).
    const targets: number[][] = [
      [1, 0], [50, 0], [150, 0], [400, 0], [80, 0],
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

    // ── Playback: Play advances the time navigator on a frame clock ──────────
    // Fire the backend `playback` action (real-time play, looped) and confirm
    // the SIGNAL CANVAS visibly changes as frames advance. NB the playback API
    // takes `speed`/`loop` only — the old `fps` param was removed (playback.py
    // is real-time paced from the time-axis scale now).
    //
    // We sample the signal window's pixels at several moments while playing and
    // count DISTINCT frames. This is the reliable signal: the backend
    // `[plot-paint] SIG` DEBUG log does NOT fire on the playback paint path (the
    // per-frame nav paint runs through _NavPainter, which paints without hitting
    // that log branch — verified the pixels DO change while 0 SIG lines are
    // logged), so counting those log lines under-reports. The pixel diff is
    // ground truth: the movie signal band moving = distinct frames on screen.
    const sig = page.getByTestId('subwindow').filter({ hasNotText: 'Navigator' }).first()
    await expect(sig).toBeVisible()
    const base = await sig.screenshot()
    await backendAction(page, 'playback', { command: 'play', loop: true })
    const frames: Buffer[] = []
    for (let i = 0; i < 8; i++) {
      await page.waitForTimeout(170)
      frames.push(await sig.screenshot())
    }
    await backendAction(page, 'playback', { command: 'pause' })
    await page.waitForTimeout(200)
    await page.screenshot({ path: 'movie_shots/02-movie-playing.png' })

    // How many sampled frames differ from the pre-play baseline (band moved)?
    let distinct = 0
    for (const f of frames) {
      const d = await pixelsDiffer(page, base, f, 1500)
      if (d.differs) distinct++
    }
    console.log(`\n===== PLAYBACK: ${distinct}/${frames.length} sampled ` +
      `signal frames differ from baseline while playing =====`)
    // INFORMATIONAL on this REAL-FILE spec: whether the signal canvas visibly
    // advances during playback depends on the specific .mrc (some local movies
    // are UNCALIBRATED — nav scale=1, no time units → the 10 fps default — and
    // some have near-black frames at the position the scrub left, so a pixel
    // diff can legitimately read 0). The AUTHORITATIVE, deterministic playback
    // VISUAL test is `insitu_playback.spec.ts` on the synthetic movie (per-frame
    // white index band, hard-asserts the band moves). Here the HARD contract is
    // the SCRUB above (5/5 moves paint a fresh frame — the harness coordinate
    // fix); playback advancement is recorded, not failed on, for arbitrary data.
    if (distinct === 0) {
      console.log(
        'NOTE: this real movie showed no visible signal change during playback ' +
        '(likely an uncalibrated/near-black-frame file); the synthetic-movie ' +
        'insitu_playback.spec.ts is the deterministic playback-visual gate.')
    }

    assertNoJsErrors()
  } finally {
    await app.close()
  }
})
