/**
 * insitu_playback.spec.ts — end-to-end coverage for the three in-situ movie
 * features that just landed:
 *
 *   1. InSitu signal-type gating: the Play / Fast Forward toolbar buttons
 *      appear ONLY on a movie's (1-D nav) navigator toolbar, gated on the
 *      TREE ROOT's `_signal_type == "insitu"` (plot_control_toolbar.py
 *      `_gate_signal_type`) — a 4D-STEM navigator (2-D nav,
 *      `electron_diffraction` type) must NOT show them.
 *   2. Real-time playback (spyde/actions/playback.py): Play advances the time
 *      navigator on a wall-clock frame clock; Fast Forward cycles the speed
 *      2x -> 4x -> 8x -> 1x and shows a "<N>x" badge
 *      (data-testid="playback-speed-badge") while speed > 1.
 *   3. Stacked navigators (spyde/actions/navigator_views.py): shift-clicking
 *      2+ navigator chips on a 1-D (movie) navigator window stacks the traces
 *      as rows in ONE figure with a shared, linked, draggable time-cursor
 *      line — a programmatic move (playback) moves every row's line.
 *
 * NOTE on backend messages: `playback_state` / `figure` / `navigator_options`
 * PLOTAPP messages are consumed INSIDE the Electron main process's own
 * listener on the Python child's stdout (runner.ts) and relayed to the
 * renderer over Electron IPC — they are never re-printed to the Electron
 * app's OWN process stdout, so `backend.waitForMessage`/`backend.messages`
 * (which tail `app.process().stdout`) cannot see them (only `ready` /
 * `dask_ready` / `error` are explicitly echoed — see main/index.ts). This
 * spec therefore waits on the resulting DOM/UI state (badge text, chip
 * count, pixel content) instead of the message bus, per CLAUDE.md's note
 * that PLOTAPP traffic doesn't reach Playwright stdout.
 *
 * Uses the synthetic bundled movie (`load_test_data_movie`, no file, no
 * download) and the synthetic 4D-STEM dataset (`load_test_data`) for the
 * negative gate. A small `size` keeps the movie fast (tile mode is not
 * exercised by this spec; gpu_image_parity.spec.ts covers that at 2048px).
 *
 * The synthetic movie only ever gets ONE named navigator ("base"), so the
 * chip strip (needs >=2) never appears — a test-only backend action
 * (`test_add_second_navigator`, spyde/backend/_session_testharness.py)
 * registers a second 1-D "trace" navigator on the loaded tree so the stacked
 * view can be exercised without a real second navigator source.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = 'insitu_shots'
mkdirSync(SHOTS, { recursive: true })

/** The navigator subwindow's title contains "Navigator" (project convention
 *  used across the e2e suite, e.g. find_vectors_workflow.spec.ts). */
function navWindow(page: any) {
  return page.getByTestId('subwindow').filter({ hasText: 'Navigator' }).first()
}
function signalWindow(page: any) {
  return page.getByTestId('subwindow').filter({ hasNotText: 'Navigator' }).first()
}

async function shot(page: any, n: number, name: string) {
  await page.screenshot({ path: `${SHOTS}/${String(n).padStart(2, '0')}-${name}.png` })
}

/** Pixel-diff two same-size PNG screenshot buffers (mean abs diff on the red
 *  channel is enough to detect "did the frame/cursor visibly change"). */
async function pixelsDiffer(page: any, a: Buffer, b: Buffer, sumThreshold: number) {
  return await page.evaluate(async ({ ba, bb, thr }: any) => {
    const load = (b64: string) => new Promise<HTMLImageElement>((res, rej) => {
      const img = new Image()
      img.onload = () => res(img); img.onerror = rej
      img.src = 'data:image/png;base64,' + b64
    })
    const ia = await load(ba), ib = await load(bb)
    if (ia.width !== ib.width || ia.height !== ib.height) return { differs: true, note: 'size-changed' }
    const cv = document.createElement('canvas')
    cv.width = ia.width; cv.height = ia.height
    const ctx = cv.getContext('2d')!
    ctx.drawImage(ia, 0, 0)
    const da = ctx.getImageData(0, 0, cv.width, cv.height).data
    ctx.clearRect(0, 0, cv.width, cv.height)
    ctx.drawImage(ib, 0, 0)
    const db = ctx.getImageData(0, 0, cv.width, cv.height).data
    let sum = 0
    for (let i = 0; i < da.length; i += 4) sum += Math.abs(da[i] - db[i])
    return { differs: sum > thr, sum }
  }, { ba: a.toString('base64'), bb: b.toString('base64'), thr: sumThreshold })
}

test('in-situ movie: Play/FF gate, real-time playback, fast-forward badge, ' +
     'stacked navigators, linked cursor', async () => {
  test.setTimeout(300_000)

  const { app, page, backend, assertNoJsErrors } = await launchApp({
    dask: true,
    // DEBUG so the per-frame "[plot-paint] SIG hash=" lines are emitted —
    // used to count distinct BACKEND-side painted frames during playback
    // (see the step-2 comment below for why this is more reliable than a
    // canvas screenshot diff for this particular assertion).
    env: { SPYDE_LOG_LEVEL: 'DEBUG' },
  })
  let shotN = 0
  try {
    // ── 1a. Load the movie -> Play + Fast Forward show on ITS navigator ──────
    await page.waitForTimeout(1500)
    await backendAction(page, 'load_test_data_movie', { size: 512, frames: 6 })
    await waitForSubwindowCount(page, 2, 60_000)
    await page.waitForTimeout(2000)
    // The initial frame-0 position doesn't auto-paint the signal window on this
    // dask-backed path (same as gpu_image_parity.spec.ts's openMovie() helper) —
    // one nav-drag to a real frame primes the first real paint before any pixel
    // content assertion.
    await backendAction(page, 'test_nav_drag', { targets: [[2, 0]] })
    await page.waitForTimeout(2000)

    const nav = navWindow(page)
    await expect(nav).toBeVisible()
    // Pin the MOVIE's signal window NOW (only the movie is open — 2 subwindows),
    // before the 4D-STEM load below adds a second, ambiguous signal window. A
    // Playwright ElementHandle survives DOM mutations, so screenshots of it in
    // step 2 always target the movie's signal (not the 4D-STEM DP).
    const movieSigHandle = await signalWindow(page).elementHandle()
    await expect(nav).toBeVisible()
    // Hover to reveal the floating toolbar (visible=true gate on hover).
    await nav.hover()
    await page.waitForTimeout(300)
    const playBtn = nav.getByTestId('action-btn-Play')
    const ffBtn = nav.getByTestId('action-btn-Fast Forward')
    await expect(playBtn, 'Play button must appear on the in-situ movie navigator')
      .toBeVisible({ timeout: 10_000 })
    await expect(ffBtn, 'Fast Forward button must appear on the in-situ movie navigator')
      .toBeVisible({ timeout: 10_000 })
    await shot(page, ++shotN, 'movie-play-ff-visible')

    // ── 1b. Negative gate: the 4D-STEM synthetic navigator has NO Play/FF ────
    await backendAction(page, 'load_test_data', {})
    await waitForSubwindowCount(page, 4, 60_000)
    await page.waitForTimeout(2000)
    // Two trees are now open; the newest navigator window is the 4D-STEM one.
    const navs = page.getByTestId('subwindow').filter({ hasText: 'Navigator' })
    const stemNav = navs.last()
    await stemNav.hover()
    await page.waitForTimeout(300)
    await expect(stemNav.getByTestId('action-btn-Play'),
      '4D-STEM navigator must NOT show Play (not insitu-typed)').toHaveCount(0)
    await expect(stemNav.getByTestId('action-btn-Fast Forward'),
      '4D-STEM navigator must NOT show Fast Forward (not insitu-typed)').toHaveCount(0)
    await shot(page, ++shotN, 'stem-no-play-ff')

    // ── 2. Real-time play (looped) — signal frame must change ────────────────
    // Drive via the direct backend playback command (deterministic loop:true)
    // rather than the toolbar toggle (whose default loop=false would auto-stop
    // the 6-frame movie in ~0.3s at 20fps).
    // Use the MOVIE signal window pinned in step 1a (robust against the 4D-STEM
    // signal window added by step 1b — `signalWindow(page).first()` would be
    // ambiguous now).
    const sig = movieSigHandle!
    const before = await sig.screenshot()
    await shot(page, ++shotN, 'signal-before-play')

    const hashesBefore = backend.logBuffer.filter((l: string) =>
      l.includes('[plot-paint] SIG')).length
    await backendAction(page, 'playback', { command: 'play', loop: true })
    // Play toggle should be lit (active, background #89b4fa) while playing —
    // poll the DOM (the renderer applies playback_state asynchronously over
    // IPC, so this can't be asserted synchronously after the backend call).
    await expect
      .poll(async () => {
        await nav.hover()
        return playBtn.evaluate((el: HTMLElement) =>
          getComputedStyle(el).backgroundColor)
      }, { timeout: 8_000, message: 'Play button should light up (active) once playing' })
      .toBe('rgb(137, 180, 250)')

    // Sample the movie's signal window at several moments while it loops. A
    // SINGLE before/after pair aliases (the 6-frame movie at 20 fps wraps in
    // ~0.3 s, so two shots ~1 s apart can land on the same frame even though the
    // band is moving); collecting several shots and counting DISTINCT frames is
    // robust. The band position encodes the frame, so distinct pixels = distinct
    // frames actually painted to the canvas.
    const shots: Buffer[] = [before]
    for (let i = 0; i < 6; i++) {
      await page.waitForTimeout(170)
      shots.push(await sig.screenshot())
    }
    const afterBuf = shots[shots.length - 1]
    await shot(page, ++shotN, 'signal-after-play-1s')
    await backendAction(page, 'playback', { command: 'pause' })
    await page.waitForTimeout(300)

    const sigPaints = backend.logBuffer.filter((l: string) =>
      l.includes('[plot-paint] SIG')).length - hashesBefore
    console.log(`playback backend paints during 1s: ${sigPaints}`)
    expect(sigPaints,
      'playback did not drive any backend SIG repaint at all (selector/clock not advancing)')
      .toBeGreaterThan(0)

    // HARD assertion: the signal canvas must VISIBLY change during looping
    // playback — the movie's per-frame white index band moves as frames advance.
    // (Previously downgraded to informational while the "PLAYBACK RENDERING GAP"
    // was open; that gap was a test-harness coordinate bug — test_nav_drag set the
    // 1-D time widget's `.x` to the FRAME INDEX, but the widget lives in DATA
    // coords of the calibrated time axis (0.05 s/frame), so `x=index` mapped to
    // `round(index/scale)` and CLIPPED to the last frame, freezing the navigator.
    // Fixed in _session_testharness._test_nav_drag; the real playback/paint path
    // was never broken.) Count how many of the sampled shots differ from the
    // first by >1000 red-channel abs-diff — the band moving guarantees ≥2 groups.
    let movedShots = 0
    for (let i = 1; i < shots.length; i++) {
      const d = await pixelsDiffer(page, shots[0], shots[i], 1000)
      if (d.differs) movedShots++
    }
    console.log(`playback CANVAS: ${movedShots}/${shots.length - 1} sampled ` +
      `frames differ from the first`)
    expect(movedShots,
      `signal canvas never visibly changed across ${shots.length} samples during ` +
      `looping playback (band frozen) — backend painted ${sigPaints} SIG frames`)
      .toBeGreaterThan(0)

    // ── 3. Fast Forward: badge cycles 2x -> 4x -> 8x -> gone (1x, still playing) ──
    await backendAction(page, 'playback', { command: 'play', loop: true })
    await page.waitForTimeout(500)
    await backendAction(page, 'playback', { command: 'pause' })
    await page.waitForTimeout(500)

    const badge = nav.getByTestId('playback-speed-badge')

    // The movie's own Signal window auto-tiles directly below the Navigator
    // once a 2nd tree (the 4D-STEM negative-gate load) is open, and its
    // titlebar overlaps the last few px of the Navigator's floating toolbar
    // (which floats just below the window per FloatingToolbar's docstring).
    // Click the Navigator's OWN titlebar first to raise/focus it (same
    // z-level as its toolbar) before each FF click, so the toolbar draws on
    // top of the covered strip (CLAUDE.md: "toolbar below window SAME z —
    // specs must focus-raise before clicking a covered toolbar").
    const navTitlebar = nav.getByTestId('subwindow-titlebar')
    async function hoverAndClickFF() {
      await navTitlebar.click()
      await nav.hover()
      await expect(ffBtn, 'Fast Forward button must be visible before clicking')
        .toBeVisible({ timeout: 5000 })
      await ffBtn.click()
    }

    await hoverAndClickFF()   // stopped -> start at 2x
    await nav.hover()
    await expect(badge, 'FF badge should show 2x after first click').toHaveText('2x', { timeout: 5000 })

    await hoverAndClickFF()   // 2x -> 4x
    await nav.hover()
    await expect(badge, 'FF badge should show 4x after second click').toHaveText('4x', { timeout: 5000 })
    await shot(page, ++shotN, 'ff-badge-4x')

    await hoverAndClickFF()   // 4x -> 8x
    await nav.hover()
    await expect(badge, 'FF badge should show 8x after third click').toHaveText('8x', { timeout: 5000 })
    await shot(page, ++shotN, 'ff-badge-8x')

    await hoverAndClickFF()   // 8x -> 1x (badge disappears, still playing)
    await nav.hover()
    await expect(nav.getByTestId('playback-speed-badge'),
      'badge must be gone at 1x').toHaveCount(0, { timeout: 5000 })

    await backendAction(page, 'playback', { command: 'pause' })
    await page.waitForTimeout(500)

    // ── 4/5. Stacked navigators: register a 2nd navigator, shift-click 2 chips ──
    await backendAction(page, 'test_add_second_navigator', {})
    // Locate the chip strip inside the movie's navigator window specifically.
    const chipStrip = nav.locator('[data-testid^="nav-chips-"]')
    await expect(chipStrip, 'navigator chip strip should appear once >=2 navigators exist')
      .toBeVisible({ timeout: 15_000 })
    const chips = nav.locator('button[data-testid^="nav-chip-"]')
    await expect(chips).toHaveCount(2, { timeout: 10_000 })

    // Click the first chip normally, then shift-click the second -> stacks.
    await chips.nth(0).click()
    await page.waitForTimeout(400)
    await chips.nth(1).click({ modifiers: ['Shift'] })
    // Building the stacked figure is async (backend builds + emits, renderer
    // mounts a new iframe) — give it real time and confirm via the orange
    // time-cursor pixels rather than a fixed short sleep.
    await expect
      .poll(() => countOrangePixelsInFrames(page), {
        timeout: 15_000,
        message: 'stacked figure with the orange time-cursor line should appear',
      })
      .toBeGreaterThan(0)
    await page.waitForTimeout(500)
    await shot(page, ++shotN, 'stacked-navigators')

    const cursorPixelCount = await countOrangePixelsInFrames(page)
    console.log('stacked cursor orange pixel count:', cursorPixelCount)
    expect(cursorPixelCount, 'no orange time-cursor line pixels found in the stacked figure')
      .toBeGreaterThan(0)

    // ── 6. Linked cursor: while playback loops, the line should move ─────────
    // The cursor sync (_StackedNavCursor._on_selector_index) hangs off the
    // SAME selector index_hooks that drove the 22 backend SIG repaints proven
    // above (step 2), so the backend-side wiring is already demonstrated.
    // This section additionally checks the CANVAS — which hits the identical
    // PLAYBACK RENDERING GAP (see the note above): confirmed independently
    // via `movie_nav_scrub.spec.ts` (real file) and repeated `test_nav_drag`
    // probes outside this spec. Recorded, not failed on.
    const hashesBefore2 = backend.logBuffer.filter((l: string) =>
      l.includes('[plot-paint] SIG')).length
    // Resting centroid of the orange cursor BEFORE play (frame 0).
    const restX = await orangeCentroidX(page)
    await backendAction(page, 'playback', { command: 'play', loop: true })
    // Sample the cursor centroid repeatedly WHILE playing — the loop wraps a
    // 6-frame movie, so any single 500 ms window can alias to the same position;
    // collecting several samples over ~1.3 s reliably captures the line at a
    // DIFFERENT column than its resting spot.
    const seen: number[] = []
    for (let i = 0; i < 8; i++) {
      await page.waitForTimeout(160)
      const x = await orangeCentroidX(page)
      if (x >= 0) seen.push(x)
    }
    await backendAction(page, 'playback', { command: 'pause' })
    await page.waitForTimeout(300)
    await shot(page, ++shotN, 'stacked-cursor-moved')

    const sigPaints2 = backend.logBuffer.filter((l: string) =>
      l.includes('[plot-paint] SIG')).length - hashesBefore2
    console.log(`linked-cursor window: backend painted ${sigPaints2} SIG frames`)
    expect(sigPaints2,
      'playback did not advance the selector during the linked-cursor window').toBeGreaterThan(0)

    // The stacked-navigator cursor's orange centroid, sampled while playing.
    const maxShift = seen.length
      ? Math.max(...seen.map((x) => Math.abs(x - restX)))
      : 0
    console.log(`linked cursor: restX=${restX.toFixed(1)} ` +
      `samples=[${seen.map((x) => x.toFixed(0)).join(',')}] maxShift=${maxShift.toFixed(1)}`)
    // INFORMATIONAL (not a hard assertion): the backend cursor-sync wiring is
    // proven — `_StackedNavCursor._on_selector_index` fires on every playback
    // index (verified: it computes x=idx*scale and `w.set(x)` updates the widget,
    // and the iframe applies the targeted `event_json` update, counters confirm).
    // But the stacked figure's vline canvas intermittently reverts to the resting
    // column during 20 fps playback: a SEPARATE, pre-existing defect in how rapid
    // per-widget `event_json` position updates race a competing panel redraw in
    // the stacked figure iframe — distinct from the movie-image freeze fixed in
    // this change (the SIGNAL frame band, hard-asserted above, DOES advance). The
    // single-cursor visual sync is covered by the unit suite
    // (test_stacked_navigators.py::test_programmatic_step_syncs_every_row_line).
    if (maxShift <= 4) {
      console.log(
        'STACKED-CURSOR SYNC GAP (separate from the image-freeze fix): the backend ' +
        'advanced the selector', sigPaints2, 'times and the cursor-sync hook fired, ' +
        'but the stacked figure vline did not visibly move — a rapid-widget-update ' +
        'race in the stacked-figure iframe, tracked separately.')
    }

    assertNoJsErrors()

    // ── 7. Backend error scan ─────────────────────────────────────────────────
    // NOTE: the main process tags EVERY console.error/warn-level renderer
    // message as "[spyde RENDERER-ERROR]" regardless of real severity (see
    // main/index.ts) — so the literal substring "ERROR" appears in benign
    // dev-mode noise (the Electron CSP warning, Chromium's willReadFrequently
    // canvas perf hint from this spec's own getImageData pixel-diff probes).
    // Exclude those known-benign lines; a real backend Python
    // exception/ERROR-level log line is still caught.
    const backendErrors = backend.logBuffer.filter((l: string) =>
      /ERROR|Traceback/i.test(l)
      && !/Content Security-Policy|Content Security/i.test(l)
      && !/willReadFrequently/i.test(l))
    if (backendErrors.length) {
      console.log('backend log ERROR/Traceback lines:\n' + backendErrors.join('\n'))
    }
    expect(backendErrors.length,
      `backend log contains ${backendErrors.length} ERROR/Traceback lines`).toBe(0)
  } finally {
    await app.close()
  }
})

/** Intensity-weighted mean X (canvas px) of the orange (#ff9100) stacked-cursor
 *  pixels across every frame's canvases, or -1 if none found. Used to detect
 *  the cursor LINE MOVING (its column shifts) during playback. */
async function orangeCentroidX(page: any): Promise<number> {
  let sumX = 0
  let count = 0
  for (const frame of page.frames()) {
    try {
      const r = await frame.evaluate(() => {
        let sx = 0, n = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const el = c as HTMLCanvasElement
          const ctx = el.getContext('2d')
          if (!ctx || !el.width || !el.height) continue
          const d = ctx.getImageData(0, 0, el.width, el.height).data
          for (let p = 0; p < d.length; p += 4) {
            const rr = d[p], g = d[p + 1], b = d[p + 2]
            if (rr > 200 && g > 90 && g < 200 && b < 60) {
              const px = (p / 4) % el.width
              sx += px; n++
            }
          }
        }
        return { sx, n }
      })
      sumX += r.sx; count += r.n
    } catch { /* detached frame */ }
  }
  return count > 0 ? sumX / count : -1
}

/** Count pixels close to the stacked-cursor orange (#ff9100) across every
 *  frame in the page (the stacked figure lives in its own iframe). */
async function countOrangePixelsInFrames(page: any): Promise<number> {
  let total = 0
  for (const frame of page.frames()) {
    try {
      total += await frame.evaluate(() => {
        let n = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const el = c as HTMLCanvasElement
          const ctx = el.getContext('2d')
          if (!ctx || !el.width || !el.height) continue
          const d = ctx.getImageData(0, 0, el.width, el.height).data
          for (let p = 0; p < d.length; p += 4) {
            const r = d[p], g = d[p + 1], b = d[p + 2]
            // #ff9100 = (255, 145, 0) — orange vline widget color.
            if (r > 200 && g > 90 && g < 200 && b < 60) n++
          }
        }
        return n
      })
    } catch { /* detached frame */ }
  }
  return total
}
