/**
 * live_view.spec.ts — Ground Crew's boundary proof.
 *
 * This spec is the argument that the split worked. A SECOND Electron app, with
 * its own fixed-pane layout and its own Python package, boots on the same shell
 * — @de/shell-main spawns the sidecar, de_shell carries the IPC and the session
 * — and shows live frames. Nothing in SpyDE is involved, and neither hyperspy
 * nor dask is installed in the path it exercises.
 *
 * If this goes red, the shell has grown an app-specific assumption.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp } = require('../../../packages/shell-testing/src/harness.cjs')

const APP_DIR = join(__dirname, '..')
const SHOTS = join(APP_DIR, 'shots')

/**
 * The fraction of BRIGHT pixels in the largest canvas of the figure iframe.
 *
 * Both failure modes this spec has actually hit are UNIFORM, and they are
 * uniform at opposite ends: a placeholder or failed decode renders black (0),
 * and a frame drawn with the placeholder's display range renders white (1).
 * The simulator's scene is seven bright discs on a dark field, so a correct
 * render lands in between and neither degenerate case can pass.
 *
 * Counting DISTINCT grey levels — what this used to do — is the wrong oracle
 * here: the scene is essentially bimodal, so a perfectly good render shows
 * about six levels and looks like a placeholder to a "> 10 levels" assertion.
 */
async function brightFraction(page: any): Promise<number> {
  let best = -1
  for (const frame of page.frames()) {
    try {
      const f = await frame.evaluate(() => {
        let out = -1, widest = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const el = c as HTMLCanvasElement
          const ctx = el.getContext('2d')
          if (!ctx || !el.width || !el.height || el.width < widest) continue
          widest = el.width
          const d = ctx.getImageData(0, 0, el.width, el.height).data
          let bright = 0, n = 0
          // Every 40th pixel: enough to characterise the frame without walking
          // megabytes in the page context.
          for (let p = 0; p < d.length; p += 4 * 40) { n++; if (d[p] > 128) bright++ }
          out = n ? bright / n : -1
        }
        return out
      })
      if (f >= 0) best = Math.max(best, f)
    } catch { /* frame detached mid-evaluate */ }
  }
  return best
}

let ctx: any

test.beforeAll(async () => {
  ctx = await launchApp({
    appDir: APP_DIR,
    appId: 'groundcrew',
    // INFO so the log-area registration is genuinely exercised: at WARNING the
    // handler short-circuits on level alone and would pass even if the app had
    // never registered itself as a verbose package.
    env: {
      GROUNDCREW_LOG_LEVEL: 'INFO',
      // Spawn deapi's own simulated DE Server and talk the real protobuf
      // protocol to it. Not a stand-in camera class: the thing under test is
      // `get_result` driving the viewer, so a fake that did not speak the
      // protocol would test nothing.
      GROUNDCREW_FAKE_SERVER: '1',
    },
    // The main process echoes this when the backend's `ready` lands. Waiting on
    // the PLOTAPP line itself does NOT work: main consumes that channel, so it
    // never reaches the app's stdout that the harness watches.
    readyLog: '[groundcrew backend] ready',
    readyMessages: [],
  })
})

// Backend `emit`/`emit_error` go down the PLOTAPP channel, which the main
// process consumes — they never reach Playwright's stdout. So a Python-side
// failure shows up here only as a missing element, with the cause invisible.
// Dump the backend's log on any failure; it is the difference between "the
// figure never appeared" and knowing why.
test.afterEach(async ({}, testInfo) => {
  if (testInfo.status !== testInfo.expectedStatus && ctx?.backend) {
    // logBuffer is an ARRAY of lines, not a string — concatenating it directly
    // yields a comma-run that reads as empty output.
    const lines: string[] = ctx.backend.logBuffer ?? []
    console.log('─── backend log (last 60 of ' + lines.length + ') ───\n' +
      lines.slice(-60).join('\n'))
  }
})

test.afterAll(async () => { await ctx?.app?.close() })

test('the status board says what it cannot see', async () => {
  const { page } = ctx

  // Runs BEFORE anything stops acquisition, so this exercises the property
  // GAP — a healthy connection to a server that simply does not expose most of
  // what the board asks for. The other degradation (a camera that has stopped
  // answering at all) is a different failure and is tested separately below;
  // conflating them let this pass for the wrong reason once already.
  await page.getByTestId('mode-status').click()
  await expect(page.getByTestId('status-headline')).toBeVisible({ timeout: 30_000 })

  const coverage = await page.getByTestId('status-coverage').innerText()
  console.log('[groundcrew] coverage =', coverage)
  expect(coverage).toMatch(/of \d+ checks reporting/)
  expect(coverage, 'the simulator cannot answer every check — that must be said')
    .toMatch(/unavailable on this server/)

  // The connection itself is fine here, so THAT card must not cry fault.
  await expect(page.getByTestId('status-card-connection')).toHaveAttribute('data-state', 'ok')

  // Nothing the server failed to report may be drawn as passing.
  const cards = page.locator('[data-testid^="status-card-"]')
  const unreported = page.locator('[data-testid^="status-card-"][data-state="unreported"]')
  expect(await unreported.count(),
    'expected unreported cards on the simulator').toBeGreaterThan(0)
  expect(await page.locator('[data-testid^="status-card-"][data-state="ok"]').count())
    .toBeLessThan(await cards.count())

  await page.screenshot({ path: join(SHOTS, '04-status.png') })
  ctx.assertNoJsErrors()
})

test('boots on the shared shell and streams live frames', async () => {
  const { page, backend } = ctx

  // 1. The fixed layout rendered — this app has a permanent control sidebar
  //    and a mode rail where SpyDE has an MDI workspace. Select the mode
  //    explicitly: tests share one app instance, so whatever ran before has
  //    left the rail somewhere, and the stats strip belongs to Imaging.
  await page.getByTestId('mode-imaging').click()
  await expect(page.getByTestId('control-panel')).toBeVisible()
  await expect(page.getByTestId('stats-strip')).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '01-boot.png') })

  // 2. The viewer figure arrived over the PLOTAPP channel and mounted.
  const frame = page.getByTestId('viewer-frame')
  await expect(frame).toBeVisible({ timeout: 60_000 })

  // 3. Frames are actually flowing. The viewer is a PULL source — anyplotlib
  //    asks the server for the region on screen — so acquisition is started
  //    explicitly and the stats strip, which rides along on the same
  //    `get_result`, is what proves a read completed.
  await page.getByTestId('start-btn').click()
  await expect
    .poll(async () => Number(await page.getByTestId('stat-max').innerText()),
      { timeout: 30_000, message: 'no frame statistics ever arrived' })
    .toBeGreaterThan(0)

  await page.screenshot({ path: join(SHOTS, '02-live.png') })

  // 4. The image actually PAINTED — and painted the CAMERA's pixels, not the
  //    figure's chrome.
  //
  //    Measured as a FRACTION inside the image canvas, not a raw bright-pixel
  //    count: an absolute count gave a false pass once, because the figure's
  //    own white background is bright and held the assertion at 322k pixels
  //    while the image pane was uniformly black.
  const bright = await brightFraction(page)
  console.log('[groundcrew] bright fraction =', bright.toFixed(3))
  // Observed ~0.009: the discs are a small part of a canvas that also carries
  // the pane's dark margins. The bound is deliberately well below that — it is
  // discriminating against an ALL-black frame, not measuring the scene.
  expect(bright, 'image pane is black — no frame reached the canvas')
    .toBeGreaterThan(0.002)
  expect(bright, 'image pane is white — the display range never left the placeholder')
    .toBeLessThan(0.6)

  // 5. Controls round-trip to Python: stopping must flip the acquisition state
  //    the backend reports back, not just the local button.
  await page.getByTestId('stop-btn').click()
  await expect(page.getByTestId('start-btn')).toBeEnabled({ timeout: 15_000 })
  await page.screenshot({ path: join(SHOTS, '03-stopped.png') })


  // 6. Nothing died on the way.
  const errs = backend.errorLines()
  if (errs.length) console.log('[groundcrew] backend errors:\n' + errs.join('\n'))
  expect(errs.join('\n')).not.toMatch(/Traceback|ModuleNotFoundError|ImportError/)
  ctx.assertNoJsErrors()
})

test('a camera that stops answering is reported, not waited on', async () => {
  const { page } = ctx

  // Runs AFTER the live-view test, which issues `stop_acquisition` — a call
  // deapi's FakeServer never returns from, taking the single connection with
  // it. Whatever the server does, the board must reach a verdict rather than
  // spin: an engineer opens this screen precisely when things are wrong.
  await page.getByTestId('mode-imaging').click()
  await page.getByTestId('mode-status').click()
  await page.getByTestId('status-refresh').click()

  const card = page.getByTestId('status-card-connection')
  await expect(card).toBeVisible({ timeout: 20_000 })
  const state = await card.getAttribute('data-state')
  const detail = await card.innerText()
  console.log('[groundcrew] connection card =', state, '|', detail.replace(/\n/g, ' '))
  expect(['ok', 'bad']).toContain(state)
  if (state === 'bad') {
    // The simulator's wedge. Naming the stuck call is the difference between a
    // usable report and "something went wrong".
    expect(detail).toMatch(/no response/)
    await page.screenshot({ path: join(SHOTS, '05-status-unresponsive.png') })
  }
  ctx.assertNoJsErrors()
})

test('controls with nothing behind them are disabled, not hidden', async () => {
  const { page } = ctx
  // A control that vanishes reads as a missing feature; one that is greyed out
  // reads as an unavailable reading, which is the truth. The simulator reports
  // neither temperature nor camera position.
  await expect(page.getByTestId('temp-btn')).toBeVisible()
  await expect(page.getByTestId('temp-btn')).toBeDisabled()
  await expect(page.getByTestId('temp-value')).toHaveText('—')
  for (const id of ['extend-btn', 'retract-btn', 'cool-btn', 'warm-btn']) {
    await expect(page.getByTestId(id), `${id} should exist`).toBeVisible()
    await expect(page.getByTestId(id), `${id} should be disabled`).toBeDisabled()
  }

  // Disabled is not enough on its own — a greyed "Extend" could be read as
  // "the camera is retracted". The reason has to be reachable.
  await expect(page.getByTestId('extend-btn'))
    .toHaveAttribute('title', /does not report camera position/)

  // And the link chip must report the LINK, not stand in for hardware state.
  await expect(page.getByTestId('link-state')).toHaveText('Ready')
})

test('the image side panels are absent on the Status board', async () => {
  const { page } = ctx

  // Both panels belong to an IMAGE. On Status they would crowd the cards and
  // imply the board is a view of the current frame, which it is not.
  await page.getByTestId('mode-imaging').click()
  await expect(page.getByTestId('plot-control')).toBeVisible()
  await expect(page.getByTestId('control-panel')).toBeVisible()

  await page.getByTestId('mode-status').click()
  await expect(page.getByTestId('plot-control')).toHaveCount(0)
  await expect(page.getByTestId('control-panel')).toHaveCount(0)

  await page.getByTestId('mode-imaging').click()
  await expect(page.getByTestId('plot-control')).toBeVisible()
})

test('the histogram is SpyDE\'s, with working contrast handles', async () => {
  const { page } = ctx
  await page.getByTestId('mode-imaging').click()

  // Drawn from real counts, not a placeholder strip.
  const hist = page.getByTestId('histogram')
  await expect(hist).toBeVisible({ timeout: 30_000 })
  expect(await hist.locator('rect').count(),
    'expected one rect per histogram bin').toBeGreaterThan(10)

  // Both drag handles present, and the clim labels reading real numbers.
  await expect(page.getByTestId('hist-min-handle')).toBeAttached()
  await expect(page.getByTestId('hist-max-handle')).toBeAttached()
  const before = await page.getByTestId('clim-max').innerText()
  expect(Number(before)).toBeGreaterThan(0)

  // Reset widens to the full data range, so the upper clim must not shrink.
  await page.getByTestId('clim-reset').click()
  await expect.poll(async () =>
    Number(await page.getByTestId('clim-max').innerText()), { timeout: 15_000 })
    .toBeGreaterThanOrEqual(Number(before))

  await page.getByTestId('clim-auto').click()
  await page.screenshot({ path: join(SHOTS, '06-histogram.png') })
  ctx.assertNoJsErrors()
})

test('every mode in the rail opens', async () => {
  const { page } = ctx
  for (const mode of ['imaging', 'motion', 'calibrate', 'status']) {
    await page.getByTestId(`mode-${mode}`).click()
    await expect(page.getByTestId(`mode-${mode}`)).toHaveAttribute('data-active', 'true')
  }
  await page.getByTestId('mode-imaging').click()
  ctx.assertNoJsErrors()
})
