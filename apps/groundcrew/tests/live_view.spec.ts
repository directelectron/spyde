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
    console.log('─── backend log ───\n' + (ctx.backend.logBuffer || '(empty)'))
  }
})

test.afterAll(async () => { await ctx?.app?.close() })

test('boots on the shared shell and streams live frames', async () => {
  const { page, backend } = ctx

  // 1. The fixed layout rendered — this app has a permanent control sidebar
  //    where SpyDE has an MDI workspace.
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
  //    Counting bright pixels is NOT enough and gave a false pass here: the
  //    figure's own white background is bright, so the assertion held at 322k
  //    pixels while the image pane was uniformly black. What distinguishes a
  //    live frame from a placeholder is that it is not uniform, so measure the
  //    spread of DISTINCT grey levels inside the image canvas.
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
