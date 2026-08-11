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
 * How many distinct grey levels the largest canvas in the figure iframe shows.
 *
 * A placeholder (zeros) or a failed decode is UNIFORM — one level. A real
 * camera frame has a beam, a noise floor and a corner marker, so it has many.
 * This is the assertion that "the image is there", precisely because a bright-
 * pixel count cannot tell the picture from the white panel behind it.
 */
async function imageGreyLevels(page: any): Promise<number> {
  let best = 0
  for (const frame of page.frames()) {
    try {
      const n = await frame.evaluate(() => {
        let levels = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const el = c as HTMLCanvasElement
          const ctx = el.getContext('2d')
          if (!ctx || !el.width || !el.height) continue
          const d = ctx.getImageData(0, 0, el.width, el.height).data
          const seen = new Set<number>()
          // Every 40th pixel: enough to characterise the histogram without
          // walking megabytes in the page context.
          for (let p = 0; p < d.length; p += 4 * 40) seen.add(d[p])
          levels = Math.max(levels, seen.size)
        }
        return levels
      })
      best = Math.max(best, n)
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
    env: { GROUNDCREW_LOG_LEVEL: 'INFO' },
    // The main process echoes this when the backend's `ready` lands. Waiting on
    // the PLOTAPP line itself does NOT work: main consumes that channel, so it
    // never reaches the app's stdout that the harness watches.
    readyLog: '[groundcrew backend] ready',
    readyMessages: [],
  })
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

  // 3. Frames are actually flowing. Poll the frame counter for a real increase
  //    — a signal from the backend, not a fixed sleep.
  await expect
    .poll(async () => Number(await page.getByTestId('stat-frame').innerText()),
      { timeout: 30_000, message: 'camera never delivered a second frame' })
    .toBeGreaterThan(1)

  await page.screenshot({ path: join(SHOTS, '02-live.png') })

  // 4. The image actually PAINTED — and painted the CAMERA's pixels, not the
  //    figure's chrome.
  //
  //    Counting bright pixels is NOT enough and gave a false pass here: the
  //    figure's own white background is bright, so the assertion held at 322k
  //    pixels while the image pane was uniformly black. What distinguishes a
  //    live frame from a placeholder is that it is not uniform, so measure the
  //    spread of DISTINCT grey levels inside the image canvas.
  const spread = await imageGreyLevels(page)
  console.log('[groundcrew] distinct grey levels in the image =', spread)
  expect(spread, 'image pane is uniform — the placeholder, not a camera frame')
    .toBeGreaterThan(10)

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
