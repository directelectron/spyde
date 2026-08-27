/**
 * dpc_live_region.spec.ts — the DPC field map must TRACK the beam region as it
 * is dragged, the same way a virtual image tracks its detector ROI.
 *
 * Nothing headless can see this. The Python suite can prove a drag frame asks
 * for a re-measure and that the frames coalesce; it cannot prove the map on
 * screen actually changed, which is the entire user-visible claim. Before the
 * live lane, this exact drag produced ZERO repaints across 6.4 s — the map sat
 * frozen for the whole gesture and only caught up on release, because
 * `_on_region_drag` armed its debounce on `pointer_up` alone.
 *
 * So the measurement is: sample the map's own CANVAS pixels on every drag step
 * and count how many DISTINCT frames appear. Measured both ways on this drag:
 * release-only scores 0, the live lane scores 12 of 24.
 *
 * Run: npx playwright test tests/dpc_live_region.spec.ts --project=electron \
 *        --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
import * as fs from 'fs'
import * as path from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = path.join(__dirname, '..', 'dpc_live_shots')
const DPC_TITLE = /DPC Field Map/

/** Steps sampled during the drag. Each is one real pointer move. */
const STEPS = 24

/**
 * How many of those steps must show a CHANGED map.
 *
 * Deliberately far below `STEPS`: the pass is superseded and restarted several
 * times a second, so how many repaints land inside a 120 ms sampling window is
 * a property of the machine, not of the code. What is being pinned is the
 * difference between "tracks the pointer" and "frozen until release", and any
 * value well above zero says that. Measured: 0 release-only, 12 with the lane.
 */
const MIN_LIVE_FRAMES = 6

test('the DPC field map tracks the beam region while it is dragged', async () => {
  test.setTimeout(600_000)
  fs.mkdirSync(SHOTS, { recursive: true })

  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    // LAZY + chunked, so the pass streams through the real cluster and takes
    // long enough for "does it restart mid-drag?" to be answerable at all. On
    // eager data every pass finishes instantly and the question is vacuous.
    await backendAction(page, 'load_test_data_dpc',
      { nav: 48, sig: 64, lazy: true, nav_chunk: 8 })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(3_000)
    await page.screenshot({ path: `${SHOTS}/01-loaded.png` })

    const sig = page.getByTestId('subwindow')
      .filter({ has: page.getByTestId('subwindow-title').filter({ hasText: /S-Synthetic DPC$/ }) })
      .last()
    await sig.getByTestId('subwindow-title').click()
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-DPC').click()
    await expect(sig.getByTestId('dpc-wizard')).toBeVisible({ timeout: 60_000 })

    // The first pass has to land before we drag, or "the map changed" would be
    // measuring the opening pass filling in rather than the drag.
    await expect.poll(
      () => sig.getByTestId('dpc-centering').getAttribute('data-worst'),
      { timeout: 180_000, message: 'the opening pass never produced a field' })
      .toMatch(/\d/)
    await page.screenshot({ path: `${SHOTS}/02-first-pass.png` })

    await sig.getByTestId('dpc-tab-Center').click()
    await sig.getByTestId('dpc-beam-circle').click()
    await expect(sig.getByTestId('dpc-beam-r')).toBeVisible()
    await page.waitForTimeout(4_000)
    await page.screenshot({ path: `${SHOTS}/03-beam-on.png` })

    const dpcWin = page.getByTestId('subwindow')
      .filter({ has: page.getByTestId('subwindow-title').filter({ hasText: DPC_TITLE }) })
      .first()
    await expect(dpcWin).toBeVisible({ timeout: 60_000 })

    /**
     * A fingerprint of the MAP'S OWN pixels, read off its canvases.
     *
     * NOT a screenshot of the window. The "Calculating…" chip sits on top of
     * the figure and pulses on a 1.6 s CSS animation, so a screenshot hash
     * changes on almost every sample while a pass is running — which is most of
     * a drag. Measured with a window screenshot, the RELEASE-ONLY behaviour this
     * spec exists to catch scored 22 of 24 "live frames" and the spec passed. A
     * canvas readback sees only what the figure drew.
     */
    const mapHash = async (): Promise<string> => {
      const host = await dpcWin.elementHandle()
      if (!host) return 'no-window'
      const parts: string[] = []
      for (const frame of page.frames()) {
        const el = await frame.frameElement().catch(() => null)
        if (!el) continue
        const inside = await host.evaluate(
          (w, f) => w.contains(f as Node), el).catch(() => false)
        if (!inside) continue
        parts.push(await frame.evaluate(() => {
          let h = 0, n = 0
          for (const c of Array.from(document.querySelectorAll('canvas'))) {
            const cv = c as HTMLCanvasElement
            const g = cv.getContext('2d')
            if (!g || !cv.width) continue
            const d = g.getImageData(0, 0, cv.width, cv.height).data
            for (let i = 0; i < d.length; i += 41) { h = (h * 31 + d[i]) | 0; n++ }
          }
          return `${n}:${h}`
        }).catch(() => 'err'))
      }
      return parts.join('|')
    }

    /**
     * Where the beam circle is on screen, from its colour (#94e2d5).
     *
     * The ring is symmetric, so the centroid of its teal pixels IS its centre —
     * which is where the drag handle sits. Guessing a point inside the circle
     * instead grabs nothing: the cursor readout still tracks, so the figure
     * looks driven while the circle stays put, and the run reports "0 frames"
     * for a drag that never happened. That is indistinguishable from the bug,
     * which is why the grab below is verified rather than assumed.
     */
    const beamCentre = async () => {
      const host = await sig.elementHandle()
      if (!host) return null
      for (const frame of page.frames()) {
        const el = await frame.frameElement().catch(() => null)
        if (!el) continue
        const inside = await host.evaluate(
          (w, f) => w.contains(f as Node), el).catch(() => false)
        if (!inside) continue
        const hit = await frame.evaluate(() => {
          let sx = 0, sy = 0, n = 0
          for (const c of Array.from(document.querySelectorAll('canvas'))) {
            const cv = c as HTMLCanvasElement
            const g = cv.getContext('2d')
            if (!g || !cv.width) continue
            const d = g.getImageData(0, 0, cv.width, cv.height).data
            const box = cv.getBoundingClientRect()
            const kx = box.width / cv.width, ky = box.height / cv.height
            for (let i = 0; i < d.length; i += 4) {
              const r = d[i], gg = d[i + 1], b = d[i + 2]
              if (!(r > 110 && r < 175 && gg > 200 && b > 185 && b < 240)) continue
              const px = (i / 4) % cv.width, py = Math.floor((i / 4) / cv.width)
              sx += box.left + px * kx; sy += box.top + py * ky; n++
            }
          }
          return n > 8 ? { x: sx / n, y: sy / n } : null
        }).catch(() => null)
        if (hit) {
          const fb = await el.boundingBox()
          return { x: (fb?.x ?? 0) + hit.x, y: (fb?.y ?? 0) + hit.y }
        }
      }
      return null
    }

    const sigBox = await sig.boundingBox()
    const start = await beamCentre()
    if (!sigBox || !start) throw new Error('could not find the beam circle')

    // Prove the grab took before measuring anything (see beamCentre).
    let grabbed = false
    for (let attempt = 0; attempt < 4 && !grabbed; attempt++) {
      await page.mouse.move(start.x, start.y)
      await page.mouse.down()
      await page.mouse.move(start.x, start.y + 12)
      await page.waitForTimeout(250)
      const now = await beamCentre()
      grabbed = !!now && Math.abs(now.y - start.y) > 3
      if (!grabbed) {
        await page.mouse.up()
        await page.waitForTimeout(400)
      }
    }
    expect(grabbed, 'never managed to grab the beam circle').toBe(true)

    const seen: string[] = [await mapHash()]
    for (let i = 1; i <= STEPS; i++) {
      const t = i / STEPS
      await page.mouse.move(
        start.x + Math.sin(t * Math.PI * 2) * sigBox.width * 0.12,
        start.y + t * sigBox.height * 0.18)
      await page.waitForTimeout(120)
      const h = await mapHash()
      if (h !== seen[seen.length - 1]) seen.push(h)
    }
    await page.mouse.up()
    const liveFrames = seen.length - 1
    await page.screenshot({ path: `${SHOTS}/04-drag-end.png` })

    console.log(`[dpc-live] distinct map frames across ${STEPS} drag steps: ${liveFrames}`)
    expect(liveFrames,
      'the field map did not repaint while the beam region was dragged — ' +
      'the re-measure is waiting for pointer_up again')
      .toBeGreaterThanOrEqual(MIN_LIVE_FRAMES)

    // The region moved, so the RESTING field must differ from the one the map
    // opened with. Without the trailing settle the map would be left showing
    // whichever superseded partial happened to be up when the pointer stopped.
    await expect.poll(mapHash, {
      timeout: 90_000,
      message: 'the settle never measured the resting region',
    }).not.toBe(seen[seen.length - 1])
    await page.screenshot({ path: `${SHOTS}/05-after-settle.png` })

    expect(backendErrorLines(backend), 'the backend reported an error')
      .toEqual([])
  } finally {
    assertNoJsErrors()
    await ctx.app?.close()
  }
})
