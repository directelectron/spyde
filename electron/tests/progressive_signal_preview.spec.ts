/**
 * progressive_signal_preview.spec.ts — a progressive Find-Vectors run must bring
 * its SIGNAL plot alive DURING the fill, not only its count-map navigator.
 *
 * The result window is a navigator + a signal plot. The navigator has always
 * filled in block by block; the signal plot used to sit on its zero placeholder
 * (black) for the whole run and only came alive when the batch finalized. This
 * spec pins both halves of the fix:
 *
 *   (a) the signal plot shows real content BEFORE `[fv-batch] finalized`, driven
 *       by one sample position from each landing block;
 *   (b) driving the count-map navigator into an already-computed region mid-run
 *       shows that position's frame instead of nothing.
 *
 * Waits are signal-based: the backend's own `[live-signal]` / `[fv-batch]` log
 * lines and pixel polls, never a fixed sleep. Real Dask + the real SPED-Ag scan
 * (208x64 patterns) so the batch spans MANY nav chunks — si_grains is 6x6 and
 * computes as a SINGLE chunk, i.e. it has no progressive phase to observe.
 *
 * (b) is asserted on the BACKEND's own served-read count, not on pixels alone:
 * the signal plot is simultaneously being repainted by landing blocks (that is
 * (a)), so "the picture changed after I dragged" cannot distinguish "the drag
 * was answered" from "a block happened to land just then". The drag itself goes
 * through the harness's `dragCrosshair` — an anyplotlib crosshair is GRABBED,
 * not clicked to, and an earlier version of this spec pressed at an arbitrary
 * point, moved the navigator not at all, and passed anyway.
 *
 * Screenshots land in electron/progressive_signal_shots/ (gitignored; the ones
 * in the PR were copied from there).
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, dragCrosshair,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'progressive_signal_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  test.setTimeout(600_000)     // sped_ag is a real 13k-pattern scan to load
  mkdirSync(SHOTS, { recursive: true })
  // INFO tees the backend's logging to stderr, which the harness buffers — the
  // `[live-signal]` lines are how we know the preview fired during the fill.
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await backendAction(ctx.page, 'load_test_data_sped_ag')
  await waitForSubwindowCount(ctx.page, 2, 420_000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test.setTimeout(600_000)

/**
 * A content signature for ONE subwindow's figure iframe: bright-pixel count plus
 * a checksum. The checksum is what matters — the window chrome (axis labels,
 * scale bar) is canvas-drawn too, so a bare bright count is non-zero even on a
 * black frame, and two different diffraction patterns can share a pixel count.
 */
async function figureSignature(win: any): Promise<{ bright: number; sum: number }> {
  const ifel = await win.locator('iframe').first().elementHandle()
  if (!ifel) return { bright: -1, sum: -1 }
  const frame = await ifel.contentFrame()
  if (!frame) return { bright: -1, sum: -1 }
  try {
    return await frame.evaluate(() => {
      let bright = 0, sum = 0
      for (const c of Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]) {
        const g = c.getContext('2d')
        if (!g || !c.width || !c.height) continue
        const d = g.getImageData(0, 0, c.width, c.height).data
        for (let p = 0; p < d.length; p += 4) {
          const v = d[p] + d[p + 1] + d[p + 2]
          if (v > 90) bright++
          sum = (sum + v * (1 + (p % 7))) % 2147483647
        }
      }
      return { bright, sum }
    })
  } catch { return { bright: -1, sum: -1 } }
}

const logLines = (needle: string): string[] =>
  ctx.backend.logBuffer.filter((l: string) => l.includes(needle))

const finalized = () => logLines('[fv-batch] finalized').length > 0

/**
 * `(N/M positions ready)` pairs from the AUTO-SAMPLE paint line — behaviour (a).
 *
 * Deliberately keyed on `preview frame at`, not on `[live-signal]`: the same
 * subsystem also logs the navigator-read line for (b), which carries the same
 * `(N/M positions ready` text, and counting both would let a navigator read
 * satisfy the "a landing block painted a frame" assertion.
 */
function previewProgress(): Array<[number, number]> {
  return logLines('preview frame at')
    .map((l) => /\((\d+)\/(\d+) positions ready/.exec(l))
    .filter(Boolean)
    .map((m) => [parseInt(m![1], 10), parseInt(m![2], 10)] as [number, number])
}

/**
 * How many navigator-driven reads the preview has ANSWERED from the
 * already-computed region — the backend's own count, parsed from the
 * `navigator read served` line.
 *
 * This is the direct evidence for behaviour (b), and a pixel signature is not a
 * substitute for it: the signal plot is also being repainted by landing blocks
 * (behaviour (a)), so "the picture changed after I dragged" cannot distinguish
 * "the drag was answered" from "a block happened to land just then".
 */
function servedCount(): number {
  const lines = logLines('navigator read served')
  if (!lines.length) return 0
  const m = /(\d+) served/.exec(lines[lines.length - 1])
  return m ? parseInt(m[1], 10) : 0
}

// The crosshair drag itself lives in the harness (`dragCrosshair`): an
// anyplotlib crosshair is GRABBED, not clicked to, so a press at an arbitrary
// point moves nothing — see the helper's own note.

test('the vectors signal plot fills in while the batch is still running', async () => {
  const { page } = ctx

  // ── kick off the batch from the wizard, exactly as a user would ───────────
  const src = sigWindow(page)
  await src.getByTestId('subwindow-title').click()
  await src.getByTestId('subwindow-titlebar').hover()
  await src.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  // NXCORR: no model download, deterministic, and CPU-only — the point of this
  // spec is the progressive display, not which detector found the spots.
  await page.getByTestId('fv-method').click()
  await page.getByTestId('fv-method-opt-nxcorr').click()
  await expect(page.getByTestId('fv-method')).toHaveAttribute('data-value', 'nxcorr')
  await page.screenshot({ path: join(SHOTS, '01-wizard-open.png') })

  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 240_000, message: 'the vectors result window never opened',
  }).toBeGreaterThan(before)

  // The result window pair is the newest two; the SIGNAL one carries an S- pill.
  const results = page.getByTestId('subwindow').filter({ hasText: /Vectors/ })
  const vecSig = results
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .last()
  await expect(vecSig).toBeVisible({ timeout: 60_000 })
  await page.screenshot({ path: join(SHOTS, '02-result-window-opened.png') })

  // ── (a) the signal plot repaints WHILE the batch is still running ─────────
  // The baseline: nothing computed yet, so the vectors DP is still its black
  // placeholder. Every later shot is compared against this one.
  const baseline = await figureSignature(vecSig)
  await page.screenshot({ path: join(SHOTS, '03-fill-start-both-black.png') })

  const during: Array<{ sum: number; bright: number; ready: number; total: number }> = []
  let shot = 4
  for (let i = 0; i < 400 && !finalized(); i++) {
    const sig = await figureSignature(vecSig)
    const prog = previewProgress()
    // Only start capturing once the PREVIEW has painted at least once —
    // before that the plot can still repaint for unrelated reasons (chrome,
    // the hover toolbar) and those frames say nothing about the fill.
    if (!prog.length) { await page.waitForTimeout(400); continue }
    const last = prog[prog.length - 1]
    // One capture per REPAINT: a changed content signature is the thing being
    // asserted, and the `(ready/total)` the preview last logged labels it.
    if (during.length === 0 || during[during.length - 1].sum !== sig.sum) {
      during.push({ ...sig, ready: last[0], total: last[1] })
      await page.screenshot({
        path: join(SHOTS, `${String(shot++).padStart(2, '0')}-during-fill-${last[0]}of${last[1]}.png`),
      })
      if (during.length >= 5) break
    }
    await page.waitForTimeout(400)
  }
  // eslint-disable-next-line no-console
  console.log('during-fill samples:', JSON.stringify(during),
              'baseline:', JSON.stringify(baseline))

  // It came off the black placeholder, not merely "changed somehow".
  expect(during.every((d) => d.bright > baseline.bright),
    `the vectors DP never got brighter than its black placeholder
     (baseline ${JSON.stringify(baseline)}): ${JSON.stringify(during)}`,
  ).toBeTruthy()

  const progress = previewProgress()
  expect(progress.length,
    'the preview never painted a frame during the fill').toBeGreaterThan(0)
  expect(progress.some(([r, t]) => r < t),
    `every preview frame arrived only after the whole scan was ready: ${JSON.stringify(progress)}`,
  ).toBeTruthy()
  expect(new Set(during.map((d) => d.sum)).size,
    `the vectors signal plot never changed while the batch was running: ${JSON.stringify(during)}`,
  ).toBeGreaterThan(1)
  expect(finalized(),
    'the batch finalized before any of this was observed — nothing was proven')
    .toBeFalsy()

  // ── (b) drag the count-map navigator over an already-computed region ──────
  const vecNav = results
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
    .last()
  // Walk LEFT from the crosshair: the result window's right edge sits under the
  // Plot Control dock, and a press that lands on the dock drives nothing.
  const servedBefore = servedCount()
  const dragSigs: number[] = []
  const walk = await dragCrosshair(page, vecNav, {
    dx: -26, steps: 5,
    onStep: async (i: number) => {
      dragSigs.push((await figureSignature(vecSig)).sum)
      await page.screenshot({ path: join(SHOTS, `8${i}-drag-step-${i}.png`) })
    },
  })

  const stillRunning = !finalized()
  const servedAfter = servedCount()
  await page.screenshot({ path: join(SHOTS, '90-drag-over-computed.png') })
  // eslint-disable-next-line no-console
  console.log('drag over computed region: crosshair moved', walk.moved, 'px',
              'signatures', JSON.stringify(dragSigs),
              'served', servedBefore, '→', servedAfter,
              'batch still running:', stillRunning)

  // The guard the previous version of this spec lacked: if the crosshair never
  // moved, everything below is measuring landing blocks, not the drag.
  expect(walk.moved,
    'the navigator crosshair never moved, so nothing about reading an '
    + 'already-computed position was exercised').toBeGreaterThan(20)
  // The backend answering navigator reads from its ready mask IS behaviour (b).
  expect(servedAfter - servedBefore,
    'dragging the navigator over the computed region did not serve a single '
    + 'frame from the already-computed vectors').toBeGreaterThan(0)
  expect(new Set(dragSigs).size,
    `the diffraction pattern did not follow the navigator over the computed
     region (identical frames at every position): ${JSON.stringify(dragSigs)}`,
  ).toBeGreaterThan(1)
  expect(stillRunning,
    'the drag happened after the batch finished, so it proves nothing about '
    + 'reading a partially-computed result').toBeTruthy()

  // Let the batch finish before the app closes (closing mid-batch wedges the
  // hidden backend's stdin tick — see find_vectors_workflow.spec.ts).
  await ctx.backend.waitForLog('[fv-batch] finalized', 420_000)
  await expect.poll(() => figureSignature(vecSig).then((s) => s.bright), {
    timeout: 30_000, message: 'the finalized vectors window is blank',
  }).toBeGreaterThan(0)
  await page.screenshot({ path: join(SHOTS, '99-finalized.png') })

  // NB no blanket backendErrorLines() audit here: a real LocalCluster on a busy
  // dev box emits its own noise (a nanny worker restart, and macOS's
  // `malloc_trim` broadcast raising "Combination not supported" from the
  // post-batch trim) that has nothing to do with this feature.
  ctx.assertNoJsErrors()
})
