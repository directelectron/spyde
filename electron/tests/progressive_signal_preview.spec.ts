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
  crosshairAt,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'progressive_signal_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  test.setTimeout(600_000)     // sped_ag is a real 13k-pattern scan to load
  mkdirSync(SHOTS, { recursive: true })
  // INFO tees the backend's logging to stderr, which the harness buffers — the
  // `[live-signal]` lines are how we know the preview fired during the fill.
  // SPYDE_TEST_HOLD parks the find-vectors batch once ~35% of the scan has
  // been computed and keeps it parked until this spec releases it
  // (backend/test_hold.py). That turns "observe a partially-computed result"
  // from a race into a state we can take our time over — which is what part
  // (b) needs, since it must have the batch RUNNING and the navigator over
  // COMPUTED data at the same instant.
  ctx = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'INFO', SPYDE_TEST_HOLD: 'fv-batch@0.35' },
  })
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
 * Has SPYDE_TEST_HOLD parked the batch? Part (a) samples the fill and part (b)
 * needs it PARKED, so (a) must stop the moment the park lands: a parked batch
 * produces no new blocks, so (a)'s "capture each repaint" loop would otherwise
 * spin out its full budget against a frozen picture — and burn the hold's own
 * MAX_HOLD_S (120 s) doing it, so (b) ran after the park had already expired.
 */
const parked = () => logLines('[test-hold] fv-batch parked').length > 0

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

/**
 * How many navigator-driven reads the preview has DECLINED (position not yet
 * computed → the last good frame stays up), from the cumulative count on the
 * backend's `navigator read declined` line.
 *
 * The diagnostic counterpart of `servedCount`: served and declined move
 * together on the backend (`ProgressiveSignalPreview` counts every read one
 * way or the other), so when a drag serves nothing this tells apart "the
 * reads ran but were over uncomputed data" (declined moves) from "no read
 * reached the preview at all" (neither moves — a wedged dispatcher or a grab
 * that missed). The line is throttled (~2 s) but carries the CUMULATIVE
 * count, and a multi-second all-declined drag always emits at least one.
 */
function declinedCount(): number {
  const lines = logLines('navigator read declined')
  if (!lines.length) return 0
  const m = /(\d+) declined/.exec(lines[lines.length - 1])
  return m ? parseInt(m[1], 10) : 0
}

// The crosshair drag itself lives in the harness (`dragCrosshair`): an
// anyplotlib crosshair is GRABBED, not clicked to, so a press at an arbitrary
// point moves nothing — see the helper's own note.

/**
 * The backend's `[test-aim] walk target …` line → where to walk, in NAV INDEX
 * deltas from wherever the crosshair currently is.
 *
 * The backend picks the target (only it can see the ready mask) but does NOT
 * move anything — the drag below does, with the mouse, from the crosshair the
 * renderer actually drew. See `_test_aim_ready_position` for why the earlier
 * "park it from the backend, then grab it" version could not work.
 */
function walkTarget(): { dix: number; diy: number; readyLeft: number;
                         navW: number; navH: number } | null {
  const line = logLines('[test-aim] walk target').pop()
  if (!line) return null
  const m = /dix=(-?\d+) diy=(-?\d+), (\d+) ready columns to the left; nav (\d+)x(\d+)/
    .exec(line)
  if (!m) return null
  return {
    dix: parseInt(m[1], 10), diy: parseInt(m[2], 10), readyLeft: parseInt(m[3], 10),
    navW: parseInt(m[4], 10), navH: parseInt(m[5], 10),
  }
}

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

  // `mid` records whether the batch was STILL RUNNING when this sample was
  // taken. Asserting `!finalized()` at the END of the test instead was a race:
  // the batch finishing is inevitable, not a failure, and on a fast runner it
  // finished before five distinct repaints had been captured — so a correct
  // progressive fill failed with "nothing was proven". What matters is when the
  // SAMPLES were taken, which is knowable exactly.
  const during: Array<{
    sum: number; bright: number; ready: number; total: number; mid: boolean
  }> = []
  let shot = 4
  for (let i = 0; i < 400 && !finalized(); i++) {
    const sig = await figureSignature(vecSig)
    const prog = previewProgress()
    // Stop as soon as the hold parks the batch: no more blocks land, so no more
    // repaints can be captured, and every further iteration spends the hold's
    // own 120 s budget that part (b) below needs (see `parked`).
    if (parked() && prog.length) break
    // Only start capturing once the PREVIEW has painted at least once —
    // before that the plot can still repaint for unrelated reasons (chrome,
    // the hover toolbar) and those frames say nothing about the fill.
    if (!prog.length) { await page.waitForTimeout(400); continue }
    const last = prog[prog.length - 1]
    // One capture per REPAINT: a changed content signature is the thing being
    // asserted, and the `(ready/total)` the preview last logged labels it.
    if (during.length === 0 || during[during.length - 1].sum !== sig.sum) {
      // Read `finalized()` at CAPTURE time, not afterwards.
      during.push({ ...sig, ready: last[0], total: last[1], mid: !finalized() })
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

  // The two claims are asserted from DIFFERENT evidence, deliberately.
  //
  // "The fill was PROGRESSIVE" is proved by the backend log (below): a preview
  // frame logged with `ready < total` can only have been painted mid-run. That
  // is exact, and it is why the handler logs at INFO.
  //
  // "The DP came off its black placeholder" is proved by PIXELS — but sampled
  // when the answer is settled, not while racing the fill. Polling for a
  // brighter frame mid-run is what made this spec flaky: samples are captured
  // when the content signature MOVES, which is not the same condition as
  // having got BRIGHTER, so a capture could legitimately land on a repaint
  // that changed `sum` and nothing else. Observed both ways — a first sample
  // exactly at baseline brightness, and a whole run where no sample was
  // brighter — while the fill itself was working fine.
  await expect
    .poll(async () => (await figureSignature(vecSig)).bright,
      { timeout: 180_000, message: 'the vectors DP never came off its black placeholder' })
    .toBeGreaterThan(baseline.bright)
  await page.screenshot({ path: join(SHOTS, '20-came-off-black.png') })

  const progress = previewProgress()
  expect(progress.length,
    'the preview never painted a frame during the fill').toBeGreaterThan(0)
  expect(progress.some(([r, t]) => r < t),
    `every preview frame arrived only after the whole scan was ready: ${JSON.stringify(progress)}`,
  ).toBeTruthy()
  // The pixel samples are kept for the screenshots and the log line above, but
  // are no longer ASSERTED on: how many distinct repaints a poll happens to
  // catch is a property of the sampling, not of the fill. `ready < total`
  // already proves the fill was progressive, and the brightness poll above
  // proves it reached the display — neither depends on catching a transient.

  // ── (b) drag the count-map navigator over an already-computed region ──────
  const vecNav = results
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
    .last()
  // The claim under test is "dragging over an ALREADY-COMPUTED region serves
  // that region's frames", which needs the drag to actually be over one.
  // The batch is PARKED at ~35% by SPYDE_TEST_HOLD (see beforeAll and
  // backend/test_hold.py), so from here to the release below the partially
  // computed state is a fact rather than a race. Two earlier attempts at this
  // without the hold both failed: polling `previewProgress()` for a fraction
  // reads the preview's throttled PAINT log, which stops updating once
  // painting stops and so never clears; and simply dragging fast enough lost
  // to a batch that had already finished (`batch still running: false`).
  await ctx.backend.waitForLog('[test-hold] fv-batch parked', 240_000)

  // AIM FROM THE BACKEND'S OWN READY MASK — never from a fill-direction guess.
  // An earlier version dragged the crosshair UP into "the top rows", assuming
  // the batch fills row-major. It does not: dispatch_chunks submits in
  // column-BANDED order (_band_key: 2-row bands, left→right), the sped_ag
  // chunk grid is uneven (2×5 — the bottom-row chunks are 2.2× smaller and
  // finish first), and completion order is scheduler-jittered across the
  // primed in-flight window. So the parked ready set differs run to run, and
  // the up-then-left walk sometimes crossed NO computed chunk at all: served
  // stayed 0 and five identical DP signatures followed — the designed
  // last-good-frame behaviour for uncomputed positions, not a serving bug
  // (that CI failure is what `declinedCount` now makes diagnosable).
  //
  // `test_aim_ready_position` REPORTS a target from the live preview's ready
  // mask (the ground truth its slice function serves from) as a nav-index
  // DELTA from the crosshair's current position, near the right end of the
  // longest computed run so the leftward walk stays over computed data. It
  // moves nothing: a previous version parked the crosshair from the backend
  // and the press on the redrawn crosshair delivered NO pointer event at all
  // (zero nav-dispatcher submits — the renderer's widget state and the
  // backend's had come apart), which read as a serving bug and was not one.
  // The whole drag below is mouse-driven from the crosshair the renderer
  // actually drew, which is the path vectors_dp_follows_nav.spec.ts already
  // proves works on this same window.
  //
  // FOCUS-RAISE FIRST (the z-order trap): the first pointerdown on an
  // unfocused subwindow raises it and is NOT delivered to the iframe widget —
  // so a walk whose press doubles as the focus click grabs nothing and the
  // crosshair never moves (observed: the hover readout walked to [0, 43]
  // while the crosshair stayed parked).
  await vecNav.getByTestId('subwindow-title').click()
  await backendAction(page, 'test_aim_ready_position')
  await ctx.backend.waitForLog('[test-aim] walk target', 30_000)
  const target = walkTarget()
  expect(target, 'the backend did not report a ready walk target').not.toBeNull()

  // Nav index → page px, from the crosshair guide lines' own extent (they span
  // the drawn image exactly, so that rect IS the navigator image on screen).
  const geom = await crosshairAt(vecNav)
  expect(geom, 'no crosshair found on the vectors navigator').not.toBeNull()
  const pxPerCol = (geom!.x1 - geom!.x0) / target!.navW
  const pxPerRow = (geom!.y1 - geom!.y0) / target!.navH
  // Walk LEFT over the computed run, in 5 steps that each move at least a
  // couple of nav columns (a sub-pixel step would repaint nothing).
  const steps = 5
  const colsPerStep = Math.max(
    2, Math.floor(Math.min(target!.readyLeft - 2, 80) / steps))
  const dxPx = -Math.max(3, Math.round(colsPerStep * pxPerCol))
  // eslint-disable-next-line no-console
  console.log('walk target:', JSON.stringify(target), 'px/col', pxPerCol.toFixed(2),
              'px/row', pxPerRow.toFixed(2), 'step px', dxPx)

  const servedBefore = servedCount()
  const declinedBefore = declinedCount()
  const dragSigs: number[] = []
  const walk = await dragCrosshair(page, vecNav, {
    dx: dxPx, steps,
    seekDx: Math.round(target!.dix * pxPerCol),
    seekDy: Math.round(target!.diy * pxPerRow),
    onStep: async (i: number) => {
      dragSigs.push((await figureSignature(vecSig)).sum)
      await page.screenshot({ path: join(SHOTS, `8${i}-drag-step-${i}.png`) })
    },
  })

  const stillRunning = !finalized()
  const servedAfter = servedCount()
  const declinedAfter = declinedCount()
  await page.screenshot({ path: join(SHOTS, '90-drag-over-computed.png') })
  // eslint-disable-next-line no-console
  console.log('drag over computed region: crosshair moved', walk.moved, 'px',
              'signatures', JSON.stringify(dragSigs),
              'served', servedBefore, '→', servedAfter,
              'declined', declinedBefore, '→', declinedAfter,
              'batch still running:', stillRunning)
  if (servedAfter - servedBefore <= 0) {
    // A zero-served drag has several distinct causes (reads declined, reads
    // never ran, the preview torn down, the hold expired mid-test) that only
    // the backend's own narration can tell apart — dump it before asserting.
    // eslint-disable-next-line no-console
    console.log('zero-served diagnostics:', JSON.stringify({
      hold: logLines('[test-hold]').slice(-4),
      aim: logLines('[test-aim]').slice(-3),
      live: logLines('[live-signal]').slice(-8),
      batch: logLines('[fv-batch]').slice(-4),
    }, null, 1))
  }

  // The guard the previous version of this spec lacked: if the crosshair never
  // moved, everything below is measuring landing blocks, not the drag. Scored
  // against the walk this run actually asked for (the step size is derived from
  // the reported ready run, so a fixed threshold would be arbitrary) — and half
  // of it, not all, because the crosshair legitimately clamps at the image edge.
  const askedPx = Math.hypot(
    Math.round(target!.dix * pxPerCol) + dxPx * steps,
    Math.round(target!.diy * pxPerRow))
  expect(walk.moved,
    `the navigator crosshair barely moved (${walk.moved.toFixed(1)} px of the `
    + `${askedPx.toFixed(1)} px asked for), so nothing about reading an `
    + 'already-computed position was exercised').toBeGreaterThan(askedPx / 2)
  // Two failure modes, told apart BEFORE the real assertion: the backend
  // counts every navigator read one way or the other (served or declined), so
  // neither moving means the reads never reached the preview at all (a wedged
  // dispatcher, a grab that missed) — a different bug from an aim that landed
  // over uncomputed data (declined moves, served does not).
  expect(
    (servedAfter - servedBefore) + (declinedAfter - declinedBefore),
    'no navigator read reached the live preview during the drag at all — the '
    + 'read path was never exercised (this is NOT an aim/ready-region problem)',
  ).toBeGreaterThan(0)
  // The backend answering navigator reads from its ready mask IS behaviour (b).
  expect(servedAfter - servedBefore,
    'dragging the navigator over the aimed READY region did not serve a single '
    + `frame (declined ${declinedBefore} → ${declinedAfter}: the reads ran but `
    + 'landed on uncomputed positions — the ready-mask aim failed)',
  ).toBeGreaterThan(0)
  expect(new Set(dragSigs).size,
    `the diffraction pattern did not follow the navigator over the computed
     region (identical frames at every position): ${JSON.stringify(dragSigs)}`,
  ).toBeGreaterThan(1)
  expect(stillRunning,
    'the drag happened after the batch finished, so it proves nothing about '
    + 'reading a partially-computed result').toBeTruthy()

  // ── (c) the window is CLOSED for business while it fills ──────────────────
  // The tree is locked for the duration of the batch (no actions, no new
  // nodes), which is what makes the preview's install-once link snapshot valid.
  // The backend refuses a click either way, but a button that looks clickable
  // and then errors is a worse app than one that shows it is unavailable — so
  // the toolbar config carries `disabled` while locked. The toolbar is
  // reveal-on-hover, hence the hover before looking.
  await vecSig.getByTestId('subwindow-titlebar').hover()
  const lockedBtn = vecSig.getByTestId(/^action-btn-/).first()
  await expect(lockedBtn).toBeVisible({ timeout: 10_000 })
  await page.screenshot({ path: join(SHOTS, '95-toolbar-greyed-while-filling.png') })
  const lockedNames = await vecSig.getByTestId(/^action-btn-/).all()
  for (const btn of lockedNames) {
    await expect(btn, 'a toolbar button stayed enabled on a window whose '
      + 'find-vectors batch is still filling it').toBeDisabled()
  }
  // eslint-disable-next-line no-console
  console.log('toolbar while locked:', lockedNames.length, 'buttons, all disabled')

  // Let the batch finish before the app closes (closing mid-batch wedges the
  // hidden backend's stdin tick — see find_vectors_workflow.spec.ts).
  // Let the parked batch run to completion — without this the finalize wait
  // below would sit until the hold's own MAX_HOLD_S timeout.
  await backendAction(page, 'test_hold_release', { name: 'fv-batch' })
  await ctx.backend.waitForLog('[fv-batch] finalized', 420_000)

  // …and the lock RELEASES: the same buttons come back live once the vectors
  // attach. (Asserted on the FIRST one only: the finished window also GAINS the
  // requires_vectors actions, so the set is not the same set.)
  await vecSig.getByTestId('subwindow-titlebar').hover()
  await expect(vecSig.getByTestId(/^action-btn-/).first(),
    'the toolbar stayed greyed after the batch finished — the lock was not '
    + 'released, or its release did not re-send the toolbar config',
  ).toBeEnabled({ timeout: 30_000 })
  await page.screenshot({ path: join(SHOTS, '96-toolbar-restored.png') })
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
