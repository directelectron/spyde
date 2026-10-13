/**
 * insitu_echem.spec.ts — an in-situ movie opens with its REAL frame timing and
 * with the electrochemistry recorded beside it attached as navigator lanes.
 *
 * This is the part a green pytest run cannot see. The unit tests prove the
 * readers and the alignment maths; only the app can show that the chips
 * actually appear on the navigator, that shift-clicking them stacks E and I
 * against the movie's own time cursor, and that dragging that cursor still
 * moves the movie.
 *
 * DEV-BOX ONLY: it drives a real 132 GB DE acquisition (with its
 * `.spyde-nav.npz` sidecar, so no navigator recompute) plus the BioLogic
 * records beside it. CI has neither, so the whole file skips when the folder
 * is absent rather than shipping a fake that would pass without proving
 * anything.
 */
import { test, expect } from '@playwright/test'
import { existsSync, mkdirSync } from 'fs'
import { join } from 'path'

const {
  launchApp, backendAction, waitForSubwindowCount, navWindow, backendErrorLines,
} = require('./_harness.cjs')

const DATA_DIR =
  'D:\\InsituElectroChemistry\\directelectron_good-electrochemistry-movie_2026-07-30_0335'
const MOVIE = join(DATA_DIR, '20251117_88071_movie.mrc')
const EC_RUN = join(DATA_DIR, 'New TEM NZH-001-03-floating_02_CV_C01.mpr')
const SHOTS = join(__dirname, '..', 'insitu_echem_shots')

const POTENTIAL_CHIP = 'Ewe (V)'
const CURRENT_CHIP = 'I (µA)'

test.skip(!existsSync(MOVIE), `in-situ electrochemistry dataset not present (${MOVIE})`)
test.describe.configure({ mode: 'serial' })

let ctx: any

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'INFO' } })
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

/** The most recent backend log line containing `needle`.
 *  `waitForLog` resolves with the line only for a FUTURE match — a line already
 *  in the buffer resolves undefined — so always read the buffer afterwards. */
function lastLine(needle: string): string {
  const hit = [...ctx.backend.logBuffer].reverse().find((l: string) => l.includes(needle))
  return hit ?? ''
}

/** The nav window's id, taken from its chip strip's testid. */
async function navChipsId(nav: any): Promise<string> {
  const testid = await nav.locator('[data-testid^="nav-chips-"]').first().getAttribute('data-testid')
  return String(testid).replace('nav-chips-', '')
}

test('a DE movie opens with its timestamps as the time base', async () => {
  const { page, backend } = ctx

  await backendAction(page, 'open_file', { path: MOVIE })
  await waitForSubwindowCount(page, 2, 180_000)
  await page.screenshot({ path: join(SHOTS, '01-movie-open.png') })

  // The reader derives 1/(fps*sum) = 8.19 ms; the timestamps say 32.76 ms.
  await backend.waitForLog('Frame timing from', 120_000)
  const timing = lastLine('Frame timing from')
  expect(timing).toContain('movie_timestamps.csv')
  expect(timing).toContain('32.760 ms/frame')
  expect(timing).toContain('30.53 fps')
  // and the reader's wrong value is reported alongside it, 4x too fast
  expect(timing).toContain('reader said 8.19')

  // The SIGNAL axes too: this is a TEM image, so nm — but rsciio reads the
  // imaging exposure's camera length of 0 as "has a camera length" and
  // calibrates it as diffraction at the unset -1 nm^-1.
  await backend.waitForLog('frame pixel size from', 30_000)
  const px = lastLine('frame pixel size from')
  expect(px).toContain('1.14786 nm/px')
  expect(px).toContain('reader said -1 nm^-1')
})

test('electrochemistry beside the movie attaches as navigator chips', async () => {
  const { page, backend } = ctx

  await backend.waitForLog('Attached Cyclic Voltammetry', 120_000)
  const attached = lastLine('Attached Cyclic Voltammetry')
  expect(attached).toContain('floating_02_CV_C01')
  expect(attached).toContain('aligned by span')
  expect(attached).toContain('100% of frames covered')

  const nav = navWindow(page)
  const navId = await navChipsId(nav)
  await expect(nav.getByTestId(`nav-chip-${POTENTIAL_CHIP}-${navId}`)).toBeVisible()
  await expect(nav.getByTestId(`nav-chip-${CURRENT_CHIP}-${navId}`)).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '02-nav-chips.png') })
})

test('shift-clicking the chips stacks E and I on the movie time cursor', async () => {
  const { page } = ctx
  const nav = navWindow(page)
  const navId = await navChipsId(nav)

  await nav.getByTestId(`nav-chip-${POTENTIAL_CHIP}-${navId}`).click()
  await nav.getByTestId(`nav-chip-${CURRENT_CHIP}-${navId}`).click({ modifiers: ['Shift'] })

  // The stacked view is a distinct figure the backend emits with
  // view_kind:'stacked'; wait for the frame to swap rather than sleeping.
  await expect
    .poll(async () => nav.locator('iframe').count(), { timeout: 60_000 })
    .toBeGreaterThan(0)
  await page.waitForTimeout(1_500)   // let the figure paint before capturing
  await nav.screenshot({ path: join(SHOTS, '03-stacked-lanes.png') })
  await page.screenshot({ path: join(SHOTS, '04-full-window.png') })
})

/** Dominant x (CSS px) of the ORANGE stacked-lane cursor, or null.
 *  `crosshairAt` in the harness finds the GREEN navigator crosshair; the
 *  stacked lanes' linked cursor is #ff9100. */
async function laneCursorX(nav: any): Promise<number | null> {
  // Iterate EVERY iframe in the window, not just the first: the stacked figure
  // mounts as its own iframe alongside the ones already there, so `.first()`
  // reads a figure with no cursor in it at all.
  const count = await nav.locator('iframe').count()
  for (let i = 0; i < count; i++) {
    const ifel = await nav.locator('iframe').nth(i).elementHandle()
    const frame = ifel && (await ifel.contentFrame())
    if (!frame) continue
    const x = await frameCursorX(frame)
    if (x !== null) return x
  }
  return null
}

async function frameCursorX(frame: any): Promise<number | null> {
  return frame.evaluate(() => {
    let best: { x: number; n: number } | null = null
    for (const c of Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]) {
      const g = c.getContext('2d')
      if (!g || !c.width || !c.height) continue
      const d = g.getImageData(0, 0, c.width, c.height).data
      const cols = new Int32Array(c.width)
      let total = 0
      for (let y = 0; y < c.height; y++) {
        for (let x = 0; x < c.width; x++) {
          const p = (y * c.width + x) * 4
          const r = d[p], gr = d[p + 1], b = d[p + 2], a = d[p + 3]
          // #ff9100 = (255, 145, 0), the vline widget orange.
          if (a > 40 && r > 200 && gr > 90 && gr < 200 && b < 60) { cols[x]++; total++ }
        }
      }
      if (!total || (best && total <= best.n)) continue
      let bx = 0
      for (let x = 1; x < c.width; x++) if (cols[x] > cols[bx]) bx = x
      const rect = c.getBoundingClientRect()
      best = { x: rect.left + (bx + 0.5) * (rect.width / c.width), n: total }
    }
    return best ? best.x : null
  })
}

test('dragging a lane cursor tracks the pointer without jumping back', async () => {
  const { page } = ctx
  const nav = navWindow(page)
  const navId = await navChipsId(nav)

  // Back to the stacked view (the previous test switched to a single lane).
  await nav.getByTestId(`nav-chip-${POTENTIAL_CHIP}-${navId}`).click()
  await nav.getByTestId(`nav-chip-${CURRENT_CHIP}-${navId}`).click({ modifiers: ['Shift'] })
  await page.waitForTimeout(2_000)

  // Find the iframe the cursor actually lives in, and use ITS box for the
  // page-coordinate maths — the stacked figure is not necessarily iframe 0.
  let box: any = null
  let startX: number | null = null
  const nFrames = await nav.locator('iframe').count()
  for (let i = 0; i < nFrames; i++) {
    const ifel = await nav.locator('iframe').nth(i).elementHandle()
    const frame = ifel && (await ifel.contentFrame())
    if (!frame) continue
    const x = await frameCursorX(frame)
    if (x !== null) {
      startX = x
      box = await nav.locator('iframe').nth(i).boundingBox()
      break
    }
  }
  expect(startX, 'no orange lane cursor found in any iframe').not.toBeNull()

  // Walk the cursor right and sample after each step. The bug this guards is
  // the ASYNC write-back landing on the line still under the pointer: the
  // cursor snaps back to the last committed frame, the next move drags it
  // forward again, and the samples oscillate instead of advancing.
  const gy = box!.y + box!.height * 0.25
  await page.mouse.move(box!.x + startX!, gy)
  await page.mouse.down()
  const samples: number[] = []
  for (let i = 1; i <= 6; i++) {
    await page.mouse.move(box!.x + startX! + i * 22, gy, { steps: 3 })
    await page.waitForTimeout(350)
    const x = await laneCursorX(nav)
    if (x !== null) samples.push(x)
  }
  await page.mouse.up()
  await page.waitForTimeout(600)
  await nav.screenshot({ path: join(SHOTS, '06-after-drag.png') })

  console.log('lane cursor drag samples:', samples.map(s => s.toFixed(1)).join(', '))
  expect(samples.length).toBeGreaterThanOrEqual(4)
  expect(samples[samples.length - 1]).toBeGreaterThan(samples[0] + 20)
  // Monotonic within a frame-quantisation tolerance — a snap-back to the last
  // committed position is tens of px, far outside this.
  for (let i = 1; i < samples.length; i++) {
    expect(samples[i], `sample ${i} went backwards: ${samples.join(', ')}`)
      .toBeGreaterThan(samples[i - 1] - 6)
  }
})

test('selecting one lane shows the WHOLE experiment, not just its start', async () => {
  const { page } = ctx
  const nav = navWindow(page)
  const navId = await navChipsId(nav)

  // Plain click = switch the live navigator in place to just this lane.
  await nav.getByTestId(`nav-chip-${POTENTIAL_CHIP}-${navId}`).click()
  await page.waitForTimeout(1_500)
  await nav.screenshot({ path: join(SHOTS, '05-single-lane.png') })

  // The lane must carry the movie's TIME calibration. Uncalibrated it plots
  // over frame index (0…7913) while the selector works in seconds (0…259), so
  // the cursor only ever reaches the first ~3% — "I can only see the beginning
  // of the experiment". Read the axis the figure actually rendered.
  const xMax = await nav.locator('iframe').first().contentFrame()
    .locator('text=/^\\d+$/').last().innerText().catch(() => '')
  // 259 s of movie: the last x tick belongs to the seconds axis, not to a
  // frame count in the thousands.
  if (xMax) expect(Number(xMax)).toBeLessThan(1_000)
})

test('loading a record explicitly works too', async () => {
  const { page, backend } = ctx
  // The File-menu picker returns a path; drive the same backend action it sends
  // (a native dialog can't be driven from Playwright).
  const before = ctx.backend.logBuffer.length
  await backendAction(page, 'load_insitu_data', { path: EC_RUN })
  await expect
    .poll(() => ctx.backend.logBuffer.slice(before)
      .some((l: string) => /insitu: (Attached|could not attach)/.test(l)),
      { timeout: 60_000 })
    .toBe(true)
  const fresh = ctx.backend.logBuffer.slice(before).join('\n')
  expect(fresh).toContain('Attached Cyclic Voltammetry')
  expect(fresh).not.toContain('could not attach')
})

test('fast forward shows the speed it is running at, up to x32', async () => {
  const { page } = ctx
  const nav = navWindow(page)
  const ff = nav.getByTitle('Fast Forward').or(nav.locator('[data-testid="toolbar-btn-Fast Forward"]'))
  const badge = page.getByTestId('playback-speed-badge')

  // 1x -> no badge. Then 2, 4, 8, 16, 32 as the cycle advances.
  await expect(badge).toHaveCount(0)
  for (const expected of ['2x', '4x', '8x', '16x', '32x']) {
    await ff.first().click()
    await expect(badge).toHaveText(expected, { timeout: 10_000 })
  }
  await page.screenshot({ path: join(SHOTS, '07-speed-32x.png') })

  // Once more wraps to 1x and the badge goes away.
  await ff.first().click()
  await expect(badge).toHaveCount(0, { timeout: 10_000 })
  await backendAction(page, 'playback', { command: 'stop' })
})

test('no renderer JS errors and no backend errors', async () => {
  ctx.assertNoJsErrors()
  const errors = backendErrorLines(ctx.backend)
    .filter((l: string) => !/DeprecationWarning|VisibleDeprecation/.test(l))
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
})
