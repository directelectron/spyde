/**
 * drift_real_scale.spec.ts — the drift caret on a REAL-SHAPE movie, on pixels.
 *
 * 245 frames x 4096^2 uint8 lazy .mrc (3.8 GiB), the shape of the maintainer's
 * "In-situ Electrochemistry Growth" dataset. Every earlier drift spec ran on
 * 96x112 or 60x2048^2, and neither could see what a real dataset does: the
 * corrected-sum panel solid BLACK, both ROI panels blank WHITE, gain 0.3x.
 *
 * This spec exists to be LOOKED AT. Its assertions are deliberately weak — that
 * the panels are not uniform — because the real verification is a human reading
 * the screenshots. A panel that is blank white or solid black is a FAILURE per
 * CLAUDE.md, and no timing can tell you which one you have.
 *
 * The fixture is written by `spyde/tests/benchmark_drift_real_scale.py`; run that
 * first (it caches the .mrc in the temp dir and reuses it).
 */
import { test, expect } from '@playwright/test'
import { mkdirSync, existsSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = 'drift_real_shots'
const MOVIE = join(tmpdir(), 'spyde-real-245x4096.mrc')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(1_800_000)

test.beforeAll(async () => {
  test.setTimeout(1_800_000)
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'INFO', SPYDE_ACTION_PROFILE: '1' },
  })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'open_file', { path: MOVIE })
  await waitForSubwindowCount(page, 2, 900_000)
  await page.waitForTimeout(4000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test('the caret opens on 245 x 4096^2 and the panels have content', async () => {
  test.skip(!existsSync(MOVIE),
    `fixture missing: run \`uv run python -m spyde.tests.benchmark_drift_real_scale\` first`)
  const { page } = ctx

  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Drift Correction').click()
  await expect(page.getByTestId('drift-wizard')).toBeVisible()
  await waitForSubwindowCount(page, 3, 900_000)

  const readout = page.getByTestId('drift-roi-readout')
  await expect.poll(() => readout.getAttribute('data-gain'),
    { timeout: 900_000, message: 'the discovery preview never reported a gain' },
  ).toBeTruthy()
  await page.waitForTimeout(3000)
  await page.screenshot({ path: `${SHOTS}/01-open.png` })

  // The two numbers the screenshot must show: a 512 box and 2 preview frames.
  // Both are decided in PYTHON, so they are the fingerprint of which backend is
  // actually running — a stale bundle cannot fake them.
  await expect(readout).toContainText('512x512 px')
  await expect(readout).toContainText('over 2 frames')

  // Gain must be ABOVE 1: below 1 means the "corrected" sum is worse than the
  // raw one, which is the correctness bug this scale exists to catch.
  const gain = Number(await readout.getAttribute('data-gain'))
  expect(gain, `gain ${gain} — the aligned sum is WORSE than the raw one`)
    .toBeGreaterThan(1.0)

  ctx.assertNoJsErrors()
})

test('Correct Drift paints a corrected sum that is not black', async () => {
  test.skip(!existsSync(MOVIE), 'fixture missing')
  const { page } = ctx

  await page.getByTestId('drift-solve').click()
  await expect(page.getByTestId('drift-result')).toBeVisible({ timeout: 900_000 })
  await page.waitForTimeout(3000)
  await page.screenshot({ path: `${SHOTS}/02-solved.png` })

  const errors = backendErrorLines(ctx.backend)
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
  ctx.assertNoJsErrors()
})
