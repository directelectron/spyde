/**
 * action_latency.spec.ts — end-to-end STAGE TIMING for user-visible actions.
 *
 * The maintainer's report was "the ROI spawns a drift check image which takes 60
 * seconds to appear" and "the Correct Drift button just seems to do nothing".
 * Neither is diagnosable from a passing functional test: `drift_wizard.spec.ts`
 * goes green in 35 s on a 40-frame 96x112 fixture, which is small enough that
 * every stage is fast for reasons that do not survive contact with a real movie.
 *
 * So this spec does two things the functional one cannot:
 *
 *   1. It runs on a REALISTIC movie — a real `.mrc` on disk, 2048² frames, more
 *      of them than fit in RAM comfortably. The backing decides the reader, and
 *      the reader is what the timings are about; `da.from_array(numpy)` would
 *      measure a path no user has.
 *   2. It reads the `[ACTION-PROFILE]` lines the backend logs (see
 *      `spyde/backend/action_profile.py`) and asserts LATENCY BUDGETS on them.
 *
 * **Why budgets and not just a print.** A number in a report is read once; a
 * budget is read by CI on every change. The 60 s regression got to a human
 * because nothing in the suite could see time. These thresholds are deliberately
 * loose — they are tripwires for "this became pathological again", not
 * micro-benchmarks, and the box they run on is not controlled.
 *
 * Profiles arrive over the LOG channel, not `emit()`. Backend `emit()` goes down
 * the PLOTAPP line protocol and the Electron main process echoes only a tiny
 * allowlist, so a spec waiting on a profile MESSAGE waits forever. INFO records
 * reach stderr, which the harness captures into `backend.logBuffer`.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = 'action_latency_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(900_000)

/** Frames at 2048² backed by a real .mrc. Big enough that a per-frame read cost
 *  of even 100 ms shows up as a wait a user would call broken. */
const FRAMES = 60

/** Budgets in ms. Loose on purpose — see the header. */
const BUDGET = {
  drift_open: 20_000,     // caret mount -> the Drift Check image is on screen
  drift_preview: 20_000,  // ROI settle -> the discovery pair repaints
  drift_commit: 3_000,    // Apply -> the corrected node exists and is shown
}

interface Profile { label: string; total: number; stages: Record<string, number>; raw: string }

/** Parse `[ACTION-PROFILE] <label> total=123.4ms  a=1.0  b=2.0  k=v ...`. */
function profiles(buf: string[], label?: string): Profile[] {
  const out: Profile[] = []
  for (const line of buf) {
    const m = /\[ACTION-PROFILE\]\s+(\S+)\s+total=([\d.]+)ms\s+(.*)$/.exec(line)
    if (!m) continue
    if (label && m[1] !== label) continue
    const stages: Record<string, number> = {}
    for (const kv of m[3].split(/\s{1,}/)) {
      const p = /^([A-Za-z_]+@?)=([\d.]+)$/.exec(kv)
      if (p) stages[p[1]] = Number(p[2])
    }
    out.push({ label: m[1], total: Number(m[2]), stages, raw: line.trim() })
  }
  return out
}

async function waitForProfile(label: string, timeout = 600_000): Promise<Profile> {
  let found: Profile | undefined
  await expect.poll(() => {
    const all = profiles(ctx.backend.logBuffer, label)
    found = all[all.length - 1]
    return Boolean(found)
  }, { timeout, message: `no [ACTION-PROFILE] ${label} line appeared` }).toBe(true)
  return found!
}

/** One line per stage, widest first — the shape a human reads to find the cost. */
function report(p: Profile) {
  const rows = Object.entries(p.stages).sort((a, b) => b[1] - a[1])
  console.log(`\n  ${p.label}  total=${p.total.toFixed(0)}ms`)
  for (const [k, v] of rows) {
    const pct = p.total > 0 ? (100 * v / p.total) : 0
    console.log(`    ${k.padEnd(20)} ${v.toFixed(1).padStart(10)} ms  ${pct.toFixed(0).padStart(3)}%`)
  }
}

test.beforeAll(async () => {
  // A hook carries its OWN timeout — `test.setTimeout` at file scope does not
  // reach it, and synthesising a multi-GB .mrc blows the 120 s default long
  // before the first assertion runs.
  test.setTimeout(900_000)
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'INFO', SPYDE_ACTION_PROFILE: '1' },
  })
  const { page } = ctx
  await page.waitForTimeout(1500)
  // A REAL .mrc, not an in-RAM dask array: the backing picks the reader and the
  // reader is the thing under measurement.
  await backendAction(page, 'load_test_data_movie',
    { frames: FRAMES, size: 2048, mrc: true })
  await waitForSubwindowCount(page, 2, 300_000)
  await page.waitForTimeout(2000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test('drift_open: the Drift Check image appears inside budget', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Drift Correction').click()
  await expect(page.getByTestId('drift-wizard')).toBeVisible()

  const p = await waitForProfile('drift_open')
  report(p)
  await page.screenshot({ path: `${SHOTS}/01-open.png` })

  expect(p.total, `drift_open took ${p.total.toFixed(0)}ms\n${p.raw}`)
    .toBeLessThan(BUDGET.drift_open)
})

test('drift_preview: the ROI discovery pair repaints inside budget', async () => {
  const { page } = ctx
  const before = profiles(ctx.backend.logBuffer, 'drift_preview').length
  // Re-tune to force a fresh preview without depending on a drag landing.
  await page.getByTestId('drift-advanced-toggle').click()
  await page.getByTestId('drift-preview-frames').fill('20')
  await page.getByTestId('drift-preview-frames').blur()

  await expect.poll(
    () => profiles(ctx.backend.logBuffer, 'drift_preview').length,
    { timeout: 600_000, message: 'the preview never re-ran' },
  ).toBeGreaterThan(before)

  const all = profiles(ctx.backend.logBuffer, 'drift_preview')
  const p = all[all.length - 1]
  report(p)
  await page.screenshot({ path: `${SHOTS}/02-preview.png` })

  expect(p.total, `drift_preview took ${p.total.toFixed(0)}ms\n${p.raw}`)
    .toBeLessThan(BUDGET.drift_preview)
})

test('drift_commit: Apply lands the corrected node promptly', async () => {
  const { page } = ctx
  await page.getByTestId('drift-advanced-toggle').click()   // collapse
  await page.getByTestId('drift-solve').click()
  await expect(page.getByTestId('drift-result')).toBeVisible({ timeout: 600_000 })

  const run = await waitForProfile('drift_run')
  report(run)

  await page.getByTestId('drift-commit').click()
  // The node must EXIST in the tree, which is the user-visible claim.
  await expect(page.getByTestId('tree-node-Drift corrected'))
    .toBeVisible({ timeout: 120_000 })

  const p = await waitForProfile('drift_commit')
  report(p)
  await page.screenshot({ path: `${SHOTS}/03-applied.png` })

  expect(p.total, `drift_commit took ${p.total.toFixed(0)}ms\n${p.raw}`)
    .toBeLessThan(BUDGET.drift_commit)

  const errors = backendErrorLines(ctx.backend)
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
})
