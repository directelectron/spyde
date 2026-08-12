/**
 * run_recipe.spec.ts — Autopilot's boundary proof.
 *
 * The THIRD app on the shell, and the one that exercises the parts Ground Crew
 * does not: the shell's progress channel, a queue rather than a free-running
 * loop, and a run that must stop promptly when asked.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp } = require('../../../packages/shell-testing/src/harness.cjs')

const APP_DIR = join(__dirname, '..')
const SHOTS = join(APP_DIR, 'shots')

let ctx: any

test.beforeAll(async () => {
  ctx = await launchApp({
    appDir: APP_DIR,
    appId: 'autopilot',
    env: { AUTOPILOT_LOG_LEVEL: 'INFO' },
    readyLog: '[autopilot backend] ready',
    readyMessages: [],
  })
})

test.afterAll(async () => { await ctx?.app?.close() })

test('runs a recipe to completion on the shared shell', async () => {
  const { page, backend } = ctx

  // 1. The recipe arrived and rendered as a queue — this app's layout, where
  //    Ground Crew has manual controls and SpyDE has an MDI workspace.
  await expect(page.getByTestId('recipe-panel')).toBeVisible()
  const steps = page.getByTestId('step-list').locator('li')
  await expect.poll(async () => steps.count(), { timeout: 30_000 }).toBeGreaterThan(0)
  const total = await steps.count()
  await page.screenshot({ path: join(SHOTS, '01-idle.png') })

  // The viewer pane exists from the start (laid out at the detector's real
  // size, so it does not jump when the first frame lands), but nothing has been
  // ACQUIRED — an unattended app that started acquiring on launch would be a
  // real bug, not a cosmetic one. The stats strip is empty until a frame paints.
  await expect(page.getByTestId('stats-strip')).toBeEmpty()

  // 2. Run it.
  await page.getByTestId('run-btn').click()

  // The shell's progress channel drives the bar — proof the shared emit_progress
  // path reaches this app's renderer.
  await expect(page.getByTestId('progress')).toContainText('/', { timeout: 30_000 })

  // 3. Every step reaches `done`, and the figure appears.
  await expect
    .poll(async () => page.getByTestId('step-list').locator('li[data-state="done"]').count(),
      { timeout: 60_000, message: 'the recipe never finished' })
    .toBe(total)
  await expect(page.getByTestId('viewer-frame')).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '02-done.png') })

  // 4. Frames were actually acquired, and the image is a real one rather than
  //    the opening placeholder.
  const acquired = Number(await page.getByTestId('stat-acquired').innerText())
  expect(acquired).toBeGreaterThan(0)
  expect(await imageGreyLevels(page),
    'image pane is uniform — the placeholder, not an acquisition').toBeGreaterThan(10)

  // 5. Nothing died.
  const errs = backend.errorLines()
  if (errs.length) console.log('[autopilot] backend errors:\n' + errs.join('\n'))
  expect(errs.join('\n')).not.toMatch(/Traceback|ModuleNotFoundError|ImportError/)
  ctx.assertNoJsErrors()
})

test('a run can be stopped part-way', async () => {
  const { page } = ctx
  const list = page.getByTestId('step-list')
  // `data-state` is on the <li> ITSELF, so it belongs in the same selector.
  // `list.locator('li').locator('[data-state="done"]')` would search each li's
  // DESCENDANTS and always count zero — a test that fails for its own reasons.
  const done = list.locator('li[data-state="done"]')
  const total = await list.locator('li').count()

  await page.getByTestId('run-btn').click()
  // Wait for real motion rather than a fixed sleep: at least one step done, but
  // not all of them.
  await expect.poll(async () => done.count(), { timeout: 30_000 }).toBeGreaterThan(0)

  await page.getByTestId('stop-btn').click()

  // Stopping must take effect PROMPTLY — the runner waits on an Event rather
  // than sleeping precisely so this holds. A generous bound still fails if the
  // run simply carries on to the end.
  await expect(page.getByTestId('run-state')).toContainText('Stopped', { timeout: 10_000 })
  expect(await done.count()).toBeLessThan(total)
  await page.screenshot({ path: join(SHOTS, '03-stopped.png') })
})

/** Distinct grey levels in the largest canvas — a placeholder is uniform, a real
 *  acquisition is not. See the note in Ground Crew's spec: a bright-pixel count
 *  cannot tell the picture from the white panel behind it. */
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
