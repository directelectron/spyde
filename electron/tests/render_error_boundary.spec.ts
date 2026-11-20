/**
 * render_error_boundary.spec.ts — a render error stays where it happened.
 *
 * Before the boundaries, one component throwing while it rendered unmounted
 * the whole interface: the window went flat to the page background colour,
 * with no message, while the backend carried on. Seen around virtual imaging.
 *
 * The errors are provoked the way a bad backend message would: a navigator
 * chip given an object for its name (drawn in a window's body) and a window
 * title given an object (drawn in its title bar, outside any window's own
 * boundary — the root's case).
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'render_error_shots')
mkdirSync(SHOTS, { recursive: true })
test.setTimeout(120_000)

const inject = (page, msg: Record<string, unknown>) =>
  page.evaluate((m) => { (window as any)._spyde_test_inject?.(m) }, msg)

test('a window that fails to draw shows its error, and the rest keeps running', async () => {
  const ctx = await launchApp({})
  const { page } = ctx
  try {
    await backendAction(page, 'load_test_data_si_grains')
    await waitForSubwindowCount(page, 2, 120_000)
    const boxes = page.locator('[data-testid^="figure-box-"]')
    await expect(boxes).toHaveCount(2)
    const id = Number((await boxes.first().getAttribute('data-testid'))!.replace('figure-box-', ''))
    // By position, not by the figure box: the fallback replaces the box.
    const windows = page.getByTestId('subwindow')
    const index = await windows.evaluateAll(
      (all, box) => all.findIndex((w) => w.querySelector(`[data-testid="${box}"]`)), `figure-box-${id}`)
    const broken = windows.nth(index)
    const other = windows.nth(1 - index)

    await inject(page, { type: 'navigator_options', window_id: id,
                         names: [{ not: 'text' }, { also: 'not' }], current: null })
    await expect(broken.getByTestId('render-error')).toBeVisible()
    await expect(broken.getByTestId('render-error')).toContainText('could not be drawn')
    // Everything else is untouched: the other window, the menu bar, the dock.
    await expect(other.locator('iframe')).toBeVisible()
    await expect(page.getByTestId('render-error')).toHaveCount(1)
    await expect(page.getByTestId('app-render-error')).toHaveCount(0)
    await expect(page.getByTestId('plot-control-dock')).toBeVisible()
    await page.screenshot({ path: join(SHOTS, '01-window-failed.png') })
    await expect.poll(() => ctx.backend.logBuffer.some(
      (line: string) => line.includes('renderer-error') && line.includes(`window ${id}`)),
      { message: 'the render error never reached the main process' }).toBe(true)

    // Repaired state + Try again brings the window back.
    await inject(page, { type: 'navigator_options', window_id: id, names: [], current: null })
    await broken.getByTestId('render-error-retry').click()
    await expect(broken.getByTestId('render-error')).toHaveCount(0)
    await expect(broken.locator('iframe')).toBeVisible()
    await page.screenshot({ path: join(SHOTS, '02-window-retried.png') })
  } finally {
    await ctx.app.close()
  }
})

test('an error outside every window shows the last-resort screen, not an empty window', async () => {
  const ctx = await launchApp({})
  const { page } = ctx
  try {
    await backendAction(page, 'load_test_data_si_grains')
    await waitForSubwindowCount(page, 2, 120_000)
    const box = page.locator('[data-testid^="figure-box-"]').first()
    const id = Number((await box.getAttribute('data-testid'))!.replace('figure-box-', ''))

    await inject(page, { type: 'window_title', window_ids: [id], title: { not: 'text' } })
    const screen = page.getByTestId('app-render-error')
    await expect(screen).toBeVisible()
    await expect(screen).toContainText('stopped on an error')
    await page.screenshot({ path: join(SHOTS, '03-app-failed.png') })

    await page.getByTestId('app-render-error-reload').click()
    await expect(page.getByTestId('mdi-area')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByTestId('app-render-error')).toHaveCount(0)
  } finally {
    await ctx.app.close()
  }
})
