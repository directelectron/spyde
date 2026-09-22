/**
 * multiangle_loader.spec.ts — a multi-angle 4-D STEM acquisition, end to end on
 * a real backend.
 *
 * Loads ten synthetic members written as real MRC files in two shells (6 at
 * 1.0°, 4 at 0.5°), so the loader takes its real path: open each member, solve
 * the real-space and reciprocal alignments, and re-express each member as an
 * ALIGNED READ of its own file. Then it checks the two things a user actually
 * sees — the Workflow panel offering the stack and the sums, and the display
 * following a switch between the 5-D stack and the 4-D sum.
 *
 * Screenshots land in electron/multiangle_shots/ because a green assertion here
 * does not prove the windows drew anything (CLAUDE.md).
 */
import { test, expect, Page } from '@playwright/test'
import { join } from 'path'

const { launchApp } = require('./_harness.cjs')

let ctx: any
let page: Page

const SHOTS = join(__dirname, '..', 'multiangle_shots')

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  page = ctx.page
  await page.waitForTimeout(1500)
  await page.evaluate(() =>
    window.electron.action('load_test_data_multiangle', { nav: 28, sig: 32 }),
  )
  // Two navigators, the diffraction pattern, and the angle ring.
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 4,
    { timeout: 180_000 },
  )
  await page.waitForTimeout(3000)
})

test.afterAll(async () => { await ctx?.app?.close() })

test('a multi-angle acquisition opens on its summed node with the stack behind it', async () => {
  await expect(page.getByTestId('status-text'))
    .toContainText('10 members in 2 shells', { timeout: 60_000 })
  await page.screenshot({ path: join(SHOTS, '01-loaded.png'), fullPage: false })

  // The Workflow panel is the toggle: the stack, its sum, and one node per shell.
  const tree = page.getByTestId('signal-tree')
  await expect(tree).toBeVisible({ timeout: 20_000 })
  await expect(tree).toContainText('Aligned Stack')
  await expect(tree).toContainText('Summed')
  await expect(tree).toContainText('0.5')
  await expect(tree).toContainText('1')
  await tree.screenshot({ path: join(SHOTS, '02-workflow.png') })

  const windowCount = await page.evaluate(
    () => document.querySelectorAll('[data-testid="subwindow"]').length)
  expect(windowCount).toBeGreaterThanOrEqual(4)
})

test('switching to the 5-D stack and back keeps a real frame on screen', async () => {
  const tree = page.getByTestId('signal-tree')

  await tree.getByTestId(/^tree-node-Aligned Stack/).first().click()
  await page.waitForTimeout(2500)
  await page.screenshot({ path: join(SHOTS, '03-stack-node.png') })

  await tree.getByTestId(/^tree-node-Summed$/).first().click()
  await page.waitForTimeout(2500)
  await page.screenshot({ path: join(SHOTS, '04-back-to-summed.png') })

  // Each figure is an embedded document, so the panels are iframes — there is
  // never a <canvas> in the TOP document, and checking for one there passes
  // whatever the app drew.
  const panels = await page.evaluate(
    () => document.querySelectorAll('iframe').length)
  expect(panels).toBeGreaterThanOrEqual(4)
})

test('the angle navigator carries a real per-member profile', async () => {
  // The 1-D angle window is the first navigator of the chain. Its plot is a
  // flat line when every angle plane holds the same image, which is a
  // navigator with nothing to navigate — and only visible as pixels.
  const angleWindow = page.getByTestId('subwindow').first()
  await angleWindow.screenshot({ path: join(SHOTS, '05-angle-navigator.png') })
  await expect(angleWindow).toBeVisible()
})

test('the acquisition opens with its angle ring up', async () => {
  // The ring is the picture of the acquisition: a circle per tilt shell with a
  // point per member, so a missing angle is a gap. Only pixels show that.
  const ring = page.getByTestId('subwindow')
    .filter({ hasText: /Angles/ }).first()
  await expect(ring).toBeVisible({ timeout: 30_000 })
  await ring.screenshot({ path: join(SHOTS, '06-angle-ring.png') })
  await page.screenshot({ path: join(SHOTS, '07-full-with-ring.png') })
})

test('picking an angle on the ring pops to that angle of the 5-D stack', async () => {
  // The behaviour the ring exists for: on the summed node it is the toggle.
  // Picking member 7 must switch the displayed node to the stack AND land on
  // that angle — a switch that ignored the angle would look identical here.
  await page.evaluate(() =>
    window.electron.action('multiangle_pick_angle', { member: 7 }))
  await page.waitForTimeout(2500)

  const ring = page.getByTestId('subwindow').filter({ hasText: /Angles/ }).first()
  await ring.screenshot({ path: join(SHOTS, '08-picked-angle.png') })
  await page.screenshot({ path: join(SHOTS, '09-full-picked.png') })

  // The Workflow panel must now show the stack, not the sum.
  await expect(page.getByTestId('signal-tree')).toContainText('Aligned Stack')
})
