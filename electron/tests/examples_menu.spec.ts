/**
 * examples_menu.spec.ts — the Examples menu, built from em-database.
 *
 * The menu is the whole feature here, so this drives it the way a user does:
 * open Examples, walk into a technique submenu, and check each dataset row
 * carries its size, its shape where known, and a marker saying whether it is
 * already on disk. Also pins the Dummy Data submenu and the data-directory
 * entry.
 *
 * Deliberately does NOT click a real dataset — that would download gigabytes.
 */
import { test, expect } from '@playwright/test'
const { launchApp } = require('./_harness.cjs')

const SHOTS = 'examples_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'INFO' } })
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(120_000)

async function openExamples() {
  const { page } = ctx
  // The menu button TOGGLES, so a menu left open by the previous test would be
  // closed by this click. Escape first, then open.
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('menu-examples-items')).toBeHidden()
  await page.getByTestId('menu-examples').click()
  await expect(page.getByTestId('menu-examples-items')).toBeVisible()
}

test('Examples groups the datasets into technique submenus', async () => {
  const { page } = ctx
  await openExamples()

  // The backend catalogue arrives async; the technique rows appear with it.
  await expect(page.getByTestId('examples-tech-4d-stem'))
    .toBeVisible({ timeout: 30_000 })
  for (const tech of ['4d-stem', 'eels', 'ebsd']) {
    await expect(page.getByTestId(`examples-tech-${tech}`),
      `no ${tech} submenu`).toBeVisible()
  }
  await page.screenshot({ path: `${SHOTS}/01-examples-techniques.png` })
  ctx.assertNoJsErrors()
})

test('each dataset shows its size, shape and download state', async () => {
  const { page } = ctx
  await openExamples()
  await page.getByTestId('examples-tech-4d-stem').hover()

  const items = page.getByTestId('examples-tech-4d-stem-items')
  await expect(items).toBeVisible({ timeout: 15_000 })

  // SPEDAg is the scan the 4D-STEM work is benchmarked on; it must be listed
  // with its size, and its shape once it has been downloaded and measured.
  const sped = page.getByTestId('example-SPEDAg')
  await expect(sped).toBeVisible()
  await expect(sped).toContainText(/\d+(\.\d+)?\s*[kMG]B/)

  // Every row carries exactly one state marker.
  const rows = await items.getByRole('button').all()
  expect(rows.length).toBeGreaterThan(3)
  let marked = 0
  for (const row of rows) {
    const text = (await row.textContent()) ?? ''
    if (text.includes('●') || text.includes('○')) marked++
  }
  expect(marked, 'dataset rows carry no downloaded/not-downloaded marker')
    .toBe(rows.length)

  await page.screenshot({ path: `${SHOTS}/02-4dstem-datasets.png` })
  ctx.assertNoJsErrors()
})

test('hovering a dataset shows a themed info card', async () => {
  const { page } = ctx
  await openExamples()
  await page.getByTestId('examples-tech-4d-stem').hover()
  await expect(page.getByTestId('examples-tech-4d-stem-items')).toBeVisible()

  await page.getByTestId('example-SPEDAg').hover()
  const card = page.getByTestId('menu-hover-card')
  await expect(card).toBeVisible({ timeout: 10_000 })
  await expect(card).toContainText('SPEDAg')
  await expect(card).toContainText('4D-STEM')
  await expect(card).toContainText(/Size/)
  await expect(card).toContainText(/On disk|Not downloaded/)
  await page.screenshot({ path: `${SHOTS}/04-hover-card.png` })

  // It is OUR panel, not the OS bubble — so the row must not also carry a
  // native title attribute racing it.
  await expect(page.getByTestId('example-SPEDAg')).not.toHaveAttribute('title', /./)

  // An undownloaded set reads differently.
  await page.getByTestId('example-FeAlStripes').hover()
  await expect(card).toContainText('Not downloaded')
  await page.screenshot({ path: `${SHOTS}/05-hover-card-undownloaded.png` })
  ctx.assertNoJsErrors()
})

test('Dummy Data is its own submenu and still loads', async () => {
  const { page } = ctx
  await openExamples()
  await page.getByTestId('examples-dummy-data').hover()
  const items = page.getByTestId('examples-dummy-data-items')
  await expect(items).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('tutorial-navigation')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/03-dummy-data.png` })

  // It is instant + no-download, so actually clicking one is fair game.
  await page.getByTestId('tutorial-navigation').click()
  await expect(page.getByTestId('subwindow').first())
    .toBeVisible({ timeout: 60_000 })
  ctx.assertNoJsErrors()
})

test('Show Example Data Directory reports a real path', async () => {
  const { page, backend } = ctx
  await openExamples()
  await expect(page.getByTestId('examples-show-dir')).toBeVisible()
  await page.getByTestId('examples-show-dir').click()
  // `open_path` is consumed by the renderer and never echoed back on the
  // PLOTAPP channel, so waiting on a message would wait forever — the backend
  // logs the reveal for exactly this reason.
  await backend.waitForLog('revealing example data directory', 20_000)
  const line = backend.logBuffer.find(
    (l: string) => l.includes('revealing example data directory'))
  expect(line, 'the backend never reported the directory').toBeTruthy()
  expect(line).toContain('em_database')
  ctx.assertNoJsErrors()
})
