/**
 * tour.spec.ts — the in-app guided coachmark tour.
 *
 * Asserts the Help "?" menu lists the guides and launching one renders the
 * coachmark overlay (bubble + step navigation) that drives the same single-source
 * guide the docs website uses. Renderer-only (SPYDE_NO_DASK=1, no Python needed).
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_NO_DASK: '1' },
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
})

test.afterAll(async () => { await app?.close() })

test.beforeEach(async () => {
  await page.reload()
  await page.waitForSelector('[data-testid="mdi-area"]')
})

test('the Help menu lists the available guided tours', async () => {
  await page.getByTestId('help-button').click()
  await expect(page.getByTestId('help-menu')).toBeVisible()
  await expect(page.getByTestId('help-guide-find-vectors')).toBeVisible()
  await expect(page.getByTestId('help-guide-find-vectors'))
    .toContainText('Finding Diffraction Vectors')
})

test('launching a tour shows the coachmark bubble with the first step', async () => {
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  await expect(page.getByTestId('tour-overlay')).toBeVisible()
  const bubble = page.getByTestId('tour-bubble')
  await expect(bubble).toBeVisible()
  // First step title from find-vectors.ts, and the step counter.
  await expect(bubble).toContainText('What you’ll do')
  await expect(bubble).toContainText('1 / 7')
})

test('Next/Back walk through steps and spotlight a real UI element', async () => {
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  const bubble = page.getByTestId('tour-bubble')

  // Step 2 anchors to the MDI area → a spotlight ring appears over the real element.
  await page.getByTestId('tour-next').click()
  await expect(bubble).toContainText('2 / 7')
  await expect(page.getByTestId('tour-spotlight')).toBeVisible()

  // Back returns to step 1.
  await page.getByTestId('tour-back').click()
  await expect(bubble).toContainText('1 / 7')
})

test('markdown bold + callout render inside the bubble', async () => {
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  const bubble = page.getByTestId('tour-bubble')
  // Step 1 body has a "> 💡" callout and **bold** text.
  await expect(bubble.locator('strong').first()).toBeVisible()
  await expect(bubble).toContainText('💡')
})

test('reaching the last step shows Done, which closes the tour', async () => {
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  // Click Next until Done appears (7 steps → 6 Nexts).
  for (let i = 0; i < 6; i++) await page.getByTestId('tour-next').click()
  await expect(page.getByTestId('tour-bubble')).toContainText('7 / 7')
  await page.getByTestId('tour-done').click()
  await expect(page.getByTestId('tour-overlay')).toHaveCount(0)
})

test('Escape closes the tour', async () => {
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  await expect(page.getByTestId('tour-overlay')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('tour-overlay')).toHaveCount(0)
})
