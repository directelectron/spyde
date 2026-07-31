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

test('the Help menu lists techniques with an Info / Guided-tour subtoolbar', async () => {
  await page.getByTestId('help-button').click()
  await expect(page.getByTestId('help-menu')).toBeVisible()
  // One row per TECHNIQUE, named for the technique…
  const row = page.getByTestId('help-technique-find-vectors')
  await expect(row).toBeVisible()
  await expect(row).toContainText('Finding Diffraction Vectors')
  // …each with its two-entry subtoolbar.
  await expect(page.getByTestId('help-info-find-vectors')).toBeVisible()
  await expect(page.getByTestId('help-guide-find-vectors')).toBeVisible()
})

test('the Help DROPDOWN mirrors the same technique → Info / Guided tour shape', async () => {
  // The title-bar Help menu (MenuBar.tsx) used to be a flat list of
  // "Guided Tour: <title>" items with no way to just read about a technique.
  await page.getByTestId('menu-help').click()
  await expect(page.getByTestId('menu-help-items')).toBeVisible()
  const row = page.getByTestId('help-technique-strain')
  await expect(row).toBeVisible()
  await expect(row).toContainText('Strain Mapping')
  await row.hover()   // fly-out opens on hover
  await expect(page.getByTestId('help-info-strain')).toBeVisible()
  await expect(page.getByTestId('help-tour-strain')).toBeVisible()
  await page.getByTestId('help-info-strain').click()
  await expect(page.getByTestId('guide-info-dialog')).toContainText('Strain Mapping')
})

test('Info opens the technique dialog with its further reading', async () => {
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-info-find-vectors').click()
  const dlg = page.getByTestId('guide-info-dialog')
  await expect(dlg).toBeVisible()
  await expect(dlg).toContainText('Finding Diffraction Vectors')
  await expect(page.getByTestId('guide-info-links')).toBeVisible()
  // The dialog can hand straight over to the walkthrough.
  await page.getByTestId('guide-info-start-tour').click()
  await expect(dlg).toHaveCount(0)
  await expect(page.getByTestId('tour-overlay')).toBeVisible()
})

test('launching a tour shows the coachmark bubble with the first step', async () => {
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  await expect(page.getByTestId('tour-overlay')).toBeVisible()
  const bubble = page.getByTestId('tour-bubble')
  await expect(bubble).toBeVisible()
  // First step title from find-vectors.ts, and the step counter. The count is
  // the guide's 7 authored steps + the trailing "More info" step the Tour
  // appends from `guide.info`.
  await expect(bubble).toContainText('What you’ll do')
  await expect(bubble).toContainText('1 / 8')
})

test('Next/Back walk through steps and spotlight a real UI element', async () => {
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  const bubble = page.getByTestId('tour-bubble')

  // Step 2 anchors to the MDI area → a spotlight ring appears over the real element.
  await page.getByTestId('tour-next').click()
  await expect(bubble).toContainText('2 / 8')
  await expect(page.getByTestId('tour-spotlight')).toBeVisible()

  // Back returns to step 1.
  await page.getByTestId('tour-back').click()
  await expect(bubble).toContainText('1 / 8')
})

test('markdown bold + callout render inside the bubble', async () => {
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  const bubble = page.getByTestId('tour-bubble')
  // Step 1 body has a "> 💡" callout and **bold** text.
  await expect(bubble.locator('strong').first()).toBeVisible()
  await expect(bubble).toContainText('💡')
})

test('the walkthrough ends on a "More info" step with external links', async () => {
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  // Click Next until Done appears (7 authored steps + More info → 7 Nexts).
  for (let i = 0; i < 7; i++) await page.getByTestId('tour-next').click()
  const bubble = page.getByTestId('tour-bubble')
  await expect(bubble).toContainText('8 / 8')
  await expect(bubble).toContainText('More info')
  const links = page.getByTestId('tour-more-info')
  await expect(links).toBeVisible()
  await expect(links.locator('[data-testid^="tour-info-link-"]').first()).toBeVisible()
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

test('clicking OUTSIDE the tour does NOT close it — only ✕ does', async () => {
  // The regression this whole change exists for: the overlay used to be a
  // full-screen `onClick={onClose}` hit-target, so any stray click killed the
  // walkthrough. Now nothing but ✕ / Done / Esc exits.
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  const overlay = page.getByTestId('tour-overlay')
  await expect(overlay).toBeVisible()

  // Click well away from the bubble, in three different regions.
  await page.getByTestId('mdi-area').click({ position: { x: 20, y: 20 } })
  await expect(overlay).toBeVisible()
  await page.getByTestId('app-bar').click({ position: { x: 300, y: 10 } })
  await expect(overlay).toBeVisible()
  await page.getByTestId('status-text').click()
  await expect(overlay).toBeVisible()
  // …and the step did not advance either.
  await expect(page.getByTestId('tour-bubble')).toContainText('1 / 8')

  // ✕ is the way out.
  await page.getByTestId('tour-close').click()
  await expect(overlay).toHaveCount(0)
})

test('the tour never blocks the app underneath (click-through)', async () => {
  // The other half of the same fix: the overlay is `pointerEvents:none`, so a
  // click while the tour is open reaches the real control — which is what makes
  // "click the peak-finding tool" a followable instruction rather than a
  // tour-ending mistake.
  await page.getByTestId('help-button').click()
  await page.getByTestId('help-guide-find-vectors').click()
  await expect(page.getByTestId('tour-overlay')).toBeVisible()

  const dock = page.getByTestId('plot-control-dock')
  const before = await dock.count()
  await page.getByTestId('toggle-sidebar').click()
  await expect.poll(async () => dock.count(), { timeout: 5_000 }).not.toBe(before)
  await expect(page.getByTestId('tour-overlay')).toBeVisible()
})
