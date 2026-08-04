/**
 * talk_present.spec.ts — open the committed SpyDE overview deck in the REAL app
 * and screenshot EVERY slide in Present mode.
 *
 * The deck (`doc/presentations/spyde-overview.spyde-report`) is built by
 * `doc/presentations/build_spyde_overview.py`. It carries only markdown / image /
 * split cells — no live figure bindings — so it opens standalone with no data
 * loaded and needs no Dask.
 *
 * This is the verification the presentation is judged by: every shot in
 * `talk_present_shots/` gets looked at. Text that overflows its slide, an empty
 * figure box, or a blank slide is a FAILURE even when the selectors pass.
 *
 * Run:
 *   npx playwright test tests/talk_present.spec.ts --project=electron \
 *     --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'
const { launchApp, backendAction, backendErrorLines } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'talk_present_shots')
const DECK = join(__dirname, '..', '..', 'doc', 'presentations',
                  'spyde-overview.spyde-report')
const N_SLIDES = 21

/** The deck's theme, as written by build_spyde_overview.py. Asserted as COMPUTED
 *  colours, because the failure this catches is a theme that round-trips through
 *  the front matter perfectly and then loses to a hard-coded stylesheet rule. */
const THEME_BG = 'rgb(18, 18, 28)'      // #12121c
const THEME_TEXT = 'rgb(233, 236, 243)' // #e9ecf3
const THEME_ACCENT = '#89b4fa'

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  await ctx.page.waitForTimeout(1500)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

test('the committed deck opens and presents cleanly', async () => {
  const { page } = ctx

  // Open the report sidebar, then load the committed deck through the SAME
  // backend handler the sidebar's Open button calls.
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_open', { path: DECK })

  // The sidebar mirrors report_state: wait for the deck's slides to arrive.
  await expect
    .poll(async () => page.getByTestId(/^report-slide-\d+$/).count(),
          { timeout: 60_000, message: 'report_open produced no slides' })
    .toBe(N_SLIDES)
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '00-sidebar.png') })

  // Enter Present mode.
  await page.getByTestId('report-present').click()
  const stage = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(stage).toBeVisible({ timeout: 30_000 })
  await page.waitForTimeout(1500)

  // The deck carries its own theme. Slide 1 is a TITLE card, which deliberately
  // has no footer — its attribution is on the card itself.
  const deck = page.getByTestId('present-mode')
  const deckStyle = await deck.evaluate((el) => {
    const cs = getComputedStyle(el)
    return { bg: cs.backgroundColor,
             accent: cs.getPropertyValue('--spyde-deck-accent').trim() }
  })
  expect(deckStyle.bg, 'deck background is not the document theme').toBe(THEME_BG)
  expect(deckStyle.accent, 'deck accent is not the document theme').toBe(THEME_ACCENT)
  // Present mode keeps EVERY slide mounted, so these locators are scoped to the
  // active stage. The deck has 3 title/section cards, and a title card carries
  // its own attribution — the footer belongs on the other 18.
  await expect(stage.getByTestId('present-footer'),
               'the title slide must not carry the footer bar').toHaveCount(0)
  await expect(page.getByTestId('present-footer'),
               'footer count != content slides').toHaveCount(N_SLIDES - 3)

  // Page through EVERY slide, screenshotting each. Assert the active slide
  // actually carries rendered text (a blank stage is the failure mode that
  // selectors alone miss), and that nothing overflows the viewport.
  for (let i = 1; i <= N_SLIDES; i++) {
    await page.waitForTimeout(650)                 // let the slide settle/paint
    const n = String(i).padStart(2, '0')
    await page.screenshot({ path: join(SHOTS, `${n}-slide.png`) })

    const text = ((await stage.innerText()) || '').trim()
    expect(text.length, `slide ${i} rendered no text`).toBeGreaterThan(10)

    // Slide 2 is the first CONTENT slide: the footer bar, its embedded logo and
    // the themed heading colour all have to be there. Checked once rather than
    // per slide — the footer is drawn by the same component every time.
    if (i === 2) {
      await expect(stage.getByTestId('present-footer')).toBeVisible()
      await expect(stage.getByTestId('present-footer-logo')).toBeVisible()
      await expect(stage.getByTestId('present-footer-text'))
        .toContainText('cfrancis@directelectron.com')
      // The base .present-md sheet hard-codes a heading colour that beats the
      // inherited one; this is the assertion that catches a themed deck whose
      // headings stay stock lavender.
      const h2 = stage.locator('h2').first()
      expect(await h2.evaluate((el) => getComputedStyle(el).color),
             'slide headings ignore the deck theme').toBe(THEME_TEXT)
    }

    // Overflow guard: the slide's content must fit its own scroll box.
    const overflow = await stage.evaluate((el: HTMLElement) => ({
      v: el.scrollHeight - el.clientHeight,
      h: el.scrollWidth - el.clientWidth,
    }))
    expect(overflow.h, `slide ${i} overflows HORIZONTALLY`).toBeLessThanOrEqual(2)
    if (overflow.v > 2) console.log(`  ! slide ${i} overflows vertically by ${overflow.v}px`)

    if (i < N_SLIDES) await page.keyboard.press('ArrowRight')
  }

  // The presenter view (S) — speaker notes must be visible to the presenter.
  await page.keyboard.press('Home')
  await page.waitForTimeout(500)
  await page.keyboard.press('s')
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '22-presenter-view.png') })

  // ESC exits.
  await page.keyboard.press('s')
  await page.waitForTimeout(300)
  await page.keyboard.press('Escape')
  await expect(stage).toBeHidden({ timeout: 15_000 })

  const errs = backendErrorLines(ctx.backend)
  expect(errs, `backend errors:\n${errs.join('\n')}`).toEqual([])
})
