/**
 * deck_theme.spec.ts — the deck THEME: colours, footer bar, logo.
 *
 * The theme belongs to the DOCUMENT, with a separate "set as default" that
 * seeds new decks. The backend contract is covered by test_report_theme.py;
 * this drives the UI end-to-end and LOOKS at the presented slide, because the
 * whole feature is visual: a footer that renders behind the pager, or a colour
 * that reaches the chrome but not the markdown, passes every headless check.
 *
 * Screenshots to deck_theme_shots/ — each Read by the author.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'deck_theme_shots')

// A 96×96 solid-magenta PNG — stands in for a real logo and is unmistakable
// both on screen and in a pixel probe. It must be a VALID image: a broken <img>
// still satisfies toBeVisible(), so an invalid fixture would let a logo that
// never draws pass every assertion (it did, until the screenshot showed it).
const LOGO_PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAYAAADimHc4AAAA7UlEQVR4nO3T' +
  'MREAIBDEwAf/SjAJMkKxa+CKzK075w6Z3U0jwAc8ICZATICYADEBYgLEBIgJEBMgJkBMgJgAMQFiAsQE' +
  'iAkQEyAmQEyAmAAxAWICxASICRATICZATICYADEBYgLEBIgJEBMgJkBMgJgAMQFiAsQEiAkQEyAmQEyA' +
  'mAAxAWICxASICRATICZATICYADEBYgLEBIgJEBMgJkBMgJgAMQFiAsQEiAkQEyAmQEyAmAAxAWICxASI' +
  'CRATICZATICYADEBYgLEBIgJEBMgJkBMgJgAMQFiAsQEiAkQEyAmQEyAmAAxAab1AEYRA2grAPRPAAAA' +
  'AElFTkSuQmCC'

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2000)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

/** The report doc's theme, straight off renderer state. */
async function docTheme(page: any) {
  return await page.evaluate(() => (window as any)._spyde_test_report?.()?.theme ?? null)
}

async function exitPresent(page: any) {
  if (await page.locator('[data-testid="present-slide"][data-active="1"]').count()) {
    await page.keyboard.press('Escape')
    await page.waitForTimeout(600)
  }
}

test('1) build a two-slide deck (title + content)', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', { type: 'presentation' })
  await expect(page.getByTestId('report-body')).toBeVisible()

  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown', source: '# SpyDE\n\nCarter Francis',
    slide_kind: 'title',
  })
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown', slide_break: true,
    source: '## Results\n\n- one\n- two',
  })
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '01-deck.png') })
})

test('2) the Theme panel opens and edits reach the document', async () => {
  const { page } = ctx
  await page.getByTestId('report-theme').click()
  await expect(page.getByTestId('theme-panel')).toBeVisible({ timeout: 5_000 })
  await page.screenshot({ path: join(SHOTS, '02-theme-panel.png') })

  // Footer identity + a logo + an accent, through the real controls.
  await page.getByTestId('theme-footer-name').fill('Carter Francis')
  await page.getByTestId('theme-footer-email').fill('cartsfrancis@gmail.com')
  await page.getByTestId('theme-footer-note').fill('Direct Electron')
  await page.getByTestId('theme-hex-accent').fill('#00ff88')
  await page.waitForTimeout(800)

  // The logo goes in through the backend action (a real file dialog can't be
  // driven here); the picker's own wiring is exercised by the panel test above.
  await backendAction(page, 'report_set_theme', { theme: { logo: LOGO_PNG, logo_height: 34 } })
  await page.waitForTimeout(800)

  const theme = await docTheme(page)
  expect(theme.footer_name).toBe('Carter Francis')
  expect(theme.footer_email).toBe('cartsfrancis@gmail.com')
  expect(theme.footer_note).toBe('Direct Electron')
  expect(theme.accent).toBe('#00ff88')
  expect(theme.logo.startsWith('data:image/png')).toBe(true)

  await page.screenshot({ path: join(SHOTS, '03-theme-edited.png') })
  await page.getByTestId('theme-close').click()
  await expect(page.getByTestId('theme-panel')).toBeHidden()
})

test('3) present: the footer shows on the content slide, NOT the title', async () => {
  const { page } = ctx
  await page.getByTestId('report-present').click()
  await expect(page.locator('[data-testid="present-slide"][data-active="1"]'))
    .toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(1200)

  // Slide 1 is the TITLE — a title card carries its own attribution, so no
  // footer bar.
  const titleSlide = page.locator('[data-testid="present-slide"][data-active="1"]')
  await expect(titleSlide).toHaveAttribute('data-kind', 'title')
  expect(await titleSlide.getByTestId('present-footer').count(),
    'a title slide must not draw the footer').toBe(0)
  await page.screenshot({ path: join(SHOTS, '04-title-no-footer.png') })

  // Slide 2 — the footer, with name / email / note, the logo, and "2 / 2".
  await page.keyboard.press('ArrowRight')
  await page.waitForTimeout(900)
  const contentSlide = page.locator('[data-testid="present-slide"][data-active="1"]')
  const footer = contentSlide.getByTestId('present-footer')
  await expect(footer).toBeVisible()
  const text = (await contentSlide.getByTestId('present-footer-text').innerText()).replace(/\s+/g, ' ')
  console.log('[deck-theme] footer text =', JSON.stringify(text))
  expect(text).toContain('Carter Francis')
  expect(text).toContain('cartsfrancis@gmail.com')
  expect(text).toContain('Direct Electron')
  await expect(contentSlide.getByTestId('present-footer-logo')).toBeVisible()
  await expect(contentSlide.getByTestId('present-footer-number')).toHaveText('2 / 2')

  // The footer must not sit UNDER the pager — a presented deck that overlaps
  // its own chrome looks broken however correct the DOM is.
  const fBox = (await footer.boundingBox())!
  const vh = page.viewportSize()?.height ?? (await page.evaluate(() => window.innerHeight))
  console.log('[deck-theme] footer bottom =', Math.round(fBox.y + fBox.height), 'of', vh)
  expect(fBox.y + fBox.height, 'the footer overflows the slide').toBeLessThanOrEqual(vh)

  await page.screenshot({ path: join(SHOTS, '05-content-footer.png') })
})

test('4) a colour change reaches the MARKDOWN, not just the chrome', async () => {
  const { page } = ctx
  await exitPresent(page)
  // Paper: a light background — the most visible possible theme switch, and the
  // one that catches text colour that never left the dark default.
  await page.getByTestId('report-theme').click()
  await expect(page.getByTestId('theme-panel')).toBeVisible()
  await page.getByTestId('theme-preset-paper').click()
  await page.waitForTimeout(700)
  await page.getByTestId('theme-close').click()

  await page.getByTestId('report-present').click()
  await expect(page.locator('[data-testid="present-slide"][data-active="1"]'))
    .toBeVisible({ timeout: 15_000 })
  await page.keyboard.press('ArrowRight')
  await page.waitForTimeout(1200)

  // The heading inside the slide's markdown must have taken the theme's text
  // colour — it is styled by an injected stylesheet, not React inline styles,
  // so this is the assertion that the CSS-variable plumbing actually works.
  const colors = await page.evaluate(() => {
    const slide = document.querySelector('[data-testid="present-slide"][data-active="1"]')
    const overlay = document.querySelector('[data-testid="present-mode"]') as HTMLElement
    const h2 = slide?.querySelector('.present-md h2') as HTMLElement | null
    return {
      deckBg: overlay ? getComputedStyle(overlay).backgroundColor : null,
      headingColor: h2 ? getComputedStyle(h2).color : null,
    }
  })
  console.log('[deck-theme] computed =', JSON.stringify(colors))
  // Paper's bg #f6f5f2 → a light rgb; the dark default would be rgb(20,20,31).
  expect(colors.deckBg).toBe('rgb(246, 245, 242)')
  expect(colors.headingColor, 'the slide markdown ignored the theme')
    .toBe('rgb(27, 27, 32)')

  await page.screenshot({ path: join(SHOTS, '06-paper-theme.png') })
  await exitPresent(page)
})

/**
 * The scope buttons have to CONFIRM themselves. "Set as default" writes to
 * settings.json and "Use my default" / "Reset" usually land on a theme that
 * looks similar to what you had — so without feedback all three can be pressed
 * with no visible consequence at all, which reads as a dead button.
 */
test('5) Set as default / Use my default / Reset each confirm visibly', async () => {
  const { page } = ctx
  await page.getByTestId('report-theme').click()
  await expect(page.getByTestId('theme-panel')).toBeVisible()

  for (const [testid, label] of [
    ['theme-set-default', '✓ Saved as default'],
    ['theme-use-default', '✓ Applied'],
    ['theme-reset', '✓ Reset'],
  ] as const) {
    const btn = page.getByTestId(testid)
    await expect(btn).toHaveAttribute('data-confirmed', '0')
    await btn.click()
    await expect(btn, `${testid} gave no confirmation`).toHaveAttribute('data-confirmed', '1')
    await expect(btn).toHaveText(label)
    await page.screenshot({ path: join(SHOTS, `07-confirm-${testid}.png`) })
    // …and it must go back on its own, or the panel is stuck looking "saved".
    await expect(btn).toHaveAttribute('data-confirmed', '0', { timeout: 5_000 })
  }
})

/**
 * Typing must echo IMMEDIATELY. The fields are owned by the backend
 * (report_state), so bound directly each keystroke would render the previous
 * value until the reply landed. This types fast and checks the input holds the
 * whole string straight away, then that it reaches the document.
 */
test('6) typing in a theme field echoes immediately and still persists', async () => {
  const { page } = ctx
  const field = page.getByTestId('theme-footer-name')
  await field.fill('')
  await page.waitForTimeout(400)

  const typed = 'Dr Carter Francis'
  await field.pressSequentially(typed, { delay: 12 })
  // No wait: the input must already show every character.
  expect(await field.inputValue(),
    'the field lagged behind typing — it is bound to the backend echo').toBe(typed)

  // And the debounced write still lands in the document.
  await expect.poll(async () => (await docTheme(page))?.footer_name, {
    timeout: 5_000, message: 'the debounced theme write never reached the backend',
  }).toBe(typed)

  await page.getByTestId('theme-close').click()
})
