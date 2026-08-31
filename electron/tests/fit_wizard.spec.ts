/**
 * fit_wizard.spec.ts — the Fit caret, driven the way a user drives it (#55/#56/#58).
 *
 * The backend tests in test_fit_wizard.py cover the handler contract. They
 * cannot see: whether the toolbar button appears on a spectrum window at all,
 * whether the palette renders its sparklines or a row of blank buttons, whether
 * the component list rebuilds after an edit, or whether Commit actually opens a
 * window. That is what this drives, and every stage is screenshotted so the
 * pixels can be looked at (CLAUDE.md: the screenshot IS the test).
 *
 * Uses the bundled synthetic EELS SI — a real 1-D signal with a background and
 * three edges, so the model being built has something to fit.
 *
 * Run: npx playwright test tests/fit_wizard.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, raiseWindowOwning, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'fit_wizard_shots'

test('Fit caret: build a model, run it, commit component maps', async () => {
  test.setTimeout(420_000)

  const ctx = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'WARNING' },
  })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    // A small SI keeps the batched fit quick; the point is the UI, not scale.
    await backendAction(page, 'load_test_data_eels', { nav: [6, 6], n_channels: 512 })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(2_000)
    await page.screenshot({ path: `${SHOTS}/01-loaded.png`, fullPage: true })

    // ── the toolbar button must EXIST on a 1-D signal window ──────────────
    const sig = sigWindow(page)
    await expect(sig).toBeVisible()
    // The floating toolbar reveals on hover, and hovering the TITLEBAR also
    // focus-raises the window above sibling toolbars — without that the click
    // lands on the MDI area instead (om_wizard_lazy.spec.ts uses the same
    // idiom for the same reason).
    await sig.getByTestId('subwindow-titlebar').hover()
    const fitBtn = sig.getByTestId('action-btn-Fit')
    await expect(fitBtn).toBeVisible({ timeout: 15_000 })
    await page.screenshot({ path: `${SHOTS}/02-toolbar.png`, fullPage: true })

    // ── open the caret ───────────────────────────────────────────────────
    await fitBtn.click()
    const caret = page.locator('[data-testid="fit-wizard"]')
    await expect(caret).toBeVisible({ timeout: 20_000 })
    await page.screenshot({ path: `${SHOTS}/03-caret-open.png`, fullPage: true })

    // ── the + picker must show SHAPES, not blank buttons (#56) ───────────
    await page.locator('[data-testid="fit-add-toggle"]').click()
    const palette = page.locator('[data-testid="fit-palette"]')
    await expect(palette).toBeVisible({ timeout: 20_000 })
    await expect(page.locator('[data-testid="fit-add-Gaussian"]')).toBeVisible()
    // Each palette entry draws its preview as an inline <svg><polyline>. A
    // catalogue that failed to sample would render the buttons with no
    // polyline at all — visible only here.
    const sparks = await palette.locator('svg polyline').count()
    expect(sparks).toBeGreaterThanOrEqual(5)
    await page.screenshot({ path: `${SHOTS}/04-palette.png`, fullPage: true })
    await page.locator('[data-testid="fit-add-toggle"]').click()   // close it again

    // Adding a component grows the caret, and FloatingToolbar re-runs its
    // placement layout effect afterwards — which can move it. Waiting on the
    // caret's box to stop changing is the honest wait: there is no event for
    // "layout settled", but there IS an observable.
    const caretSettled = async () => {
      let prev = ''
      for (let i = 0; i < 25; i++) {
        const cur = JSON.stringify(await caret.boundingBox())
        if (cur === prev) return
        prev = cur
        await page.waitForTimeout(120)
      }
    }


    const addComponent = async (kind: string) => {
      // The picker is a POPUP behind "+ Component" and closes after each pick.
      await page.locator('[data-testid="fit-add-toggle"]').click()
      const pal = page.locator('[data-testid="fit-palette"]')
      await expect(pal).toBeVisible({ timeout: 10_000 })
      await caretSettled()
      await page.locator(`[data-testid="fit-add-${kind}"]`).click()
      // The status line is the backend's own acknowledgement — assert on it as
      // well as the row, so a click that never reached the backend is
      // distinguishable from a row that failed to render.
      await expect(page.locator('[data-testid="fit-status"]'))
        .toContainText(`Added ${kind}`, { timeout: 15_000 })
      await expect(page.locator(`[data-testid="fit-comp-${kind}"]`))
        .toBeVisible({ timeout: 15_000 })
    }

    await addComponent('PowerLaw')
    await addComponent('Gaussian')
    await page.screenshot({ path: `${SHOTS}/05-model-built.png`, fullPage: true })

    // The parameter rows are the caret rebuilt from the backend's fit_state —
    // if they are absent the model shown is not the model being fitted.
    await expect(page.locator('[data-testid="fit-p-Gaussian-centre"]')).toBeVisible()

    // ── edit a parameter and confirm the backend echoes it back ──────────
    const centre = page.locator('[data-testid="fit-p-Gaussian-centre"]')
    await centre.fill('532')
    await centre.blur()
    await page.waitForTimeout(1_500)
    await expect(centre).toHaveValue(/^53[12]/, { timeout: 10_000 })
    // This screenshot is where the LIVE PREVIEW and the on-plot drag handles
    // (#57) are checked — both live inside the anyplotlib figure, so there is
    // no DOM assertion for them and the pixels are the test. A silently broken
    // preview looks exactly like a working one to every other check here: the
    // first version passed this whole spec while drawing nothing, because
    // add_line takes `label` and was being given `name`.
    await page.screenshot({ path: `${SHOTS}/06-param-edited.png`, fullPage: true })

    // ── the model must SURVIVE close / reopen ────────────────────────────
    // The model and the fit live on the TREE, not on the caret controller: a
    // model costs real effort to build and a fit costs minutes, so closing the
    // caret to get it out of the way must not discard either.
    await page.locator('[data-testid="fit-close"]').click()
    await expect(caret).toBeHidden({ timeout: 15_000 })
    await sig.getByTestId('subwindow-titlebar').hover()
    await fitBtn.click()
    // Exactly ONE caret — the StrictMode open/close/open contract.
    await expect(page.locator('[data-testid="fit-wizard"]')).toHaveCount(1, { timeout: 15_000 })
    await expect(page.locator('[data-testid="fit-comp-Gaussian"]'))
      .toBeVisible({ timeout: 15_000 })
    await expect(page.locator('[data-testid="fit-p-Gaussian-centre"]'))
      .toHaveValue(/^53[12]/, { timeout: 10_000 })
    await caretSettled()
    await page.screenshot({ path: `${SHOTS}/06b-model-restored.png`, fullPage: true })

    // ── run the fit (second tab) ─────────────────────────────────────────
    // The live-preview window opened by adding components can sit over the caret
    // by now; raise its own window first (the click a user makes without noticing).
    await raiseWindowOwning(page, "fit-wizard")
    await page.locator('[data-testid="fit-tab-Run"]').click()
    await expect(page.locator('[data-testid="fit-run"]')).toBeVisible({ timeout: 10_000 })
    await caretSettled()
    await page.locator('[data-testid="fit-run"]').click()
    const status = page.locator('[data-testid="fit-status"]')
    await expect(status).toContainText(/converged/i, { timeout: 180_000 })
    await page.screenshot({ path: `${SHOTS}/07-fitted.png`, fullPage: true })

    // ── commit: one map per component, in a NEW window ───────────────────
    const commit = page.locator('[data-testid="fit-commit"]')
    await expect(commit).toBeVisible({ timeout: 20_000 })
    await commit.click()
    await waitForSubwindowCount(page, 3, 120_000)
    await page.waitForTimeout(2_500)
    // The committed window carries one chip per component — the same
    // click-one / cmd-click-to-tile toggle the strain components use (#58).
    //
    // Match the BREADCRUMB, anchored. Two maps windows exist by design — the
    // caret's live one (refreshed as positions fit) and this snapshot — and
    // `filter({hasText})` is a substring match over the whole subwindow, so
    // the old locator matched both and failed strict mode. `$` is what
    // separates "Fit components" from "Fit components (live)".
    const committed = page.getByTestId('subwindow').filter({
      has: page.getByTestId('window-breadcrumb').filter({ hasText: /Fit components$/ }),
    })
    await expect(committed).toBeVisible({ timeout: 20_000 })
    await expect(committed, 'the committed snapshot must be exactly one window')
      .toHaveCount(1)
    // And the live one is still there under its own name, so a user can tell
    // the moving window from the kept one.
    await expect(page.getByTestId('window-breadcrumb')
      .filter({ hasText: 'Fit components (live)' })).toHaveCount(1)
    await page.screenshot({ path: `${SHOTS}/09-committed.png`, fullPage: true })

    const errs = backend.logBuffer.filter((l: string) => /Traceback|CRITICAL/.test(l))
    expect(errs, `backend errors:\n${errs.join('\n')}`).toHaveLength(0)
    assertNoJsErrors()
  } finally {
    // Backend emit/emit_error are the PLOTAPP line protocol and never reach
    // Playwright's stdout (CLAUDE.md), so dump the captured log here — without
    // it a backend exception shows up only as a UI element that never appears.
    const tail = backend.logBuffer.slice(-40).join('\n')
    console.log(`\n──── backend log (tail) ────\n${tail}\n────────────────────────────`)
    await ctx.app.close().catch(() => {})
  }
})
