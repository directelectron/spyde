/**
 * console_preview.spec.ts — the math console's PERSISTENT overlay pills and
 * eye-toggled live preview (built after console_math.spec.ts). Covers:
 *
 *   1. Launch + load si_grains.
 *   2. Pill persistence: drag the signal breadcrumb pill into the console,
 *      confirm console-pill-<name> renders, then keep typing and confirm the
 *      pill survives (the old cosmetic pill auto-cleared after 2.6s — that
 *      behavior is GONE, this is the regression test for it).
 *   3. Word-boundary correctness: `<name>2 + <name>` pills ONLY the bound
 *      name, never as a substring of the longer identifier.
 *   4. Eye toggle + auto preview: an operator-only expression previews as an
 *      image with visible non-transparent pixels in the fixed slot.
 *   5. Tier gating: a call-containing auto-tier expression is quietly refused
 *      ("Ctrl+Enter" in the reason); the explicit (Ctrl+Enter) tier then
 *      either computes a real preview or refuses as "too expensive" — the
 *      cost guard's outcome depends on the dataset's chunking, so this test
 *      OBSERVES which branch fires and pins its own record of that.
 *   6. Quiet failure: an incomplete expression produces no error styling.
 *   7. Eye off: the preview slot unmounts.
 *   8. No renderer JS errors; no ERROR/Traceback in the backend log.
 *
 * DnD choreography copied from console_math.spec.ts / breadcrumb_header.spec.ts
 * (construct a real DataTransfer in-page, dispatch native drag events on the
 * source/target found via data-testid — Playwright can't hand a DataTransfer
 * across independently-evaluated locators).
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = 'console_shots'
mkdirSync(SHOTS, { recursive: true })

async function shot(page: any, n: number, name: string) {
  await page.screenshot({ path: `${SHOTS}/${String(n).padStart(2, '0')}-${name}.png` })
}

/**
 * Poll the ConsolePreviewPanel's rendered DOM (the pop-out floating above the
 * bar) until it settles on a NEW state (different from `prevSignature`), then
 * classify it.
 *
 * NOTE on why this doesn't use `backend.waitForMessage('console_preview_result')`:
 * that message travels the `PLOTAPP:` stdout line protocol, which is consumed
 * INSIDE the Electron main process's own listener on the Python child's stdout
 * (runner.ts) and relayed to the renderer over Electron IPC — it is never
 * re-printed to the Electron app's OWN process stdout, so `backend.messages`
 * (which tails `app.process().stdout`) never sees it (confirmed empirically:
 * `backend.messages` stayed `[]` for the whole run). Same rule CLAUDE.md
 * documents for `playback_state`/`figure` and insitu_playback.spec.ts already
 * works around — wait on rendered DOM state instead of the message bus.
 *
 * Returns a signature string classifying what's showing: 'image:<litPixels>',
 * 'sparkline:<litPixels>', 'scalar:<text>', 'reason:<text>', or 'empty'.
 */
async function readPreviewSignature(page: any): Promise<string> {
  const slot = page.getByTestId('console-preview-panel')
  if (await slot.count() === 0) return 'no-slot'
  const canvas = page.getByTestId('console-preview-canvas')
  if (await canvas.count() > 0) {
    const stats = await canvas.evaluate((el: HTMLCanvasElement) => {
      const ctx = el.getContext('2d')
      if (!ctx) return { lit: -1, nonAlphaZero: -1, maxVal: -1, sum: -1 }
      const d = ctx.getImageData(0, 0, el.width, el.height).data
      let lit = 0, nonAlphaZero = 0, maxVal = 0, sum = 0
      for (let p = 0; p < d.length; p += 4) {
        if (d[p + 3] > 0) {
          nonAlphaZero++
          sum += d[p]
          if (d[p] > maxVal) maxVal = d[p]
          if (d[p] > 30) lit++
        }
      }
      return { lit, nonAlphaZero, maxVal, sum }
    })
    // `sum` is a full-content checksum so two DIFFERENT frames with a
    // coincidentally equal lit-count still produce distinct signatures (needed
    // by the nav-move step, which asserts the thumbnail CHANGED).
    return `canvas:${stats.lit}:${stats.nonAlphaZero}:${stats.maxVal}:${stats.sum}`
  }
  const text = (await slot.textContent().catch(() => '')) || ''
  return `text:${text}`
}

/** True once a signature reflects ACTUAL content, not just the slot mounting
 *  empty (a freshly-mounted canvas reads 'canvas:0' for a beat before the
 *  async preview reply paints it) or the empty '·' placeholder. */
function isSettledSignature(sig: string): boolean {
  if (sig.startsWith('canvas:')) return Number(sig.split(':')[1]) > 0
  if (sig.startsWith('text:')) return sig.slice('text:'.length).trim().length > 0
  return false
}

/**
 * Poll until the preview slot's signature differs from `prevSig` AND reflects
 * settled content (not just an empty just-mounted canvas), or timeout.
 */
async function waitForPreviewChange(page: any, prevSig: string, timeoutMs = 15_000): Promise<string> {
  const start = Date.now()
  let sig = prevSig
  while (Date.now() - start < timeoutMs) {
    sig = await readPreviewSignature(page)
    if (sig !== prevSig && isSettledSignature(sig)) return sig
    await page.waitForTimeout(150)
  }
  return sig
}

/**
 * Poll until the preview signature has been STABLE for `quietMs` (no change),
 * then return it — used to let an in-flight nav-refresh land before the next
 * assertion baselines against the signature.
 */
async function waitForPreviewSettle(page: any, quietMs = 1_200, timeoutMs = 10_000): Promise<string> {
  const start = Date.now()
  let sig = await readPreviewSignature(page)
  let stableSince = Date.now()
  while (Date.now() - start < timeoutMs) {
    await page.waitForTimeout(200)
    const cur = await readPreviewSignature(page)
    if (cur !== sig) {
      sig = cur
      stableSince = Date.now()
    } else if (Date.now() - stableSince >= quietMs) {
      return sig
    }
  }
  return sig
}

async function dragAndDrop(page: any, srcTestId: string, dstTestId: string) {
  return await page.evaluate(({ srcTestId, dstTestId }: any) => {
    function el(testId: string): HTMLElement {
      const found = document.querySelector(`[data-testid="${testId}"]`)
      if (!found) throw new Error(`no element with data-testid="${testId}"`)
      return found as HTMLElement
    }
    const src = el(srcTestId)
    const dst = el(dstTestId)
    const dt = new DataTransfer()
    function fire(target: HTMLElement, type: string) {
      const rect = target.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true,
        cancelable: true,
        clientX: rect.left + rect.width / 2,
        clientY: rect.top + rect.height / 2,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
      return ev
    }
    fire(src, 'dragstart')
    const typesAfterStart = Array.from(dt.types)
    fire(dst, 'dragenter')
    fire(dst, 'dragover')
    fire(dst, 'drop')
    fire(src, 'dragend')
    return { types: typesAfterStart }
  }, { srcTestId, dstTestId })
}

test('console preview: persistent pills, word boundary, eye-toggle live preview, ' +
     'tier gating, quiet failure', async () => {
  test.setTimeout(300_000)

  const { app, page, backend, assertNoJsErrors } = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'WARNING' },
  })
  let shotN = 20   // console_math.spec.ts uses 01-11; start well clear of it.
  try {
    // ── 1. Launch + load si_grains ────────────────────────────────────────
    await page.waitForTimeout(1500)
    await backendAction(page, 'load_test_data_si_grains')
    await waitForSubwindowCount(page, 2, 60_000)   // navigator + signal
    await page.waitForTimeout(2000)

    const input = page.getByTestId('console-input')
    await expect(page.getByTestId('console-bar')).toBeVisible({ timeout: 15_000 })
    await shot(page, ++shotN, 'loaded')

    // ── 2. Pill persistence (the core regression) ────────────────────────
    const sigWin = page.getByTestId('subwindow')
      .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
      .first()
    const pill = sigWin.getByTestId('window-breadcrumb')
    await expect(pill).toBeVisible({ timeout: 10_000 })
    await pill.evaluate((el: HTMLElement) => el.setAttribute('data-console-drag-src', '1'))
    await input.click()
    await input.fill('')

    const dragResult = await page.evaluate(() => {
      const src = document.querySelector('[data-console-drag-src="1"]') as HTMLElement
      const dst = document.querySelector('[data-testid="console-input"]') as HTMLElement
      if (!src || !dst) throw new Error('drag src/dst not found')
      const dt = new DataTransfer()
      function fire(target: HTMLElement, type: string) {
        const rect = target.getBoundingClientRect()
        const ev = new DragEvent(type, {
          bubbles: true, cancelable: true,
          clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2,
        })
        Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
        target.dispatchEvent(ev)
        return ev
      }
      fire(src, 'dragstart')
      const typesAfterStart = Array.from(dt.types)
      fire(dst, 'dragenter')
      fire(dst, 'dragover')
      fire(dst, 'drop')
      fire(src, 'dragend')
      return { types: typesAfterStart }
    })
    expect(dragResult.types, 'breadcrumb pill dragstart must set the signal-ref MIME type')
      .toContain('application/x-spyde-signal-ref')

    await expect.poll(async () => (await input.inputValue()).trim(), {
      timeout: 10_000,
      message: 'console input should contain the resolved signal variable name after the drop',
    }).not.toBe('')
    const name = (await input.inputValue()).trim()
    console.log('pill persistence: inserted name =', JSON.stringify(name))
    expect(name.length).toBeGreaterThan(0)

    const namePill = page.getByTestId(`console-pill-${name}`)
    await expect(namePill, 'dropped var must render as a pill').toBeVisible({ timeout: 5_000 })
    await shot(page, ++shotN, 'pill-after-drop')

    // Keep typing after the drop — the OLD cosmetic pill auto-cleared after
    // 2.6s; the new overlay pill must persist for as long as the identifier
    // remains in the text.
    await input.click()
    await input.press('End')
    await input.pressSequentially(' > 100')
    await expect(namePill, 'pill must survive continued typing (no 2.6s auto-clear)')
      .toBeVisible({ timeout: 3_000 })
    expect(await input.inputValue()).toBe(`${name} > 100`)
    await shot(page, ++shotN, 'pill-persists-after-typing')

    // ── 3. Word boundary ──────────────────────────────────────────────────
    await input.fill(`${name}2 + ${name}`)
    const pillMatches = page.locator(`[data-testid="console-pill-${name}"]`)
    await expect(pillMatches).toHaveCount(1)
    // Also confirm no accidental pill for the longer identifier itself.
    expect(await page.locator(`[data-testid="console-pill-${name}2"]`).count()).toBe(0)
    await shot(page, ++shotN, 'word-boundary')
    await input.fill('')

    // ── 4. Eye + auto preview ──────────────────────────────────────────────
    // The eye ALWAYS starts OFF (deliberately NOT persisted — an auto-opening
    // panel on launch reads as noise; user call). Pin that: no panel exists
    // before the first click.
    const eye = page.getByTestId('console-preview-eye')
    await expect(page.getByTestId('console-preview-panel'),
      'the preview panel must NOT be open at startup (eye starts off, not persisted)')
      .toHaveCount(0)
    await eye.click()
    await expect(page.getByTestId('console-preview-panel'), 'eye click must mount the preview slot')
      .toHaveCount(1)
    const sigBeforeFirstPreview = await readPreviewSignature(page)
    // Use an arithmetic (non-comparison) expression on the raw signal so the
    // thumbnail always carries the diffraction pattern's real intensity
    // variation. A `> 100` BOOLEAN mask was tried first here and (correctly)
    // rendered an all-zero/all-black thumbnail at the default (0,0) navigator
    // cursor — this synthetic dataset's corner pixel's diffraction pattern
    // apparently never exceeds 100, so the mask is legitimately all-False
    // there (a data fact, not a preview-pipeline bug: `canvas:0:484:0` — 484
    // opaque, all-zero-value pixels — confirms the payload rendered and
    // painted correctly, just an all-False frame). `+ 0` guarantees a real
    // greyscale gradient regardless of cursor position.
    await input.fill(`${name} + 0`)
    const sig1 = await waitForPreviewChange(page, sigBeforeFirstPreview, 15_000)
    console.log('auto preview (arithmetic) signature:', sig1)
    expect(sig1.startsWith('canvas:'), `expected an image/sparkline canvas, got "${sig1}"`).toBe(true)
    const litPixels = Number(sig1.split(':')[1])
    expect(litPixels, 'preview thumbnail must paint visible pixels').toBeGreaterThan(0)

    const slot = page.getByTestId('console-preview-panel')
    await expect(slot).toBeVisible({ timeout: 5_000 })
    await expect(page.getByTestId('console-preview-canvas')).toBeVisible({ timeout: 5_000 })
    await shot(page, ++shotN, 'eye-on-auto-preview')

    // ── 4b. The preview TRACKS THE NAVIGATOR ───────────────────────────────
    // Moving the crosshair re-runs the last AUTO preview backend-side
    // (base_selector.NAV_CHANGE_HOOKS → ConsoleSession nav_refresh) with the
    // SAME preview id — no typing involved. Drive the REAL selector
    // server-side via the test_nav_drag action (the lever
    // nav_drag_distributed.spec.ts uses).
    //
    // si_grains is 6×6 with GRAINS — frames within a grain are byte-identical
    // (verified: frame(3,3) == frame(5,5), the default centre cursor's grain),
    // so a single arbitrary move can land on an identical frame and the
    // checksum legitimately can't change. Sequence two positions verified to
    // hold DIFFERENT frames: settle at (0,0) first, then move to (5,5)
    // (frame(0,0) != frame(5,5)) and require the thumbnail to change.
    await page.evaluate(() => (window as any).electron.action('test_nav_drag', { targets: [[0, 0]] }))
    const sigAtOrigin = await waitForPreviewSettle(page)
    console.log('preview signature settled at (0,0):', sigAtOrigin)
    await page.evaluate(() => (window as any).electron.action('test_nav_drag', { targets: [[5, 5]] }))
    const sigAfterNavMove = await waitForPreviewChange(page, sigAtOrigin, 15_000)
    console.log('nav-move refreshed preview signature:', sigAfterNavMove)
    expect(sigAfterNavMove.startsWith('canvas:'),
      `expected a refreshed image after the nav move, got "${sigAfterNavMove}"`).toBe(true)
    expect(sigAfterNavMove, 'preview must re-render at the new navigator position')
      .not.toBe(sigAtOrigin)
    await shot(page, ++shotN, 'nav-move-refreshes-preview')

    // ── 5. Tier gating ───────────────────────────────────────────────────
    const sigBeforeCall = await readPreviewSignature(page)
    await input.fill(`${name}.sum(axis=(0, 1))`)
    const sigAutoCall = await waitForPreviewChange(page, sigBeforeCall, 15_000)
    console.log('auto-tier call-expression signature:', sigAutoCall)
    // The AUTO tier refuses any call-containing expression, so the slot must
    // fall to the reason text (no canvas) mentioning Ctrl+Enter.
    expect(sigAutoCall.startsWith('text:'), `expected a reason (no canvas) for auto-tier call refusal, got "${sigAutoCall}"`).toBe(true)
    expect(sigAutoCall).toContain('Ctrl+Enter')
    await shot(page, ++shotN, 'tier-gate-auto-refused')

    // Explicit tier: Ctrl+Enter always fires a one-shot preview even with a
    // call present. The cost guard (8 source chunks / 256 MB) may or may not
    // trip depending on the dataset's actual chunking — OBSERVE which branch
    // fires and assert against that.
    const sigBeforeExplicit = await readPreviewSignature(page)
    await input.press('Control+Enter')
    const sigExplicit = await waitForPreviewChange(page, sigBeforeExplicit, 15_000)
    console.log('explicit-tier (Ctrl+Enter) call-expression signature:', sigExplicit)
    // Disambiguate a `scalar` result from an `unavailable` reason (both render
    // as plain text with no canvas — ConsolePreviewPanel gives them no distinct
    // testid): a refusal's reason is one of the fixed backend strings
    // ("too expensive", "can't estimate cost", "incomplete expression", a
    // "<Type>: message" exception repr for a manual-tier eval error); a scalar
    // result is the value's repr, which for `<name>.sum(axis=(0,1))` is a
    // numeric-looking string. Treat known refusal phrasing as a refusal, and
    // "text:" content that DOESN'T match any refusal phrasing as a real
    // (scalar) result — an image/sparkline canvas is unambiguously a result.
    const isExpensiveRefusal = sigExplicit.startsWith('text:') && /expensive/i.test(sigExplicit)
    const isOtherRefusal = sigExplicit.startsWith('text:')
      && /can't estimate cost|incomplete expression|Error:/i.test(sigExplicit)
    const isRealResult = sigExplicit.startsWith('canvas:')
      || (sigExplicit.startsWith('text:') && !isExpensiveRefusal && !isOtherRefusal)
    // OBSERVED (see the console.log above / the test's stdout in the report):
    // record here which branch actually fired on this run so the assertion
    // tracks backend truth rather than a guess.
    console.log(isExpensiveRefusal
      ? 'OBSERVED: explicit tier hit the "too expensive" cost-guard refusal.'
      : isOtherRefusal
        ? 'OBSERVED: explicit tier hit a non-cost refusal (see signature above).'
        : 'OBSERVED: explicit tier computed and returned a real preview (cost guard did not trip).')
    expect(isExpensiveRefusal || isRealResult,
      `explicit preview must be either a real result or an "expensive" refusal, got "${sigExplicit}"`)
      .toBe(true)
    await shot(page, ++shotN, 'tier-gate-explicit-ctrl-enter')

    // ── 6. Quiet failure ───────────────────────────────────────────────────
    const errorTogglesBefore = await page.locator('[data-testid="console-error-toggle"]').count()
    await input.fill(`${name} >`)   // incomplete expression
    await page.waitForTimeout(800)
    const errorTogglesAfter = await page.locator('[data-testid="console-error-toggle"]').count()
    expect(errorTogglesAfter, 'an incomplete preview expression must not surface error styling')
      .toBe(errorTogglesBefore)
    // The reason slot for an incomplete expression should not read as an
    // exception traceback (it's the quiet "incomplete expression" reason or a
    // dimmed prior frame — never a Python exception string).
    const slotText = await page.getByTestId('console-preview-panel').textContent().catch(() => '')
    console.log('quiet-failure slot text:', JSON.stringify(slotText))
    expect(slotText || '').not.toMatch(/Traceback|Error:/)
    await shot(page, ++shotN, 'quiet-failure-incomplete-expr')

    // ── 7. Eye off ──────────────────────────────────────────────────────────
    await eye.click()
    await expect(page.getByTestId('console-preview-panel')).toHaveCount(0)
    await shot(page, ++shotN, 'eye-off-slot-gone')

    // ── 8. No renderer JS errors; scan backend log for real errors ─────────
    assertNoJsErrors()
    const backendErrors = backend.logBuffer.filter((l: string) =>
      /ERROR|Traceback/i.test(l)
      && !/Content Security-Policy|Content Security/i.test(l)
      && !/willReadFrequently/i.test(l),
    )
    if (backendErrors.length) {
      console.log('backend log ERROR/Traceback lines:\n' + backendErrors.join('\n'))
    }
    expect(backendErrors.length,
      `backend log contains ${backendErrors.length} ERROR/Traceback lines`).toBe(0)
  } finally {
    await app.close()
  }
})
