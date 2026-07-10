/**
 * console_math.spec.ts — first integration run of the math console (backend
 * ConsoleSession + renderer ConsoleBar) built in this session. Covers:
 *
 *   1. Console bar visible at app start.
 *   2. `np.random.rand(256, 256)` echo + out1 chip; drag the chip onto the MDI
 *      area (synthetic HTML5 DnD) to open a Signal window showing the noise.
 *   3. Load the synthetic in-situ movie, drag the `»` console-ref grip from
 *      its Signal window onto the console input to insert the variable name,
 *      then build a lazy-safe boolean mask and open it (binary image).
 *   4. Arithmetic on the bound signal name + double-click-to-open.
 *   5. Error path: 1/0 → red echo, click-to-expand traceback, collapse on the
 *      next exec.
 *   6. History: ArrowUp recalls the last submitted line.
 *
 * DnD: Playwright's `dispatchEvent` can carry a `dataTransfer` value that Chromium
 * treats as `DataTransfer` (per playwright docs, `JSHandle`s for `dataTransfer` are
 * not directly settable from the Node side across independent elements, so we do
 * the whole drag choreography — dragstart/dragenter/dragover/drop — INSIDE one
 * `page.evaluate` call, constructing a real `DataTransfer` in-page and dispatching
 * native drag events on the source/target elements found via `data-testid`. This
 * exercises the SAME `onDragStart`/`onDrop` React handlers the real user drag
 * would (they read `e.dataTransfer`), unlike a scripted `sendAction` shortcut.
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
 * Perform a full native HTML5 drag-and-drop from the element matching
 * `srcTestId` to the element matching `dstTestId`, entirely inside the page
 * (so the constructed `DataTransfer` is shared across dragstart/dragover/drop
 * the way a real user drag would be). Returns the dataTransfer types seen by
 * the source's dragstart handler, for assertions.
 */
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
      // jsdom/Chromium DragEvent doesn't accept dataTransfer via the
      // constructor dict in all versions — set it directly (real Chromium
      // DragEvent has a writable dataTransfer property backing store here).
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

test('math console: exec, chip drag-to-MDI, signal drag-in, mask, arithmetic, ' +
     'error traceback, history', async () => {
  test.setTimeout(300_000)

  const { app, page, backend, assertNoJsErrors } = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'WARNING' },
  })
  let shotN = 0
  try {
    // ── 1. Console bar visible at app start ──────────────────────────────────
    await page.waitForTimeout(1000)
    const consoleBar = page.getByTestId('console-bar')
    const input = page.getByTestId('console-input')
    await expect(consoleBar).toBeVisible({ timeout: 15_000 })
    await expect(input).toBeVisible()
    await shot(page, ++shotN, 'console-visible')

    // ── 2. np.random.rand(256, 256) → echo + out1 chip ───────────────────────
    await input.click()
    await input.fill('np.random.rand(256, 256)')
    await input.press('Enter')

    const echo = page.getByTestId('console-echo')
    await expect(echo, 'echo strip should show the out1 array repr')
      .toContainText('array', { timeout: 10_000 })
    const chip = page.getByTestId('console-chip-out1')
    await expect(chip, 'out1 chip should appear after the exec').toBeVisible({ timeout: 10_000 })
    await expect(chip).toContainText('256')
    await shot(page, ++shotN, 'out1-echo-and-chip')

    // Verify the drag wiring: dragstart on the chip must populate the
    // CONSOLE_VAR_DRAG_MIME dataTransfer type before we rely on the full drop.
    const mdiArea = page.getByTestId('mdi-area')
    const dragResult = await dragAndDrop(page, 'console-chip-out1', 'mdi-area')
    expect(dragResult.types, 'chip dragstart must set the console-var MIME type')
      .toContain('application/x-spyde-console-var')

    // The drop should have fired console_create_window -> a new Signal window.
    await waitForSubwindowCount(page, 1, 30_000)
    const noiseWindow = page.getByTestId('subwindow').first()
    await expect(noiseWindow).toBeVisible({ timeout: 15_000 })
    await page.waitForTimeout(1500)   // let the figure iframe paint

    // KNOWN PRE-EXISTING RACE (not introduced by the console, reproduces on a
    // plain `load_test_data_si_grains` too): anyplotlib's standalone-HTML iframe
    // loads its widget ESM via an async `import(blobUrl)`; the FIRST state push
    // (image pixels) can arrive and call `model.set()`/`save_changes()` before
    // `render()` has registered its `model.on('change', …)` listener. The model's
    // `_data` is updated but nothing repaints — so a window that never gets a
    // SECOND organic paint (true of every console-created window; a
    // navigator-driven window instead self-heals on the first drag) can render
    // permanently blank. Confirmed independent of SpyDE's own message plumbing
    // (which now retains+replays both the base64 AND raw-binary pixel push for a
    // late-mounting iframe — see SpyDEContext.tsx's `latestBinaryStates` fix in
    // this change) — the loss happens INSIDE anyplotlib's ESM bootstrap, out of
    // this repo's control. A real user interaction that re-triggers `_set_array`
    // (e.g. changing the colormap) reliably un-blanks it — do that here so the
    // screenshot shows the actual noise instead of a placeholder black square.
    await page.getByRole('combobox', { name: 'Colormap' }).selectOption('viridis')
    await page.waitForTimeout(1000)
    await shot(page, ++shotN, 'noise-window-open')

    // ── 3. Load the movie, signal drag-IN, lazy-safe mask ────────────────────
    await backendAction(page, 'load_test_data_movie', { size: 256, frames: 6 })
    await waitForSubwindowCount(page, 3, 60_000)
    await page.waitForTimeout(2000)

    // Movie's Signal window: subwindow titles are generic ("Signal" /
    // "Navigator"), not the dataset name — so identify it as the newest
    // non-Navigator subwindow (the noise window from step 2 is titled "out1",
    // distinct from the movie's default "Signal" title).
    const movieSignalWindow = page.getByTestId('subwindow')
      .filter({ hasNotText: 'Navigator' })
      .filter({ hasNotText: 'out1' })
      .last()
    await expect(movieSignalWindow).toBeVisible({ timeout: 15_000 })
    const gripTestId = 'console-ref-handle'
    // Scope the grip lookup to the movie signal window specifically, since
    // multiple SubWindows each have their own grip with the same test id.
    const grip = movieSignalWindow.getByTestId(gripTestId)
    await expect(grip).toBeVisible({ timeout: 10_000 })

    // Give each SubWindow's grip a unique marker attribute so the in-page
    // evaluate can target THIS window's grip specifically (data-testid is not
    // unique across windows).
    await grip.evaluate((el: HTMLElement) => el.setAttribute('data-console-drag-src', '1'))
    await input.click()
    await input.fill('')
    const signalDragResult = await page.evaluate(() => {
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
    expect(signalDragResult.types, 'signal grip dragstart must set the signal-ref MIME type')
      .toContain('application/x-spyde-signal-ref')

    // The console input should now contain the resolved variable name (the
    // movie tree's positional alias, e.g. "s1" or "s2" depending on load order).
    await expect.poll(async () => await input.inputValue(), {
      timeout: 10_000,
      message: 'console input should contain the resolved signal variable name after the drop',
    }).not.toBe('')
    const insertedName = (await input.inputValue()).trim()
    console.log('signal drag-in inserted name:', JSON.stringify(insertedName))
    expect(insertedName.length, 'dropped signal name must be non-empty').toBeGreaterThan(0)
    await shot(page, ++shotN, 'signal-dragged-into-console')

    // Build a lazy-safe boolean mask off the inserted name.
    await input.fill(`mask = ${insertedName} > 500`)
    await input.press('Enter')
    const maskChip = page.getByTestId('console-chip-mask')
    await expect(maskChip, 'mask chip should appear after the assignment').toBeVisible({ timeout: 10_000 })
    await expect(maskChip).toContainText('lazy')
    await expect(maskChip).toContainText('bool')
    await shot(page, ++shotN, 'mask-chip')

    // Open the mask's window (double-click is the documented fallback path).
    const subwindowCountBeforeMask = await page.locator('[data-testid="subwindow"]').count()
    await maskChip.dblclick()
    await waitForSubwindowCount(page, subwindowCountBeforeMask + 1, 30_000)
    await page.waitForTimeout(1500)
    await shot(page, ++shotN, 'mask-window-open')

    // ── 4. Arithmetic on the bound name + double-click-to-open ───────────────
    await input.fill(`${insertedName} * 2 + 10`)
    await input.press('Enter')
    const arithChip = page.getByTestId('console-chip-out2')
    await expect(arithChip, 'out2 chip should appear for the arithmetic result')
      .toBeVisible({ timeout: 10_000 })
    await shot(page, ++shotN, 'arith-out2-chip')

    const subwindowCountBeforeArith = await page.locator('[data-testid="subwindow"]').count()
    await arithChip.dblclick()
    await waitForSubwindowCount(page, subwindowCountBeforeArith + 1, 30_000)
    await page.waitForTimeout(1500)
    await shot(page, ++shotN, 'arith-window-open')

    // ── 5. Error path: 1/0 → red echo, click-to-expand traceback ─────────────
    await input.fill('1/0')
    await input.press('Enter')
    const errorToggle = page.getByTestId('console-error-toggle')
    await expect(errorToggle, 'error echo should appear for 1/0').toBeVisible({ timeout: 10_000 })
    await expect(errorToggle).toContainText('ZeroDivisionError')
    await errorToggle.click()
    const traceback = page.getByTestId('console-traceback')
    await expect(traceback).toBeVisible({ timeout: 5_000 })
    await expect(traceback).toContainText('ZeroDivisionError')
    await shot(page, ++shotN, 'error-traceback-open')

    // Next exec collapses the traceback panel.
    await input.fill('1 + 1')
    await input.press('Enter')
    await expect(page.getByTestId('console-echo')).toBeVisible({ timeout: 10_000 })
    await expect(traceback, 'traceback panel should collapse after the next exec').toHaveCount(0)
    await shot(page, ++shotN, 'traceback-collapsed-after-next-exec')

    // ── 6. History: ArrowUp recalls the last submitted line ──────────────────
    await input.fill('')
    await input.press('ArrowUp')
    await expect.poll(async () => await input.inputValue(), { timeout: 5_000 })
      .toBe('1 + 1')
    await shot(page, ++shotN, 'history-arrowup')

    // ── 7. No renderer JS errors; scan backend log for real errors ───────────
    assertNoJsErrors()

    const backendErrors = backend.logBuffer.filter((l: string) =>
      /ERROR|Traceback/i.test(l)
      && !/Content Security-Policy|Content Security/i.test(l)
      && !/willReadFrequently/i.test(l)
      // The 1/0 ZeroDivisionError is EXPECTED user-code output captured by the
      // console engine itself (not a backend fault) — it's carried in the
      // console_result IPC payload, not logged at ERROR level, so no exclusion
      // should be needed here; kept only as a defensive note.
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
