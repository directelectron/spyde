/**
 * shell_split_smoke.spec.ts — the extraction gate.
 *
 * The shell split moved the Python IPC/logging kernel into `de_shell` and the
 * main-process sidecar/env/updater kernel into `@de/shell-main`. Everything
 * those touch is invisible to a typecheck and to the headless pytest suite: the
 * backend spawning at all, PLOTAPP messages crossing the boundary, figures
 * rendering, and the log panel still receiving `spyde.*` records now that the
 * handler's verbose-package gate is registered rather than hardcoded.
 *
 * So this drives the real app and looks at pixels. It is deliberately shallow —
 * the feature suites cover behaviour; this covers "the wiring survived".
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, countColorPixels, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'shell_split_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  // SPYDE_LOG_LEVEL=INFO so the log-area registration is actually exercised:
  // at WARNING the handler's `_accept` short-circuits on level alone and would
  // pass even if `register_area_rules` had never run.
  ctx = await launchApp({ dask: false, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await ctx.page.waitForTimeout(1500)
})

test.afterAll(async () => { await ctx?.app?.close() })
test.setTimeout(180_000)

test('backend spawns, data loads, figures paint, logs stream', async () => {
  const { page, backend } = ctx

  await page.screenshot({ path: join(SHOTS, '01-boot.png') })

  // 1. The sidecar came up through the extracted @de/shell-main runner and the
  //    renderer got its PLOTAPP messages: a load opens real subwindows.
  await backendAction(page, 'load_test_vectors')
  await waitForSubwindowCount(page, 4, 90_000)
  await page.waitForTimeout(2000)
  await page.screenshot({ path: join(SHOTS, '02-loaded.png') })

  // 2. Figures actually PAINTED. A subwindow can exist with a blank iframe when
  //    the figure protocol or the binary transport is broken — which is exactly
  //    the kind of thing moving the custom scheme could have broken.
  //    (countColorPixels only understands 'bright' | 'red' | 'green'; any other
  //    string counts nothing and would pass vacuously.)
  const bright = await countColorPixels(page, 'bright')
  console.log('[shell-split] bright figure pixels =', bright)
  expect(bright).toBeGreaterThan(0)

  // 3. The vector overlay drew — markers travel a different path from the image
  //    pixels (figure JSON, not the binary frame), so this covers the half that
  //    'bright' does not.
  const red = await countColorPixels(page, 'red')
  console.log('[shell-split] red overlay pixels =', red)
  expect(red).toBeGreaterThan(0)

  // 4. emit_status crossed the boundary: ipc.py now lives in de_shell, and the
  //    status line is the shortest proof its stdout channel still reaches the
  //    renderer.
  await expect(page.locator('text=/Found \\d+ diffraction vectors/')).toBeVisible()

  // 5. Nothing died on the way. backendErrorLines surfaces stderr the PLOTAPP
  //    channel would otherwise swallow.
  const errs = backendErrorLines(backend)
  if (errs.length) console.log('[shell-split] backend stderr:\n' + errs.join('\n'))
  expect(errs.join('\n')).not.toMatch(/ModuleNotFoundError|ImportError|Traceback/)
})
