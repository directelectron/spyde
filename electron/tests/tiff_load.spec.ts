/**
 * tiff_load.spec.ts — a plain .tif must OPEN, with a real Dask cluster up.
 *
 * The bug: rosettasciio's lazy TIFF graph closes over an open BufferedReader,
 * so it cannot be pickled to a Dask worker. `client.compute()` on it raised at
 * submit ("Could not serialize object of type _HLGExprSequence") — and because
 * that call sits between window_computing().start() and the _on_plot_ready that
 * stops it, every .tif opened as a permanently BLACK window stuck on
 * "Calculating…", with "Failed to load …" on the status bar.
 *
 * `dask: true` is the whole point — the failure only exists when a client
 * exists, which is why the headless suite (SPYDE_NO_DASK=1) never saw it and
 * why the unit tests in test_unpicklable_graph_display.py are not sufficient on
 * their own.
 *
 * Two files, because the two lazy-array→compute paths are different code:
 * single-page takes the no-navigation display (compute_display_future), and the
 * 8-page stack takes the progressive navigator fill (which must run in-process,
 * naming its scheduler).
 */
import { test, expect } from '@playwright/test'
import { execFileSync } from 'child_process'
import { existsSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'

const {
  launchApp, backendAction, waitForSubwindowCount, countColorPixels,
  backendErrorLines,
} = require('./_harness.cjs')

const OUT = join(tmpdir(), 'spyde-tiff-fixture')
const SHOTS = join(__dirname, '..', 'tiff_load_shots')

test.beforeAll(() => {
  const root = join(__dirname, '..', '..')
  const py = [join(root, '.venv', 'Scripts', 'python.exe'),
              join(root, '.venv', 'bin', 'python')].find((p) => existsSync(p))
    ?? (process.platform === 'win32' ? 'python' : 'python3')
  execFileSync(py, ['-m', 'spyde.tests.gen_tiff_fixture', OUT], { cwd: root })
})

for (const [label, file, windows] of [
  ['single-page (display path)', 'single.tif', 1],
  ['multi-page stack (navigator fill)', 'stack.tif', 2],
] as [string, string, number][]) {
  test(`a .tif opens and paints — ${label}`, async () => {
    test.setTimeout(180_000)
    const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
    const { page, backend } = ctx
    try {
      await backendAction(page, 'open_file', { path: join(OUT, file) })

      // A raw multi-page .tif is an ambiguous scan grid, so the backend asks
      // for the shape first (by design — see _wants_nav_prompt). Accept it.
      if (windows > 1) {
        const open = page.getByRole('button', { name: 'Open', exact: true })
        await expect(open).toBeVisible({ timeout: 30_000 })
        await open.click()
      }

      await waitForSubwindowCount(page, windows, 90_000)
      await page.waitForTimeout(6_000)
      await page.screenshot({ path: join(SHOTS, `${file}.png`) })

      // 1. The window actually carries an image. This is the assertion that
      //    fails on the bug: the figure stayed black forever.
      const bright = await countColorPixels(page, 'bright')
      console.log(`[tiff_load] ${file}: bright=${bright}`)
      expect(bright, 'the figure never painted').toBeGreaterThan(1_000)

      // 2. The computing overlay was RELEASED. It is started before the compute
      //    and only _on_plot_ready stops it, so a raise there stranded it up.
      expect(await page.getByText('Calculating').count(),
             'the Calculating… overlay never cleared').toBe(0)

      // 3. No serialization traceback (or anything else) in the backend log.
      const errs = backendErrorLines(backend)
      if (errs.length) console.log('[tiff_load] backend errors:\n' + errs.join('\n'))
      expect(errs).toHaveLength(0)
      ctx.assertNoJsErrors()
    } finally {
      await ctx.app.close()
    }
  })
}
