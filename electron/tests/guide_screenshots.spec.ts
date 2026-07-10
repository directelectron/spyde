/**
 * guide_screenshots.spec.ts — generate a screenshot PER STEP of each guide.
 *
 * Single source: the guides in `guides/*.ts` carry both the documentation (anchor
 * + body + image filename) AND an optional `drive` screenplay (how to reach the
 * step). This run walks every step's `drive`, waits on SIGNAL-based conditions
 * (subwindow count / element visible / canvas pixels — never fixed sleeps beyond
 * a tiny paint settle), and writes the step's `image` into the docs media dir so
 * the website (DocsApp reads `./media/{guide.id}/{step.image}`) shows real,
 * current screenshots. The user also gets these while away from the dev machine.
 *
 * Real-data tier only (`electron-real` project): needs a Dask client + the
 * bundled/real datasets the drive blocks load. Run:
 *   SPYDE_E2E_REAL=1 npx playwright test guide_screenshots.spec.ts --project=electron-real
 */
import { test } from '@playwright/test'
import { mkdirSync } from 'fs'
import { join } from 'path'
// Guides are pure-data modules (no React) — safe to import here.
import { GUIDES } from '../../guides/index'
const {
  launchApp, backendAction, waitForSubwindowCount, countColorPixels,
} = require('./_harness.cjs')

const MEDIA_ROOT = join(__dirname, '..', '..', 'docs-site', 'public', 'media')

test.describe('guide screenshots', () => {
  for (const guide of GUIDES) {
    // Only guides that define at least one drivable step are auto-captured.
    if (!guide.steps.some((s) => s.drive || s.image)) continue

    test(`capture "${guide.title}"`, async () => {
      const ctx = await launchApp({ dask: true })
      const { page } = ctx
      const outDir = join(MEDIA_ROOT, guide.id)
      mkdirSync(outDir, { recursive: true })

      try {
        for (const step of guide.steps) {
          const d = step.drive
          // --- perform the step's action -------------------------------------
          if (d?.action === 'backend' && d.backend) {
            await backendAction(page, d.backend)
          } else if (d?.action === 'click' || d?.action === 'hover') {
            const tid = d.testid || step.anchor
            if (tid) {
              // Act within the signal (non-navigator) window when the target is a
              // per-window control (toolbar/titlebar/action button).
              const sig = page.getByTestId('subwindow')
                .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) }).first()
              const scope = (await sig.count()) ? sig : page
              // Toolbar action buttons + sub-actions are hidden until the window
              // is hovered — reveal the floating toolbar first so the click lands.
              if (/^(action-btn-|subaction-)/.test(tid) && (await sig.count())) {
                await sig.getByTestId('subwindow-titlebar').hover()
              }
              const el = scope.getByTestId(tid).first()
              if (d.action === 'hover') await el.hover()
              else await el.click()
            }
          }
          // --- wait on a SIGNAL before capturing -----------------------------
          const w = d?.waitFor
          const timeout = d?.timeoutMs ?? 60_000
          if (w?.subwindows) await waitForSubwindowCount(page, w.subwindows, timeout)
          if (w?.visible) {
            await page.getByTestId(w.visible).first().waitFor({ state: 'visible', timeout })
          }
          if (w?.pixels) {
            await test.expect.poll(() => countColorPixels(page, w.pixels!), {
              timeout, message: `pixels ${w.pixels} never appeared for "${step.title}"`,
            }).toBeGreaterThan(0)
          }
          if (d?.settleMs) await page.waitForTimeout(d.settleMs)

          // --- screenshot ----------------------------------------------------
          if (!step.image) continue
          const dest = join(outDir, step.image)
          if (d?.shotTarget && d.shotTarget !== 'page') {
            const target = d.shotTarget === 'subwindow'
              ? page.getByTestId('subwindow').filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) }).first()
              : page.getByTestId(d.shotTarget).first()
            await target.screenshot({ path: dest })
          } else {
            await page.screenshot({ path: dest })
          }
          console.log(`[guide:${guide.id}] wrote ${step.image}`)
        }
        ctx.assertNoJsErrors()
      } finally {
        await ctx.app.close()
      }
    })
  }
})
