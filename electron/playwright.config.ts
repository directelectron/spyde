import { defineConfig } from '@playwright/test'
import { readdirSync } from 'fs'
import { join } from 'path'

// A FRESH Electron profile + settings dir for every app launch, so no spec can
// inherit another's localStorage/IndexedDB or settings.json. This file is
// evaluated in every worker process, which is what lets one hook reach all ~60
// specs — including the ~30 that call `_electron.launch()` directly instead of
// going through _harness.cjs. See tests/_clean_slate.cjs for the full why.
// eslint-disable-next-line @typescript-eslint/no-var-requires
require('./tests/_clean_slate.cjs').installCleanSlate()

/**
 * Every spec file that costs >= ~45 s on a hosted runner, SLOWEST FIRST — the
 * order IS the CI balancing (round-robin below). Playwright's `--shard`
 * balances by FILE COUNT, not duration, and reliably parks one shard at 2-3x
 * the others, so CI does not use `--shard` at all.
 *
 * Durations: median of the two most recent green main runs (2026-08-03).
 * `reportSlowTests` prints every file over 30 s, so CI says when this list
 * has drifted.
 */
export const SLOW_SPECS = [
  '**/progressive_signal_preview.spec.ts',      // 194s
  '**/progressive_orientation_preview.spec.ts', // 154s
  '**/ipf_perf.spec.ts',                        // 149s
  '**/report_edit2.spec.ts',                    // 111s
  '**/sped_ag_grid.spec.ts',                    // 105s
  '**/insitu_playback.spec.ts',                 // 90s
  '**/gpu_image_parity.spec.ts',                // 90s
  '**/ipf_two_window_vom.spec.ts',              // 90s
  '**/eels_edge_component.spec.ts',             // 84s
  '**/ipf_two_window.spec.ts',                  // 81s
  '**/laundry_visual.spec.ts',                  // 80s
  '**/vector_om_lazy.spec.ts',                  // 70s
  '**/orientation_lazy.spec.ts',                // 70s
  '**/ebsd_workflow.spec.ts',                   // 66s
  '**/action_scoping.spec.ts',                  // 65s
  '**/compose_real_drag.spec.ts',               // 61s
  '**/om_wizard_lazy.spec.ts',                  // 60s
  '**/nav_drag_distributed.spec.ts',            // 58s
  '**/actions_lifecycle.spec.ts',               // 57s
  '**/fit_handles.spec.ts',                     // 57s
  '**/shutdown.spec.ts',                        // 56s
  '**/grid_present.spec.ts',                    // 56s
  '**/vector_overlay.spec.ts',                  // 54s
  '**/find_vectors_result.spec.ts',             // 51s
  '**/vector_vi_lazy.spec.ts',                  // 51s
  '**/fit_from_composition.spec.ts',            // 50s
  '**/ipf_refine_render.spec.ts',               // 49s
  '**/vi_lazy.spec.ts',                         // 47s
]

// Which slice to run. CI sets both; unset (every local run) means
// "everything", so nobody has to know this exists to run the suite.
//   SPYDE_E2E_SLICE:  slow -> ONLY the files above;  fast -> everything else
//   SPYDE_E2E_GROUP:  'k/n' -> every n-th file of the slice, offset k (1-based)
// Round-robin over the slice's file order spreads duration evenly: SLOW_SPECS
// is sorted slowest-first, and the fast slice is all sub-45s files, so any
// contiguous alphabetical cluster of heavy files lands on different jobs.
const bare = (glob: string) => glob.slice(glob.lastIndexOf('/') + 1)
const FAST_SPECS = readdirSync(join(__dirname, 'tests'))
  .filter((f) => f.endsWith('.spec.ts'))
  .filter((f) => !f.endsWith('.real.spec.ts') && f !== 'guide_screenshots.spec.ts')
  .filter((f) => !SLOW_SPECS.some((g) => bare(g) === f))
  .sort()
const SLICE = process.env.SPYDE_E2E_SLICE
let sliceFiles = SLICE === 'slow' ? SLOW_SPECS.map(bare) : SLICE === 'fast' ? FAST_SPECS : null
const GROUP = process.env.SPYDE_E2E_GROUP
if (sliceFiles && GROUP) {
  const [k, n] = GROUP.split('/').map(Number)
  sliceFiles = sliceFiles.filter((_, i) => i % n === k - 1)
}
const SLICE_MATCH = sliceFiles ? sliceFiles.map((f) => '**/' + f) : null

export default defineConfig({
  testDir: './tests',
  // Settings isolation for EVERY spec — ~30 of them call _electron.launch()
  // directly rather than going through _harness.cjs, and on a fresh CI runner
  // they get the first-run welcome tour whose overlay eats pointer events.
  // They all spread ...process.env, so setting it there is what reaches them.
  globalSetup: require.resolve('./tests/global-setup.cjs'),
  globalTeardown: require.resolve('./tests/global-teardown.cjs'),
  timeout: 120_000,
  expect: { timeout: 15_000 },
  // CI: 0 — a retry of a timed-out app boot doubles the damage (a 10-min hang
  // became 20). Locally: 1, as before.
  retries: process.env.CI ? 0 : 1,
  // line: one timestamped row per test WITH its duration (the dot reporter made
  // CI stalls unattributable — 9 silent minutes with no test name). html: the CI
  // workflow uploads playwright-report/ as an artifact; without an html reporter
  // that folder never exists and the upload is silently empty.
  reporter: [['line'], ['html', { open: 'never' }]],
  // Every spec file boots its own Electron + Python backend (~20s on a hosted
  // runner), so file-level durations are THE optimization target — list all
  // files slower than 30s in the summary, not just the default top 5.
  reportSlowTests: { max: 0, threshold: 30_000 },
  // Several specs launch a REAL Electron app + Dask LocalCluster. Running them in
  // parallel made the cluster-ready handshake contend → intermittent flakiness
  // (om_wizard_lazy / vector_om_lazy / vector_vi_lazy / vi_lazy). Serialise the
  // whole suite: slower but deterministic (one Electron+cluster at a time).
  workers: 1,
  fullyParallel: false,

  use: {
    // _electronPath is resolved at test setup time in the fixture
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    {
      // CI / default tier: synthetic + bundled data only, no network. Excludes
      // the real-data specs (*.real.spec.ts) and the screenshot generator, which
      // need downloaded pyxem datasets.
      name: 'electron',
      testMatch: SLICE_MATCH ?? '**/*.spec.ts',
      testIgnore: ['**/*.real.spec.ts', '**/guide_screenshots.spec.ts'],
    },
    {
      // Local / nightly tier: real pyxem datasets + per-step screenshot
      // generation. Opt-in — run with `SPYDE_E2E_REAL=1 npx playwright test
      // --project=electron-real`. Longer per-test budget (downloads + heavy
      // compute) and one retry stripped (real runs are expensive).
      name: 'electron-real',
      testMatch: ['**/*.real.spec.ts', '**/guide_screenshots.spec.ts'],
      timeout: 600_000,
      retries: 0,
    },
  ],
})
