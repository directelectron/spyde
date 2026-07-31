import { defineConfig } from '@playwright/test'
import { join } from 'path'

/**
 * The specs that dominate wall-clock, newest measurement first.
 *
 * Playwright's `--shard` balances by FILE COUNT, not duration, and this suite's
 * files differ by ~40x. Measured on one CI run: shard 3 ran 127 tests in 6.8
 * min while shard 1 ran 90 in 22.6 — and re-sharding never fixed it, because
 * count-balancing keeps making the same bad split (the workflow records ×2 and
 * ×3 both landing one shard over 20 min, one of them cancelled at the timeout).
 *
 * Splitting THESE out and sharding the rest evens it up without another runner:
 * 22.6 / 11.7 / 6.8 / 5.8  ->  roughly 13 / 12 / 11 / 10.
 *
 * `reportSlowTests` above prints every file over 30 s, so CI itself tells you
 * when this list has drifted. Keep it to files that genuinely dominate — a long
 * list defeats the point by leaving the sharded half too thin.
 */
export const SLOW_SPECS = [
  '**/fit_quality.spec.ts',          // 3.7m
  '**/gpu_image_parity.spec.ts',     // 1.5m
  '**/ipf_perf.spec.ts',             // 1.3m
  '**/action_scoping.spec.ts',       // 1.1m
  '**/fit_handles.spec.ts',          // 57s
  '**/fit_from_composition.spec.ts', // 53s
]

// Which half of the split to run. CI sets it per job; unset (every local run)
// means "everything", so nobody has to know this exists to run the suite.
//   slow  -> ONLY the files above, one job, unsharded
//   fast  -> everything EXCEPT them, sharded --shard=N/4
const SLICE = process.env.SPYDE_E2E_SLICE
const SLICE_IGNORE = SLICE === 'fast' ? SLOW_SPECS : []
const SLICE_MATCH = SLICE === 'slow' ? SLOW_SPECS : null

export default defineConfig({
  testDir: './tests',
  // Settings isolation for EVERY spec — ~30 of them call _electron.launch()
  // directly rather than going through _harness.cjs, and on a fresh CI runner
  // they get the first-run welcome tour whose overlay eats pointer events.
  // They all spread ...process.env, so setting it there is what reaches them.
  globalSetup: require.resolve('./tests/global-setup.cjs'),
  timeout: 120_000,
  expect: { timeout: 15_000 },
  retries: 1,
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
      testIgnore: [
        '**/*.real.spec.ts', '**/guide_screenshots.spec.ts',
        ...SLICE_IGNORE,
      ],
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
