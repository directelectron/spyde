import { defineConfig } from '@playwright/test'
import { join } from 'path'

export default defineConfig({
  testDir: './tests',
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
      testMatch: '**/*.spec.ts',
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
