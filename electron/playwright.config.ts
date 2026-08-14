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
 * Every spec file that costs >= ~45 s on a hosted runner, with its CI SLOT
 * cost in seconds (in-file test time + ~35 s app boot/teardown). The weights
 * are BALANCING DATA, not contracts: the slow slice is bin-packed by them
 * (greedy longest-first below), so editing a weight rebalances the groups.
 * Playwright's `--shard` balances by FILE COUNT, not duration, and reliably
 * parks one shard at 2-3x the others, so CI does not use `--shard` at all.
 *
 * Weights: measured slot times from the PR-tier validation run (2026-08-07);
 * progressive_signal_preview's slot ran 2.3x its in-file time (a slow batch
 * finalize on CI runners), so slots, not file times, are what balance.
 * `reportSlowTests` prints every file over 30 s, so CI says when this list
 * has drifted.
 */
export const SLOW_SPECS: Array<[string, number]> = [
  ['**/progressive_signal_preview.spec.ts', 400],
  ['**/gpu_image_parity.spec.ts', 125],
  ['**/progressive_orientation_preview.spec.ts', 113],
  ['**/ipf_perf.spec.ts', 113],
  ['**/report_edit2.spec.ts', 101],
  ['**/ipf_two_window.spec.ts', 101],
  ['**/action_scoping.spec.ts', 101],
  ['**/fit_handles.spec.ts', 95],
  ['**/shutdown.spec.ts', 93],
  ['**/fit_from_composition.spec.ts', 87],
  ['**/eels_edge_component.spec.ts', 81],
  ['**/compose_real_drag.spec.ts', 81],
  ['**/laundry_visual.spec.ts', 76],
  ['**/grid_present.spec.ts', 75],
  ['**/ebsd_workflow.spec.ts', 73],
  ['**/ipf_two_window_vom.spec.ts', 71],
  ['**/sped_ag_grid.spec.ts', 69],
  ['**/orientation_lazy.spec.ts', 69],
  ['**/actions_lifecycle.spec.ts', 58],
  ['**/insitu_playback.spec.ts', 55],
  ['**/vector_om_lazy.spec.ts', 55],
  ['**/nav_drag_distributed.spec.ts', 55],
  ['**/find_vectors_result.spec.ts', 55],
  ['**/om_wizard_lazy.spec.ts', 55],
  ['**/vector_overlay.spec.ts', 55],
  ['**/vi_lazy.spec.ts', 55],
  ['**/ipf_refine_render.spec.ts', 55],
  ['**/vector_vi_lazy.spec.ts', 50],
]
const SLOW_GLOBS = SLOW_SPECS.map(([g]) => g)

// Which slice to run. CI sets both; unset (every local run) means
// "everything", so nobody has to know this exists to run the suite.
//   SPYDE_E2E_SLICE:  slow -> ONLY the files above;  fast -> everything else
//   SPYDE_E2E_GROUP:  'k/n' -> group k of n (1-based)
// fast: all files are sub-45s, so round-robin over the sorted list is even.
// slow: greedy longest-first bin-packing on the declared weights — round-robin
// cannot isolate the one ~7-min spec that must ride alone in its group, this
// can, and it rebalances itself whenever a weight is edited. Deterministic
// (stable sort, ties by list order), so every CI job computes the same bins.
const bare = (glob: string) => glob.slice(glob.lastIndexOf('/') + 1)
// Recursive: a spec added in a SUBDIRECTORY must land in a slice too, not
// silently fall out of CI. '**/'+bare matches at any depth.
const ALL_SPEC_FILES = readdirSync(join(__dirname, 'tests'), { recursive: true })
  .map((f) => bare(String(f).replace(/\\/g, '/')))
  .filter((f) => f.endsWith('.spec.ts'))
  .sort()
for (const g of SLOW_GLOBS) {
  // A renamed/deleted slow spec must fail loudly here, not silently carry a
  // stale glob while the file drifts into the fast slice.
  if (!ALL_SPEC_FILES.includes(bare(g)))
    throw new Error(`SLOW_SPECS entry has no file under tests/: ${g} — update the list`)
}
const FAST_SPECS = ALL_SPEC_FILES
  .filter((f) => !f.endsWith('.real.spec.ts') && f !== 'guide_screenshots.spec.ts')
  .filter((f) => !SLOW_GLOBS.some((g) => bare(g) === f))
const SLICE = process.env.SPYDE_E2E_SLICE
const GROUP = process.env.SPYDE_E2E_GROUP
let sliceFiles: string[] | null = null
if (SLICE === 'fast') {
  sliceFiles = FAST_SPECS
  if (GROUP) {
    const [k, n] = GROUP.split('/').map(Number)
    sliceFiles = sliceFiles.filter((_, i) => i % n === k - 1)
  }
} else if (SLICE === 'slow') {
  sliceFiles = SLOW_GLOBS.map(bare)
  if (GROUP) {
    const [k, n] = GROUP.split('/').map(Number)
    const bins = Array.from({ length: n }, () => ({ total: 0, files: [] as string[] }))
    for (const [g, w] of [...SLOW_SPECS].sort((a, b) => b[1] - a[1])) {
      const lightest = bins.reduce((m, b) => (b.total < m.total ? b : m))
      lightest.total += w
      lightest.files.push(bare(g))
    }
    sliceFiles = bins[k - 1].files
  }
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
  // 1 everywhere. CI was 0 to stop a timed-out app boot doubling its damage
  // (a 10-min hang became 20), but that predates the 120 s per-test timeout,
  // which now caps a retried hang at ~2 extra minutes. What CI actually sees
  // today is a singleton timing flake per run somewhere in ~380 specs
  // (caret-restore poll, log-row count, a pixel-ratio probe — a different
  // spec each run), and one retry absorbs those without hiding them: a
  // retried pass is reported as "flaky", not green.
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
