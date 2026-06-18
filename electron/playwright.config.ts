import { defineConfig } from '@playwright/test'
import { join } from 'path'

export default defineConfig({
  testDir: './tests',
  timeout: 120_000,
  expect: { timeout: 15_000 },
  retries: 1,
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
  },

  projects: [
    {
      name: 'electron',
      testMatch: '**/*.spec.ts',
    },
  ],
})
