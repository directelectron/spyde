import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  // One worker: each spec launches a real Electron app with a Python sidecar,
  // and several at once contend for CPU badly enough to look like app bugs.
  workers: 1,
  fullyParallel: false,
  timeout: 120_000,
  reporter: 'line',
  projects: [{ name: 'autopilot' }],
})
