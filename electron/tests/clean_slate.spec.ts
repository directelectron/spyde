/**
 * clean_slate.spec.ts — every app launch starts with NO state from the last one.
 *
 * This is the contract `_clean_slate.cjs` exists to provide, and it is the kind
 * of thing that rots silently: nothing else in the suite would notice if the
 * launch patch stopped installing, and the symptom would not be "isolation
 * broke" — it would be some unrelated spec failing only in a particular shard
 * order, which is exactly the failure that is miserable to trace.
 *
 * Both launch styles are covered on purpose. `_harness.cjs` already minted a
 * fresh settings dir per launch, but the ~30 specs that call
 * `_electron.launch()` directly did not, and NOTHING isolated Electron's own
 * profile (localStorage / IndexedDB) for either.
 */
import { test, expect } from '@playwright/test'
const { _electron: electron } = require('@playwright/test')
const { join } = require('path')
const { launchApp } = require('./_harness.cjs')

const MAIN = join(__dirname, '..', 'out', 'main', 'index.js')
const KEY = '__clean_slate_probe__'

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

/** Read the settings dir the backend was actually told to use. */
function settingsDirOf(app: any) {
  return app.evaluate(async () => process.env.SPYDE_SETTINGS_DIR)
}

/** localStorage is synchronous to JS but persisted to disk ASYNCHRONOUSLY, so
 *  closing straight after the write can simply lose it — and a leak test whose
 *  value never reached disk passes for the wrong reason. Give Chromium a beat
 *  to flush before the app goes away. (Measured: without this, the harness case
 *  passed even with isolation disabled.) */
async function writeAndFlush(page: any) {
  await page.evaluate((k: string) => localStorage.setItem(k, 'stale'), KEY)
  expect(await page.evaluate((k: string) => localStorage.getItem(k), KEY)).toBe('stale')
  await page.waitForTimeout(1500)
}

test('localStorage does not survive into the next launch (harness launch)', async () => {
  const a = await launchApp({})
  await writeAndFlush(a.page)
  await a.app.close()

  const b = await launchApp({})
  const leaked = await b.page.evaluate((k: string) => localStorage.getItem(k), KEY)
  await b.app.close()
  expect(leaked, 'localStorage leaked from the previous launch — the Electron ' +
    'profile is being shared between specs').toBeNull()
})

test('localStorage does not survive into the next launch (direct _electron.launch)', async () => {
  // Deliberately the raw call the ~30 non-harness specs make, spreading
  // process.env exactly as they do.
  const launch = () => electron.launch({ args: [MAIN], env: { ...process.env, SPYDE_NO_DASK: '1' } })

  const a = await launch()
  const pa = await a.firstWindow()
  await writeAndFlush(pa)
  await a.close()

  const b = await launch()
  const pb = await b.firstWindow()
  const leaked = await pb.evaluate((k: string) => localStorage.getItem(k), KEY)
  await b.close()
  expect(leaked, 'a direct _electron.launch() reused the previous profile').toBeNull()
})

test('each launch gets its own settings dir, and an explicit one is honoured', async () => {
  const a = await electron.launch({ args: [MAIN], env: { ...process.env, SPYDE_NO_DASK: '1' } })
  const da = await settingsDirOf(a)
  await a.close()

  const b = await electron.launch({ args: [MAIN], env: { ...process.env, SPYDE_NO_DASK: '1' } })
  const db = await settingsDirOf(b)
  await b.close()

  expect(da, 'no settings dir was set at all').toBeTruthy()
  expect(db).not.toBe(da)

  // …and a spec that brings its own keeps it. first_run.spec.ts depends on
  // this: it launches twice with one dir to test that a setting PERSISTS.
  const mine = da as string
  const c = await electron.launch({
    args: [MAIN], env: { ...process.env, SPYDE_NO_DASK: '1', SPYDE_SETTINGS_DIR: mine },
  })
  const dc = await settingsDirOf(c)
  await c.close()
  expect(dc, 'an explicitly-passed SPYDE_SETTINGS_DIR was overridden').toBe(mine)
})
