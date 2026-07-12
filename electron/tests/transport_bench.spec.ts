/**
 * transport_bench.spec.ts — measure the per-frame `transport` stage of a
 * navigator scrub on a big movie, with the anyplotlib BINARY pixel transport
 * active. `transport` is the anyplotlib set_data push (the paint half of a nav
 * update). The raw-bytes side-channel change means set_data no longer base64-
 * encodes the frame (and _electron no longer decodes it back) — so on a 4k
 * movie the transport stage should drop from ~75-160 ms to a small fraction.
 *
 * Parses the [PAINT-PROFILE] lines the backend logs at INFO (teed to stderr via
 * SPYDE_LOG_LEVEL, captured in ctx.backend.logBuffer) and reports the median
 * transport ms across the scrub.
 */
import { test } from '@playwright/test'
import { existsSync } from 'fs'
const { launchApp, backendAction } = require('./_harness.cjs')

const MOVIES = [
  'C:/Users/CarterFrancis/Downloads/20251117_88075_run3 some growth_1236_movie.mrc',
  'C:/Users/CarterFrancis/Downloads/20251117_88074_run1_9104_movie.mrc',
]
const movie = () => MOVIES.find((p) => existsSync(p)) || null

function medianTransport(log: string): { n: number; median: number; max: number; frames: number[] } {
  const re = /\[PAINT-PROFILE\][^\n]*?transport=([\d.]+)/g
  const xs: number[] = []
  let m: RegExpExecArray | null
  while ((m = re.exec(log))) xs.push(parseFloat(m[1]))
  if (!xs.length) return { n: 0, median: -1, max: -1, frames: [] }
  const sorted = [...xs].sort((a, b) => a - b)
  return {
    n: xs.length,
    median: +sorted[sorted.length >> 1].toFixed(1),
    max: +sorted[sorted.length - 1].toFixed(1),
    frames: xs.map((v) => +v.toFixed(1)),
  }
}

async function runMode(binary: string): Promise<any> {
  const path = movie()
  const { app, page, backend } = await launchApp({
    dask: true,
    env: {
      APL_BINARY_TRANSPORT: binary,   // '1' fast path vs '0' old base64
      SPYDE_NAV_PROFILE: '1',         // emit [PAINT-PROFILE] lines
      SPYDE_LOG_LEVEL: 'INFO',        // tee logging to stderr (harness captures)
    },
  })
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'open_file', { path })
    await page.waitForFunction(
      () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
      { timeout: 180_000 })
    await page.waitForFunction(
      () => !/Reading .*\.mrc/i.test(document.body.textContent || ''),
      { timeout: 300_000 })

    // Scrub the navigator across a handful of distinct frames. One batched
    // multi-target drag paints each frame in turn (the harness dwells per
    // target) — fast enough to stay well under the timeout while still giving
    // ~10 clean per-frame transport samples.
    const targets: number[][] = []
    for (let i = 0; i < 12; i++) targets.push([i * 12, 0])
    await backendAction(page, 'test_nav_drag', { targets })
    await page.waitForTimeout(3000)

    const res = medianTransport(backend.logBuffer)
    // Emit a few raw PAINT-PROFILE lines so we can see the stage breakdown.
    const raw = (backend.logBuffer.match(/\[PAINT-PROFILE\][^\n]*/g) || []).slice(-6)
    return { binary, res, raw }
  } finally {
    await app.close()
  }
}

test('transport bench binary side-channel (=1)', async () => {
  test.setTimeout(420_000)
  test.skip(!movie(), 'no movie')
  const out = await runMode('1')
  console.log('TRANSPORT[binary=1]:', JSON.stringify(out.res))
  for (const l of out.raw) console.log('  ', l.replace(/^.*\[PAINT-PROFILE\]/, '[PAINT-PROFILE]'))
})

test('transport bench base64 baseline (=0)', async () => {
  test.setTimeout(420_000)
  test.skip(!movie(), 'no movie')
  const out = await runMode('0')
  console.log('TRANSPORT[binary=0]:', JSON.stringify(out.res))
  for (const l of out.raw) console.log('  ', l.replace(/^.*\[PAINT-PROFILE\]/, '[PAINT-PROFILE]'))
})
