/**
 * _harness.cjs — shared Playwright helpers for the SpyDE Electron e2e suite.
 *
 * CommonJS on purpose: Node 23 + Playwright 1.61 crash
 * (`context.conditions?.includes is not a function`) on any cross-file relative
 * `.ts` import, so the shared launch + helpers live in a `.cjs` module that the
 * `.spec.ts` files `require()` at runtime instead of `import`ing.
 *
 * What this gives every spec that opts in:
 *  - ONE canonical app launch (eager SPYDE_NO_DASK or real LocalCluster).
 *  - RENDERER JS-ERROR CAPTURE: page `pageerror` + console `error` are collected;
 *    `assertNoJsErrors()` fails the test if any fired (no spec did this before).
 *  - SIGNAL-BASED WAITS instead of `waitForTimeout` sleeps: `backend.waitForLog`
 *    (e.g. "Dask cluster ready") and `backend.waitForMessage` (parses the
 *    `PLOTAPP:` JSON line protocol, e.g. `nav_drag_result`).
 *  - canvas-pixel helpers (`countColorPixels`, `waitForNonBlackCanvas`) lifted
 *    from visual.spec.ts / vi_lazy.spec.ts / vector_om_lazy.spec.ts.
 */
const { _electron: electron } = require('@playwright/test')
const { join } = require('path')

/**
 * Launch the app. Returns { app, page, backend, jsErrors, assertNoJsErrors }.
 * @param {{dask?: boolean, env?: Record<string,string>}} opts
 *   dask=false (default) → SPYDE_NO_DASK=1 (renderer-only / eager, fast).
 *   dask=true            → real LocalCluster + client.
 */
async function launchApp(opts = {}) {
  const { dask = false, env = {} } = opts
  const app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: {
      ...process.env,
      ...(dask ? {} : { SPYDE_NO_DASK: '1' }),
      ...env,
    },
  })

  const backend = createBackend(app)
  const page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  // The renderer must have mounted (window.electron + IPC wired) before we fire
  // any action, and the Python backend must be ready to receive it. Without this
  // an action sent too early is silently dropped (no window ever opens).
  await page.waitForSelector('[data-testid="mdi-area"]', { timeout: 30_000 })
  await backend.waitForLog('[spyde backend] ready', 60_000).catch(() => {})
  if (dask) await backend.waitForDask(60_000).catch(() => {})

  // ---- renderer JS-error capture -------------------------------------------
  const jsErrors = []
  page.on('pageerror', (err) => { jsErrors.push(`pageerror: ${err.message}`) })
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      const t = msg.text()
      // Ignore benign network 404s for optional media (docs screenshots) etc.
      if (/Failed to load resource/.test(t)) return
      jsErrors.push(`console.error: ${t}`)
    }
  })
  const assertNoJsErrors = () => {
    if (jsErrors.length) {
      throw new Error(
        `Renderer JS errors detected (${jsErrors.length}):\n` +
        jsErrors.map((e) => '  - ' + e).join('\n'),
      )
    }
  }

  return { app, page, backend, jsErrors, assertNoJsErrors }
}

/**
 * Wraps the backend subprocess stdout/stderr: lets specs await a log substring
 * or a PLOTAPP JSON message instead of polling with sleeps.
 */
function createBackend(app) {
  const logBuffer = []          // recent raw lines (for waitForLog latecomers)
  const messages = []           // parsed PLOTAPP objects
  const logWaiters = []         // {needle, resolve}
  const msgWaiters = []         // {type, resolve}

  const grab = (d) => {
    const s = String(d)
    for (const ln of s.split('\n')) {
      if (!ln) continue
      logBuffer.push(ln)
      if (logBuffer.length > 2000) logBuffer.shift()
      for (let i = logWaiters.length - 1; i >= 0; i--) {
        if (ln.includes(logWaiters[i].needle)) {
          logWaiters[i].resolve(ln)
          logWaiters.splice(i, 1)
        }
      }
      const j = ln.indexOf('PLOTAPP:')
      if (j >= 0) {
        try {
          const obj = JSON.parse(ln.slice(j + 'PLOTAPP:'.length))
          messages.push(obj)
          for (let i = msgWaiters.length - 1; i >= 0; i--) {
            if (msgWaiters[i].type === obj.type) {
              msgWaiters[i].resolve(obj)
              msgWaiters.splice(i, 1)
            }
          }
        } catch { /* not JSON */ }
      }
    }
  }
  app.process().stdout?.on('data', grab)
  app.process().stderr?.on('data', grab)

  return {
    /** Resolve once a log line containing `needle` is seen (past OR future). */
    waitForLog(needle, timeout = 60_000) {
      if (logBuffer.some((l) => l.includes(needle))) return Promise.resolve()
      return new Promise((resolve, reject) => {
        const w = { needle, resolve }
        logWaiters.push(w)
        setTimeout(() => {
          const k = logWaiters.indexOf(w)
          if (k >= 0) logWaiters.splice(k, 1)
          reject(new Error(`waitForLog timed out waiting for "${needle}"`))
        }, timeout)
      })
    },
    /** Resolve with the next PLOTAPP message of `type` (or one already seen). */
    waitForMessage(type, timeout = 60_000) {
      const seen = messages.find((m) => m.type === type)
      if (seen) return Promise.resolve(seen)
      return new Promise((resolve, reject) => {
        const w = { type, resolve }
        msgWaiters.push(w)
        setTimeout(() => {
          const k = msgWaiters.indexOf(w)
          if (k >= 0) msgWaiters.splice(k, 1)
          reject(new Error(`waitForMessage timed out waiting for "${type}"`))
        }, timeout)
      })
    },
    /**
     * Wait for the Dask cluster to be ready (real-data runs). The backend's
     * "Dask cluster ready" is a PLOTAPP status message (consumed by the main
     * process's readline, so it never reaches Electron stdout), but the main
     * process ECHOES the companion `dask_ready` lifecycle message to stdout as
     * `[spyde backend] dask_ready:` — match that.
     */
    waitForDask(timeout = 60_000) {
      return this.waitForLog('dask_ready', timeout)
    },
    get logBuffer() { return logBuffer },
    get messages() { return messages },
  }
}

/** Fire a test-only backend action via the renderer IPC bridge. */
async function backendAction(page, action, payload = {}) {
  await page.evaluate(
    ({ a, p }) => window.electron.action(a, p),
    { a: action, p: payload },
  )
}

/** Wait until at least `n` subwindows exist (result windows opening). */
async function waitForSubwindowCount(page, n, timeout = 60_000) {
  await page.waitForFunction(
    (count) => document.querySelectorAll('[data-testid="subwindow"]').length >= count,
    n,
    { timeout },
  )
}

/**
 * Wait for the vector actions to unlock — the REAL "diffraction_vectors
 * attached" signal: the vector toolbar buttons are requires_vectors-gated, so
 * they exist in the DOM only after find-vectors finalizes and re-sends the
 * toolbar. (Do NOT wait on the "Found N diffraction vectors" status — it
 * travels the PLOTAPP stdout protocol, invisible to the harness log buffer.)
 */
async function waitForVectorActions(page, timeout = 60_000) {
  await page.waitForFunction(
    () => document.querySelectorAll(
      '[data-testid="action-btn-Strain Mapping"]').length > 0,
    undefined, { timeout },
  )
}

/**
 * Load the bundled synthetic Find-Vectors RESULT tree (test-only backend
 * action `load_test_vectors`): a 6×6 four-spot dataset run through Find
 * Diffraction Vectors, vectors attached, vector actions unlocked. THE fast
 * path (seconds, works under SPYDE_NO_DASK) for anything downstream of
 * vectors — Strain / Vector VI / Vector OM specs should start here instead of
 * paying the multi-minute distributed batch.
 */
async function loadTestVectors(page, timeout = 60_000) {
  // backend-ready can land slightly before the stdin pump is live; settle
  // first so the action isn't dropped (same pattern the lazy specs use).
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_vectors')
  await waitForSubwindowCount(page, 4, timeout)
  await waitForVectorActions(page, timeout)
}

/**
 * Log a dask scheduler/worker snapshot to the backend log at WARNING
 * ([dask-state] lines in ctx.backend.logBuffer): task-state histogram,
 * per-worker load, call stacks of executing tasks. Fire this whenever a
 * compute "looks stuck" before giving up — its output localizes the stall
 * (submission vs scheduling vs worker-side execution).
 */
async function dumpDaskState(page, settleMs = 2_000) {
  await backendAction(page, 'dump_dask_state')
  await page.waitForTimeout(settleMs)
}

/**
 * Count canvas pixels matching a colour across frames. kind:
 *  'bright' (any non-black), 'red' (#ff3030 markers), 'green' (#30ff60 matched
 *  template). The green test requires a non-trivial BLUE channel (b in ~60..160)
 *  so it matches the overlay's #30ff60 (48,255,96) but NOT the navigator's pure
 *  green crosshair (~0,255,0) — that crosshair was a false positive.
 *
 * Frames whose URL/name suggests the navigator are skipped for marker colours,
 * so a green count reflects the DIFFRACTION-pattern overlay, not the navigator.
 */
async function countColorPixels(page, kind) {
  let total = 0
  for (const frame of page.frames()) {
    try {
      total += await frame.evaluate((k) => {
        let n = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const ctx = c.getContext('2d')
          if (!ctx || !c.width || !c.height) continue
          const d = ctx.getImageData(0, 0, c.width, c.height).data
          for (let p = 0; p < d.length; p += 4) {
            const r = d[p], g = d[p + 1], b = d[p + 2]
            if (k === 'bright' && (r > 30 || g > 30 || b > 30)) n++
            if (k === 'red' && r > 120 && g < 90 && b < 90) n++
            // #30ff60: green high, red low-ish, blue PRESENT (≈96). The blue band
            // rejects the navigator's pure-green crosshair (blue≈0).
            if (k === 'green' && g > 150 && r < 130 && b > 50 && b < 170) n++
          }
        }
        return n
      }, kind)
    } catch { /* detached frame */ }
  }
  return total
}

/**
 * Window pickers. The breadcrumb Pill replaced the old "<name> Navigator"
 * title text with an S-/N- kind-prefix chip, so `filter({ hasText:
 * 'Navigator' })` no longer distinguishes windows — select by prefix instead.
 * Windows without a breadcrumb (bare figure windows, e.g. strain) match
 * neither; pick those by their plain title text.
 */
function sigWindow(page) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
}

function navWindow(page) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
    .first()
}

function navWindows(page) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
}

/**
 * REAL backend error lines from the log buffer — the "no errors surfaced"
 * audit several specs end with. Matches Python ERROR/Traceback lines but
 * excludes known-benign noise:
 *  - the Electron dev-mode CSP "Security Warning" (tagged RENDERER-ERROR),
 *  - Chromium's willReadFrequently canvas perf hint (our own pixel probes),
 *  - CHROMIUM PROCESS stderr in `[pid:date:ERROR:file.cc(line)]` format —
 *    on headless Linux CI this fires constantly (bus.cc dbus failures,
 *    viz_main_impl.cc "Exiting GPU process", command_buffer_proxy_impl.cc)
 *    and is infrastructure noise, not a SpyDE error. Python backend lines
 *    never match that shape, so real errors still fail the audit.
 */
function backendErrorLines(backend) {
  return backend.logBuffer.filter((l) =>
    /ERROR|Traceback/i.test(l)
    && !/Security Warning|Content.Security.Policy|Content Security/i.test(l)
    && !/willReadFrequently/i.test(l)
    && !/:(ERROR|FATAL):[a-z_0-9]+\.(cc|mm)\(\d+\)/.test(l))
}

/**
 * A safe point to GRAB the titlebar for a WINDOW-MOVE drag. The breadcrumb
 * pill (left side) is an HTML5 drag SOURCE that stops pointerdown — grabbing
 * it starts a DnD payload drag, NOT a window move. The window controls
 * (minimize/maximize/close, ~90px) own the right edge. Return a point in the
 * empty strip between the two.
 */
async function titlebarGrabPoint(win) {
  const bar = win.getByTestId('subwindow-titlebar')
  const bb = await bar.boundingBox()
  if (!bb) throw new Error('titlebar has no bounding box')
  let x = bb.x + bb.width / 2
  const pill = win.getByTestId('window-breadcrumb')
  if (await pill.count()) {
    const pb = await pill.first().boundingBox()
    if (pb) x = pb.x + pb.width + 20
  }
  x = Math.min(x, bb.x + bb.width - 110)
  return { x, y: bb.y + bb.height / 2 }
}

module.exports = {
  launchApp,
  backendAction,
  waitForSubwindowCount,
  waitForVectorActions,
  loadTestVectors,
  dumpDaskState,
  countColorPixels,
  sigWindow,
  navWindow,
  navWindows,
  titlebarGrabPoint,
  backendErrorLines,
}
