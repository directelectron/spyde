/**
 * global-teardown.cjs — remove the run-scoped scratch root.
 *
 * `_clean_slate.cjs` mints a fresh Electron profile + settings dir per launch,
 * which is ~100 directories over a full suite run. On CI the runner is thrown
 * away so it would not matter, but on a dev box those would accumulate in the
 * system temp run after run. global-setup puts them all under one root
 * (SPYDE_E2E_TMP_ROOT) precisely so this can be one delete.
 *
 * Best-effort: a profile still held open by an Electron process that outlived
 * its spec would make this throw on Windows, and failing the whole run at
 * teardown over a temp directory would be worse than leaving it behind.
 */
const { rmSync } = require('fs')

module.exports = async () => {
  const root = process.env.SPYDE_E2E_TMP_ROOT
  if (!root) return
  try {
    rmSync(root, { recursive: true, force: true, maxRetries: 3 })
  } catch (e) {
    console.warn(`[e2e] could not remove scratch root ${root}: ${e.message}`)
  }
}
