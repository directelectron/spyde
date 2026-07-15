/**
 * envProgress.ts — parse `uv` first-run output into structured setup progress.
 *
 * `uv sync` / `uv pip install` stream human-readable status to stderr. On first
 * packaged launch that stream is the ONLY signal that anything is happening
 * (hundreds of MB of wheels, incl. PyTorch, are being fetched). The renderer
 * turns these events into a floating "Setting up…" overlay with a live phase,
 * a friendly current-step line, an optional download %, and a raw log tail —
 * so first launch never looks frozen.
 *
 * This module is pure string→event, no I/O, so it is unit-testable
 * (test_env_progress.ts) without spawning uv.
 */

export type EnvPhase =
  | 'resolving'    // building the dependency graph
  | 'downloading'  // fetching wheels/sdists
  | 'installing'   // unpacking into the venv
  | 'building'     // building the editable spyde package (setuptools_scm)
  | 'torch'        // the per-machine torch step (the big one)
  | 'working'      // fallback: uv said something we don't specifically classify

export interface EnvProgressEvent {
  phase: EnvPhase
  /** Short human sentence for the overlay's headline step, e.g. "Downloading PyTorch". */
  step: string
  /** 0–100 when uv reports a download percentage for a large artifact; else null. */
  percent: number | null
}

const KNOWN_BIG = /\btorch\b/i

/**
 * Classify ONE raw line of uv output. Returns null for noise (blank lines,
 * lines we can't meaningfully turn into a step) — the caller keeps the last
 * non-null event as the current state and always appends the raw line to the
 * log tail regardless.
 */
export function parseUvLine(raw: string): EnvProgressEvent | null {
  const line = raw.replace(/\r/g, '').trim()
  if (!line) return null

  // Our own [env-setup] breadcrumbs (pythonEnv.ts) — use them verbatim, they're
  // already user-facing phase announcements.
  if (line.startsWith('[env-setup]')) {
    const msg = line.slice('[env-setup]'.length).trim()
    // Order matters: the "two-step install … torch deferred" breadcrumb mentions
    // torch but announces the RESOLVE phase, so match the sync/plan keywords
    // first and only fall to the torch phase for an actual torch-install line.
    if (/two-step|full locked|uv sync|deferred/i.test(msg)) {
      return { phase: 'resolving', step: 'Preparing the analysis environment', percent: null }
    }
    if (/torch/i.test(msg)) {
      return { phase: 'torch', step: 'Installing PyTorch for your GPU', percent: null }
    }
    return { phase: 'working', step: capitalize(msg), percent: null }
  }

  // A download progress line. uv renders these like:
  //   "torch     ======>          123.4 MiB/825.0 MiB"  (bar form), or
  //   "Downloading torch (825.0 MiB)"                    (start), or a trailing
  //   percentage. Pull a percent if the two sizes are present.
  const dl = line.match(/^([\w.\-]+)\s+.*?([\d.]+)\s*([KMG]i?B)\s*\/\s*([\d.]+)\s*([KMG]i?B)/i)
  if (dl) {
    const pkg = dl[1]
    const cur = toBytes(dl[2], dl[3])
    const tot = toBytes(dl[4], dl[5])
    const percent = tot > 0 ? Math.min(100, Math.round((cur / tot) * 100)) : null
    const big = KNOWN_BIG.test(pkg)
    return {
      phase: big ? 'torch' : 'downloading',
      step: big ? 'Downloading PyTorch' : `Downloading ${pkg}`,
      percent,
    }
  }

  // uv's phase verbs (it prints "Resolved N packages", "Downloaded …",
  // "Prepared …", "Installed N packages", "Building …").
  if (/^Resolv(ing|ed)\b/i.test(line)) {
    return { phase: 'resolving', step: 'Resolving dependencies', percent: null }
  }
  if (/^Download(ing|ed)\b/i.test(line)) {
    const m = line.match(/^Download(?:ing|ed)\s+([\w.\-]+)/i)
    const pkg = m?.[1]
    const big = pkg ? KNOWN_BIG.test(pkg) : false
    return {
      phase: big ? 'torch' : 'downloading',
      step: big ? 'Downloading PyTorch' : (pkg ? `Downloading ${pkg}` : 'Downloading packages'),
      percent: null,
    }
  }
  if (/^Prepar(ing|ed)\b/i.test(line)) {
    return { phase: 'installing', step: 'Preparing packages', percent: null }
  }
  if (/^Install(ing|ed)\b/i.test(line)) {
    const m = line.match(/(\d+)\s+packages?/i)
    return {
      phase: 'installing',
      step: m ? `Installing ${m[1]} packages` : 'Installing packages',
      percent: null,
    }
  }
  if (/^Building\b/i.test(line) || /setuptools[_-]scm|Building wheel|Building editable/i.test(line)) {
    return { phase: 'building', step: 'Building the SpyDE package', percent: null }
  }
  if (/^Audit(ing|ed)\b/i.test(line)) {
    return { phase: 'installing', step: 'Finalizing the environment', percent: null }
  }

  // Recognized-but-unclassified: keep the phase we can't infer as generic
  // "working" so the overlay still updates its step text and never looks stuck.
  return { phase: 'working', step: 'Working', percent: null }
}

function toBytes(n: string, unit: string): number {
  const v = parseFloat(n)
  const u = unit.toUpperCase()
  if (u.startsWith('G')) return v * 1024 ** 3
  if (u.startsWith('M')) return v * 1024 ** 2
  if (u.startsWith('K')) return v * 1024
  return v
}

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s
}
