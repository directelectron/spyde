/**
 * GpuHelpDialog.tsx — Help -> GPU & CUDA
 *
 * Static, factual help content about SpyDE's GPU acceleration (what ships, what
 * hardware/driver it needs, CPU fallback, the large first-run torch download)
 * PLUS a live "triage" section: it probes the machine (nvidia-smi via the main
 * process) and the installed torch (the backend's get_gpu_status), states a
 * plain-language verdict, and — in a packaged install — offers "Fix PyTorch
 * install", which re-installs the locked torch release with
 * `--torch-backend=auto` so the build matches THIS machine's GPU/driver.
 *
 * Distinct from GpuStatusDialog (Help -> GPU Status…), which shows the LIVE
 * detected device; this explains the setup + can repair it.
 */
import React, { useEffect, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'

interface TriageProbe {
  nvidia: { name: string; driver: string } | null
  managedEnv: boolean
  envExists: boolean
  lockedTorch: string | null
  busy: boolean
}

interface GpuStatusResult {
  torch_available: boolean
  torch_version: string | null
  device: string | null
  gpu_available: boolean
  reason: string
}

interface Verdict {
  tone: 'good' | 'warn' | 'info'
  text: string
  canFix: boolean
}

// Combine the machine probe (nvidia-smi) with the torch-side facts (backend
// get_gpu_status) into one plain-language verdict.
function computeVerdict(triage: TriageProbe, status: GpuStatusResult): Verdict {
  if (status.gpu_available) {
    return {
      tone: 'good',
      text: `Nothing to do — PyTorch is already using your GPU (${status.device ?? 'accelerated device'}).`,
      canFix: false,
    }
  }
  if (triage.nvidia) {
    const { name, driver } = triage.nvidia
    const driverNum = parseFloat(driver)
    if (!status.torch_available) {
      return {
        tone: 'warn',
        text: `An NVIDIA GPU was found (${name}, driver ${driver}) but PyTorch is not installed in the analysis environment. Fix will install the build matching this machine.`,
        canFix: true,
      }
    }
    if (Number.isFinite(driverNum) && driverNum < 550) {
      return {
        tone: 'warn',
        text: `An NVIDIA GPU was found (${name}) but driver ${driver} is older than the ~550 the CUDA 12.4 runtime needs. Update the NVIDIA driver first, then run Fix.`,
        canFix: true,
      }
    }
    return {
      tone: 'warn',
      text: `An NVIDIA GPU was found (${name}, driver ${driver}) but the installed PyTorch (${status.torch_version ?? 'unknown version'}) cannot use it — most likely a CPU-only build. Fix will install the matching CUDA build.`,
      canFix: true,
    }
  }
  return {
    tone: 'info',
    text: 'No NVIDIA GPU detected — the CPU build of PyTorch is recommended (a much smaller download). Fix will switch to it if the current install is the larger CUDA build.',
    canFix: true,
  }
}

type FixState =
  | { phase: 'idle' }
  | { phase: 'running' }
  | { phase: 'done' }
  | { phase: 'error'; error: string }

export function GpuHelpDialog({ onClose }: { onClose: () => void }) {
  const { openGpuStatusDialog, sendAction } = useSpyDE()

  // Triage: probe on open. Machine facts from the main process (gpu:triage),
  // torch facts from the backend (get_gpu_status → the gpu_status_result DOM
  // CustomEvent SpyDEContext re-broadcasts — same wiring as GpuStatusDialog).
  const [triage, setTriage] = useState<TriageProbe | null>(null)
  const [status, setStatus] = useState<GpuStatusResult | null>(null)
  const [fix, setFix] = useState<FixState>({ phase: 'idle' })

  useEffect(() => {
    let alive = true
    const onResult = (e: Event) => setStatus((e as CustomEvent).detail as GpuStatusResult)
    window.addEventListener('spyde:gpu_status_result', onResult)
    sendAction('get_gpu_status')
    window.electron.gpuTriage?.()
      .then((t) => { if (alive) setTriage(t) })
      .catch(() => { if (alive) setTriage(null) })
    return () => {
      alive = false
      window.removeEventListener('spyde:gpu_status_result', onResult)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const verdict = triage && status ? computeVerdict(triage, status) : null
  // Why the Fix button is disabled (tooltip); null = enabled.
  const fixDisabledReason = !triage
    ? 'Still probing…'
    : !triage.managedEnv
      ? 'SpyDE is running from a development environment — PyTorch is managed by uv in the repo, not by the app.'
      : !triage.envExists
        ? 'The Python environment has not been created yet — restart SpyDE to run first-time setup.'
        : triage.busy
          ? 'Environment setup or another fix is already running.'
          : fix.phase === 'running'
            ? 'Fix is running…'
            : null

  const onFix = async () => {
    if (fixDisabledReason || fix.phase === 'running') return
    setFix({ phase: 'running' })
    try {
      const r = await window.electron.gpuFixTorch()
      setFix(r.ok ? { phase: 'done' } : { phase: 'error', error: r.error ?? 'unknown error' })
    } catch (err) {
      setFix({ phase: 'error', error: (err as Error)?.message ?? String(err) })
    }
  }

  return (
    <div style={styles.overlay} data-testid="gpu-help-dialog" onClick={onClose}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>GPU &amp; CUDA</h3>

        <div style={styles.body}>
          <p style={styles.p}>
            SpyDE&apos;s heavy compute (diffraction-vector finding and orientation
            mapping) is GPU-accelerated. At install time PyTorch is resolved for
            your machine: a CUDA build when a suitable NVIDIA GPU is found, the
            much smaller CPU build otherwise.
          </p>

          <Section title="Triage: is PyTorch matched to this machine?">
            <div style={styles.triageBox} data-testid="gpu-triage-section">
              {!verdict ? (
                <p style={{ ...styles.p, margin: 0, color: '#a6adc8' }} data-testid="gpu-triage-verdict">
                  Running checks…
                </p>
              ) : (
                <p
                  style={{ ...styles.p, margin: 0, color: TONE_COLOR[verdict.tone] }}
                  data-testid="gpu-triage-verdict"
                >
                  {verdict.text}
                  {triage && !triage.managedEnv ? (
                    <span style={{ color: '#a6adc8' }}>
                      {' '}(Development run — report only.)
                    </span>
                  ) : null}
                </p>
              )}
              {verdict?.canFix && (
                <div style={{ marginTop: 8 }}>
                  <button
                    data-testid="gpu-triage-fix"
                    style={{
                      ...styles.fixBtn,
                      opacity: fixDisabledReason ? 0.45 : 1,
                      cursor: fixDisabledReason ? 'default' : 'pointer',
                    }}
                    disabled={Boolean(fixDisabledReason)}
                    title={fixDisabledReason ??
                      `Re-install PyTorch${triage?.lockedTorch ? ` ${triage.lockedTorch}` : ''} with the build resolved for this machine (--torch-backend=auto)`}
                    onClick={onFix}
                  >
                    {fix.phase === 'running' ? 'Fixing… (this can take a while)' : 'Fix PyTorch install'}
                  </button>
                  {fix.phase === 'running' && (
                    <p style={{ ...styles.p, marginTop: 6, color: '#a6adc8' }}>
                      Downloading and installing — progress streams into the Log
                      panel&apos;s <b>Raw output</b> view.
                    </p>
                  )}
                  {fix.phase === 'done' && (
                    <p style={{ ...styles.p, marginTop: 6, color: '#a6e3a1' }} data-testid="gpu-triage-fix-done">
                      Done. Restart SpyDE to use the new PyTorch.
                    </p>
                  )}
                  {fix.phase === 'error' && (
                    <p style={{ ...styles.p, marginTop: 6, color: '#f38ba8' }} data-testid="gpu-triage-fix-error">
                      Fix failed: {fix.error}. See the Log panel&apos;s Raw output
                      view for the full install log.
                    </p>
                  )}
                </div>
              )}
            </div>
          </Section>

          <Section title="What you need for GPU acceleration">
            <ul style={styles.ul}>
              <li>An <b>NVIDIA GPU</b> with a reasonably current driver. The
                CUDA-12.4 runtime wheels need an NVIDIA driver of roughly
                <b> version 550 or newer</b>. If the app runs but reports no GPU,
                a driver update is the most common fix.</li>
              <li>GPUs from roughly the last decade are supported — <b>Maxwell /
                Pascal and newer</b> (e.g. GTX 900/1000 series, RTX, and their
                data-center equivalents).</li>
              <li><b>No separate CUDA Toolkit install is needed.</b> The PyTorch
                wheels bundle the CUDA runtime they require, so you do not install
                CUDA yourself.</li>
              <li>On Apple Silicon Macs, SpyDE uses the Metal (MPS) backend
                instead; on Linux and machines without an NVIDIA GPU, it runs on
                CPU.</li>
            </ul>
          </Section>

          <Section title="No compatible GPU? Everything still works">
            <p style={styles.p}>
              A GPU is an accelerator, not a requirement. Every GPU code path has
              a CPU fallback, so SpyDE runs fully without one — the accelerated
              steps are simply slower. You will not lose any functionality.
            </p>
          </Section>

          <Section title="Check what SpyDE detected">
            <p style={styles.p}>
              Open <b>Help → GPU Status…</b> to see whether acceleration is
              active, which device was selected, the installed torch version, and
              — if it fell back to CPU — the exact reason.
            </p>
            <button
              data-testid="gpu-help-open-status"
              style={styles.linkBtn}
              onClick={() => { onClose(); openGpuStatusDialog() }}
            >
              Open GPU Status…
            </button>
          </Section>

          <Section title="First launch: a large download">
            <p style={styles.p}>
              On first launch SpyDE builds its Python environment and downloads
              PyTorch. The CUDA torch package is large (roughly <b>2.4 GiB</b>),
              so the very first start needs a good internet connection and some
              patience — later launches reuse the installed environment and start
              quickly. (On machines without an NVIDIA GPU the CPU build is
              fetched instead, which is far smaller.)
            </p>
            <p style={styles.p}>
              The environment is installed under your user-data folder. If you
              ever need to reset it, deleting that folder&apos;s
              <code style={styles.code}>python-env</code> directory forces a clean
              re-install on the next launch.
            </p>
          </Section>
        </div>

        <div style={styles.footer}>
          <button data-testid="gpu-help-close" style={styles.close} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={styles.section}>
      <div style={styles.sectionTitle}>{title}</div>
      {children}
    </div>
  )
}

const TONE_COLOR: Record<Verdict['tone'], string> = {
  good: '#a6e3a1',
  warn: '#f9e2af',
  info: '#cdd6f4',
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 9500,
    background: 'rgba(17,17,27,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: 24,
  },
  dialog: {
    width: 480, maxWidth: '100%', maxHeight: '90%',
    display: 'flex', flexDirection: 'column',
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 10,
    padding: 18, color: '#cdd6f4', boxShadow: '0 16px 40px rgba(0,0,0,0.55)',
    fontSize: 13,
  },
  title: { margin: '0 0 12px', fontSize: 16, fontWeight: 600, flexShrink: 0 },
  body: {
    flex: 1, minHeight: 0, overflowY: 'auto',
    paddingRight: 4,
  },
  section: { marginBottom: 14 },
  sectionTitle: {
    fontSize: 12.5, fontWeight: 600, color: '#89b4fa', marginBottom: 5,
  },
  triageBox: {
    background: '#11111b', border: '1px solid #313244', borderRadius: 6,
    padding: '10px 12px',
  },
  p: { margin: '0 0 8px', fontSize: 12.5, lineHeight: 1.55, color: '#cdd6f4' },
  ul: {
    margin: '0 0 4px', paddingLeft: 18, fontSize: 12.5, lineHeight: 1.55,
    display: 'flex', flexDirection: 'column', gap: 5,
  },
  code: {
    background: '#11111b', border: '1px solid #313244', borderRadius: 4,
    padding: '0 4px', fontSize: 11.5,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  linkBtn: {
    background: '#313244', border: 'none', color: '#cdd6f4',
    borderRadius: 6, padding: '5px 12px', cursor: 'pointer', fontSize: 12,
    marginTop: 2,
  },
  fixBtn: {
    background: '#89b4fa', border: 'none', color: '#11111b', fontWeight: 600,
    borderRadius: 6, padding: '6px 14px', fontSize: 12,
  },
  footer: {
    display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 14,
    flexShrink: 0,
  },
  close: {
    background: 'transparent', border: '1px solid #313244', color: '#cdd6f4',
    borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 12,
  },
}
