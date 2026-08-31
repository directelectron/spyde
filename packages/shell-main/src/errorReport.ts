/**
 * errorReport.ts — "Report a Problem": what went wrong, on what machine, sent
 * to the maintainers.
 *
 * The install base is small and the failures that matter are environmental —
 * a GPU driver, a Windows install directory, a Python wheel that resolved
 * differently on one machine. So a report is worth much more than a stack
 * trace: it carries the OS, the app and runtime versions, the state of the
 * managed Python environment, the last thing the updater said, and the tail of
 * the backend's own output.
 *
 * NOTHING IS SENT WITHOUT A CLICK. Problems are recorded into a bounded
 * in-memory ring as they happen so a report written minutes later still knows
 * what failed, but the ring never leaves the machine on its own. There is no
 * background transport, no crash handler phoning home, and no first-run consent
 * screen to write, because there is nothing to consent to until the user opens
 * the dialog and presses Send.
 *
 * Where it goes: a Sentry project, when a DSN is configured for the build (see
 * `initErrorReporting`). With no DSN — or when the machine is offline, which
 * for a microscope-room PC is the normal case — the same report is written to
 * `<userData>/reports/` and the path handed back, so the user can attach it to
 * an email. That fallback is not a degraded mode; it is the offline path.
 */
import { app, net } from 'electron'
import { randomBytes } from 'crypto'
import { mkdirSync, writeFileSync } from 'fs'
import { join } from 'path'
import { arch, cpus, freemem, platform, release, totalmem, type as osType } from 'os'
import { shellConfig } from './config'
import { recentBackendOutput } from './backendProcess'
import { recordedProblems, type Problem } from './problemLog'
import { getLastUpdateStatus, readUpdateChannel, updatesSupported } from './updater'
import {
  buildEnvelope, formatEventId, parseSentryDsn, sentryAuthHeader, type SentryTarget,
} from './sentryEnvelope'

/** Everything a report carries about the machine and the run. */
export interface Diagnostics {
  app: { name: string; version: string; packaged: boolean; channel: string; updatesSupported: boolean }
  os: { platform: string; type: string; release: string; arch: string; totalMemoryGb: number; freeMemoryGb: number; cpu: string; locale: string }
  runtime: { electron: string; chrome: string; node: string; v8: string }
  update: { lastStatus: unknown }
  problems: Problem[]
  backendOutput: string[]
  /** Whatever the host app added — SpyDE contributes its GPU and Python-env triage. */
  host: Record<string, unknown>
}

/** What `submitReport` managed to do. */
export interface ReportResult {
  /** True when the report reached the reporting service. */
  sent: boolean
  /** Sentry's id for the event, when it was sent — worth quoting in an email. */
  eventId?: string
  /** Where the report was written locally. Always set, sent or not. */
  bundlePath?: string
  /** Why sending failed, when it did. */
  error?: string
}

/** A report is small; a slow network should not hang the dialog. */
const SEND_TIMEOUT_MS = 15_000

let target: SentryTarget | null = null
let hostDiagnostics: () => Promise<Record<string, unknown>> = async () => ({})

/**
 * Configure reporting. Call once at startup.
 *
 * `dsn` is the Sentry DSN for the build. It is a public key — it grants writing
 * events and nothing else — so shipping it in the app is how Sentry is designed
 * to be used; still, keep it in CI's environment rather than in the repo so it
 * can be rotated without a code change. Absent or malformed, reports still work
 * and go to disk only.
 *
 * `collectHostDiagnostics` lets the app add what only it can answer (SpyDE
 * passes its GPU probe and managed-Python-environment triage).
 */
export function initErrorReporting(options: {
  dsn?: string | null
  collectHostDiagnostics?: () => Promise<Record<string, unknown>>
}): void {
  target = parseSentryDsn(options.dsn)
  if (options.collectHostDiagnostics) hostDiagnostics = options.collectHostDiagnostics
}

/** Whether reports can reach the maintainers, or will only be written to disk. */
export function reportingConfigured(): boolean {
  return target !== null
}

const GIGABYTE = 1024 ** 3

/** Gather everything a report carries. Safe to call at any time. */
export async function collectDiagnostics(): Promise<Diagnostics> {
  const cfg = shellConfig()
  let host: Record<string, unknown> = {}
  try {
    host = await hostDiagnostics()
  } catch (err) {
    host = { error: String(err) }
  }
  return {
    app: {
      name: cfg.appName,
      version: app.getVersion(),
      packaged: app.isPackaged,
      channel: safely(() => readUpdateChannel(), 'unknown'),
      updatesSupported: safely(() => updatesSupported(), false),
    },
    os: {
      platform: platform(),
      type: osType(),
      release: release(),
      arch: arch(),
      totalMemoryGb: round1(totalmem() / GIGABYTE),
      freeMemoryGb: round1(freemem() / GIGABYTE),
      cpu: cpus()[0]?.model ?? 'unknown',
      locale: safely(() => app.getLocale(), 'unknown'),
    },
    runtime: {
      electron: process.versions.electron ?? '',
      chrome: process.versions.chrome ?? '',
      node: process.versions.node ?? '',
      v8: process.versions.v8 ?? '',
    },
    update: { lastStatus: safely(() => getLastUpdateStatus(), null) },
    problems: recordedProblems(),
    backendOutput: recentBackendOutput(),
    host,
  }
}

/**
 * Write the report to disk and, when a DSN is configured, send it.
 *
 * The local copy is written FIRST and unconditionally: if the send fails the
 * user still has something to attach to an email, and if it succeeds they have
 * a record of what they sent. `contact` is whatever the reporter chose to type
 * — it is optional, and an empty one is simply omitted rather than sent blank.
 */
export async function submitReport(input: {
  message: string
  contact?: string
}): Promise<ReportResult> {
  const diagnostics = await collectDiagnostics()
  const eventId = formatEventId(randomBytes(16))
  const bundlePath = writeBundle(eventId, input, diagnostics)

  if (!target) {
    return {
      sent: false,
      bundlePath,
      error: 'No reporting service is configured for this build, so the report was saved locally.',
    }
  }
  try {
    await sendToSentry(target, eventId, input, diagnostics)
    return { sent: true, eventId, bundlePath }
  } catch (err) {
    return { sent: false, eventId, bundlePath, error: (err as Error)?.message ?? String(err) }
  }
}

/** The report as a file under `<userData>/reports/`. Returns '' if unwritable. */
function writeBundle(
  eventId: string,
  input: { message: string; contact?: string },
  diagnostics: Diagnostics,
): string {
  try {
    const dir = join(app.getPath('userData'), 'reports')
    mkdirSync(dir, { recursive: true })
    // Sortable, filename-safe, and second-resolution is plenty for one report.
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    const file = join(dir, `${shellConfig().appId}-report-${stamp}.json`)
    writeFileSync(file, JSON.stringify({ eventId, ...input, diagnostics }, null, 2), 'utf8')
    return file
  } catch {
    return ''
  }
}

/** POST one envelope. Rejects on a network failure, a timeout, or a 4xx/5xx. */
async function sendToSentry(
  to: SentryTarget,
  eventId: string,
  input: { message: string; contact?: string },
  diagnostics: Diagnostics,
): Promise<void> {
  const cfg = shellConfig()
  const event: Record<string, unknown> = {
    event_id: eventId,
    timestamp: new Date().toISOString(),
    platform: 'javascript',
    level: 'error',
    logger: 'user-report',
    release: `${cfg.appId}@${diagnostics.app.version}`,
    environment: diagnostics.app.packaged ? 'production' : 'development',
    message: { formatted: firstLine(input.message) },
    tags: {
      os: diagnostics.os.platform,
      os_release: diagnostics.os.release,
      arch: diagnostics.os.arch,
      channel: diagnostics.app.channel,
      packaged: String(diagnostics.app.packaged),
    },
    contexts: {
      os: { name: diagnostics.os.type, version: diagnostics.os.release },
      device: { arch: diagnostics.os.arch, memory_size: diagnostics.os.totalMemoryGb, model: diagnostics.os.cpu },
      runtime: { name: 'electron', version: diagnostics.runtime.electron },
    },
    extra: {
      report: input.message,
      diagnostics,
    },
  }
  if (input.contact?.trim()) event.user = { email: input.contact.trim() }

  const body = buildEnvelope(to, event, new Date().toISOString())
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), SEND_TIMEOUT_MS)
  try {
    // Electron's `net` rather than global fetch: it uses Chromium's stack, so a
    // machine configured with a system proxy or a corporate root certificate —
    // which describes a lot of instrument PCs — works without extra setup.
    const response = await net.fetch(to.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-sentry-envelope',
        'X-Sentry-Auth': sentryAuthHeader(to, `${cfg.appId}-shell/1.0`),
      },
      body,
      signal: controller.signal,
    })
    if (!response.ok) {
      throw new Error(`The reporting service refused the report (HTTP ${response.status}).`)
    }
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') {
      throw new Error('Sending timed out. The report was saved on this machine instead.')
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

function firstLine(text: string): string {
  const line = text.trim().split('\n')[0]
  return line.length > 200 ? `${line.slice(0, 200)}...` : (line || 'Problem report')
}

function round1(value: number): number {
  return Math.round(value * 10) / 10
}

function safely<T>(read: () => T, fallback: T): T {
  try { return read() } catch { return fallback }
}
