/**
 * sentryEnvelope.ts — the wire format for a problem report, with no I/O.
 *
 * Sentry's ingest API is a documented HTTP endpoint, and a report is a small
 * JSON document posted to it. That is the whole integration, so this is written
 * against the protocol rather than against `@sentry/electron`: the SDK's value
 * is automatic crash capture, and reports here are only ever sent because
 * somebody clicked a button. Skipping it keeps a native crash handler out of a
 * hardened-runtime, notarized macOS build and keeps the payload something a
 * maintainer can read.
 *
 * Everything here is pure, so the parts that are easy to get wrong — the DSN
 * split, the auth header, the envelope's length-prefixed framing — are unit
 * tested on every OS without a network.
 *
 * Protocol reference: https://develop.sentry.dev/sdk/envelopes/
 */

/** The pieces of a DSN that the ingest request actually needs. */
export interface SentryTarget {
  /** Full URL to POST an envelope to. */
  endpoint: string
  /** The DSN's public key, for the X-Sentry-Auth header. */
  publicKey: string
  /** The DSN with any secret stripped — envelope headers carry it verbatim. */
  dsn: string
}

/**
 * Split a Sentry DSN into what an ingest POST needs, or null if it isn't one.
 *
 * A DSN looks like `https://<publicKey>@<host>/<projectId>`; older ones carry
 * `<publicKey>:<secret>@`, and the secret is not used by the envelope endpoint.
 * Returning null (rather than throwing) is what lets a build ship with no DSN
 * configured and fall back to writing the report to disk.
 */
export function parseSentryDsn(dsn: string | undefined | null): SentryTarget | null {
  if (!dsn) return null
  let url: URL
  try {
    url = new URL(dsn.trim())
  } catch {
    return null
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') return null
  const publicKey = url.username
  const projectId = url.pathname.replace(/^\/+/, '').replace(/\/+$/, '')
  if (!publicKey || !projectId || !/^\d+$/.test(projectId)) return null
  const base = `${url.protocol}//${url.host}`
  return {
    endpoint: `${base}/api/${projectId}/envelope/`,
    publicKey,
    dsn: `${url.protocol}//${publicKey}@${url.host}/${projectId}`,
  }
}

/** The `X-Sentry-Auth` header value for a target. */
export function sentryAuthHeader(target: SentryTarget, client: string): string {
  return [
    'Sentry sentry_version=7',
    `sentry_client=${client}`,
    `sentry_key=${target.publicKey}`,
  ].join(', ')
}

/**
 * A 32-character lowercase hex id, the shape Sentry requires for `event_id`.
 * Takes its randomness from the caller so this file stays pure and testable.
 */
export function formatEventId(bytes: Uint8Array): string {
  let hex = ''
  for (const b of bytes) hex += b.toString(16).padStart(2, '0')
  return hex.slice(0, 32).padEnd(32, '0')
}

/**
 * Frame an event as a Sentry envelope: newline-delimited JSON, where each item
 * is preceded by a header naming its type and byte length. The length is in
 * BYTES, not characters — a report containing a non-ASCII path or a stack trace
 * with a smart quote would be rejected if this counted characters.
 */
export function buildEnvelope(
  target: SentryTarget,
  event: Record<string, unknown>,
  sentAt: string,
): string {
  const body = JSON.stringify(event)
  const length = Buffer.byteLength(body, 'utf8')
  const envelopeHeader = JSON.stringify({
    event_id: event.event_id,
    sent_at: sentAt,
    dsn: target.dsn,
  })
  const itemHeader = JSON.stringify({
    type: 'event',
    content_type: 'application/json',
    length,
  })
  return `${envelopeHeader}\n${itemHeader}\n${body}\n`
}
