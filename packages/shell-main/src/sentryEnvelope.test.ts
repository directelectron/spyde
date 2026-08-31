/**
 * sentryEnvelope.test.ts — node:test unit tests for the problem-report wire
 * format.
 *
 * This file is hand-written against a protocol, so the tests stand in for the
 * SDK that would otherwise be guaranteeing it. They pin the two things a
 * malformed report would fail on silently — the envelope's byte-length framing
 * and the auth header — and the DSN parse, whose whole job is to fail softly so
 * a build with no DSN configured still writes reports to disk.
 *
 * Run: `node --test src/sentryEnvelope.test.ts`, or via the `test:unit` script.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  buildEnvelope, formatEventId, parseSentryDsn, sentryAuthHeader,
} from './sentryEnvelope.ts'

test('parses a modern DSN into endpoint, key and project', () => {
  const target = parseSentryDsn('https://abc123@o44.ingest.sentry.io/1234567')
  assert.equal(target?.endpoint, 'https://o44.ingest.sentry.io/api/1234567/envelope/')
  assert.equal(target?.publicKey, 'abc123')
  assert.equal(target?.dsn, 'https://abc123@o44.ingest.sentry.io/1234567')
})

test('drops the secret from a legacy key:secret DSN', () => {
  const target = parseSentryDsn('https://pub:secret@o44.ingest.sentry.io/9')
  assert.equal(target?.publicKey, 'pub')
  assert.ok(!target?.dsn.includes('secret'), 'the secret must not be echoed back')
})

test('a self-hosted DSN keeps its port and host', () => {
  const target = parseSentryDsn('https://key@sentry.example.org:8443/42')
  assert.equal(target?.endpoint, 'https://sentry.example.org:8443/api/42/envelope/')
})

test('an absent or unusable DSN yields null rather than throwing', () => {
  // Each of these must leave the app in the "write to disk only" mode.
  for (const dsn of [
    undefined, null, '', '   ',
    'not-a-url',
    'https://o44.ingest.sentry.io/1234567',   // no public key
    'https://abc123@o44.ingest.sentry.io/',   // no project id
    'https://abc123@o44.ingest.sentry.io/notanumber',
    'ftp://abc123@o44.ingest.sentry.io/1',    // not HTTP
  ]) {
    assert.equal(parseSentryDsn(dsn as string | undefined), null, `should reject: ${String(dsn)}`)
  }
})

test('auth header carries version, client and key', () => {
  const target = parseSentryDsn('https://abc123@o44.ingest.sentry.io/1')!
  const header = sentryAuthHeader(target, 'spyde-shell/1.0')
  assert.match(header, /^Sentry sentry_version=7/)
  assert.match(header, /sentry_client=spyde-shell\/1\.0/)
  assert.match(header, /sentry_key=abc123/)
})

test('event ids are 32 lowercase hex characters', () => {
  const id = formatEventId(Uint8Array.from({ length: 16 }, (_, i) => i))
  assert.match(id, /^[0-9a-f]{32}$/)
  assert.equal(id, '000102030405060708090a0b0c0d0e0f')
})

test('a short byte array is padded rather than producing a short id', () => {
  assert.match(formatEventId(new Uint8Array([1, 2])), /^[0-9a-f]{32}$/)
})

test('envelope framing is three newline-delimited JSON lines', () => {
  const target = parseSentryDsn('https://abc123@o44.ingest.sentry.io/7')!
  const event = { event_id: 'a'.repeat(32), message: 'boom' }
  const lines = buildEnvelope(target, event, '2026-08-28T00:00:00.000Z').split('\n')

  const envelopeHeader = JSON.parse(lines[0])
  assert.equal(envelopeHeader.event_id, 'a'.repeat(32))
  assert.equal(envelopeHeader.dsn, target.dsn)

  const itemHeader = JSON.parse(lines[1])
  assert.equal(itemHeader.type, 'event')

  assert.deepEqual(JSON.parse(lines[2]), event)
  assert.equal(lines[3], '', 'the envelope ends with a newline')
})

test('the item length is BYTES, not characters', () => {
  // A report naming a file with an accent, or a stack trace with a smart quote,
  // is rejected outright if this counts characters — hence the explicit test.
  const target = parseSentryDsn('https://abc123@o44.ingest.sentry.io/7')!
  const event = { event_id: 'b'.repeat(32), message: 'Ångström — µm' }
  const lines = buildEnvelope(target, event, '2026-08-28T00:00:00.000Z').split('\n')
  const declared = JSON.parse(lines[1]).length
  assert.equal(declared, Buffer.byteLength(lines[2], 'utf8'))
  assert.ok(declared > lines[2].length, 'multi-byte content must declare more bytes than characters')
})
