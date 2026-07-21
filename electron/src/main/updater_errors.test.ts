/**
 * updater_errors.test.ts — node:test unit tests for the pure updater error mapper.
 *
 * Runs on every CI push, every OS (`node --test`, native TS type-stripping on
 * Node 24+). Guards the "mac auto-update failed spectacularly" regression: a raw
 * releases.atom XML body must be collapsed to a short line, never passed through.
 *
 * Run: `node --test src/main/updater_errors.test.ts` (from electron/), or via the
 * `test:unit` npm script.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  friendlyError, truncateMessage, isPrereleaseVersion, defaultChannelForVersion,
} from './updater_errors.ts'

test('offline / DNS errors → offline message', () => {
  for (const raw of ['net::ERR_INTERNET_DISCONNECTED', 'getaddrinfo ENOTFOUND github.com', 'EAI_AGAIN']) {
    assert.match(friendlyError(raw), /offline/i)
  }
})

test('timeout errors → try again', () => {
  for (const raw of ['ETIMEDOUT', 'net::ERR_TIMED_OUT', 'request timed out']) {
    assert.match(friendlyError(raw), /too long|try again/i)
  }
})

test('connection reset/refused → could not reach', () => {
  for (const raw of ['ECONNRESET', 'net::ERR_CONNECTION_REFUSED', 'socket hang up']) {
    assert.match(friendlyError(raw), /reach the update server/i)
  }
})

test('missing feed / 404 → no update info', () => {
  for (const raw of ['Cannot find latest-mac.yml', 'HttpError: 404', 'status code 404']) {
    assert.match(friendlyError(raw), /no update information/i)
  }
})

test('RAW ATOM/HTML FEED is collapsed, never passed through (the regression)', () => {
  const atom = '<?xml version="1.0" encoding="UTF-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">'
    + '<entry><id>tag:github.com,2008:Repository/1/v0.2.0-rc.6</id>'.repeat(300) + '</feed>'
  const out = friendlyError(atom)
  assert.doesNotMatch(out, /<entry|<feed|<\?xml/i, 'markup leaked into the user message')
  assert.ok(out.length < 200, `message not collapsed (len ${out.length})`)
  assert.match(out, /unexpected response|manually/i)
})

test('a raw <html> error page is collapsed too', () => {
  const html = '<!DOCTYPE html><html><body>' + '<div>error</div>'.repeat(500) + '</body></html>'
  const out = friendlyError(html)
  assert.doesNotMatch(out, /<html|<div|<!DOCTYPE/i)
  assert.ok(out.length < 200)
})

test('an unrecognised long single-line blob is truncated, not verbatim', () => {
  const blob = 'SomeUpstreamError: ' + 'x'.repeat(5000)
  const out = friendlyError(blob)
  assert.ok(out.length <= 300, `not truncated (len ${out.length})`)
  assert.ok(out.endsWith('…'))
})

test('a short recognisable-enough message passes through (collapsed whitespace)', () => {
  assert.equal(friendlyError('Update   failed\n  for reason X'), 'Update failed for reason X')
})

test('truncateMessage collapses whitespace and caps length', () => {
  assert.equal(truncateMessage('a\n\n  b   c'), 'a b c')
  const long = 'y'.repeat(400)
  const t = truncateMessage(long, 100)
  assert.equal(t.length, 100)
  assert.ok(t.endsWith('…'))
})

test('empty / nullish input → empty string, never throws', () => {
  assert.equal(friendlyError(''), '')
  // @ts-expect-error deliberately exercising a nullish raw
  assert.equal(friendlyError(null), '')
  // @ts-expect-error deliberately exercising an undefined raw
  assert.equal(friendlyError(undefined), '')
})

// ── channel detection (rc build → beta, stable build → stable) ────────────────

test('prerelease versions are detected', () => {
  for (const v of ['0.2.0-rc.8', '0.2.0-rc.1', '1.0.0-beta.2', '2.3.4-alpha.1',
                   'v0.2.0-rc.8', '0.2.0-rc.8+build.5']) {
    assert.equal(isPrereleaseVersion(v), true, `${v} should be prerelease`)
  }
})

test('plain releases are NOT prerelease', () => {
  for (const v of ['0.2.0', '1.0.0', 'v2.3.4', '0.2.0+build.5', '10.20.30']) {
    assert.equal(isPrereleaseVersion(v), false, `${v} should be stable`)
  }
})

test('defaultChannelForVersion: rc build → beta, stable build → stable', () => {
  assert.equal(defaultChannelForVersion('0.2.0-rc.8'), 'beta')
  assert.equal(defaultChannelForVersion('v1.0.0-beta.1'), 'beta')
  assert.equal(defaultChannelForVersion('0.2.0'), 'stable')
  assert.equal(defaultChannelForVersion('1.2.3'), 'stable')
})

test('channel detection tolerates empty/garbage version', () => {
  assert.equal(isPrereleaseVersion(''), false)
  // @ts-expect-error nullish
  assert.equal(isPrereleaseVersion(null), false)
  assert.equal(defaultChannelForVersion(''), 'stable')
})
