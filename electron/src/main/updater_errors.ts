/**
 * updater_errors.ts — PURE, dependency-free error mapping for the auto-updater.
 *
 * Split out of updater.ts (which imports `electron`, so it can't be loaded by a
 * plain node unit test) so `friendlyError` can be unit-tested with `node:test` on
 * every CI push, on every OS. The whole point: an updater error must NEVER reach
 * the UI as a raw blob — the "mac auto-update failed spectacularly" screenshot was
 * ~10 KB of raw releases.atom XML rendered full-screen because the fallback
 * returned the provider's message verbatim. See updater_errors.test.ts.
 */

/** Bound an unrecognised error message so a huge single-line payload can't blow
 *  out the error box. Collapse whitespace, cap length, keep it one readable line. */
export function truncateMessage(s: string, max = 300): string {
  const flat = String(s ?? '').replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat
}

/** Map a raw electron-updater / Chromium-net error to something a user can act
 *  on. Falls back to a BOUNDED, sanitised version of the raw message when we
 *  don't recognise it — never the raw blob verbatim. */
export function friendlyError(raw: string): string {
  const s = String(raw || '')
  if (/ERR_INTERNET_DISCONNECTED|ENOTFOUND|EAI_AGAIN|ERR_NAME_NOT_RESOLVED|getaddrinfo/i.test(s)) {
    return 'You appear to be offline — check your connection and try again.'
  }
  if (/ETIMEDOUT|ERR_TIMED_OUT|ERR_CONNECTION_TIMED_OUT|timed out/i.test(s)) {
    return 'The update server took too long to respond — please try again.'
  }
  if (/ERR_CONNECTION_(REFUSED|RESET|CLOSED)|ECONNRESET|ECONNREFUSED|socket hang up/i.test(s)) {
    return 'Could not reach the update server — please try again.'
  }
  if (/latest.*\.yml|Cannot find .*\.yml|status code 404|HttpError: 404|ERR_HTTP_RESPONSE_CODE_FAILURE/i.test(s)) {
    return 'No update information available right now — please try again later.'
  }
  // A provider that can't resolve the platform feed can surface the GitHub
  // releases.atom body (or an HTML error page) as the error message. That raw
  // XML/HTML must NEVER reach the UI verbatim — it rendered as a full-screen wall
  // of markup ("mac auto-update failed spectacularly"). Detect markup / an
  // oversized blob and collapse it to a short, actionable line.
  if (/<\?xml|<!DOCTYPE|<feed\b|<entry\b|<html\b|<rss\b/i.test(s)) {
    return 'The update server returned an unexpected response — please try again later or update manually from GitHub.'
  }
  return truncateMessage(s)
}
