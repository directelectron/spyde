/**
 * theme.ts — the app's tokens, in one place.
 *
 * Extracted from the styles that were previously inlined and duplicated across
 * the single App file. Having them named is what lets the status board use
 * `state.unreported` rather than a grey someone picked once, and it is why a
 * card cannot accidentally be drawn in the same colour as a passing one.
 *
 * Continuous with SpyDE's dark palette on purpose — these are sibling apps and
 * an engineer moves between them.
 */

export const C = {
  bg: '#12141a',
  bgSunken: '#0d0f14',
  panel: '#171a22',
  panelRaised: '#1e2230',
  border: '#262a35',
  borderStrong: '#2f3442',

  text: '#e6e9f0',
  textDim: '#c8cee0',
  textMuted: '#7a8296',

  accent: '#8ab4ff',
  accentSunken: '#22304a',
} as const

/**
 * Status colours.
 *
 * `unreported` and `noCriteria` are deliberately BOTH muted and NOT green:
 * neither is a passing check. They differ from each other because the fixes
 * differ — a server that does not expose the property, versus a value nobody
 * has agreed a threshold for yet.
 */
export const STATE = {
  ok: { fg: '#41d18a', bg: '#41d18a1a', label: 'OK' },
  warn: { fg: '#e2b04a', bg: '#e2b04a1a', label: 'Check' },
  bad: { fg: '#ef6b6b', bg: '#ef6b6b1a', label: 'Fault' },
  unreported: { fg: '#5d6478', bg: '#5d647814', label: 'Not reported' },
  no_criteria: { fg: '#7d86a8', bg: '#7d86a814', label: 'No threshold' },
} as const

export type StateKey = keyof typeof STATE

export const stateOf = (s: string): typeof STATE[StateKey] =>
  STATE[(s as StateKey)] ?? STATE.unreported

export const FONT_MONO =
  'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace'
