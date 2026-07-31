/**
 * units.tsx — render a HyperSpy axis `units` string for DISPLAY.
 *
 * HyperSpy stores units as raw LaTeX ("$\AA^{-1}$" for Å⁻¹, "$\mu$m" for µm)
 * because that is what its matplotlib axis labels consume. That raw string is
 * the value that must be written BACK to `axes_manager`, but shown literally in
 * a sidebar cell it is unreadable. So the dock splits the two: the cell DISPLAYS
 * `<UnitText>` (rendered), the click-to-edit input keeps the raw string, and the
 * commit sends the raw string on unchanged — the same value/`display` split the
 * axes table already uses for rounded scale/offset.
 *
 * KaTeX with `output:'mathml'` — the same pipeline the report markdown uses
 * (kernel/markdown.ts). MathML renders natively in Chromium with NO KaTeX
 * stylesheet and no web fonts, so this costs the dock nothing at load and the
 * glyphs inherit the surrounding colour/size. `throwOnError:false` +
 * `strict:'ignore'` mean a malformed or non-TeX unit string degrades to KaTeX's
 * own inline error markup instead of throwing (and `\AA` doesn't warn about
 * math-vs-text accents on every render).
 *
 * `dangerouslySetInnerHTML` is safe here for the same reason markdown.ts trusts
 * its substituted fragment: the HTML is generated locally by KaTeX from text,
 * with no `trust` option, so nothing in the source string can become markup.
 */
import React from 'react'
import katex from 'katex'

/** Does this units string contain TeX syntax? `$…$` delimiters, a control
 *  sequence, a group, or a super/subscript. Plain units ("px", "nm", "e/s",
 *  "1/nm", "") have none of these and must render as ORDINARY TEXT — KaTeX
 *  would set them as italic maths variables, which is worse than the raw
 *  string it replaced. */
const TEX_RE = /[\\^_{}$]/

export function isLatexUnits(raw: string): boolean {
  return TEX_RE.test(raw)
}

/** Drop the `$` math delimiters. They become EMPTY GROUPS rather than nothing,
 *  because hyperspy's other common form is "$\mu$m" — deleting the `$` outright
 *  would weld the control sequence to the next letter ("\mum", undefined),
 *  while "{}\mu{}m" renders "μm" exactly as matplotlib does. `{}` is a no-op
 *  everywhere else ("$\AA^{-1}$" → "{}\AA^{-1}{}", "nm$^{-1}$" → "nm{}^{-1}{}").
 *  A units string with no `$` at all (bare "\AA^{-1}") is already the whole
 *  expression and passes through untouched. */
function stripDelimiters(raw: string): string {
  return raw.replace(/\$/g, '{}')
}

// Units repeat across every row and the dock re-renders on every histogram
// push, so memoise the (pure) render. The key space is tiny — a handful of
// distinct unit strings per session.
const cache = new Map<string, string | null>()

const OPTS = {
  output: 'mathml',
  strict: 'ignore',        // `\AA` is a text accent in strict LaTeX — don't warn
  errorColor: '#f38ba8',
} as const

/** KaTeX MathML for *raw*, or null when it isn't (or can't be) maths. */
export function unitsToMathML(raw: string): string | null {
  if (!raw || !isLatexUnits(raw)) return null
  const hit = cache.get(raw)
  if (hit !== undefined) return hit
  const tex = stripDelimiters(raw)
  // Nothing left to typeset (a stray "$", "{}") — show the raw text instead of
  // an empty cell.
  if (!/[A-Za-z0-9\\]/.test(tex)) {
    cache.set(raw, null)
    return null
  }
  let html: string | null = null
  try {
    // UPRIGHT: a unit is never a variable, so it is set roman (SI, and what the
    // figure's own scale bar / axis label already show) — maths mode would
    // italicise "Å" and "m". `throwOnError` is ON for this attempt only, so a
    // string `\mathrm{}` can't hold falls through to the plain render below
    // rather than to KaTeX's red error markup.
    html = katex.renderToString(`\\mathrm{${tex}}`, { ...OPTS, throwOnError: true })
  } catch {
    try {
      html = katex.renderToString(tex, { ...OPTS, throwOnError: false })
    } catch {
      // Defensive: throwOnError:false already downgrades parse errors, but an
      // unexpected KaTeX failure must not blank the cell — null falls back to
      // the raw text in the plain-text branch below.
      html = null
    }
  }
  cache.set(raw, html)
  return html
}

/** A units string as the user should SEE it: rendered when it is LaTeX, plain
 *  text otherwise. `title` carries the raw string either way, so the source of
 *  a rendered cell is one hover away. */
export function UnitText({ raw }: { raw: string }) {
  const html = unitsToMathML(raw)
  if (html == null) return <>{raw}</>
  return (
    <span
      data-testid="unit-latex"
      title={raw}
      // Inherit the cell's colour; nudge the size up (the same 1.1× the report
      // markdown gives KaTeX) because a 10 px dock cell renders the exponent
      // near-illegibly at 1em.
      style={{ color: 'inherit', fontSize: '1.1em' }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
