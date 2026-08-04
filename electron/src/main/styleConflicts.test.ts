/**
 * styleConflicts.test.ts — no React style object may mix a border SHORTHAND with
 * a border LONGHAND on the same element.
 *
 * React warns "Removing a style property during rerender (borderColor) when a
 * conflicting property is set (border)" and then drops one of them, so the
 * element renders with the wrong border. The pattern that causes it is the
 * ordinary base+modifier spread:
 *
 *     seg:     { border: '1px solid #313244' }          // base — SHORTHAND
 *     segOpen: { borderColor: '#45475a' }               // modifier — LONGHAND
 *     <div style={{ ...S.seg, ...(open ? S.segOpen : {}) }} />
 *
 * It fires on every toggle, and there were 14 of these across the app before
 * this guard existed. The fix is always the same: give the modifier the full
 * shorthand (`border: '1px solid #45475a'`).
 *
 * This scans source text rather than rendering anything — the conflict is a
 * property of the style objects, so it is catchable without a DOM.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
// This test lives in src/main/ because the renderer tsconfig has no node
// types; it only READS the renderer sources, so the location is immaterial.
const RENDERER = join(HERE, '..', 'renderer', 'src')

/** Every .tsx under the renderer source tree. */
function tsxFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) tsxFiles(p, out)
    else if (name.endsWith('.tsx')) out.push(p)
  }
  return out
}

// Longhands that a `border` shorthand resets, and the shorthand each conflicts
// with. `borderTop`/`borderBottom` are themselves per-edge shorthands, so a
// blanket `borderColor` conflicts with those too.
const LONGHAND = /\bborder(Color|Width|Style)\s*:/

test('no style object mixes border shorthand and longhand', () => {
  const offenders: string[] = []
  for (const file of tsxFiles(RENDERER)) {
    const text = readFileSync(file, 'utf8')
    // Style objects that set a border LONGHAND.
    const longhandNames = new Set<string>()
    for (const m of text.matchAll(/(\w+):\s*\{[^{}]*\}/g)) {
      if (LONGHAND.test(m[0])) longhandNames.add(m[1])
    }
    if (!longhandNames.size) continue
    // Style objects that set the `border` (or per-edge) SHORTHAND.
    const shorthandNames = new Set<string>()
    for (const m of text.matchAll(/(\w+):\s*\{[\s\S]*?\n\s*\},/g)) {
      if (/\bborder(Top|Right|Bottom|Left)?\s*:/.test(m[0])) shorthandNames.add(m[1])
    }
    // A spread that combines one of each on the SAME element.
    for (const name of longhandNames) {
      const spread = new RegExp(
        String.raw`\{\s*\.\.\.[^}]*?\.\.\.\(?[^}]*?\b${name}\b[^}]*?\}`, 'g')
      for (const m of text.matchAll(spread)) {
        if ([...shorthandNames].some(b => m[0].includes(`.${b}`))) {
          offenders.push(`${file.slice(RENDERER.length + 1)} — '${name}' ` +
                         `sets a border longhand but is spread over a shorthand base`)
        }
      }
    }
  }
  assert.deepEqual(
    offenders, [],
    'React drops one of the two and the border renders wrong:\n  ' +
    offenders.join('\n  ') +
    '\nGive the modifier the full shorthand instead, e.g. ' +
    "border: '1px solid #89b4fa'.")
})
