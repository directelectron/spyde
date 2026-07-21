#!/usr/bin/env node
/**
 * update-download-links.mjs — regenerate the "Download the latest version" tables
 * in README.md and doc/user_guide/installing.rst with version-pinned links.
 *
 * Run by the Prepare Release workflow so the front-page + docs download links
 * always point at the release being cut. Idempotent: rewrites only the content
 * BETWEEN the marker comments, so re-running with the same version is a no-op.
 *
 *   node scripts/update-download-links.mjs <version> [repo]
 *   e.g. node scripts/update-download-links.mjs 0.2.0-rc.8 CSSFrancis/spyde
 *
 * Markers (must already exist in each file):
 *   README.md (HTML comments):
 *     <!-- spyde:download-table:start --> … <!-- spyde:download-table:end -->
 *   installing.rst (rST comments):
 *     .. spyde:download-table:start  …  .. spyde:download-table:end
 *
 * Asset names mirror electron-builder.yml's outputs (see release.yml):
 *   Windows  SpyDE-Setup-<v>.exe
 *   macOS    SpyDE-<v>-arm64.dmg
 *   Linux    SpyDE-<v>.AppImage
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')

const version = (process.argv[2] || '').replace(/^v/, '').trim()
const repo = (process.argv[3] || 'CSSFrancis/spyde').trim()
if (!version) {
  console.error('usage: node scripts/update-download-links.mjs <version> [owner/repo]')
  process.exit(1)
}

const base = `https://github.com/${repo}/releases/download/v${version}`
const rel = `https://github.com/${repo}/releases/tag/v${version}`
const assets = {
  win: `SpyDE-Setup-${version}.exe`,
  mac: `SpyDE-${version}-arm64.dmg`,
  linux: `SpyDE-${version}.AppImage`,
}

// ── Markdown table (README) ───────────────────────────────────────────────────
const md = [
  `**[⬇ Download SpyDE v${version}](${rel})** — pick your platform:`,
  '',
  '| Platform | Download |',
  '|----------|----------|',
  `| **Windows** | [${assets.win}](${base}/${assets.win}) |`,
  `| **macOS** (Apple Silicon) | [${assets.mac}](${base}/${assets.mac}) |`,
  `| **Linux** | [${assets.linux}](${base}/${assets.linux}) |`,
  '',
  `All releases: <https://github.com/${repo}/releases>`,
].join('\n')

// ── rST table (docs) ──────────────────────────────────────────────────────────
// list-table is the robust rST table (no column-width fiddling).
const rst = [
  `**Download SpyDE v${version}:** \`all releases <https://github.com/${repo}/releases>\`__`,
  '',
  '.. list-table::',
  '   :header-rows: 1',
  '   :widths: 30 70',
  '',
  '   * - Platform',
  '     - Download',
  '   * - **Windows**',
  `     - \`${assets.win} <${base}/${assets.win}>\`__`,
  '   * - **macOS** (Apple Silicon)',
  `     - \`${assets.mac} <${base}/${assets.mac}>\`__`,
  '   * - **Linux**',
  `     - \`${assets.linux} <${base}/${assets.linux}>\`__`,
].join('\n')

/** Replace the text between start/end markers; error if markers are missing.
 *  Surrounds the body with BLANK lines so it renders in both Markdown and rST —
 *  in rST a `.. marker` comment swallows the next line unless a blank line breaks
 *  the comment block, and the end marker likewise needs a blank line before it. */
function replaceBetween(text, startMarker, endMarker, body, file) {
  const s = text.indexOf(startMarker)
  const e = text.indexOf(endMarker)
  if (s === -1 || e === -1 || e < s) {
    throw new Error(`markers not found (or out of order) in ${file}: ` +
      `${startMarker} … ${endMarker}`)
  }
  const before = text.slice(0, s + startMarker.length)
  const after = text.slice(e)
  return `${before}\n\n${body}\n\n${after}`
}

let changed = 0

// README.md
{
  const file = join(root, 'README.md')
  const text = readFileSync(file, 'utf8')
  const out = replaceBetween(
    text,
    '<!-- spyde:download-table:start -->',
    '<!-- spyde:download-table:end -->',
    md, 'README.md')
  if (out !== text) { writeFileSync(file, out); changed++; console.log(`updated README.md → v${version}`) }
}

// doc/user_guide/installing.rst
{
  const file = join(root, 'doc', 'user_guide', 'installing.rst')
  const text = readFileSync(file, 'utf8')
  const out = replaceBetween(
    text,
    '.. spyde:download-table:start',
    '.. spyde:download-table:end',
    rst, 'doc/user_guide/installing.rst')
  if (out !== text) { writeFileSync(file, out); changed++; console.log(`updated installing.rst → v${version}`) }
}

console.log(changed ? `done (${changed} file(s) changed)` : 'no changes (already up to date)')
