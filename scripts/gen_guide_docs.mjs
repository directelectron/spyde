#!/usr/bin/env node
/**
 * gen_guide_docs.mjs — render the guides/ walkthroughs into the Sphinx docs.
 *
 * A Guide (guides/*.ts) is the SINGLE SOURCE for a technique walkthrough. It was
 * already rendered in two places — the in-app coachmark Tour and the docs-site
 * React app — and this script adds the third: `doc/tutorials/<id>.rst`, so the
 * full click-by-click example is in the project DOCUMENTATION and someone on the
 * web can read the same steps without installing anything.
 *
 * Because the guides are TypeScript and Sphinx is Python, the bridge is
 * esbuild: bundle guides/index.ts to a throwaway CJS file (the guides are pure
 * data — no JSX, no React — so this is a plain, dependency-free bundle), require
 * it, and emit rST. Regenerate after editing any guide:
 *
 *     node scripts/gen_guide_docs.mjs           # write doc/tutorials/
 *     node scripts/gen_guide_docs.mjs --check   # fail if it would change
 *
 * The generated pages are COMMITTED, so a docs build needs only Python — Node is
 * required to regenerate, not to build. `--check` is the drift guard.
 *
 * Screenshots: a step's `image` is the file `guide_screenshots.spec.ts` writes
 * into docs-site/public/media/<guide>/. conf.py copies that tree into
 * doc/tutorials/media/ at build time (see `setup(app)` there) rather than this
 * script duplicating PNGs into git. A step whose screenshot has not been
 * captured yet simply renders without one.
 */
import { execFileSync } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync, existsSync, rmSync, readdirSync, unlinkSync } from 'node:fs'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const OUT_DIR = join(root, 'doc', 'tutorials')
const MEDIA_SRC = join(root, 'docs-site', 'public', 'media')
const ESBUILD = join(root, 'electron', 'node_modules', '.bin', 'esbuild')

const check = process.argv.includes('--check')

// ---------------------------------------------------------------------------
// Load the guides (TS -> CJS -> require)
// ---------------------------------------------------------------------------
function loadGuides() {
  if (!existsSync(ESBUILD)) {
    console.error(
      `esbuild not found at ${ESBUILD}\n` +
      'Run `npm install` in electron/ first — this script bundles the ' +
      'TypeScript guides so it can read them.')
    process.exit(2)
  }
  const tmp = join(root, '.guides-bundle.cjs')
  try {
    execFileSync(ESBUILD, [
      join(root, 'guides', 'index.ts'),
      '--bundle', '--platform=node', '--format=cjs', `--outfile=${tmp}`,
      '--log-level=error',
    ], { stdio: 'inherit' })
    const require_ = createRequire(import.meta.url)
    // Bust any cache so repeated runs in one process see fresh content.
    delete require_.cache?.[tmp]
    return require_(tmp).GUIDES
  } finally {
    if (existsSync(tmp)) rmSync(tmp)
  }
}

// ---------------------------------------------------------------------------
// Markdown (the guide body dialect) -> reStructuredText
//
// The dialect is deliberately tiny (see guides/markdown.tsx): paragraphs,
// `**bold**`, `` `code` ``, and `> 💡` callout blockquotes. Anything else is
// passed through verbatim.
// ---------------------------------------------------------------------------
function inlineMd(s) {
  return s
    // Escape rST's own inline markup characters that markdown does not use.
    .replace(/\|/g, '\\|')
    // `code` -> ``code``  (do this BEFORE bold so ** inside code is untouched)
    .replace(/`([^`]+)`/g, '``$1``')
    // **bold** -> **bold** (identical in rST, but normalise stray spacing)
    .replace(/\*\*\s*([^*]+?)\s*\*\*/g, '**$1**')
}

function bodyToRst(md, indent = '') {
  const out = []
  for (const raw of md.split('\n\n')) {
    const block = raw.trim()
    if (!block) continue
    if (block.startsWith('>')) {
      // `> 💡 …` callout -> a Sphinx tip directive.
      const text = block
        .split('\n')
        .map((l) => l.replace(/^>\s?/, '').trim())
        .join(' ')
        .replace(/^💡\s*/, '')
      out.push(`${indent}.. tip::`)
      out.push('')
      out.push(...wrap(inlineMd(text), `${indent}   `))
      out.push('')
    } else {
      out.push(...wrap(inlineMd(block.replace(/\n/g, ' ')), indent))
      out.push('')
    }
  }
  return out
}

/** Soft-wrap at ~78 columns so the generated rST stays readable in a diff. */
function wrap(text, indent) {
  const width = 78 - indent.length
  const words = text.split(/\s+/)
  const lines = []
  let line = ''
  for (const w of words) {
    if (line && (line + ' ' + w).length > width) { lines.push(indent + line); line = w }
    else line = line ? line + ' ' + w : w
  }
  if (line) lines.push(indent + line)
  return lines
}

const underline = (s, ch) => ch.repeat(Math.max(s.length, 3))

// ---------------------------------------------------------------------------
// One guide -> one rST page
// ---------------------------------------------------------------------------
function guideToRst(guide) {
  const L = []
  L.push('..')
  L.push('   GENERATED FILE — do not edit by hand.')
  L.push('   Source: guides/' + guide.id + '.ts (the same walkthrough the in-app')
  L.push('   guided tour renders). Regenerate with:')
  L.push('       node scripts/gen_guide_docs.mjs')
  L.push('')
  L.push(`.. _tutorial-${guide.id}:`)
  L.push('')
  L.push(guide.title)
  L.push(underline(guide.title, '='))
  L.push('')
  L.push(...wrap(inlineMd(guide.summary), ''))
  L.push('')
  L.push('.. admonition:: Follow along in the app')
  L.push('   :class: note')
  L.push('')
  L.push('   Every step below is also a live walkthrough inside SpyDE:')
  L.push(`   **Help → ${guide.title} → Guided tour**. The tour loads the same small`)
  L.push('   tutorial dataset for you (no download), highlights each control as you')
  L.push('   go, and closes the example data again when you exit.')
  L.push('')

  // --- the steps ----------------------------------------------------------
  L.push('Steps')
  L.push(underline('Steps', '-'))
  L.push('')

  let n = 0
  for (const step of guide.steps) {
    n += 1
    const heading = `${n}. ${step.title}`
    L.push(heading)
    L.push(underline(heading, '~'))
    L.push('')
    L.push(...bodyToRst(step.body))
    if (step.image && existsSync(join(MEDIA_SRC, guide.id, step.image))) {
      L.push(`.. image:: media/${guide.id}/${step.image}`)
      L.push(`   :alt: ${step.title}`)
      L.push('   :width: 100%')
      L.push('')
    }
  }

  // --- more information ---------------------------------------------------
  if (guide.info) {
    L.push('More information')
    L.push(underline('More information', '-'))
    L.push('')
    L.push(...bodyToRst(guide.info.blurb))
    if (guide.info.links?.length) {
      L.push('Further reading')
      L.push(underline('Further reading', '~'))
      L.push('')
      L.push(...wrap(
        'SpyDE wraps pyxem, HyperSpy, eXSpy, kikuchipy and orix; those projects ' +
        'document the underlying methods in far more depth than a walkthrough can.',
        ''))
      L.push('')
      for (const link of guide.info.links) {
        L.push(`* \`${link.label} <${link.url}>\`_`);
        if (link.note) {
          L.push('')
          L.push(...wrap(inlineMd(link.note), '  '))
          L.push('')
        }
      }
      L.push('')
    }
  }
  return L.join('\n').replace(/\n{3,}/g, '\n\n') + '\n'
}

function indexRst(guides) {
  const L = []
  L.push('..')
  L.push('   GENERATED FILE — do not edit by hand.')
  L.push('   Regenerate with: node scripts/gen_guide_docs.mjs')
  L.push('')
  L.push('.. _tutorials-index:')
  L.push('')
  L.push('Tutorials')
  L.push('=========')
  L.push('')
  L.push(...wrap(
    'Click-by-click walkthroughs of a complete technique in the SpyDE ' +
    'interface. Each one is generated from the same source as the guided tour ' +
    'built into the app, so what you read here is exactly what the app shows ' +
    'you — open one in SpyDE from **Help → <technique> → Guided tour**, or ' +
    'read it through here first.', ''))
  L.push('')
  L.push(...wrap(
    'Every tutorial is self-contained: the in-app tour loads its own small ' +
    'example dataset (no download, a couple of seconds), and closes it again ' +
    'when you finish. Each one ends with **More information** — background on ' +
    'the technique and links to the upstream documentation.', ''))
  L.push('')
  L.push('.. toctree::')
  L.push('   :maxdepth: 1')
  L.push('')
  for (const g of guides) L.push(`   ${g.id}`)
  L.push('')
  L.push('.. list-table::')
  L.push('   :header-rows: 1')
  L.push('   :widths: 25 75')
  L.push('')
  L.push('   * - Tutorial')
  L.push('     - What it covers')
  for (const g of guides) {
    L.push(`   * - :ref:\`${g.title} <tutorial-${g.id}>\``)
    L.push(`     - ${inlineMd(g.summary).replace(/\n/g, ' ')}`)
  }
  L.push('')
  return L.join('\n') + '\n'
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
const guides = loadGuides()
const files = new Map()
files.set('index.rst', indexRst(guides))
for (const g of guides) files.set(`${g.id}.rst`, guideToRst(g))

if (check) {
  let bad = 0
  for (const [name, text] of files) {
    const path = join(OUT_DIR, name)
    const cur = existsSync(path) ? readFileSync(path, 'utf8') : null
    if (cur !== text) { console.error(`OUT OF DATE: doc/tutorials/${name}`); bad += 1 }
  }
  // Also flag a page for a guide that no longer exists.
  if (existsSync(OUT_DIR)) {
    for (const name of readdirSync(OUT_DIR)) {
      if (name.endsWith('.rst') && !files.has(name)) {
        console.error(`STALE: doc/tutorials/${name}`); bad += 1
      }
    }
  }
  if (bad) {
    console.error('\nRun `node scripts/gen_guide_docs.mjs` and commit the result.')
    process.exit(1)
  }
  console.log(`doc/tutorials/ is up to date (${files.size} files).`)
  process.exit(0)
}

mkdirSync(OUT_DIR, { recursive: true })
for (const [name, text] of files) writeFileSync(join(OUT_DIR, name), text)
// Drop pages for guides that were removed.
for (const name of readdirSync(OUT_DIR)) {
  if (name.endsWith('.rst') && !files.has(name)) {
    unlinkSync(join(OUT_DIR, name))
    console.log(`removed stale doc/tutorials/${name}`)
  }
}
console.log(`wrote ${files.size} files to doc/tutorials/`)
