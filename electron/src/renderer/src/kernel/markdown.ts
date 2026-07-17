/**
 * markdown.ts — the single markdown render pipeline for the Report Builder.
 *
 * `marked` (+ two custom math tokenizers) → `DOMPurify.sanitize` → KaTeX
 * substitution. Used for BOTH the in-app rendered view (ReportCell) and the
 * sanitized `html` fragment shipped on every markdown commit
 * (`report_add_cell`/`report_update_cell`) so static HTML export embeds real
 * rendered HTML rather than an ugly `<pre>` fallback. Factoring it here keeps
 * the display render and the export-cache render byte-identical.
 *
 * Math: `$$…$$` (display) and `$…$` (inline) are captured by marked TOKENIZER
 * extensions — so a `$` inside a code span/fence is never treated as math —
 * and rendered by KaTeX with `output:'mathml'`. MathML needs NO stylesheet or
 * fonts (Chromium/Firefox/Safari all render MathML Core natively), so the
 * cached fragment stays self-contained in the exported HTML/PDF for free.
 *
 * Sanitizer ordering: the tokenizers emit an EMPTY placeholder
 * `<span class="md-math" data-tex data-display>` through marked/DOMPurify
 * (plain span + data attrs — survives sanitization untouched), and KaTeX HTML
 * is substituted in AFTERWARD. DOMPurify's default profile would otherwise
 * strip KaTeX's `<semantics>/<annotation>` MathML and leak the raw TeX as
 * text. The substituted fragment is trusted by construction: it is generated
 * locally by KaTeX (`throwOnError:false`, no `trust` option) from text.
 */
import { marked, type TokenizerAndRendererExtension } from 'marked'
import DOMPurify from 'dompurify'
import katex from 'katex'

// ── math tokenizer extensions ─────────────────────────────────────────────────

interface MathToken {
  type: 'blockMath' | 'inlineMath'
  raw: string
  tex: string
  display: boolean
}

const mathPlaceholder = (tex: string, display: boolean): string =>
  `<span class="md-math" data-tex="${encodeURIComponent(tex)}" data-display="${display ? '1' : '0'}"></span>`

/** `$$…$$` opening a block (may span lines). */
const blockMath: TokenizerAndRendererExtension = {
  name: 'blockMath',
  level: 'block',
  start: (src: string) => src.indexOf('$$'),
  tokenizer(src: string): MathToken | undefined {
    const m = /^\$\$([\s\S]+?)\$\$(?:\n+|$)/.exec(src)
    if (m && m[1].trim()) {
      return { type: 'blockMath', raw: m[0], tex: m[1].trim(), display: true }
    }
    return undefined
  },
  renderer(token) {
    return mathPlaceholder((token as unknown as MathToken).tex, true)
  },
}

/** `$$…$$` inside a paragraph (still display math), and `$…$` inline math.
 *  Inline `$` follows the pandoc rule: the opening `$` must be immediately
 *  followed by a non-space, the closing `$` immediately preceded by one and
 *  not followed by a digit — so "$5 and $10" is never math. `\$` escapes stay
 *  literal (marked's escape rule owns them; inside math `\$` passes through
 *  to KaTeX). */
const inlineMath: TokenizerAndRendererExtension = {
  name: 'inlineMath',
  level: 'inline',
  start: (src: string) => src.indexOf('$'),
  tokenizer(src: string): MathToken | undefined {
    const dd = /^\$\$([^$]+?)\$\$/.exec(src)
    if (dd && dd[1].trim()) {
      return { type: 'inlineMath', raw: dd[0], tex: dd[1].trim(), display: true }
    }
    const m = /^\$(?!\s)((?:\\.|[^\\$\n])+?)\$(?!\d)/.exec(src)
    if (m && m[1].trim() && !/\s$/.test(m[1])) {
      return { type: 'inlineMath', raw: m[0], tex: m[1], display: false }
    }
    return undefined
  },
  renderer(token) {
    const t = token as unknown as MathToken
    return mathPlaceholder(t.tex, t.display)
  },
}

marked.use({ extensions: [blockMath, inlineMath] })

// ── render pipeline ───────────────────────────────────────────────────────────

/** Substitute each sanitized `.md-math` placeholder with its KaTeX-rendered
 *  MathML fragment. String-parse via DOMParser (attribute order after
 *  sanitization is not guaranteed, so no regex on the raw string). */
function substituteMath(html: string): string {
  if (!html.includes('md-math')) return html
  const doc = new DOMParser().parseFromString(html, 'text/html')
  const spans = doc.querySelectorAll('span.md-math[data-tex]')
  if (!spans.length) return html
  for (const el of Array.from(spans)) {
    let tex = ''
    try { tex = decodeURIComponent(el.getAttribute('data-tex') ?? '') } catch { /* keep '' */ }
    const display = el.getAttribute('data-display') === '1'
    let rendered = ''
    try {
      rendered = katex.renderToString(tex, {
        output: 'mathml',
        displayMode: display,
        throwOnError: false,
        errorColor: '#f38ba8',
      })
      // MathML-only output skips the `.katex-display` wrapper the HTML output
      // has; add it so the display-math CSS (centered block, x-scroll) applies.
      if (display) rendered = `<span class="katex-display">${rendered}</span>`
    } catch {
      // Defensive: throwOnError:false already downgrades parse errors, but a
      // truly unexpected KaTeX failure must not blank the whole cell.
      const fb = doc.createElement('code')
      fb.textContent = display ? `$$${tex}$$` : `$${tex}$`
      el.replaceWith(fb)
      continue
    }
    const holder = doc.createElement('template')
    holder.innerHTML = rendered
    el.replaceWith(holder.content)
  }
  return doc.body.innerHTML
}

/** Render markdown source → sanitized HTML fragment (safe for the DOM AND for
 *  the exported page). `$…$` / `$$…$$` come back as KaTeX MathML. */
export function renderMarkdown(src: string): string {
  const html = marked.parse(src ?? '', { async: false }) as string
  return substituteMath(DOMPurify.sanitize(html))
}
