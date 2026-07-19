/**
 * ReportCell.tsx — a markdown cell in the Report sidebar.
 *
 * Rendered view: `marked` (+ KaTeX math) → `DOMPurify.sanitize` →
 * dangerouslySetInnerHTML, styled for the dark theme via a scoped wrapper class
 * (`spyde-md`) whose stylesheet is injected once (the ConsoleBar/StatusBar
 * keyframe idiom). Double-click → an autosized monospace <textarea> with a
 * FORMATTING TOOLBAR (bold/italic/strike/code/headings/lists/quote/link/math);
 * commit on blur AND Ctrl/Cmd-Enter (report_update_cell), Escape reverts.
 * Ctrl/Cmd-B and Ctrl/Cmd-I work inside the editor. Raw mode (report-level
 * toggle) forces the textarea for every cell.
 *
 * Cell chrome on hover: an HTML5 drag-handle (reorder within the list →
 * report_move_cell, wired by the parent) + a delete button (report_remove_cell).
 */
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { renderMarkdown } from '../kernel/markdown'
import { reportClipboard } from '../kernel/reportClipboard'
import type { ReportCell as ReportCellType } from '../kernel/protocol'
import { CellChrome, ColumnBadge, type CellColumn } from './CellChrome'
import { SlideNotesEditor } from './SlideNotesEditor'

// One-time scoped markdown stylesheet for the dark theme. Injected under a
// `.spyde-md` wrapper so it never leaks into the rest of the app. Sizes are in
// EM off a `--spyde-md-fs` base so the sidebar's A−/A+ text-size control scales
// everything (headings, code, tables) together.
if (typeof document !== 'undefined' && !document.getElementById('spyde-md-css')) {
  const el = document.createElement('style')
  el.id = 'spyde-md-css'
  el.textContent = `
.spyde-md { color: #cdd6f4; font-size: var(--spyde-md-fs, 13px); line-height: 1.55; word-break: break-word; }
.spyde-md > *:first-child { margin-top: 0; }
.spyde-md > *:last-child { margin-bottom: 0; }
.spyde-md h1, .spyde-md h2, .spyde-md h3, .spyde-md h4 {
  color: #cdd6f4; font-weight: 600; line-height: 1.3;
  margin: 14px 0 6px; }
.spyde-md h1 { font-size: 1.5em; border-bottom: 1px solid #313244; padding-bottom: 4px; }
.spyde-md h2 { font-size: 1.25em; border-bottom: 1px solid #313244; padding-bottom: 3px; }
.spyde-md h3 { font-size: 1.1em; }
.spyde-md h4 { font-size: 1em; color: #a6adc8; }
.spyde-md p { margin: 6px 0; }
.spyde-md strong { font-weight: 700; color: #ffffff; }
.spyde-md em { color: #f5e0dc; }
.spyde-md del { color: #7f849c; }
.spyde-md a { color: #89b4fa; text-decoration: none; }
.spyde-md a:hover { text-decoration: underline; }
.spyde-md ul, .spyde-md ol { margin: 6px 0; padding-left: 22px; }
.spyde-md li { margin: 2px 0; }
.spyde-md li input[type="checkbox"] { margin-right: 6px; accent-color: #89b4fa; }
.spyde-md code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em;
  background: #11111b; border: 1px solid #313244; border-radius: 4px;
  padding: 1px 5px; color: #f5c2e7; }
.spyde-md pre {
  background: #11111b; border: 1px solid #313244; border-radius: 6px;
  padding: 10px 12px; overflow-x: auto; margin: 8px 0; }
.spyde-md pre code { background: none; border: none; padding: 0; color: #cdd6f4; }
.spyde-md blockquote {
  border-left: 3px solid #45475a; margin: 8px 0; padding: 2px 12px;
  color: #a6adc8; }
.spyde-md table { border-collapse: collapse; margin: 8px 0; font-size: 0.92em; }
.spyde-md th, .spyde-md td { border: 1px solid #313244; padding: 4px 8px; }
.spyde-md th { background: #1e1e2e; color: #cdd6f4; }
.spyde-md img { max-width: 100%; border-radius: 4px; }
.spyde-md hr { border: none; border-top: 1px solid #313244; margin: 12px 0; }
/* KaTeX MathML (output:'mathml' — no KaTeX stylesheet needed). */
.spyde-md .katex { font-size: 1.12em; }
.spyde-md .katex-display { display: block; margin: 10px 0; text-align: center;
  overflow-x: auto; overflow-y: hidden; }
.spyde-md math { color: #cdd6f4; }
`
  document.head.appendChild(el)
}

interface Props {
  cell: ReportCellType
  /** Report-level raw/rendered toggle — forces the editor for every cell. */
  rawMode: boolean
  /** Commit a new source + its rendered (sanitized) html fragment. The html
   *  rides along so static export embeds real HTML, not a `<pre>` fallback. */
  onUpdate: (source: string, html: string) => void
  onRemove: () => void
  /** Own index in the cell list (for Duplicate → insert at index+1). */
  index: number
  /** This cell STARTS a slide (first cell or a slide_break) — offer the
   *  per-slide "Title slide" toggle in the chrome. */
  slideStart?: boolean
  /** HTML5 DnD reorder wiring supplied by the parent list. */
  dragProps: {
    onDragStart: (e: React.DragEvent) => void
    onDragOver: (e: React.DragEvent) => void
    onDrop: (e: React.DragEvent) => void
    onDragEnd: () => void
    dragging: boolean
    dropBefore: boolean
  }
}

// ── textarea formatting helpers ──────────────────────────────────────────────
// All pure (draft, selStart, selEnd) → { next, selStart, selEnd } so the
// toolbar and the keyboard shortcuts share one implementation.

interface EditResult { next: string; selStart: number; selEnd: number }

/** Wrap/unwrap the selection in `prefix`/`suffix` (toggle). An empty selection
 *  wraps `placeholder` and selects it. */
function wrapSelection(draft: string, s: number, e: number,
                       prefix: string, suffix: string = prefix,
                       placeholder = 'text'): EditResult {
  const before = draft.slice(0, s)
  const after = draft.slice(e)
  let sel = draft.slice(s, e)
  // Toggle OFF: markers just outside the selection…
  if (before.endsWith(prefix) && after.startsWith(suffix)) {
    return {
      next: before.slice(0, before.length - prefix.length) + sel + after.slice(suffix.length),
      selStart: s - prefix.length, selEnd: e - prefix.length,
    }
  }
  // …or inside it.
  if (sel.length >= prefix.length + suffix.length &&
      sel.startsWith(prefix) && sel.endsWith(suffix)) {
    const inner = sel.slice(prefix.length, sel.length - suffix.length)
    return { next: before + inner + after, selStart: s, selEnd: s + inner.length }
  }
  if (!sel) sel = placeholder
  return {
    next: before + prefix + sel + suffix + after,
    selStart: s + prefix.length, selEnd: s + prefix.length + sel.length,
  }
}

/** Expand [s, e) to whole lines of `draft`. */
function lineSpan(draft: string, s: number, e: number): [number, number] {
  const start = draft.lastIndexOf('\n', s - 1) + 1
  let end = draft.indexOf('\n', Math.max(e, s))
  if (end === -1) end = draft.length
  return [start, end]
}

/** Set the heading level of every selected line (toggle off when the line is
 *  already at that level). */
function setHeading(draft: string, s: number, e: number, level: number): EditResult {
  const [ls, le] = lineSpan(draft, s, e)
  const target = '#'.repeat(level) + ' '
  const lines = draft.slice(ls, le).split('\n').map(line => {
    const stripped = line.replace(/^#{1,6}\s+/, '')
    return line.startsWith(target) ? stripped : target + stripped
  })
  const block = lines.join('\n')
  return { next: draft.slice(0, ls) + block + draft.slice(le), selStart: ls, selEnd: ls + block.length }
}

/** Prefix every selected line (toggle): '- ' bullets, '> ' quote, or 'n. '
 *  numbering for ordered lists. */
function setLinePrefix(draft: string, s: number, e: number,
                       kind: 'bullet' | 'ordered' | 'quote'): EditResult {
  const [ls, le] = lineSpan(draft, s, e)
  const raw = draft.slice(ls, le).split('\n')
  const strip = (line: string) => line.replace(/^(\s*)(?:[-*+]\s+|\d+\.\s+|>\s+)/, '$1')
  const already = raw.every(line => !line.trim() ||
    (kind === 'bullet' ? /^\s*[-*+]\s+/.test(line)
      : kind === 'ordered' ? /^\s*\d+\.\s+/.test(line)
        : /^\s*>\s+/.test(line)))
  let n = 0
  const lines = raw.map(line => {
    if (!line.trim()) return line
    const bare = strip(line)
    if (already) return bare
    n += 1
    return kind === 'bullet' ? `- ${bare}` : kind === 'ordered' ? `${n}. ${bare}` : `> ${bare}`
  })
  const block = lines.join('\n')
  return { next: draft.slice(0, ls) + block + draft.slice(le), selStart: ls, selEnd: ls + block.length }
}

/** Insert a display-math block around the selection (or a starter formula). */
function insertMathBlock(draft: string, s: number, e: number): EditResult {
  const sel = draft.slice(s, e) || 'E = mc^2'
  const before = draft.slice(0, s)
  const needsNL = before.length > 0 && !before.endsWith('\n')
  const prefix = (needsNL ? '\n' : '') + '$$\n'
  const next = before + prefix + sel + '\n$$\n' + draft.slice(e)
  return { next, selStart: s + prefix.length, selEnd: s + prefix.length + sel.length }
}

/** Wrap the selection as a markdown link, selecting the url placeholder. */
function insertLink(draft: string, s: number, e: number): EditResult {
  const sel = draft.slice(s, e) || 'link text'
  const before = draft.slice(0, s)
  const url = 'https://'
  const next = `${before}[${sel}](${url})${draft.slice(e)}`
  const urlStart = s + 1 + sel.length + 2
  return { next, selStart: urlStart, selEnd: urlStart + url.length }
}

type ToolbarCommand =
  | 'bold' | 'italic' | 'strike' | 'code'
  | 'h1' | 'h2' | 'h3'
  | 'bullet' | 'ordered' | 'quote'
  | 'math' | 'link'

function applyCommand(cmd: ToolbarCommand, draft: string, s: number, e: number): EditResult {
  switch (cmd) {
    case 'bold': return wrapSelection(draft, s, e, '**')
    case 'italic': return wrapSelection(draft, s, e, '*')
    case 'strike': return wrapSelection(draft, s, e, '~~')
    case 'code': return wrapSelection(draft, s, e, '`', '`', 'code')
    case 'h1': return setHeading(draft, s, e, 1)
    case 'h2': return setHeading(draft, s, e, 2)
    case 'h3': return setHeading(draft, s, e, 3)
    case 'bullet': return setLinePrefix(draft, s, e, 'bullet')
    case 'ordered': return setLinePrefix(draft, s, e, 'ordered')
    case 'quote': return setLinePrefix(draft, s, e, 'quote')
    case 'math': return insertMathBlock(draft, s, e)
    case 'link': return insertLink(draft, s, e)
  }
}

// The toolbar rows: [command, label, tooltip, extra label style].
const TOOLBAR: Array<[ToolbarCommand, string, string, React.CSSProperties?]> = [
  ['bold', 'B', 'Bold (Ctrl+B)', { fontWeight: 800 }],
  ['italic', 'I', 'Italic (Ctrl+I)', { fontStyle: 'italic', fontFamily: 'serif' }],
  ['strike', 'S', 'Strikethrough', { textDecoration: 'line-through' }],
  ['code', '<>', 'Inline code', { fontFamily: 'ui-monospace, monospace', fontSize: 9.5 }],
  ['h1', 'H1', 'Heading 1'],
  ['h2', 'H2', 'Heading 2'],
  ['h3', 'H3', 'Heading 3'],
  ['bullet', '•', 'Bullet list'],
  ['ordered', '1.', 'Numbered list'],
  ['quote', '❝', 'Quote'],
  ['math', '√x', 'Math block ($$…$$)', { fontStyle: 'italic', fontFamily: 'serif' }],
  ['link', '🔗', 'Link', { fontSize: 10 }],
]

export function ReportCell({ cell, rawMode, onUpdate, onRemove, index, slideStart, dragProps }: Props) {
  const { sendAction } = useSpyDE()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(cell.source ?? '')
  const [hover, setHover] = useState(false)
  const [notesOpen, setNotesOpen] = useState(false)
  const taRef = useRef<HTMLTextAreaElement>(null)

  // Sync the draft when the backing source changes and we're NOT actively
  // editing (a live report_state update from elsewhere).
  useEffect(() => { if (!editing) setDraft(cell.source ?? '') }, [cell.source, editing])

  // Raw mode forces the editor open; leaving raw mode drops back to rendered
  // (unless the user had explicitly double-clicked into edit).
  const showEditor = rawMode || editing

  // Autosize the textarea to its content.
  useLayoutEffect(() => {
    const ta = taRef.current
    if (!ta || !showEditor) return
    ta.style.height = 'auto'
    ta.style.height = `${ta.scrollHeight}px`
  }, [draft, showEditor])

  const commit = () => {
    setEditing(false)
    if (draft !== (cell.source ?? '')) onUpdate(draft, renderMarkdown(draft))
  }
  const revert = () => {
    setDraft(cell.source ?? '')
    setEditing(false)
  }

  // Apply a toolbar/shortcut command to the textarea selection, then restore
  // focus + selection after React re-renders the controlled value.
  const runCommand = (cmd: ToolbarCommand) => {
    const ta = taRef.current
    if (!ta) return
    const r = applyCommand(cmd, draft, ta.selectionStart, ta.selectionEnd)
    setDraft(r.next)
    requestAnimationFrame(() => {
      const el = taRef.current
      if (!el) return
      el.focus()
      el.setSelectionRange(r.selStart, r.selEnd)
    })
  }

  const rendered = React.useMemo(() => renderMarkdown(cell.source ?? ''), [cell.source])
  const empty = !(cell.source ?? '').trim()

  // The serialized clipboard form of THIS cell (source + its rendered html, so a
  // paste static-exports real HTML). Rendered from the live source so a paste
  // reflects any un-committed-but-persisted edit already in `cell.source`.
  const serialize = () => ({
    cell_type: 'markdown' as const,
    source: cell.source ?? '',
    html: rendered,
  })
  const doCopy = () => reportClipboard.set(serialize())
  const doDuplicate = () =>
    sendAction('report_paste_cell', { cell: serialize(), index: index + 1 })

  return (
    <div
      data-testid={`report-cell-${cell.id}`}
      draggable={!showEditor}
      onDragStart={dragProps.onDragStart}
      onDragOver={dragProps.onDragOver}
      onDrop={dragProps.onDrop}
      onDragEnd={dragProps.onDragEnd}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        ...styles.cell,
        ...(dragProps.dragging ? styles.cellDragging : {}),
        ...(dragProps.dropBefore ? styles.cellDropBefore : {}),
      }}
    >
      {/* Always-visible slide-column badge (◧ Left / ◨ Right). */}
      <ColumnBadge column={cell.column} />
      {/* Hover chrome: drag handle (reorder) + column toggle + copy + duplicate + delete. */}
      {(hover || showEditor) && (
        <CellChrome
          cellId={cell.id}
          styles={{ chrome: styles.chrome, chromeBtn: styles.chromeBtn, deleteBtn: styles.deleteBtn, columnBtnActive: styles.columnBtnActive }}
          onCopy={doCopy}
          onDuplicate={doDuplicate}
          onDelete={onRemove}
          column={cell.column}
          onSetColumn={(c: CellColumn) => sendAction('report_set_cell_column', { cell_id: cell.id, column: c })}
          slideStart={slideStart}
          slideKind={cell.slide_kind}
          onToggleTitle={() => sendAction('report_set_slide_kind', { cell_id: cell.id })}
          slideStyle={cell.slide_style}
          onCycleStyle={(style) => sendAction('report_set_slide_style', { cell_id: cell.id, slide_style: style })}
          slideNotes={cell.notes}
          notesOpen={notesOpen}
          onToggleNotes={() => setNotesOpen((v) => !v)}
          deleteTestid={`report-cell-delete-${cell.id}`}
          deleteTitle="Delete cell"
          leading={
            <span
              data-testid={`report-cell-drag-${cell.id}`}
              style={styles.dragHandle}
              title="Drag to reorder"
            >⠿</span>
          }
        />
      )}

      {showEditor ? (
        <div>
          {/* Formatting toolbar. onMouseDown preventDefault keeps the textarea
              focused so a button click can't blur-commit mid-edit. */}
          <div style={styles.fmtBar} data-testid={`report-cell-toolbar-${cell.id}`}>
            {TOOLBAR.map(([cmd, label, tip, extra]) => (
              <button
                key={cmd}
                data-testid={`report-fmt-${cmd}-${cell.id}`}
                style={{ ...styles.fmtBtn, ...(extra ?? {}) }}
                title={tip}
                tabIndex={-1}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => runCommand(cmd)}
              >{label}</button>
            ))}
          </div>
          <textarea
            ref={taRef}
            data-testid={`report-cell-textarea-${cell.id}`}
            style={styles.textarea}
            value={draft}
            autoFocus={editing && !rawMode}
            spellCheck={false}
            placeholder="Write markdown…  ($x^2$ and $$…$$ render as math)"
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => { if (!rawMode) commit(); else if (draft !== (cell.source ?? '')) onUpdate(draft, renderMarkdown(draft)) }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey || e.shiftKey)) {
                e.preventDefault()
                ;(e.target as HTMLTextAreaElement).blur()
              } else if (e.key === 'Escape' && !rawMode) {
                e.preventDefault()
                revert()
              } else if ((e.ctrlKey || e.metaKey) && !e.altKey) {
                const k = e.key.toLowerCase()
                if (k === 'b') { e.preventDefault(); runCommand('bold') }
                else if (k === 'i') { e.preventDefault(); runCommand('italic') }
              }
            }}
          />
        </div>
      ) : (
        <div
          data-testid={`report-cell-rendered-${cell.id}`}
          className="spyde-md"
          onDoubleClick={() => { setDraft(cell.source ?? ''); setEditing(true) }}
          title="Double-click to edit"
          style={styles.rendered}
        >
          {empty
            ? <span style={styles.emptyHint}>Empty text cell — double-click to edit</span>
            : <span dangerouslySetInnerHTML={{ __html: rendered }} />}
        </div>
      )}

      {/* Speaker-notes editor (slide-starting cells only), toggled from the chrome
          📝 button. Debounced → report_set_slide_notes. */}
      {slideStart && notesOpen && (
        <SlideNotesEditor
          cellId={cell.id}
          notes={cell.notes ?? ''}
          onCommit={(notes) => sendAction('report_set_slide_notes', { cell_id: cell.id, notes })}
          onClose={() => setNotesOpen(false)}
        />
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  cell: {
    position: 'relative',
    borderRadius: 6,
    padding: '4px 6px',
    marginBottom: 2,
    borderTop: '2px solid transparent',
  },
  cellDragging: { opacity: 0.4 },
  cellDropBefore: { borderTop: '2px solid #89b4fa' },
  chrome: {
    position: 'absolute', top: 2, right: 4, zIndex: 2,
    display: 'flex', alignItems: 'center', gap: 4,
    background: 'rgba(24,24,37,0.9)', borderRadius: 5, padding: '1px 3px',
  },
  dragHandle: {
    cursor: 'grab', color: '#6c7086', fontSize: 13, userSelect: 'none',
    lineHeight: 1,
  },
  chromeBtn: {
    background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer',
    fontSize: 12, padding: '0 2px', lineHeight: 1,
  },
  columnBtnActive: {
    background: 'none', border: 'none', color: '#89b4fa', cursor: 'pointer',
    fontSize: 12, padding: '0 2px', lineHeight: 1,
  },
  deleteBtn: {
    background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer',
    fontSize: 11, padding: '0 2px', lineHeight: 1,
  },
  rendered: {
    cursor: 'text', minHeight: 20, padding: '4px 4px',
    borderRadius: 4,
  },
  emptyHint: { color: '#585b70', fontSize: 12, fontStyle: 'italic' },
  fmtBar: {
    display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 2,
    background: '#181825', border: '1px solid #313244', borderBottom: 'none',
    borderRadius: '5px 5px 0 0', padding: '3px 4px',
  },
  fmtBtn: {
    background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer',
    fontSize: 11, lineHeight: 1, padding: '3px 5px', borderRadius: 4,
    minWidth: 20, textAlign: 'center',
  },
  textarea: {
    width: '100%', boxSizing: 'border-box', resize: 'none',
    background: '#11111b', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: '0 0 5px 5px',
    padding: '6px 8px', fontSize: 12.5,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    lineHeight: 1.5, outline: 'none', overflow: 'hidden',
    minHeight: 40,
  },
}
