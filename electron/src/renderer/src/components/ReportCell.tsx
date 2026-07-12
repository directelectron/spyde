/**
 * ReportCell.tsx — a markdown cell in the Report sidebar.
 *
 * Rendered view: `marked` → `DOMPurify.sanitize` → dangerouslySetInnerHTML,
 * styled for the dark theme via a scoped wrapper class (`spyde-md`) whose
 * stylesheet is injected once (the ConsoleBar/StatusBar keyframe idiom).
 * Double-click → an autosized monospace <textarea>; commit on blur AND
 * Ctrl/Cmd-Enter (report_update_cell), Escape reverts. Raw mode (report-level
 * toggle) forces the textarea for every cell.
 *
 * Cell chrome on hover: an HTML5 drag-handle (reorder within the list →
 * report_move_cell, wired by the parent) + a delete button (report_remove_cell).
 */
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import type { ReportCell as ReportCellType } from '../kernel/protocol'

// One-time scoped markdown stylesheet for the dark theme. Injected under a
// `.spyde-md` wrapper so it never leaks into the rest of the app.
if (typeof document !== 'undefined' && !document.getElementById('spyde-md-css')) {
  const el = document.createElement('style')
  el.id = 'spyde-md-css'
  el.textContent = `
.spyde-md { color: #cdd6f4; font-size: 13px; line-height: 1.55; word-break: break-word; }
.spyde-md > *:first-child { margin-top: 0; }
.spyde-md > *:last-child { margin-bottom: 0; }
.spyde-md h1, .spyde-md h2, .spyde-md h3, .spyde-md h4 {
  color: #cdd6f4; font-weight: 600; line-height: 1.3;
  margin: 14px 0 6px; }
.spyde-md h1 { font-size: 19px; border-bottom: 1px solid #313244; padding-bottom: 4px; }
.spyde-md h2 { font-size: 16px; border-bottom: 1px solid #313244; padding-bottom: 3px; }
.spyde-md h3 { font-size: 14px; }
.spyde-md h4 { font-size: 13px; color: #a6adc8; }
.spyde-md p { margin: 6px 0; }
.spyde-md a { color: #89b4fa; text-decoration: none; }
.spyde-md a:hover { text-decoration: underline; }
.spyde-md ul, .spyde-md ol { margin: 6px 0; padding-left: 22px; }
.spyde-md li { margin: 2px 0; }
.spyde-md code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
  background: #11111b; border: 1px solid #313244; border-radius: 4px;
  padding: 1px 5px; color: #f5c2e7; }
.spyde-md pre {
  background: #11111b; border: 1px solid #313244; border-radius: 6px;
  padding: 10px 12px; overflow-x: auto; margin: 8px 0; }
.spyde-md pre code { background: none; border: none; padding: 0; color: #cdd6f4; }
.spyde-md blockquote {
  border-left: 3px solid #45475a; margin: 8px 0; padding: 2px 12px;
  color: #a6adc8; }
.spyde-md table { border-collapse: collapse; margin: 8px 0; font-size: 12px; }
.spyde-md th, .spyde-md td { border: 1px solid #313244; padding: 4px 8px; }
.spyde-md th { background: #1e1e2e; color: #cdd6f4; }
.spyde-md img { max-width: 100%; border-radius: 4px; }
.spyde-md hr { border: none; border-top: 1px solid #313244; margin: 12px 0; }
`
  document.head.appendChild(el)
}

function renderMarkdown(src: string): string {
  // `marked` is configured synchronously (no async extensions), so parse
  // returns a string here; sanitize before it ever touches the DOM.
  const html = marked.parse(src ?? '', { async: false }) as string
  return DOMPurify.sanitize(html)
}

interface Props {
  cell: ReportCellType
  /** Report-level raw/rendered toggle — forces the editor for every cell. */
  rawMode: boolean
  onUpdate: (source: string) => void
  onRemove: () => void
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

export function ReportCell({ cell, rawMode, onUpdate, onRemove, dragProps }: Props) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(cell.source ?? '')
  const [hover, setHover] = useState(false)
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
    if (draft !== (cell.source ?? '')) onUpdate(draft)
  }
  const revert = () => {
    setDraft(cell.source ?? '')
    setEditing(false)
  }

  const rendered = React.useMemo(() => renderMarkdown(cell.source ?? ''), [cell.source])
  const empty = !(cell.source ?? '').trim()

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
      {/* Hover chrome: drag handle (reorder) + delete. */}
      {(hover || showEditor) && (
        <div style={styles.chrome}>
          <span
            data-testid={`report-cell-drag-${cell.id}`}
            style={styles.dragHandle}
            title="Drag to reorder"
          >⠿</span>
          <button
            data-testid={`report-cell-delete-${cell.id}`}
            style={styles.deleteBtn}
            title="Delete cell"
            onClick={onRemove}
          >✕</button>
        </div>
      )}

      {showEditor ? (
        <textarea
          ref={taRef}
          data-testid={`report-cell-textarea-${cell.id}`}
          style={styles.textarea}
          value={draft}
          autoFocus={editing && !rawMode}
          spellCheck={false}
          placeholder="Write markdown…"
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => { if (!rawMode) commit(); else if (draft !== (cell.source ?? '')) onUpdate(draft) }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault()
              ;(e.target as HTMLTextAreaElement).blur()
            } else if (e.key === 'Escape' && !rawMode) {
              e.preventDefault()
              revert()
            }
          }}
        />
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
  deleteBtn: {
    background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer',
    fontSize: 11, padding: '0 2px', lineHeight: 1,
  },
  rendered: {
    cursor: 'text', minHeight: 20, padding: '4px 4px',
    borderRadius: 4,
  },
  emptyHint: { color: '#585b70', fontSize: 12, fontStyle: 'italic' },
  textarea: {
    width: '100%', boxSizing: 'border-box', resize: 'none',
    background: '#11111b', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 5,
    padding: '6px 8px', fontSize: 12.5,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    lineHeight: 1.5, outline: 'none', overflow: 'hidden',
    minHeight: 40,
  },
}
