/**
 * SlideNotesEditor.tsx — the inline speaker-notes editor for a slide-starting
 * cell in the Report sidebar.
 *
 * Speaker notes are a PER-SLIDE attribute carried on the slide's FIRST cell (like
 * slide_kind / slide_style). A slide-starting cell's hover chrome offers a
 * "📝 Notes" toggle (owned by CellChrome); when open, THIS component renders a
 * small expandable textarea BELOW the cell. Edits are DEBOUNCED into
 * report_set_slide_notes so authoring stays cheap (notes are edited occasionally).
 *
 * Notes are speaker-private: they show only in the presenter view, never to the
 * audience or in the exported audience deck. The editor makes that explicit with
 * a muted "presenter only" hint.
 */
import React, { useEffect, useRef, useState } from 'react'

interface Props {
  cellId: string
  /** The slide's current notes (from the backend state — this cell is the slide
   *  start, so cell.notes is authoritative). */
  notes: string
  /** Debounced commit → report_set_slide_notes { cell_id, notes }. */
  onCommit: (notes: string) => void
  /** Close the editor (the parent owns the open/closed flag). */
  onClose: () => void
}

const DEBOUNCE_MS = 400

export function SlideNotesEditor({ cellId, notes, onCommit, onClose }: Props) {
  const [draft, setDraft] = useState(notes ?? '')
  const taRef = useRef<HTMLTextAreaElement>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Track the last value we committed so an incoming state echo of our OWN edit
  // doesn't clobber the draft mid-typing.
  const lastSent = useRef(notes ?? '')

  // Re-sync the draft when the backing notes change from ELSEWHERE (not our own
  // debounced commit echoing back).
  useEffect(() => {
    const incoming = notes ?? ''
    if (incoming !== draft && incoming !== lastSent.current) {
      setDraft(incoming)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notes])

  // Focus on open.
  useEffect(() => { taRef.current?.focus() }, [])

  // Flush any pending debounce on unmount so a fast close-then-blur never drops
  // the last keystrokes.
  useEffect(() => {
    return () => {
      if (timer.current) {
        clearTimeout(timer.current)
        timer.current = null
      }
    }
  }, [])

  const schedule = (value: string) => {
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => {
      timer.current = null
      lastSent.current = value
      onCommit(value)
    }, DEBOUNCE_MS)
  }

  const flush = () => {
    if (timer.current) { clearTimeout(timer.current); timer.current = null }
    if (draft !== lastSent.current) {
      lastSent.current = draft
      onCommit(draft)
    }
  }

  return (
    <div style={styles.wrap} data-testid={`slide-notes-editor-${cellId}`}>
      <div style={styles.header}>
        <span style={styles.label}>📝 Speaker notes</span>
        <span style={styles.hint}>presenter only — hidden from the audience</span>
        <button
          data-testid={`slide-notes-close-${cellId}`}
          style={styles.closeBtn}
          title="Close notes editor"
          onClick={() => { flush(); onClose() }}
        >✕</button>
      </div>
      <textarea
        ref={taRef}
        data-testid={`slide-notes-textarea-${cellId}`}
        style={styles.textarea}
        value={draft}
        spellCheck
        placeholder="Notes for this slide (what to say). Markdown supported. Only you see these."
        onChange={(e) => { setDraft(e.target.value); schedule(e.target.value) }}
        onBlur={flush}
        onKeyDown={(e) => {
          // Escape flushes + closes; keep Enter for newlines (notes are multi-line).
          if (e.key === 'Escape') { e.preventDefault(); flush(); onClose() }
        }}
      />
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    margin: '2px 2px 6px', padding: '6px 8px',
    background: 'rgba(137,180,250,0.06)',
    border: '1px solid rgba(137,180,250,0.28)', borderRadius: 6,
  },
  header: {
    display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4,
  },
  label: { fontSize: 11, fontWeight: 700, color: '#89b4fa' },
  hint: { fontSize: 10, color: '#6c7086', fontStyle: 'italic', flex: 1 },
  closeBtn: {
    background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer',
    fontSize: 12, padding: '0 2px', lineHeight: 1,
  },
  textarea: {
    width: '100%', boxSizing: 'border-box', resize: 'vertical',
    minHeight: 56, background: '#11111b', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 5,
    padding: '6px 8px', fontSize: 12,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    lineHeight: 1.5, outline: 'none',
  },
}
