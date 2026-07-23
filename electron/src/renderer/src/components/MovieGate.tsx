/**
 * MovieGate.tsx — owns the full-screen Movie editor's lifecycle inside the SpyDE
 * provider (mirrors PresentGate).
 *
 * Two ways the editor opens for a movie cell:
 *   • spyde:movie_edit {cell_id} — dispatched by a movie CARD's "Edit ▶" button
 *     (renderer-local CustomEvent).
 *   • spyde:movie_edit_open {cell_id} — re-broadcast from the BACKEND when the
 *     sidebar "Movie" card creates a placeholder movie with open:true, so a fresh
 *     Movie card jumps straight into the editor.
 *
 * On open the gate fires movie_open (the backend resolves the cell's source, seeds
 * defaults, and emits movie_state); on close it fires movie_close (which cancels
 * any in-flight export and drops the session). Only ONE editor is open at a time
 * (editing is inherently single-cell).
 *
 * Also PUBLISHES the surfaced signal window's id via movieEditorClaim (laundry
 * #6/#13): the editor's SeamlessFigureFrame mounts a SECOND iframe for the same
 * fig_id as the source's ordinary MDI window (this gate lays the editor on TOP
 * of the MDI area, not in place of it) — without the claim, MDIArea keeps that
 * MDI window's iframe mounted too and the two race for the fig_id's live
 * pushes (see movieEditorClaim.ts for the full mechanism). The claim tracks
 * `movie_state`'s `signal_window_id` for THIS cell and is cleared whenever the
 * editor closes for any reason (explicit close, cell switch, unmount).
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { MovieEditor } from './MovieEditor'
import { setMovieEditorClaim } from '../kernel/movieEditorClaim'
import type { MovieStateMessage } from '../kernel/protocol'

export function MovieGate() {
  const { state, sendAction } = useSpyDE()
  const [cellId, setCellId] = React.useState<string | null>(null)

  React.useEffect(() => {
    const onEdit = (e: Event) => {
      const id = (e as CustomEvent).detail?.cell_id
      if (typeof id === 'string' && id) setCellId(id)
    }
    window.addEventListener('spyde:movie_edit', onEdit)
    window.addEventListener('spyde:movie_edit_open', onEdit)
    return () => {
      window.removeEventListener('spyde:movie_edit', onEdit)
      window.removeEventListener('spyde:movie_edit_open', onEdit)
    }
  }, [])

  // Open the backend session when a cell is selected; close it on teardown /
  // switch. Keyed by cellId so switching cells closes the old session first.
  // The claim is cleared in this SAME cleanup — unconditionally, so a stale
  // claim can never survive past movie_close firing (explicit close, cell
  // switch, or MovieGate/App unmount all run this cleanup).
  React.useEffect(() => {
    if (!cellId) return
    sendAction('movie_open', { cell_id: cellId })
    return () => {
      sendAction('movie_close', { cell_id: cellId })
      setMovieEditorClaim(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cellId])

  // Track movie_state for OUR cell to know which MDI window the editor is
  // currently surfacing (may change if the cell's source is reassigned).
  React.useEffect(() => {
    if (!cellId) return
    const onState = (e: Event) => {
      const d = (e as CustomEvent).detail as MovieStateMessage
      if (d.cell_id !== cellId) return
      setMovieEditorClaim(d.signal_window_id ?? null)
    }
    window.addEventListener('spyde:movie_state', onState)
    return () => window.removeEventListener('spyde:movie_state', onState)
  }, [cellId])

  // Auto-close if the report closes (or a NEW/different report is opened) or
  // our cell disappears from the document (deleted, undo) — MovieGate is
  // mounted at the app root regardless of report state, so unlike a normal
  // report-sidebar view it does NOT naturally unmount when the report goes
  // away. Without this the editor would sit open on a cell the backend has
  // already dropped (report_new / report_close / a movie cell delete tear down
  // the session server-side — see handlers.py `_clear_movie_sessions` /
  // `report_remove_cell`), so every action would silently no-op against a
  // session that no longer exists.
  React.useEffect(() => {
    if (!cellId) return
    const report = state.report
    const stillOpen = !!report?.open && report.cells.some(c => c.id === cellId)
    if (!stillOpen) setCellId(null)
  }, [cellId, state.report])

  if (!cellId) return null
  return (
    <MovieEditor
      cellId={cellId}
      sendAction={(action, payload) => sendAction(action, payload)}
      onClose={() => setCellId(null)}
    />
  )
}
