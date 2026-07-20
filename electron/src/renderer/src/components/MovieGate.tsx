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
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { MovieEditor } from './MovieEditor'

export function MovieGate() {
  const { sendAction } = useSpyDE()
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
  React.useEffect(() => {
    if (!cellId) return
    sendAction('movie_open', { cell_id: cellId })
    return () => { sendAction('movie_close', { cell_id: cellId }) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cellId])

  if (!cellId) return null
  return (
    <MovieEditor
      cellId={cellId}
      sendAction={(action, payload) => sendAction(action, payload)}
      onClose={() => setCellId(null)}
    />
  )
}
