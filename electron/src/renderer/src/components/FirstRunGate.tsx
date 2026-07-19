/**
 * FirstRunGate.tsx — Phase 4 of the docs overhaul: auto-open the "welcome"
 * guided walkthrough on a genuine first launch.
 *
 * Lives inside SpyDEProvider (like PresentGate/GpuStatusGate) so it can use
 * `useSpyDE()`; App.tsx owns the actual tour state (`tour`/`setTour`) and
 * passes `onAutoOpen` down, the same shape as PresentGate's `onStartGuide`.
 *
 * Sequence:
 *   1. Wait for `state.ready` (backend/dask ready — mirrors the "Ready" gate
 *      MDIArea already uses) so we never race the tour's own autoload, which
 *      needs a live backend to load the tutorial dataset.
 *   2. Once ready (fired exactly once via a guard ref), request
 *      `get_first_run` — the staged action mirroring `get_gpu_status`
 *      (sendAction + a `spyde:first_run_result` DOM CustomEvent reply; see
 *      GpuStatusDialog.tsx for the same request/response shape).
 *   3. If the reply says `first_run: true` AND we haven't already auto-opened
 *      in this session, open the welcome guide and immediately fire
 *      `mark_tutorial_seen` so the flag is persisted the moment the tour is
 *      shown (even if the user closes it right away) — it never auto-shows
 *      again on a later launch.
 *
 * The welcome guide stays reachable any time afterwards from the "?" Help
 * menu / Help → First Steps (guides/index.ts lists it) — this gate only
 * controls the ONE-TIME automatic open.
 */
import React, { useEffect, useRef } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { getGuide, type Guide } from '@guides/index'

export function FirstRunGate({ onAutoOpen }: { onAutoOpen: (g: Guide) => void }) {
  const { state, sendAction } = useSpyDE()
  // sendAction's identity changes each render; route through a ref so the
  // effects below don't re-fire on every render (same guard used elsewhere,
  // e.g. Tour.tsx's sendRef).
  const sendRef = useRef(sendAction)
  sendRef.current = sendAction
  // Guards so we query at most once, and auto-open at most once, per app
  // lifetime — a re-render or a reconnect can't re-fire either.
  const queriedRef = useRef(false)
  const openedRef = useRef(false)

  // Step 1+2: once the backend is ready, ask if this is a first run.
  useEffect(() => {
    if (!state.ready || queriedRef.current) return
    queriedRef.current = true
    sendRef.current('get_first_run')
  }, [state.ready])

  // Step 3: react to the reply.
  useEffect(() => {
    const onResult = (e: Event) => {
      const detail = (e as CustomEvent).detail as { first_run?: boolean } | undefined
      if (openedRef.current || !detail?.first_run) return
      const guide = getGuide('welcome')
      if (!guide) return
      openedRef.current = true
      onAutoOpen(guide)
      // Persist immediately on open (not on dismissal) so a user who closes
      // the tour right away still won't see it auto-open next launch.
      sendRef.current('mark_tutorial_seen')
    }
    window.addEventListener('spyde:first_run_result', onResult)
    return () => window.removeEventListener('spyde:first_run_result', onResult)
  }, [onAutoOpen])

  return null
}
