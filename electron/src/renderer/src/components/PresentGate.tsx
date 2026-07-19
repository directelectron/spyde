/**
 * PresentGate.tsx — owns Present mode's lifecycle inside the SpyDE provider.
 *
 * The Report sidebar's "Present" button dispatches a `spyde:report_present`
 * window event; this gate opens PresentMode. It holds the state that must SURVIVE
 * a "go live" excursion:
 *   • the current slide index (a ref, persisted across the exit so re-entry lands
 *     on the SAME slide),
 *   • an `excursion` flag → the floating "⤺ Back to presentation" pill.
 *
 * The go-live handoff (design piece 3): a slide's "Launch live ▶" fires
 * `onLaunchLive`. We EXIT Present mode, fire the live action — `tutorial_load`
 * (Phase 1) to load the dataset, plus optionally start the guide tour (Phase 2's
 * auto-drive lands separately) — and show the Back pill. Pressing "P" or the pill
 * RE-ENTERS Present mode at the persisted slide index.
 *
 * GRACEFUL DEGRADATION (Phase 1/2 may not be merged): `tutorial_load` is fired
 * best-effort — if the backend doesn't know it, it just no-ops and the excursion
 * is simply "exited Present mode with a way back". The guide tour is started only
 * when a `guide` id is present AND `onStartGuide` resolves it. We never hard-
 * depend on those phases; the Back pill is what makes the round-trip work either
 * way.
 */
import React, { useEffect, useRef, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { PresentMode, type LiveAction } from './PresentMode'
import { getGuide, type Guide } from '@guides/index'

export function PresentGate({ onStartGuide }: { onStartGuide: (g: Guide) => void }) {
  const { state, sendAction } = useSpyDE()
  const [presenting, setPresenting] = useState(false)
  // A go-live excursion is active: Present mode is closed but a "Back" pill lets
  // the user resume the deck at the persisted slide.
  const [excursion, setExcursion] = useState(false)
  // The slide index survives BOTH a plain exit and a go-live excursion (a ref, so
  // reopening resumes where the user was). Reset only on an explicit full exit.
  const slideRef = useRef(0)

  // The Report sidebar's Present button (and any other launcher) opens us via
  // this event. Starting fresh resets to slide 0 unless we're resuming.
  useEffect(() => {
    const onPresent = (e: Event) => {
      const detail = (e as CustomEvent).detail as { resume?: boolean } | undefined
      if (!detail?.resume) slideRef.current = 0
      setExcursion(false)
      setPresenting(true)
    }
    window.addEventListener('spyde:report_present', onPresent)
    return () => window.removeEventListener('spyde:report_present', onPresent)
  }, [])

  // "P" resumes the deck from an active excursion (a hardware-key shortcut that
  // doesn't require finding the pill). Only while an excursion is pending and we
  // aren't already presenting; ignore it when typing in an input.
  useEffect(() => {
    if (!excursion || presenting) return
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      const typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)
      if (typing) return
      if (e.key === 'p' || e.key === 'P') { e.preventDefault(); resume() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [excursion, presenting])

  const resume = () => { setExcursion(false); setPresenting(true) }

  const onExit = () => {
    setPresenting(false)
    setExcursion(false)
    slideRef.current = 0
  }

  const onLaunchLive = (action: LiveAction) => {
    // Exit Present mode but KEEP the slide index (slideRef untouched) so the Back
    // pill / "P" resumes at this slide.
    setPresenting(false)
    setExcursion(true)
    // Best-effort dataset load (Phase 1). Absent action → backend no-op.
    if (action.tutorial) {
      sendAction('tutorial_load', { name: action.tutorial })
    }
    // Optionally start the guide tour (Phase 2's auto-drive is separate; here we
    // just open the coachmark tour if the guide resolves).
    if (action.guide) {
      const g = getGuide(action.guide)
      if (g) onStartGuide(g)
    }
  }

  // Nothing to show unless presenting or mid-excursion.
  if (!presenting && !excursion) return null

  return (
    <>
      {presenting && (
        <PresentMode
          initialSlide={slideRef.current}
          onSlideChange={(i) => { slideRef.current = i }}
          onExit={onExit}
          onLaunchLive={onLaunchLive}
        />
      )}
      {/* Floating "Back to presentation" pill — the deliberate return path from a
          go-live side-trip. Bottom-center, above the status bar. */}
      {excursion && !presenting && (
        <button
          data-testid="present-resume-pill"
          style={styles.pill}
          title="Resume the presentation (P)"
          onClick={resume}
        >⤺ Back to presentation</button>
      )}
    </>
  )
}

const styles: Record<string, React.CSSProperties> = {
  pill: {
    position: 'fixed', bottom: 48, left: '50%', transform: 'translateX(-50%)',
    zIndex: 9400,
    background: '#89b4fa', color: '#11111b', border: 'none',
    borderRadius: 22, padding: '9px 20px', fontSize: 14, fontWeight: 700,
    cursor: 'pointer', boxShadow: '0 6px 20px rgba(0,0,0,0.5)',
  },
}
