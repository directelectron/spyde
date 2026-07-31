/**
 * GuideInfoDialog.tsx — the "Info" half of a technique.
 *
 * The help bar lists a TECHNIQUE and offers two things under it: **Info** (this
 * dialog — what the technique is, and where to read more) and **Guided tour**
 * (the in-app coachmark walkthrough). Both render from the SAME `Guide` object
 * in guides/, so the app, the docs page and the tour can never disagree:
 * `guide.info.blurb` is the background, `guide.info.links` the further reading.
 *
 * Links are EXTERNAL and open in the user's browser (openExternal) — SpyDE
 * wraps pyxem / HyperSpy / eXSpy / kikuchipy / orix and those projects own the
 * science, so we cite and link rather than restating their documentation here.
 *
 * Unlike the Tour (deliberately click-through and ✕-only), this IS a modal and
 * follows the app's dialog idiom — backdrop click closes, exactly like
 * PeriodicTable and GpuHelpDialog.
 */
import React from 'react'
import type { Guide } from '@guides/index'
import { Markdown } from '@guides/markdown'

export function GuideInfoDialog({
  guide,
  onClose,
  onStartGuide,
}: {
  guide: Guide
  onClose: () => void
  onStartGuide: (g: Guide) => void
}) {
  const info = guide.info
  return (
    <div style={S.backdrop} data-testid="guide-info-dialog" onClick={onClose}>
      <div style={S.modal} onClick={(e) => e.stopPropagation()}>
        <div style={S.head}>
          <div>
            <div style={S.kicker}>Technique</div>
            <h2 style={S.title}>{guide.title}</h2>
          </div>
          <button
            data-testid="guide-info-close"
            style={S.x}
            title="Close"
            aria-label="Close"
            onClick={onClose}
          >
            ✕
          </button>
        </div>

        <p style={S.summary}>{guide.summary}</p>

        {info && (
          <div style={S.body}>
            <Markdown text={info.blurb} styles={{ paragraph: S.p, callout: S.callout }} />
          </div>
        )}

        {info && info.links.length > 0 && (
          <div style={S.links} data-testid="guide-info-links">
            <div style={S.linksLabel}>Further reading</div>
            {info.links.map((l) => (
              <button
                key={l.url}
                data-testid={`guide-info-link-${l.url}`}
                style={S.linkBtn}
                onClick={() => window.electron?.openExternal?.(l.url)}
              >
                <span style={S.linkLabel}>{l.label} ↗</span>
                {l.note && <span style={S.linkNote}>{l.note}</span>}
              </button>
            ))}
          </div>
        )}

        <div style={S.footer}>
          <button data-testid="guide-info-dismiss" style={S.ghost} onClick={onClose}>
            Close
          </button>
          <button
            data-testid="guide-info-start-tour"
            style={S.primary}
            onClick={() => { onClose(); onStartGuide(guide) }}
          >
            Guided tour ›
          </button>
        </div>
      </div>
    </div>
  )
}

const ACCENT = '#89b4fa'
const S: Record<string, React.CSSProperties> = {
  backdrop: {
    position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 9500,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  modal: {
    width: 520, maxWidth: 'calc(100vw - 32px)', maxHeight: 'calc(100vh - 60px)',
    overflowY: 'auto', background: '#1e1e2e', border: '1px solid #313244',
    borderRadius: 10, padding: 18, color: '#cdd6f4',
    boxShadow: '0 18px 48px rgba(0,0,0,0.6)',
    fontSize: 13,
  },
  head: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' },
  kicker: {
    fontSize: 10, fontWeight: 700, letterSpacing: 0.7, textTransform: 'uppercase',
    color: ACCENT,
  },
  title: { margin: '4px 0 0', fontSize: 19, fontWeight: 600 },
  x: {
    background: 'transparent', border: 'none', color: '#6c7086',
    cursor: 'pointer', fontSize: 14, padding: 2,
  },
  summary: { margin: '10px 0 0', color: '#a6adc8', lineHeight: 1.5, fontSize: 13 },
  body: { marginTop: 6, lineHeight: 1.6, color: '#bac2de' },
  p: { margin: '8px 0' },
  callout: {
    margin: '10px 0', padding: '9px 11px', borderRadius: 7,
    background: 'rgba(137,180,250,0.10)', borderLeft: `3px solid ${ACCENT}`,
    color: '#cdd6f4',
  },
  links: {
    marginTop: 14, paddingTop: 12, borderTop: '1px solid #313244',
    display: 'flex', flexDirection: 'column', gap: 6,
  },
  linksLabel: {
    fontSize: 10.5, fontWeight: 700, letterSpacing: 0.6, color: '#6c7086',
    textTransform: 'uppercase',
  },
  linkBtn: {
    display: 'block', width: '100%', textAlign: 'left', cursor: 'pointer',
    background: 'rgba(137,180,250,0.08)', border: '1px solid #313244',
    borderRadius: 7, padding: '8px 10px', color: '#cdd6f4',
  },
  linkLabel: { display: 'block', fontSize: 12.5, color: ACCENT, fontWeight: 500 },
  linkNote: { display: 'block', fontSize: 11.5, color: '#7f849c', marginTop: 2, lineHeight: 1.4 },
  footer: {
    marginTop: 18, display: 'flex', justifyContent: 'flex-end', gap: 8,
    // Sticky at the bottom of the scrolling modal: a technique with a long
    // blurb + five links overflows, and Close / Guided tour must stay reachable
    // without scrolling to the end first (same trick as the Tour bubble footer).
    position: 'sticky', bottom: -18, background: '#1e1e2e',
    paddingTop: 10, marginBottom: -18, paddingBottom: 18,
  },
  ghost: {
    background: 'transparent', border: '1px solid #313244', color: '#cdd6f4',
    borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 12,
  },
  primary: {
    background: ACCENT, border: 'none', color: '#11111b', fontWeight: 600,
    borderRadius: 6, padding: '7px 16px', cursor: 'pointer', fontSize: 12,
  },
}
