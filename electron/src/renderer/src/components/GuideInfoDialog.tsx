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
import { ModalDialog, S as Shell } from './WizardShell'

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
    <ModalDialog testid="guide-info-dialog" width={520} maxHeight="calc(100vh - 60px)"
      title={<>
        <span style={S.kicker}>Technique</span>
        <span style={S.title}>{guide.title}</span>
      </>}
      onClose={onClose} closeTestid="guide-info-close" dismissOnBackdrop
      footer={<>
        <button data-testid="guide-info-dismiss" style={Shell.dialogCancel} onClick={onClose}>
          Close
        </button>
        <button
          data-testid="guide-info-start-tour"
          style={Shell.dialogConfirm}
          onClick={() => { onClose(); onStartGuide(guide) }}
        >
          Guided tour ›
        </button>
      </>}>
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
    </ModalDialog>
  )
}

const ACCENT = '#89b4fa'
const S: Record<string, React.CSSProperties> = {
  kicker: {
    display: 'block', fontSize: 10, fontWeight: 700, letterSpacing: 0.7,
    textTransform: 'uppercase', color: ACCENT,
  },
  title: { display: 'block', marginTop: 4, fontSize: 19 },
  summary: { margin: '6px 0 0', color: '#a6adc8', lineHeight: 1.5, fontSize: 13 },
  body: { lineHeight: 1.6, color: '#bac2de' },
  p: { margin: '8px 0' },
  callout: {
    margin: '10px 0', padding: '9px 11px', borderRadius: 7,
    background: 'rgba(137,180,250,0.10)', borderLeft: `3px solid ${ACCENT}`,
    color: '#cdd6f4',
  },
  links: {
    margin: '6px 0 28px', paddingTop: 12, borderTop: '1px solid #313244',
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
}
