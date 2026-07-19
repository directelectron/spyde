/**
 * DocsApp.tsx — the SpyDE docs website.
 *
 * Renders the SAME guides the in-app coachmark tour uses (imported from the
 * repo-root guides/ via the @guides alias), so the website and the app never
 * drift. Each guide is shown as a numbered, scrollable walkthrough; steps with a
 * screenshot show it, matching what the in-app tour spotlights live.
 */
import React, { useEffect, useState } from 'react'
import { GUIDES, type Guide } from '@guides/index'
import { Markdown } from '@guides/markdown'

/**
 * InteractiveEmbed — renders a step's self-contained interactive HTML embed in a
 * sandboxed iframe. The embed (built by spyde/tests/gen_guide_embeds.py into
 * public/media/<guide>/) is a standalone page that runs entirely in the browser
 * — navigate, integrate, and virtual-imaging all recompute in JS from
 * precomputed data, with ZERO runtime
 * Python (no pyodide). `sandbox="allow-scripts"` lets the embed's ESM module run
 * while denying same-origin access — the same isolation the app's report export
 * uses. A little "interactive — try it" badge tells the reader to click/drag.
 *
 * Graceful degradation: the embed .html is optional media (like a screenshot). A
 * broken/blank iframe is worse than nothing, so we HEAD-probe the file first and
 * render nothing if it's missing — the step keeps its text, just no demo.
 */
function InteractiveEmbed({ guideId, embed, title }:
  { guideId: string; embed: string; title: string }) {
  const src = `./media/${guideId}/${embed}`
  // 'checking' → probing; 'ok' → file present, mount the iframe; 'missing' → hide.
  const [state, setState] = useState<'checking' | 'ok' | 'missing'>('checking')
  useEffect(() => {
    let live = true
    // A HEAD (falling back to GET) confirms the embed exists before we mount the
    // iframe, so a missing file degrades to hidden instead of a broken frame.
    fetch(src, { method: 'HEAD' })
      .then((r) => { if (live) setState(r.ok ? 'ok' : 'missing') })
      .catch(() => { if (live) setState('missing') })
    return () => { live = false }
  }, [src])

  if (state === 'missing') return null
  return (
    <div style={styles.embedWrap} data-testid={`docs-embed-${embed}`}>
      <div style={styles.embedBadge}>
        <span style={styles.embedDot} />
        interactive — try it
      </div>
      {state === 'ok' && (
        <iframe
          src={src}
          title={`${title} — interactive`}
          // allow-scripts only: the embed's ESM runs, but it stays cross-origin
          // isolated (no cookies, no same-origin fetch of the parent site).
          sandbox="allow-scripts"
          style={styles.embedFrame}
          // NOT loading="lazy": the explorer measures its panel rects with
          // requestAnimationFrame/ResizeObserver at mount, which stalls if the
          // frame is deferred off-screen — it must lay out eagerly to initialise.
        />
      )}
    </div>
  )
}

export function DocsApp() {
  const [guide, setGuide] = useState<Guide>(GUIDES[0])
  return (
    <div style={styles.root}>
      <aside style={styles.sidebar}>
        <div style={styles.brand}>
          <span style={styles.logoDot} />
          <span style={styles.brandText}>SpyDE Docs</span>
        </div>
        <div style={styles.navLabel}>Guides</div>
        {GUIDES.map((g) => (
          <button
            key={g.id}
            data-testid={`docs-nav-${g.id}`}
            onClick={() => setGuide(g)}
            style={{
              ...styles.navItem,
              background: g.id === guide.id ? 'rgba(137,180,250,0.14)' : 'transparent',
              color: g.id === guide.id ? '#cdd6f4' : '#a6adc8',
            }}
          >
            {g.title}
          </button>
        ))}
      </aside>

      <main style={styles.main}>
        <article style={styles.article}>
          <h1 style={styles.h1}>{guide.title}</h1>
          <p style={styles.summary}>{guide.summary}</p>

          {guide.steps.map((step, i) => (
            <section key={i} data-testid={`docs-step-${i}`} style={styles.step}>
              <div style={styles.stepHead}>
                {step.anchor !== null && <span style={styles.stepNum}>{i + 1}</span>}
                <h2 style={styles.h2}>{step.title}</h2>
              </div>
              <div style={styles.stepBody}>
                <Markdown
                  text={step.body}
                  styles={{ paragraph: styles.p, callout: styles.callout }}
                />
              </div>
              {step.embed ? (
                <InteractiveEmbed guideId={guide.id} embed={step.embed} title={step.title} />
              ) : (
                step.image && (
                  <img
                    src={`./media/${guide.id}/${step.image}`}
                    alt={step.title}
                    style={styles.shot}
                    // Screenshots are optional; hide the broken-image icon if absent.
                    onError={(e) => { (e.currentTarget.style.display = 'none') }}
                  />
                )
              )}
            </section>
          ))}
        </article>
      </main>
    </div>
  )
}

const ACCENT = '#89b4fa'
const styles: Record<string, React.CSSProperties> = {
  root: {
    display: 'flex', minHeight: '100vh', margin: 0,
    background: '#11111b', color: '#cdd6f4',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
  },
  sidebar: {
    width: 240, flexShrink: 0, borderRight: '1px solid #1e1e2e',
    padding: '18px 12px', position: 'sticky', top: 0, height: '100vh',
    boxSizing: 'border-box', background: '#181825',
  },
  brand: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 22, paddingLeft: 6 },
  logoDot: {
    width: 10, height: 10, borderRadius: '50%',
    background: 'linear-gradient(135deg, #89b4fa, #cba6f7)',
    boxShadow: '0 0 8px rgba(137,180,250,0.6)',
  },
  brandText: { fontSize: 15, fontWeight: 700, letterSpacing: 0.3 },
  navLabel: {
    fontSize: 10.5, color: '#6c7086', letterSpacing: 0.7, textTransform: 'uppercase',
    padding: '4px 8px',
  },
  navItem: {
    display: 'block', width: '100%', textAlign: 'left', border: 'none',
    borderRadius: 6, padding: '8px 10px', cursor: 'pointer', fontSize: 13.5,
    marginBottom: 2,
  },
  main: { flex: 1, display: 'flex', justifyContent: 'center', padding: '40px 24px' },
  article: { width: '100%', maxWidth: 760 },
  h1: { fontSize: 30, fontWeight: 700, margin: '0 0 8px' },
  summary: { fontSize: 15, color: '#a6adc8', lineHeight: 1.5, margin: '0 0 32px' },
  step: { marginBottom: 34, paddingBottom: 24, borderBottom: '1px solid #1e1e2e' },
  stepHead: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 },
  stepNum: {
    flexShrink: 0, width: 26, height: 26, borderRadius: '50%',
    background: ACCENT, color: '#11111b', fontWeight: 700, fontSize: 13,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  h2: { fontSize: 19, fontWeight: 600, margin: 0 },
  stepBody: { fontSize: 14.5, lineHeight: 1.6, color: '#bac2de' },
  p: { margin: '8px 0' },
  callout: {
    margin: '12px 0', padding: '12px 14px', borderRadius: 8,
    background: 'rgba(137,180,250,0.10)', borderLeft: `3px solid ${ACCENT}`,
    color: '#cdd6f4',
  },
  shot: {
    display: 'block', width: '100%', marginTop: 16, borderRadius: 8,
    border: '1px solid #313244',
  },
  // Interactive embed: a dark-themed well with a "try it" badge above a
  // sandboxed iframe that fills the article width. A fixed-ish height (via
  // aspect-ratio, clamped) keeps it from collapsing before the embed lays out.
  embedWrap: {
    marginTop: 16, borderRadius: 8, border: '1px solid #313244',
    background: '#181825', overflow: 'hidden',
  },
  embedBadge: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '7px 12px', fontSize: 11, fontWeight: 600, letterSpacing: 0.3,
    color: '#89b4fa', background: 'rgba(137,180,250,0.10)',
    borderBottom: '1px solid #1e1e2e', textTransform: 'uppercase',
  },
  embedDot: {
    width: 7, height: 7, borderRadius: '50%', background: '#a6e3a1',
    boxShadow: '0 0 5px rgba(166,227,161,0.8)',
  },
  embedFrame: {
    display: 'block', width: '100%', height: 520, border: 'none',
    background: '#1e1e2e',
  },
}
