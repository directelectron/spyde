/**
 * DocsApp.tsx — the SpyDE docs website.
 *
 * Two things live here, both imported from the repo root rather than copied:
 *
 * - **Guides** — the SAME guides the in-app coachmark tour uses (@guides), so
 *   the website and the app never drift. Each is a numbered, scrollable
 *   walkthrough; steps with a screenshot show it, matching what the in-app tour
 *   spotlights live.
 * - **Reports** — published SpyDE reports (@reports): whole analyses of real
 *   datasets, exported by the app's own interactive HTML export and hosted here
 *   verbatim. The site frames them, it does not re-render them.
 */
import React, { useEffect, useState } from 'react'
import { GUIDES, type Guide } from '@guides/index'
import { Markdown } from '@guides/markdown'
import { REPORTS, type Report } from '@reports/index'

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

/**
 * ReportView — a published report, framed.
 *
 * The report file is a COMPLETE self-contained page (its own article CSS, its
 * own interactive panels), so it gets an iframe of its own rather than being
 * pulled apart and re-styled: whatever the app exported is what the reader sees.
 * Above it sits the catalogue entry — summary, facts, source, and the commands
 * that rebuild it.
 *
 * Same graceful degradation as the guide embeds: a report .html can be large
 * enough that it is generated rather than committed, so HEAD-probe it first and
 * say "not built" instead of mounting a broken frame.
 */
function ReportView({ report }: { report: Report }) {
  const src = `./media/reports/${report.file}`
  const [state, setState] = useState<'checking' | 'ok' | 'missing'>('checking')
  useEffect(() => {
    let live = true
    setState('checking')
    fetch(src, { method: 'HEAD' })
      .then((r) => { if (live) setState(r.ok ? 'ok' : 'missing') })
      .catch(() => { if (live) setState('missing') })
    return () => { live = false }
  }, [src])

  return (
    <article style={styles.article} data-testid={`docs-report-${report.id}`}>
      <h1 style={styles.h1}>{report.title}</h1>
      <p style={styles.summary}>{report.summary}</p>

      <div style={styles.facts}>
        {report.facts.map((f) => (
          <div key={f.label} style={styles.fact}>
            <span style={styles.factLabel}>{f.label}</span>
            <span style={styles.factValue}>{f.value}</span>
          </div>
        ))}
      </div>

      {report.source && (
        <p style={styles.sourceLine}>
          Data:{' '}
          {report.source.url ? (
            <a href={report.source.url} target="_blank" rel="noreferrer noopener"
               style={styles.sourceLink}>
              {report.source.label} ↗
            </a>
          ) : report.source.label}
        </p>
      )}

      {state === 'missing' ? (
        <div style={styles.notBuilt} data-testid="docs-report-missing">
          <strong>Not in this checkout.</strong> The exported report file is
          missing, so there is nothing to show. Rebuild it with:
          {report.build && (
            <pre style={styles.buildPre}>{report.build.join('\n')}</pre>
          )}
        </div>
      ) : (
        <div style={styles.reportWrap}>
          <div style={styles.embedBadge}>
            <span style={styles.embedDot} />
            full report — scroll inside
          </div>
          {state === 'ok' && (
            <iframe
              src={src}
              title={report.title}
              // allow-scripts only: the report's own interactive panels run, but
              // it stays cross-origin isolated from the docs site.
              sandbox="allow-scripts"
              style={styles.reportFrame}
            />
          )}
        </div>
      )}

      {state !== 'missing' && (
        <p style={styles.openWhole}>
          <a href={src} target="_blank" rel="noreferrer noopener"
             style={styles.sourceLink} data-testid="docs-report-open">
            Open the report on its own ↗
          </a>
        </p>
      )}
    </article>
  )
}

type Selection =
  | { kind: 'guide'; guide: Guide }
  | { kind: 'report'; report: Report }

export function DocsApp() {
  const [sel, setSel] = useState<Selection>({ kind: 'guide', guide: GUIDES[0] })
  const guide = sel.kind === 'guide' ? sel.guide : null
  const activeId = sel.kind === 'guide' ? sel.guide.id : sel.report.id
  const navStyle = (id: string) => ({
    ...styles.navItem,
    background: id === activeId ? 'rgba(137,180,250,0.14)' : 'transparent',
    color: id === activeId ? '#cdd6f4' : '#a6adc8',
  })
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
            onClick={() => setSel({ kind: 'guide', guide: g })}
            style={navStyle(g.id)}
          >
            {g.title}
          </button>
        ))}
        {REPORTS.length > 0 && (
          <>
            <div style={{ ...styles.navLabel, marginTop: 18 }}>Reports</div>
            {REPORTS.map((r) => (
              <button
                key={r.id}
                data-testid={`docs-nav-${r.id}`}
                onClick={() => setSel({ kind: 'report', report: r })}
                style={navStyle(r.id)}
              >
                {r.title}
              </button>
            ))}
          </>
        )}
      </aside>

      <main style={styles.main}>
        {guide === null ? <ReportView report={(sel as { report: Report }).report} /> : (
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

          {/* "More information" — the same `guide.info` the in-app tour shows as
              its final step and Help → <technique> → Info… shows in a dialog.
              One source, three renderings. */}
          {guide.info && (
            <section data-testid="docs-more-info" style={styles.step}>
              <div style={styles.stepHead}>
                <h2 style={styles.h2}>More information</h2>
              </div>
              <div style={styles.stepBody}>
                <Markdown
                  text={guide.info.blurb}
                  styles={{ paragraph: styles.p, callout: styles.callout }}
                />
              </div>
              {guide.info.links.length > 0 && (
                <div style={styles.links}>
                  <div style={styles.linksLabel}>Further reading</div>
                  {guide.info.links.map((l) => (
                    <a
                      key={l.url}
                      href={l.url}
                      target="_blank"
                      rel="noreferrer noopener"
                      style={styles.linkCard}
                    >
                      <span style={styles.linkLabel}>{l.label} ↗</span>
                      {l.note && <span style={styles.linkNote}>{l.note}</span>}
                    </a>
                  ))}
                </div>
              )}
            </section>
          )}
        </article>
        )}
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
  // Report view: a facts strip, then the exported report in its own tall frame.
  // The frame is deliberately large — a report is a document, not a thumbnail —
  // and scrolls internally so the reader never loses the docs chrome.
  facts: {
    display: 'flex', flexWrap: 'wrap', gap: 10, margin: '0 0 14px',
  },
  fact: {
    display: 'flex', flexDirection: 'column', gap: 2,
    background: 'rgba(137,180,250,0.08)', border: '1px solid #313244',
    borderRadius: 8, padding: '8px 12px', minWidth: 120,
  },
  factLabel: {
    fontSize: 10.5, letterSpacing: 0.6, textTransform: 'uppercase',
    color: '#6c7086', fontWeight: 700,
  },
  factValue: { fontSize: 13.5, color: '#cdd6f4' },
  sourceLine: { fontSize: 12.5, color: '#7f849c', margin: '0 0 18px' },
  sourceLink: { color: ACCENT, textDecoration: 'none' },
  reportWrap: {
    borderRadius: 8, border: '1px solid #313244', background: '#181825',
    overflow: 'hidden',
  },
  reportFrame: {
    display: 'block', width: '100%', height: '78vh', minHeight: 560,
    border: 'none', background: '#ffffff',
  },
  openWhole: { fontSize: 13, margin: '12px 0 0' },
  notBuilt: {
    borderRadius: 8, border: '1px solid #313244',
    background: 'rgba(249,226,175,0.08)', padding: '14px 16px',
    fontSize: 13.5, color: '#bac2de', lineHeight: 1.6,
  },
  buildPre: {
    margin: '10px 0 0', padding: '10px 12px', borderRadius: 6,
    background: '#11111b', border: '1px solid #313244', color: '#a6adc8',
    fontSize: 12.5, overflowX: 'auto',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
  },
  links: { marginTop: 18, display: 'flex', flexDirection: 'column', gap: 8 },
  linksLabel: {
    fontSize: 11, fontWeight: 700, letterSpacing: 0.7, color: '#6c7086',
    textTransform: 'uppercase', marginBottom: 2,
  },
  linkCard: {
    display: 'block', textDecoration: 'none',
    background: 'rgba(137,180,250,0.08)', border: '1px solid #313244',
    borderRadius: 8, padding: '10px 12px',
  },
  linkLabel: { display: 'block', fontSize: 14, color: ACCENT, fontWeight: 500 },
  linkNote: { display: 'block', fontSize: 12.5, color: '#7f849c', marginTop: 3, lineHeight: 1.45 },
}
