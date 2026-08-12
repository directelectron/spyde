/**
 * Status.tsx — the camera status board.
 *
 * The design problem here is not layout, it is honesty. This server answers 7
 * of the 22 properties the board wants, so most cards have nothing behind them.
 * A board that rendered those as green would be worse than no board: it would
 * be an engineer's checklist that lies.
 *
 * So the headline counts only what is actually being reported, and says so:
 * "9 of 14 checks reporting". Unreported and unjudged cards are visibly
 * inert — muted, dashed, and grouped after the live ones — and they carry the
 * reason, because "not reported by this server" and "no threshold agreed yet"
 * send an engineer to two completely different places.
 */
import React from 'react'
import { C, FONT_MONO, stateOf } from '../theme'
import { Btn, Pill } from '../ui'

export interface StatusCard {
  key: string; label: string; group: string
  state: string; detail: string
  readings: Record<string, unknown>
  missing: string[]
}

export interface StatusSummary {
  overall: string
  counts: Record<string, number>
  reporting: number
  total: number
}

const LIVE = new Set(['ok', 'warn', 'bad'])

const HEADLINE: Record<string, string> = {
  ok: 'All reporting checks pass',
  warn: 'Attention needed',
  bad: 'Fault',
  unreported: 'Nothing to report',
}

export function StatusMode({ cards, summary, onRefresh }: {
  cards: StatusCard[]; summary: StatusSummary | null; onRefresh: () => void
}) {
  if (!summary) {
    return (
      <div style={{ padding: 32, color: C.textMuted, fontSize: 13 }}>
        Reading camera status…
      </div>
    )
  }

  const groups = Array.from(new Set(cards.map((c) => c.group)))
  const silent = cards.filter((c) => !LIVE.has(c.state))

  return (
    <div style={{ padding: '20px 24px', overflowY: 'auto', height: '100%' }}>
      <Headline summary={summary} onRefresh={onRefresh} />

      {groups.map((g) => {
        const inGroup = cards.filter((c) => c.group === g)
        // Live cards first: what the board can actually tell you should not be
        // buried among the things it cannot.
        const ordered = [...inGroup.filter((c) => LIVE.has(c.state)),
                         ...inGroup.filter((c) => !LIVE.has(c.state))]
        return (
          <div key={g} style={{ marginBottom: 22 }}>
            <h3 style={{
              margin: '0 0 9px', fontSize: 10.5, fontWeight: 700,
              letterSpacing: '.09em', textTransform: 'uppercase', color: C.textMuted,
            }}>{g}</h3>
            <div style={{
              display: 'grid', gap: 9,
              gridTemplateColumns: 'repeat(auto-fill, minmax(268px, 1fr))',
            }}>
              {ordered.map((c) => <Card key={c.key} card={c} />)}
            </div>
          </div>
        )
      })}

      {silent.length > 0 && <Footnote silent={silent} />}
    </div>
  )
}

function Headline({ summary, onRefresh }: { summary: StatusSummary; onRefresh: () => void }) {
  const s = stateOf(summary.overall)
  const gaps = summary.total - summary.reporting
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 16, marginBottom: 22,
      padding: '14px 16px', background: C.panel,
      border: `1px solid ${C.border}`, borderLeft: `3px solid ${s.fg}`,
      borderRadius: 8,
    }}>
      <div style={{ flex: 1 }}>
        <div data-testid="status-headline"
          style={{ fontSize: 15, fontWeight: 650, color: s.fg, marginBottom: 3 }}>
          {HEADLINE[summary.overall] ?? 'Status'}
        </div>
        {/* The coverage line is the point of the whole screen: it stops a board
            that can only see four things from reading as a healthy camera. */}
        <div data-testid="status-coverage" style={{ fontSize: 12, color: C.textMuted }}>
          {summary.reporting} of {summary.total} checks reporting
          {gaps > 0 && ` · ${gaps} unavailable on this server`}
        </div>
      </div>
      <Counts counts={summary.counts} />
      <Btn onClick={onRefresh} testId="status-refresh">Refresh</Btn>
    </div>
  )
}

function Counts({ counts }: { counts: Record<string, number> }) {
  const order = ['bad', 'warn', 'ok', 'no_criteria', 'unreported']
  return (
    <div style={{ display: 'flex', gap: 6 }}>
      {order.filter((k) => counts[k]).map((k) => (
        <Pill key={k} state={k}>{counts[k]} {stateOf(k).label}</Pill>
      ))}
    </div>
  )
}

function Card({ card }: { card: StatusCard }) {
  const s = stateOf(card.state)
  const live = LIVE.has(card.state)
  return (
    <div
      data-testid={`status-card-${card.key}`}
      data-state={card.state}
      style={{
        padding: '11px 13px', background: live ? C.panel : 'transparent',
        border: `1px ${live ? 'solid' : 'dashed'} ${live ? C.border : C.borderStrong}`,
        borderRadius: 7,
        // Inert cards recede. They are still readable — an engineer needs to
        // know the check EXISTS and is dark — but they must not compete with
        // the ones carrying real information.
        opacity: live ? 1 : 0.72,
      }}
    >
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        gap: 8, marginBottom: 5,
      }}>
        <span style={{ fontSize: 13, fontWeight: 550, color: live ? C.text : C.textMuted }}>
          {card.label}
        </span>
        <Pill state={card.state} />
      </div>
      <div style={{
        fontSize: 11.5, color: C.textMuted, fontFamily: FONT_MONO,
        lineHeight: 1.5, wordBreak: 'break-word',
      }}>
        {card.detail}
      </div>
    </div>
  )
}

/**
 * Names the properties the server did not answer.
 *
 * Deliberately concrete: a list of exact property names is what someone needs
 * to decide whether the server is old, the channel is down, or the check is
 * pointed at the wrong name.
 */
function Footnote({ silent }: { silent: StatusCard[] }) {
  const missing = Array.from(new Set(silent.flatMap((c) => c.missing)))
  const unjudged = silent.filter((c) => c.state === 'no_criteria')
  return (
    <div style={{
      marginTop: 8, padding: '13px 15px', background: C.bgSunken,
      border: `1px solid ${C.border}`, borderRadius: 7,
      fontSize: 12, color: C.textMuted, lineHeight: 1.65,
    }}>
      {missing.length > 0 && (
        <div style={{ marginBottom: unjudged.length ? 9 : 0 }}>
          <strong style={{ color: C.textDim, fontWeight: 600 }}>
            Not exposed by this server ({missing.length})
          </strong>
          <div style={{ marginTop: 4, fontFamily: FONT_MONO, fontSize: 11 }}>
            {missing.join(' · ')}
          </div>
        </div>
      )}
      {unjudged.length > 0 && (
        <div>
          <strong style={{ color: C.textDim, fontWeight: 600 }}>
            Reported but not judged ({unjudged.length})
          </strong>
          <div style={{ marginTop: 4 }}>
            {unjudged.map((c) => c.label).join(', ')} — no threshold has been
            agreed, so these are shown as readings and are not counted as passing.
          </div>
        </div>
      )}
    </div>
  )
}
