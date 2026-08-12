/**
 * Status.tsx — the camera status board.
 *
 * A go/no-go board you check before committing microscope time. Everything
 * should read green; the design's whole job is to make the one thing that is
 * NOT read instantly.
 *
 * One card per SUBSYSTEM, each with a headline value, its supporting readings,
 * and the property family it came from — because "amber" is useless if you
 * cannot tell what to go and fix. When something is wrong the card grows a
 * `fix` line saying what to do about it.
 *
 * Against this server most cards have nothing behind them: it answers 7 of the
 * 22 properties the board wants. A board that drew those green would be an
 * engineer's checklist that lies, so the banner reports coverage and inert
 * cards are visibly dashed and dimmed.
 */
import React from 'react'
import { C, FONT_MONO, stateOf } from '../theme'
import { Btn } from '../ui'

export interface StatusRow { label: string; value: string; state: string }

export interface StatusCard {
  key: string; title: string; source: string
  state: string; big: string; big_tone?: string
  rows: StatusRow[]; chips: string[]; fix: string; missing: string[]
}

export interface StatusSummary {
  overall: string
  counts: Record<string, number>
  reporting: number
  total: number
}

const LIVE = new Set(['ok', 'warn', 'bad'])

const HEADLINE: Record<string, string> = {
  ok: 'Ready to acquire',
  warn: 'Attention needed',
  bad: 'Not fit to use',
  unreported: 'Nothing to report',
}

export function StatusMode({ cards, summary, onRefresh }: {
  cards: StatusCard[]; summary: StatusSummary | null; onRefresh: () => void
}) {
  if (!summary) {
    return <div style={{ padding: 32, color: C.textMuted, fontSize: 13 }}>
      Reading camera status…
    </div>
  }

  // Live cards first. What the board can actually tell you should not be
  // buried among the things it cannot.
  const live = cards.filter((c) => LIVE.has(c.state))
  const inert = cards.filter((c) => !LIVE.has(c.state))

  return (
    <div style={{ padding: '16px 18px', overflowY: 'auto', height: '100%' }}>
      <Banner summary={summary} onRefresh={onRefresh} />
      <Grid cards={live} />
      {inert.length > 0 && (
        <>
          <SectionRule>Not reporting on this server</SectionRule>
          <Grid cards={inert} />
        </>
      )}
    </div>
  )
}

function Grid({ cards }: { cards: StatusCard[] }) {
  return (
    <div style={{
      display: 'grid', gap: 12, marginBottom: 16,
      gridTemplateColumns: 'repeat(auto-fill, minmax(232px, 1fr))',
      alignItems: 'start',
    }}>
      {cards.map((c) => <CardView key={c.key} card={c} />)}
    </div>
  )
}

/** A section label with a rule running off to the right. */
function SectionRule({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8, margin: '4px 0 10px',
      font: `600 9.5px/1 ${FONT_MONO}`, letterSpacing: '.11em',
      textTransform: 'uppercase', color: C.textMuted,
    }}>
      {children}
      <span style={{ flex: 1, height: 1, background: C.border }} />
    </div>
  )
}

function Banner({ summary, onRefresh }: {
  summary: StatusSummary; onRefresh: () => void
}) {
  const s = stateOf(summary.overall)
  const gaps = summary.total - summary.reporting
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 11, marginBottom: 16,
      padding: '11px 14px', borderRadius: 8,
      background: s.bg, border: `1px solid ${s.fg}55`,
    }}>
      <Dot state={summary.overall} size={9} />
      <strong data-testid="status-headline" style={{ fontSize: 13.5, color: s.fg }}>
        {HEADLINE[summary.overall] ?? 'Status'}
      </strong>
      {/* The coverage line is the point of the whole screen: it stops a board
          that can only see two things from reading as a healthy camera. */}
      <span data-testid="status-coverage" style={{ fontSize: 12, color: C.textMuted }}>
        {summary.reporting} of {summary.total} checks reporting
        {gaps > 0 && ` · ${gaps} unavailable on this server`}
      </span>
      <span style={{ flex: 1 }} />
      <Btn onClick={onRefresh} testId="status-refresh">Re-check all</Btn>
    </div>
  )
}

/** Disc for pass, diamond for everything else — readable without colour vision. */
function Dot({ state, size = 7 }: { state: string; size?: number }) {
  return <span style={{
    width: size, height: size, flex: 'none', background: stateOf(state).fg,
    borderRadius: state === 'ok' ? 999 : 1,
    transform: state === 'ok' ? 'none' : 'rotate(45deg)',
  }} />
}

function CardView({ card }: { card: StatusCard }) {
  const s = stateOf(card.state)
  const live = LIVE.has(card.state)
  const tone = card.big_tone === 'cryo' ? C.cryo : s.fg

  return (
    <div
      data-testid={`status-card-${card.key}`}
      data-state={card.state}
      style={{
        padding: '10px 12px', borderRadius: 8,
        background: live ? C.panel : 'transparent',
        border: `1px ${live ? 'solid' : 'dashed'} ${live ? C.border : C.borderStrong}`,
        // The state reads from the top edge before you get to the words.
        borderTop: `2px solid ${s.fg}`,
        opacity: live ? 1 : 0.75,
      }}
    >
      <div style={{
        display: 'flex', alignItems: 'center', gap: 7, marginBottom: 7,
        fontSize: 11.5, fontWeight: 600,
      }}>
        <Dot state={card.state} />
        <span style={{ color: live ? C.text : C.textMuted }}>{card.title}</span>
        <span style={{ flex: 1 }} />
        {/* Which property family to go and look at. */}
        <span title={card.source} style={{
          font: `400 9.5px/1 ${FONT_MONO}`, color: C.textMuted,
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          maxWidth: 104,
        }}>{card.source}</span>
      </div>

      <div style={{
        font: `16px/1.25 ${FONT_MONO}`, fontVariantNumeric: 'tabular-nums',
        marginBottom: card.rows.length ? 6 : 0, color: tone,
      }}>{card.big}</div>

      {card.rows.map((r, i) => (
        <div key={i} style={{
          display: 'flex', justifyContent: 'space-between', gap: 8, padding: '2.5px 0',
        }}>
          <span style={{ fontSize: 11.5, color: C.textMuted }}>{r.label}</span>
          <span style={{
            font: `11.5px ${FONT_MONO}`, fontVariantNumeric: 'tabular-nums',
            color: r.state === 'ok' ? C.textDim : stateOf(r.state).fg,
            textAlign: 'right', wordBreak: 'break-word',
          }}>{r.value}</span>
        </div>
      ))}

      {card.chips.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 6 }}>
          {card.chips.map((c, i) => (
            <span key={i} style={{
              padding: '3px 8px', borderRadius: 999, fontSize: 11,
              border: `1px solid ${C.borderStrong}`, color: C.textDim,
            }}>{c}</span>
          ))}
        </div>
      )}

      {/* What to DO. Only present when something is wrong — a fix line on a
          healthy card is noise that trains people to stop reading them. */}
      {card.fix && (
        <div style={{
          marginTop: 8, paddingTop: 7, borderTop: `1px solid ${C.border}`,
          fontSize: 11, lineHeight: 1.45, color: C.textMuted,
        }}>{card.fix}</div>
      )}
    </div>
  )
}
