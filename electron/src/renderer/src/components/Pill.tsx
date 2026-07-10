/**
 * Pill.tsx — the one draggable "pill" (rounded button-like chip) used across the
 * app: the window-header breadcrumb, minimized-window chips, workflow nodes, and
 * console tokens. One component so drag payloads, theming, and hover/droppable
 * states are consistent everywhere.
 *
 * A window pill is the SINGLE drag source for its dataset: on dragStart it stamps
 * ALL relevant MIME payloads (window / navigator / signal-ref), so the DROP TARGET
 * decides what happens — MDI → new navigator/dataset, console → bind signal,
 * navigator titlebar → add as navigator. Titlebar empty space still moves the
 * window (the pill's drag is HTML5 drag, separate from the window-move gesture).
 */
import React, { useState } from 'react'
import {
  WINDOW_DRAG_MIME, NAVIGATOR_DRAG_MIME, SIGNAL_REF_DRAG_MIME, CONSOLE_VAR_DRAG_MIME,
  WORKFLOW_NODE_DRAG_MIME,
} from '../kernel/dnd'

export interface PillSegment {
  text: string
  /** accent = filled accent bg (the dataset Name); muted = subdued middle/last. */
  tone?: 'accent' | 'primary' | 'muted'
  /** A tight kind prefix (e.g. "S-"/"N-") — rendered snug against the NEXT
   *  segment with no `|` separator, so it reads as `S-name` not `S- | name`. */
  prefix?: boolean
  /** double-click this segment (e.g. the Name → rename). */
  onDoubleClick?: (e: React.MouseEvent) => void
  testid?: string
}

/** What a window pill carries so any drop target can consume it. */
export interface WindowPillPayload {
  windowId: number
  isNavigator: boolean
  /** the displayed navigator's name (navigator windows only) */
  navName?: string | null
}

export interface PillProps {
  /** breadcrumb segments rendered `a | b | c`. */
  segments: PillSegment[]
  /** window payload → drag stamps window/navigator/signal-ref MIMEs. */
  window?: WindowPillPayload
  /** console-var payload → drag stamps CONSOLE_VAR_DRAG_MIME (result chips). */
  consoleVar?: { name: string }
  /** workflow-node payload → drag stamps a node ref (see WORKFLOW_NODE_DRAG_MIME). */
  workflowNode?: { windowId: number; signalId: number; name: string }
  active?: boolean
  size?: 'sm' | 'md'
  title?: string
  onClick?: (e: React.MouseEvent) => void
  /** e.g. stopPropagation so grabbing the pill in a titlebar doesn't also start
   *  a window-move gesture. */
  onPointerDown?: (e: React.PointerEvent) => void
  /** extra content rendered after the segments (e.g. minimize/close affordance). */
  trailing?: React.ReactNode
  testid?: string
  style?: React.CSSProperties
}

const C = {
  bg: '#1e1e2e', bgHover: '#2a2a3e', border: '#313244', borderHover: '#585b70',
  accent: '#89b4fa', accentText: '#11111b', primary: '#cdd6f4', muted: '#a6adc8',
  sep: '#585b70',
}

export function Pill({
  segments, window: win, consoleVar, workflowNode, active = false,
  size = 'md', title, onClick, onPointerDown, trailing, testid, style,
}: PillProps) {
  const [hover, setHover] = useState(false)

  const onDragStart = (e: React.DragEvent) => {
    const dt = e.dataTransfer
    dt.effectAllowed = 'copy'
    if (win) {
      // Whole pill = the dataset's drag source: stamp every payload a target
      // might want. The drop target picks the MIME it understands.
      dt.setData(WINDOW_DRAG_MIME, String(win.windowId))
      dt.setData(SIGNAL_REF_DRAG_MIME, JSON.stringify({ windowId: win.windowId }))
      if (win.isNavigator) {
        dt.setData(NAVIGATOR_DRAG_MIME, JSON.stringify({
          windowId: win.windowId, name: win.navName || 'base',
        }))
      }
    }
    if (consoleVar) dt.setData(CONSOLE_VAR_DRAG_MIME, JSON.stringify({ name: consoleVar.name }))
    if (workflowNode) dt.setData(WORKFLOW_NODE_DRAG_MIME, JSON.stringify(workflowNode))
  }

  const pad = size === 'sm' ? '2px 8px' : '3px 10px'
  const fontSize = size === 'sm' ? 10 : 11
  const bg = active ? C.accent : hover ? C.bgHover : C.bg
  const border = active ? C.accent : hover ? C.borderHover : C.border

  return (
    <span
      data-testid={testid}
      draggable
      onDragStart={onDragStart}
      onClick={onClick}
      onPointerDown={onPointerDown}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      title={title}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 0,
        background: bg, border: `1px solid ${border}`, borderRadius: 10,
        padding: pad, fontSize, lineHeight: 1.3, cursor: 'grab',
        userSelect: 'none', maxWidth: '100%', minWidth: 0,
        transition: 'background 90ms, border-color 90ms',
        ...style,
      }}
    >
      {segments.map((s, i) => {
        const tone = s.tone ?? (i === 0 ? 'accent' : 'muted')
        const color = active
          ? (tone === 'accent' ? C.accentText : 'rgba(17,17,27,0.75)')
          : tone === 'accent' ? C.accent : tone === 'primary' ? C.primary : C.muted
        // A `|` separator sits between segments UNLESS this one or the previous
        // one is a tight kind prefix (S-/N-), which hugs the next segment.
        const prev = segments[i - 1]
        const sep = i > 0 && !s.prefix && !prev?.prefix
        // Spacing before this segment: none for a prefix's following name (tight),
        // a small gap otherwise.
        const ml = i === 0 ? 0 : s.prefix ? 5 : prev?.prefix ? 0 : 3
        return (
          <React.Fragment key={i}>
            {sep && (
              <span style={{
                color: active ? 'rgba(17,17,27,0.5)' : C.sep,
                fontSize: fontSize - 1, margin: '0 4px',
              }}>|</span>
            )}
            <span
              data-testid={s.testid}
              onDoubleClick={s.onDoubleClick
                ? (e) => { e.stopPropagation(); s.onDoubleClick!(e) }
                : undefined}
              style={{
                color,
                marginLeft: sep ? 0 : ml,
                fontWeight: tone === 'accent' ? 600 : 500,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                cursor: s.onDoubleClick ? 'text' : undefined,
                maxWidth: 220,
              }}
            >
              {s.text}
            </span>
          </React.Fragment>
        )
      })}
      {trailing}
    </span>
  )
}
