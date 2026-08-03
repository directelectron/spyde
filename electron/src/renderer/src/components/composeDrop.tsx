/**
 * composeDrop.tsx — the 5-zone COMPOSE drop overlay, shared by every report cell
 * that hosts a live figure.
 *
 * Dropping a window/figure pill onto a figure that is ALREADY in the report is
 * how you build a multi-panel (subplot-grid) figure: hover an EDGE → "Tile ↑ ↓ ←
 * →" → the backend re-lays the cell out as an anyplotlib `subplots()` grid;
 * hover the CENTER → the richer "Overlay / Combine" prompt.
 *
 * This lived inside ReportFigureCell, so it only existed on a PLAIN figure cell.
 * A SPLIT block's figure side (the text-beside-a-figure slide layout, and the
 * shape most presentation slides actually use) had a bare drop handler that
 * REPLACED its figure instead — so on those slides "drop a second figure to
 * combine them" silently swapped the figure and the grid was unreachable. Both
 * cells now mount the same overlay from here.
 */
import React from 'react'
import type { RepfigSpec, RepfigPanel } from '../kernel/protocol'

// The compose modes the backend can return (subset of these per drop).
export type ComposeMode =
  | 'overlay' | 'callout'
  | 'tile-up' | 'tile-down' | 'tile-left' | 'tile-right'

// The five drop zones on a figure cell (or, on a multi-panel grid, within the
// hovered PANEL's cell rect).
export type Zone = 'center' | 'up' | 'down' | 'left' | 'right'

export const ZONE_TILE: Record<Exclude<Zone, 'center'>, ComposeMode> = {
  up: 'tile-up', down: 'tile-down', left: 'tile-left', right: 'tile-right',
}

// The currently-hovered zone PLUS which panel it's relative to (for a grid
// figure) and the panel's on-screen rect (fraction of the shield box, 0..1) so
// ComposeZones can position itself over just that cell. `panelId` is null for a
// single-panel figure (whole-box zones) and `panelRect` is the full box then.
export interface HoverZone {
  zone: Zone
  panelId: string | null
  panelLabel: string | null
  panelRect: { left: number; top: number; width: number; height: number }
}

export const FULL_RECT = { left: 0, top: 0, width: 1, height: 1 }

// A1, B2… panel labels from grid position: row-major letter per panel index.
export const PANEL_LETTERS = 'ABCDEFGHIJKLMNOP'
export function panelLabel(index: number): string {
  return `Panel ${PANEL_LETTERS[index] ?? String(index + 1)}`
}

// Map a cursor fraction (fx, fy) WITHIN a cell rect (0..1 local to that rect)
// to a Zone: a ~30%-wide edge strip on each side, center otherwise. Shared by
// the single-panel (whole box) and grid (per-panel cell) paths.
export function zoneFromLocalFraction(fx: number, fy: number): Zone {
  const edge = 0.3
  const dl = fx, dr = 1 - fx, dt = fy, db = 1 - fy
  const m = Math.min(dl, dr, dt, db)
  if (m > edge) return 'center'
  if (m === dl) return 'left'
  if (m === dr) return 'right'
  if (m === dt) return 'up'
  return 'down'
}

/**
 * Resolve the hovered zone for a drag event over a figure's shield box.
 *
 * On a multi-panel GRID figure this first resolves WHICH panel cell the cursor
 * is over (uniform rows/cols — report grids carry no width ratios), then
 * computes the zone from the LOCAL fraction within that cell. On a single-panel
 * figure it's the whole box (panelId null).
 */
export function hoverZoneAt(e: React.DragEvent, figure: RepfigSpec | undefined | null): HoverZone {
  const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
  const fx = (e.clientX - r.left) / Math.max(1, r.width)
  const fy = (e.clientY - r.top) / Math.max(1, r.height)

  const gridPanels: RepfigPanel[] = figure?.layout?.kind === 'grid' ? (figure.panels ?? []) : []
  if (gridPanels.length <= 1) {
    return { zone: zoneFromLocalFraction(fx, fy), panelId: null, panelLabel: null, panelRect: FULL_RECT }
  }
  const rows = Math.max(1, Number(figure?.layout?.rows) || 1)
  const cols = Math.max(1, Number(figure?.layout?.cols) || 1)
  const col = Math.min(cols - 1, Math.max(0, Math.floor(fx * cols)))
  const row = Math.min(rows - 1, Math.max(0, Math.floor(fy * rows)))
  let panel = gridPanels.find(p => p.grid_pos[0] === row && p.grid_pos[1] === col) ?? null
  if (!panel) {
    // Hole in a sparse grid — target the NEAREST occupied panel, but keep the
    // highlight on the hovered (empty) cell so the overlay tracks the cursor.
    let bestDist = Infinity
    for (const p of gridPanels) {
      const [pr, pc] = p.grid_pos
      const d = (pr - row) ** 2 + (pc - col) ** 2
      if (d < bestDist) { bestDist = d; panel = p }
    }
  }
  const idx = gridPanels.indexOf(panel as RepfigPanel)
  return {
    zone: zoneFromLocalFraction(fx * cols - col, fy * rows - row),
    panelId: panel ? panel.id : null,
    panelLabel: panel ? panelLabel(idx) : null,
    panelRect: { left: col / cols, top: row / rows, width: 1 / cols, height: 1 / rows },
  }
}

// ── 5-zone drop overlay ───────────────────────────────────────────────────────

/**
 * `panelRect` positions the zones overlay INSIDE the hovered grid cell (percent
 * of the shield box); defaults to the full box on a single-panel figure.
 * `panelLabel` (e.g. "Panel B"), when present, is appended to the tile labels
 * so a multi-panel drop reads as "Tile → of Panel B".
 */
export function ComposeZones({ active, cellId, panelRect, panelLabel: targetLabel, centerLabel }: {
  active: Zone
  cellId: string
  panelRect?: { left: number; top: number; width: number; height: number }
  panelLabel?: string | null
  /** Override the CENTER zone's label — a split block's centre REPLACES its
   *  figure rather than opening the overlay/combine prompt, and the zone has to
   *  say so or the drop reads as a silent no-op. */
  centerLabel?: string
}) {
  const zStyle = (z: Zone): React.CSSProperties => ({
    ...styles.zone,
    ...(active === z ? styles.zoneHot : {}),
  })
  const rootStyle: React.CSSProperties = panelRect
    ? {
        ...styles.zonesRoot,
        inset: 'auto',
        left: `${panelRect.left * 100}%`, top: `${panelRect.top * 100}%`,
        width: `${panelRect.width * 100}%`, height: `${panelRect.height * 100}%`,
        right: 'auto', bottom: 'auto',
      }
    : styles.zonesRoot
  const ofSuffix = targetLabel ? ` of ${targetLabel}` : ''
  // Only the cell under the cursor shows zones, so the bare spec testids
  // (figcell-zone-<zone>) are unambiguous; data-cell disambiguates in the DOM.
  return (
    <div style={rootStyle} data-testid="figcell-zones" data-cell={cellId}>
      {/* Edges first (thin strips), center last so it sits between them. */}
      <div data-testid="figcell-zone-up" style={{ ...zStyle('up'), ...styles.zoneUp }}>
        <span style={styles.zoneLabel}>{`Tile ↑${ofSuffix}`}</span>
      </div>
      <div data-testid="figcell-zone-down" style={{ ...zStyle('down'), ...styles.zoneDown }}>
        <span style={styles.zoneLabel}>{`Tile ↓${ofSuffix}`}</span>
      </div>
      <div data-testid="figcell-zone-left" style={{ ...zStyle('left'), ...styles.zoneLeft }}>
        <span style={styles.zoneLabel}>{`Tile ←${ofSuffix}`}</span>
      </div>
      <div data-testid="figcell-zone-right" style={{ ...zStyle('right'), ...styles.zoneRight }}>
        <span style={styles.zoneLabel}>{`Tile →${ofSuffix}`}</span>
      </div>
      <div data-testid="figcell-zone-center" style={{ ...zStyle('center'), ...styles.zoneCenter }}>
        <span style={styles.zoneLabel}>
          {centerLabel ?? (targetLabel ? `Combine${ofSuffix}` : 'Overlay / Combine')}
        </span>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  zonesRoot: {
    position: 'absolute', inset: 0, pointerEvents: 'none',
  },
  // The zones are a DROP TARGET, not decoration: at the old 0.35-alpha dashed
  // border over a bright figure they were near-invisible, which read as "it
  // flashed but I couldn't see it". Solid border, a dark scrim behind the label
  // so it survives on white data, and a much louder hot state.
  zone: {
    position: 'absolute', display: 'flex', alignItems: 'center',
    justifyContent: 'center',
    border: '1.5px solid rgba(137,180,250,0.55)',
    background: 'rgba(17,17,27,0.45)',
    boxSizing: 'border-box',
    transition: 'background 70ms, border-color 70ms',
  },
  zoneHot: {
    borderColor: '#89b4fa', borderWidth: 2,
    background: 'rgba(137,180,250,0.42)',
    boxShadow: 'inset 0 0 0 1px rgba(255,255,255,0.25)',
  },
  zoneUp: { left: '28%', right: '28%', top: 0, height: '28%' },
  zoneDown: { left: '28%', right: '28%', bottom: 0, height: '28%' },
  zoneLeft: { top: '28%', bottom: '28%', left: 0, width: '28%' },
  zoneRight: { top: '28%', bottom: '28%', right: 0, width: '28%' },
  zoneCenter: { left: '28%', right: '28%', top: '28%', bottom: '28%' },
  zoneLabel: {
    fontSize: 10.5, fontWeight: 700, color: '#ffffff',
    textShadow: '0 1px 3px rgba(0,0,0,0.95), 0 0 2px rgba(0,0,0,0.9)',
    textAlign: 'center', padding: '0 2px', lineHeight: 1.15,
  },
}
