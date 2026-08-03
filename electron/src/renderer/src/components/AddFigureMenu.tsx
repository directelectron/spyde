/**
 * AddFigureMenu.tsx — the ＋ chrome button's window picker: "add another figure
 * to this cell", i.e. build a subplot grid by CLICKING rather than by dragging.
 *
 * WHY. Combining two figures into one multi-panel figure was reachable only by
 * dragging a window pill onto an existing figure cell and releasing inside a
 * 28%-wide edge strip. That gesture is fiddly to aim, the zone overlay only
 * exists mid-drag, and on a split block it did not exist at all — so the feature
 * read as missing. This is the same backend verb (`repfig_compose` with a
 * `tile-*` mode) behind a target you cannot miss.
 *
 * The menu lists every open window; picking one tiles it in on the chosen side.
 * Direction defaults to → (beside), matching how a 2-up slide is usually read,
 * with the other three available from the same row.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { AnchoredMenu } from './AnchoredMenu'
import type { ComposeMode } from './composeDrop'

const DIRECTIONS: { mode: ComposeMode; glyph: string; label: string }[] = [
  { mode: 'tile-right', glyph: '→', label: 'Beside (right)' },
  { mode: 'tile-left', glyph: '←', label: 'Beside (left)' },
  { mode: 'tile-down', glyph: '↓', label: 'Below' },
  { mode: 'tile-up', glyph: '↑', label: 'Above' },
]

/**
 * Renders the picker panel anchored to this cell's ＋ button.
 *
 * `onFill` (optional) is used INSTEAD of a tile when the cell has no figure yet
 * — a split block's empty figure side fills rather than tiles.
 */
export function AddFigureMenu({ cellId, hasFigure, onClose, onFill }: {
  cellId: string
  /** False → the cell has no figure yet, so a pick FILLS it (no grid). */
  hasFigure: boolean
  onClose: () => void
  onFill?: (windowId: number) => void
}) {
  const { state, sendAction } = useSpyDE()
  const [dir, setDir] = React.useState<ComposeMode>('tile-right')

  // Anchor on the ＋ button itself. It lives inside CellChrome (which owns the
  // markup), so it's resolved from the DOM rather than threaded through as a
  // ref — the chrome is only mounted while hovered, and the menu is only ever
  // opened BY that button, so it is always present here.
  const [anchorEl, setAnchorEl] = React.useState<HTMLElement | null>(null)
  React.useEffect(() => {
    setAnchorEl(document.querySelector<HTMLElement>(`[data-testid="cell-add-figure-${cellId}"]`))
  }, [cellId])

  // Loading the same dataset twice gives several windows the SAME title, so the
  // list read as four identical rows with nothing to choose between. Disambiguate
  // only the ones that actually collide (an unambiguous name stays clean).
  const windows = React.useMemo(() => {
    const list = Array.from(state.windows.values()).filter(w => w.visible !== false)
    const seen = new Map<string, number>()
    for (const w of list) {
      const k = `${w.isNavigator ? 'N' : 'S'}:${w.title}`
      seen.set(k, (seen.get(k) ?? 0) + 1)
    }
    const used = new Map<string, number>()
    return list.map(w => {
      const k = `${w.isNavigator ? 'N' : 'S'}:${w.title}`
      if ((seen.get(k) ?? 0) < 2) return { ...w, label: w.title }
      const n = (used.get(k) ?? 0) + 1
      used.set(k, n)
      return { ...w, label: `${w.title} (${n})` }
    })
  }, [state.windows])

  const pick = (windowId: number) => {
    onClose()
    if (!hasFigure && onFill) { onFill(windowId); return }
    sendAction('repfig_compose', {
      cell_id: cellId, mode: dir, source_window_id: windowId,
    })
  }

  if (!anchorEl) return null

  return (
    <AnchoredMenu
      anchorEl={anchorEl}
      testid={`add-figure-menu-${cellId}`}
      onClose={onClose}
      align="right"
      minWidth={232}
    >
      <div style={styles.head}>
        {hasFigure ? 'Add a figure to this cell' : 'Choose a figure'}
      </div>

      {/* Where the new panel goes. Hidden when there's nothing to tile beside. */}
      {hasFigure && (
        <div style={styles.dirRow} data-testid={`add-figure-dirs-${cellId}`}>
          {DIRECTIONS.map(d => (
            <button
              key={d.mode}
              data-testid={`add-figure-dir-${d.mode}-${cellId}`}
              title={d.label}
              style={dir === d.mode ? styles.dirBtnActive : styles.dirBtn}
              onClick={() => setDir(d.mode)}
            >{d.glyph}</button>
          ))}
        </div>
      )}

      <div style={styles.list}>
        {windows.length === 0 && (
          <div style={styles.empty}>No open windows — open a dataset first.</div>
        )}
        {windows.map(w => (
          <button
            key={w.windowId}
            data-testid={`add-figure-win-${w.windowId}-${cellId}`}
            style={styles.item}
            title={w.title}
            onClick={() => pick(w.windowId)}
            onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(137,180,250,0.18)' }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'none' }}
          >
            <span style={styles.itemKind}>{w.isNavigator ? 'N' : 'S'}</span>
            <span style={styles.itemName}>{w.label}</span>
          </button>
        ))}
      </div>
    </AnchoredMenu>
  )
}

const styles: Record<string, React.CSSProperties> = {
  head: {
    fontSize: 11, fontWeight: 600, color: '#7f849c',
    textTransform: 'uppercase', letterSpacing: 0.4,
    padding: '7px 10px 5px',
  },
  dirRow: {
    display: 'flex', gap: 4, padding: '0 10px 7px',
    borderBottom: '1px solid #313244', marginBottom: 5,
  },
  dirBtn: {
    flex: 1, background: 'none', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 5,
    height: 24, fontSize: 13, cursor: 'pointer',
  },
  dirBtnActive: {
    flex: 1, background: '#89b4fa', color: '#11111b',
    border: '1px solid #89b4fa', borderRadius: 5,
    height: 24, fontSize: 13, fontWeight: 700, cursor: 'pointer',
  },
  list: { maxHeight: 260, overflowY: 'auto', paddingBottom: 5 },
  item: {
    display: 'flex', alignItems: 'center', gap: 8, width: '100%',
    background: 'none', border: 'none', color: '#cdd6f4',
    padding: '6px 10px', fontSize: 12.5, cursor: 'pointer', textAlign: 'left',
  },
  itemKind: {
    flex: '0 0 auto', fontSize: 10, fontWeight: 700, color: '#89b4fa',
    border: '1px solid #313244', borderRadius: 4, padding: '1px 5px',
  },
  itemName: {
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  empty: { padding: '8px 10px 10px', fontSize: 12, color: '#585b70' },
}
