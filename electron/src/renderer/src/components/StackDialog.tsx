/**
 * StackDialog.tsx — build an ordered stack of datasets to combine into one
 * higher-dimensional dataset (e.g. several 4D-STEM MRC scans → a 5D stack with
 * an extra leading index axis).
 *
 * Each chosen dataset is a rounded chip in a vertical "stack"; drag a chip to
 * reorder, × to remove, and the "Add datasets…" tile at the bottom appends more
 * (native multi-select picker). Confirming sends `open_stack` with the paths in
 * chip order — the backend (`Session.open_stack`) stacks them lazily, cropping to
 * a common shape if they differ.
 */
import React, { useState } from 'react'

const ACCENT = '#89b4fa'

function basename(p: string): string {
  const parts = p.split(/[/\\]/)
  return parts[parts.length - 1] || p
}

export function StackDialog({
  initialPaths = [],
  onConfirm,
  onCancel,
}: {
  initialPaths?: string[]
  onConfirm: (paths: string[]) => void
  onCancel: () => void
}) {
  const [paths, setPaths] = useState<string[]>(initialPaths)
  const [dragIdx, setDragIdx] = useState<number | null>(null)
  const [overIdx, setOverIdx] = useState<number | null>(null)

  const append = (picked: string[] | undefined): void => {
    if (picked && picked.length) {
      // Append in selection order; allow the same path twice (a user may
      // intentionally repeat) — they can remove duplicates with ×.
      setPaths((prev) => [...prev, ...picked])
    }
  }

  const addDatasets = async (): Promise<void> => {
    append(
      await window.electron.pickFiles({
        name: 'EM Data',
        extensions: ['hspy', 'zspy', 'mrc', 'tif', 'tiff', 'de5'],
      }),
    )
  }

  // .zspy / .zarr are folders, not files — a separate directory picker (the
  // native dialog can't select files and folders at once, esp. on Windows).
  const addFolders = async (): Promise<void> => {
    append(await window.electron.pickFolders())
  }

  const removeAt = (i: number): void =>
    setPaths((prev) => prev.filter((_, j) => j !== i))

  // Move the dragged chip to just before `target` (or end if target is null).
  const reorder = (from: number, target: number): void => {
    setPaths((prev) => {
      if (from === target) return prev
      const next = prev.slice()
      const [moved] = next.splice(from, 1)
      // After removing `from`, indices > from shift down by one.
      const insertAt = target > from ? target - 1 : target
      next.splice(insertAt, 0, moved)
      return next
    })
  }

  const canStack = paths.length >= 2

  return (
    <div style={styles.overlay} data-testid="stack-dialog">
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>Load Stack</h3>
        <p style={styles.sub}>
          Combine several datasets into one — they stack along a new index axis
          (top = 0). Drag to reorder. Mismatched shapes are cropped to the common
          size.
        </p>

        <div style={styles.stack}>
          {paths.map((p, i) => (
            <div
              key={`${p}#${i}`}
              data-testid={`stack-chip-${i}`}
              draggable
              onDragStart={() => setDragIdx(i)}
              onDragEnd={() => {
                setDragIdx(null)
                setOverIdx(null)
              }}
              onDragOver={(e) => {
                e.preventDefault()
                if (overIdx !== i) setOverIdx(i)
              }}
              onDrop={(e) => {
                e.preventDefault()
                if (dragIdx !== null) reorder(dragIdx, i)
                setDragIdx(null)
                setOverIdx(null)
              }}
              style={{
                ...styles.chip,
                ...(dragIdx === i ? styles.chipDragging : null),
                ...(overIdx === i && dragIdx !== null && dragIdx !== i
                  ? styles.chipDropTarget
                  : null),
              }}
              title={p}
            >
              <span style={styles.grip} aria-hidden>
                ⠿
              </span>
              <span style={styles.badge}>{i}</span>
              <span style={styles.chipName}>{basename(p)}</span>
              <button
                data-testid={`stack-remove-${i}`}
                style={styles.remove}
                onClick={() => removeAt(i)}
                title="Remove from stack"
              >
                ×
              </button>
            </div>
          ))}

          {/* Drop zone at the very end (reorder to last). */}
          {dragIdx !== null && (
            <div
              data-testid="stack-drop-end"
              onDragOver={(e) => {
                e.preventDefault()
                setOverIdx(paths.length)
              }}
              onDrop={(e) => {
                e.preventDefault()
                if (dragIdx !== null) reorder(dragIdx, paths.length)
                setDragIdx(null)
                setOverIdx(null)
              }}
              style={{
                ...styles.dropEnd,
                ...(overIdx === paths.length ? styles.chipDropTarget : null),
              }}
            />
          )}

          <button
            data-testid="stack-add"
            style={styles.addTile}
            onClick={addDatasets}
          >
            + Add datasets…
          </button>
          <button
            data-testid="stack-add-folders"
            style={styles.addTile}
            onClick={addFolders}
          >
            + Add .zspy/.zarr folders…
          </button>
        </div>

        <div style={styles.footer}>
          <button data-testid="stack-cancel" style={styles.cancel} onClick={onCancel}>
            Cancel
          </button>
          <button
            data-testid="stack-confirm"
            style={{ ...styles.confirm, opacity: canStack ? 1 : 0.5 }}
            disabled={!canStack}
            onClick={() => onConfirm(paths)}
            title={canStack ? '' : 'Add at least two datasets'}
          >
            Stack {paths.length > 0 ? `(${paths.length})` : ''}
          </button>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    inset: 0,
    zIndex: 9500,
    background: 'rgba(17,17,27,0.6)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dialog: {
    width: 420,
    maxHeight: '80vh',
    display: 'flex',
    flexDirection: 'column',
    background: '#1e1e2e',
    border: '1px solid #313244',
    borderRadius: 10,
    padding: 18,
    color: '#cdd6f4',
    boxShadow: '0 16px 40px rgba(0,0,0,0.55)',
    fontSize: 13,
  },
  title: { margin: '0 0 4px', fontSize: 16, fontWeight: 600 },
  sub: { margin: '0 0 14px', fontSize: 12, color: '#a6adc8', lineHeight: 1.4 },
  stack: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    overflowY: 'auto',
    padding: 2,
    marginBottom: 14,
  },
  chip: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    background: '#11111b',
    border: '1px solid #313244',
    borderRadius: 10,
    padding: '10px 12px',
    cursor: 'grab',
    userSelect: 'none',
  },
  chipDragging: { opacity: 0.4, cursor: 'grabbing' },
  chipDropTarget: { borderColor: ACCENT, boxShadow: `inset 0 0 0 1px ${ACCENT}` },
  grip: { color: '#6c7086', fontSize: 14, lineHeight: 1, cursor: 'grab' },
  badge: {
    minWidth: 18,
    height: 18,
    borderRadius: 9,
    background: ACCENT,
    color: '#11111b',
    fontSize: 11,
    fontWeight: 700,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '0 5px',
  },
  chipName: {
    flex: 1,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    fontSize: 12.5,
  },
  remove: {
    background: 'transparent',
    border: 'none',
    color: '#6c7086',
    fontSize: 18,
    lineHeight: 1,
    cursor: 'pointer',
    padding: '0 2px',
  },
  dropEnd: {
    height: 10,
    borderRadius: 6,
    border: '1px dashed #313244',
  },
  addTile: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'transparent',
    border: '1px dashed #45475a',
    borderRadius: 10,
    padding: '12px',
    color: '#a6adc8',
    cursor: 'pointer',
    fontSize: 13,
  },
  footer: { display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 'auto' },
  cancel: {
    background: 'transparent',
    border: '1px solid #313244',
    color: '#cdd6f4',
    borderRadius: 6,
    padding: '6px 14px',
    cursor: 'pointer',
    fontSize: 12,
  },
  confirm: {
    background: ACCENT,
    border: 'none',
    color: '#11111b',
    fontWeight: 600,
    borderRadius: 6,
    padding: '6px 18px',
    cursor: 'pointer',
    fontSize: 12,
  },
}
