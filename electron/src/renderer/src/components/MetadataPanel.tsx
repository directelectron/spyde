/**
 * MetadataPanel.tsx — the dock's metadata section.
 *
 * DESIGN. The dock has one scarce resource (vertical space) and this panel used
 * to spend it on a one-per-row list of mostly-unset fields, several of which
 * repeated something already on screen. It is now a curated SUMMARY plus a
 * detail box:
 *
 *   • Dataset  → a chip strip (shape · dtype · lazy/chunks · fps), not four rows.
 *     What each chip elides (chunk block size, the full shape) is in its tooltip.
 *   • Instrument → ONE four-across row; the four fields are short and always
 *     worth showing, set or not, because that row is also how you set them.
 *   • Experiment → the three fields that identify the run (Name / Date / e-/s).
 *   • Everything else (SECONDARY below, and the whole Movie group) is hidden
 *     from the summary and reachable through the ⋯ detail box.
 *
 * Clicking ANY field opens that box: a caret-pointed popover with the field's
 * description (the YAML text, via `MetadataMessage.info`), its full untruncated
 * value, where hyperspy stores it, and — for a writable field — the editor. So
 * abbreviating a value in the summary never loses it, and editing gets a
 * properly sized input instead of one squeezed into a quarter-width cell.
 */
import React from 'react'
import type { ChunkInfo, MetadataDict, MetadataInfo } from '../kernel/SpyDEContext'
import { CaretBox } from './CaretBox'
import { ChunkViewer } from './ChunkViewer'

/** Rendered by the chip strip rather than as a field grid. */
const DATASET = 'Dataset'
/** Folded into the chip strip (its values are derived from the time axis, which
 *  the Axes table already shows — a whole group repeating it was noise). */
const MOVIE = 'Movie / In-Situ'
const INSTRUMENT = 'Instrument Metadata'

/** Fields kept OUT of the summary, per group. Each is either a duplicate of
 *  something already on screen or near-never populated; all stay one click away
 *  in the group's ⋯ box, still editable. */
const SECONDARY: Record<string, string[]> = {
  'Root Experiment Details': [
    'Dtype',   // same value as the Dataset strip's dtype chip
    'Dim.',    // same information as the Dataset strip's shape chip
    'Mode',    // detector exposure mode — rarely present outside DE data
    'Cam.',    // camera model — ditto
  ],
}

/** Shorter headings than the config's group names (the dock is 300px wide). */
const GROUP_TITLE: Record<string, string> = {
  [INSTRUMENT]: 'Instrument',
  'Root Experiment Details': 'Experiment',
}

/** The Dataset group is synthetic (built in metadata_extract, not the YAML), so
 *  it has no config description to send. Its copy lives here instead. */
const DATASET_INFO: Record<string, string> = {
  Shape: 'Navigation × signal shape of the DISPLAYED node, which may differ from the root.',
  Dtype: 'Element type of the displayed data array.',
  Chunks: 'Dask block shape and size. Chunks that split the signal axes make the navigator slow.',
  Lazy: 'Whether the data is a lazy dask array (read on demand) or already in memory.',
}

/** A value the backend renders as "unset" — "--", "-- kV", "YYYY-MM-DD". */
const isUnset = (v: string) => !v || v.startsWith('--') || v === 'YYYY-MM-DD'

export interface MetadataPanelProps {
  meta: MetadataDict
  editable: Record<string, Record<string, string>>
  info: MetadataInfo
  /** Dask block layout, when the displayed node is lazy — the "lazy" chip opens
   *  the chunk viewer instead of a plain field detail. */
  chunking?: ChunkInfo
  onEdit: (group: string, prop: string, value: string) => void
}

/** Which field the detail popover is showing: a single field, or a whole group
 *  (the ⋯ box, which lists every field including the hidden ones). `el` is the
 *  anchor itself, so a press on it is not treated as an outside click and its
 *  own onClick can toggle the box shut. */
type Focus = { group: string; prop: string | null; anchor: DOMRect; el: HTMLElement }

export function MetadataPanel({ meta, editable, info, chunking, onEdit }: MetadataPanelProps) {
  const [focus, setFocus] = React.useState<Focus | null>(null)
  // The chunk viewer is anchored the same way a field detail is; null = closed.
  const [chunks, setChunks] = React.useState<{ anchor: DOMRect; el: HTMLElement } | null>(null)
  const open = (group: string, prop: string | null) => (e: React.MouseEvent) => {
    // The second click of a DOUBLE click is the same intent as the first, not a
    // second toggle — otherwise double-clicking a field opens and immediately
    // closes the box.
    if (e.detail > 1) return
    const el = e.currentTarget as HTMLElement
    if (chunking && group === DATASET && (prop === 'Lazy' || prop === 'Chunks')) {
      setFocus(null)
      setChunks((c) => c ? null : { anchor: el.getBoundingClientRect(), el })
      return
    }
    setChunks(null)
    setFocus((f) => (f && f.group === group && f.prop === prop)
      ? null                                     // clicking it again closes it
      : { group, prop, anchor: el.getBoundingClientRect(), el })
  }

  const groups = Object.keys(meta)
  const ds = meta[DATASET]
  const movie = meta[MOVIE]

  return (
    <div style={S.panel} data-testid="metadata-panel"
      // A popover is positioned against a rect captured on click, so scrolling
      // the panel would leave it pointing at nothing. Cheaper and steadier than
      // tracking the anchor.
      onScroll={() => setFocus(null)}>

      {(ds || movie) && <DatasetStrip ds={ds} movie={movie} onOpen={open} />}

      {groups.filter((g) => g !== DATASET && g !== MOVIE).map((group) => {
        const fields = meta[group]
        const hidden = SECONDARY[group] ?? []
        const shown = Object.keys(fields).filter((p) => !hidden.includes(p))
        if (!shown.length) return null
        return (
          <div key={group} style={S.group}>
            <div style={S.groupHead}>
              <span style={S.groupTitle}>{GROUP_TITLE[group] ?? group}</span>
              <button data-testid={`meta-more-${group}`} style={S.more}
                title={`All ${GROUP_TITLE[group] ?? group} fields`}
                onClick={open(group, null)}>⋯</button>
            </div>
            {/* Column count = the group's field count, so each group is ONE row:
                four instrument settings, three experiment identifiers. */}
            <div style={group === INSTRUMENT ? S.grid4
              : shown.length === 3 ? S.grid3 : S.grid2}>
              {shown.map((prop) => (
                <Cell key={prop} group={group} prop={prop} value={fields[prop]}
                  editable={editable[group]?.[prop] !== undefined} onOpen={open} />
              ))}
            </div>
          </div>
        )
      })}

      {focus && (
        <FieldDetail focus={focus} meta={meta} editable={editable} info={info}
          onPick={(group, prop) => setFocus((f) => f && { ...f, group, prop })}
          onClose={() => setFocus(null)}
          onCommit={(group, prop, v) => { onEdit(group, prop, v); setFocus(null) }} />
      )}

      {chunks && chunking && (
        <ChunkViewer info={chunking} anchor={chunks.anchor} el={chunks.el}
          onClose={() => setChunks(null)} />
      )}
    </div>
  )
}

/** One summary cell: tiny key over its value. Truncates rather than wraps — the
 *  full string is one click away, which is the point of the detail box. */
function Cell({ group, prop, value, editable, onOpen }: {
  group: string; prop: string; value: string; editable: boolean
  onOpen: (g: string, p: string) => (e: React.MouseEvent) => void
}) {
  const [hover, setHover] = React.useState(false)
  const unset = isUnset(value)
  return (
    <div style={S.cell}
      onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      <span style={S.key} title={prop}>{prop}</span>
      <span
        data-testid={`meta-${group}-${prop}`}
        // The laundry suite keys editability off this exact title; it is also
        // the honest hover hint, so it stays the wording it is.
        title={editable ? 'click to edit' : prop}
        onClick={onOpen(group, prop)}
        style={{
          ...S.val,
          color: unset ? '#585b70' : '#cdd6f4',
          background: hover ? '#1e1e2e' : 'transparent',
        }}
      >{value || '—'}</span>
    </div>
  )
}

/** Dataset + movie as a wrapping chip strip. Each chip carries the detail the
 *  old four-row block spelled out (chunk shape, exact fps) in its tooltip. */
function DatasetStrip({ ds, movie, onOpen }: {
  ds?: Record<string, string>; movie?: Record<string, string>
  onOpen: (g: string, p: string) => (e: React.MouseEvent) => void
}) {
  const chips: Array<{ group: string; prop: string; text: string; title: string; mono?: boolean }> = []
  if (ds?.Shape) chips.push({ group: DATASET, prop: 'Shape', text: ds.Shape, title: 'Shape' })
  if (ds?.Dtype) chips.push({ group: DATASET, prop: 'Dtype', text: ds.Dtype, title: 'Dtype', mono: true })
  if (ds?.Lazy) {
    chips.push({
      group: DATASET, prop: ds.Chunks ? 'Chunks' : 'Lazy',
      text: ds.Lazy === 'yes' ? 'lazy' : 'eager',
      // The chunk shape is the thing worth knowing about a lazy array, and it
      // is far too long for a chip — so it becomes the chip's tooltip, and the
      // chip itself opens the block diagram.
      title: ds.Chunks ? `chunks ${ds.Chunks} · click for the block layout` : 'in memory',
    })
  }
  // Only a movie that actually HAS a frame rate earns a chip.
  if (movie?.FPS && !isUnset(movie.FPS)) {
    chips.push({ group: MOVIE, prop: 'FPS', text: movie.FPS, title: 'Frames per second' })
  }
  if (!chips.length) return null
  return (
    <div style={S.strip} data-testid="dataset-strip">
      {chips.map((c) => (
        <span key={`${c.group}-${c.prop}`} data-testid={`meta-${c.group}-${c.prop}`}
          title={c.title} style={{ ...S.chip, fontFamily: c.mono ? 'ui-monospace, monospace' : undefined }}
          onClick={onOpen(c.group, c.prop)}>{c.text}</span>
      ))}
    </div>
  )
}

/** Contents of the field-detail caret box (the box shell is CaretBox). */
function FieldDetail({ focus, meta, editable, info, onPick, onClose, onCommit }: {
  focus: Focus
  meta: MetadataDict
  editable: Record<string, Record<string, string>>
  info: MetadataInfo
  onPick: (group: string, prop: string) => void
  onClose: () => void
  onCommit: (group: string, prop: string, value: string) => void
}) {
  const { group, prop, anchor, el } = focus
  const fields = meta[group] ?? {}
  const raw = prop != null ? editable[group]?.[prop] : undefined
  const field = prop != null ? info[group]?.[prop] : undefined
  const description = prop == null ? ''
    : (group === DATASET ? DATASET_INFO[prop] : field?.description) ?? ''

  return (
    <CaretBox anchor={anchor} el={el} testid="meta-detail" onClose={onClose}>
      {prop == null ? (
        // ⋯ mode: every field in the group, hidden ones included.
        <>
          <div style={S.popTitle}>{GROUP_TITLE[group] ?? group}</div>
          {Object.keys(fields).map((p) => (
            <button key={p} data-testid={`meta-detail-row-${p}`} style={S.row}
              onClick={() => onPick(group, p)}>
              <span style={S.rowKey}>{p}</span>
              <span style={{ ...S.rowVal, color: isUnset(fields[p]) ? '#585b70' : '#cdd6f4' }}>
                {fields[p] || '—'}
              </span>
            </button>
          ))}
        </>
      ) : (
        <>
          <div style={S.popTitle}>{prop}</div>
          <div style={S.popValue}>{fields[prop] || '—'}</div>
          {description && <div style={S.popDesc}>{description}</div>}
          {raw !== undefined ? (
            <Editor initial={raw} units={field?.units ?? ''} testid={`meta-${group}-${prop}-input`}
              onCommit={(v) => onCommit(group, prop, v)} onCancel={onClose} />
          ) : (
            <div style={S.popNote}>Derived — read only.</div>
          )}
          {field?.key && <div style={S.popKey} title={field.key}>{field.key}</div>}
        </>
      )}
    </CaretBox>
  )
}

/** The popover's editor. Pre-filled with the RAW unit-free value (the display
 *  string has units baked in and would fail the backend's float parse). */
function Editor({ initial, units, testid, onCommit, onCancel }: {
  initial: string; units: string; testid: string
  onCommit: (v: string) => void; onCancel: () => void
}) {
  const [draft, setDraft] = React.useState(initial)
  React.useEffect(() => { setDraft(initial) }, [initial, testid])
  return (
    <div style={S.editRow}>
      <input data-testid={testid} autoFocus style={S.input} value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') onCommit(draft)
          else if (e.key === 'Escape') onCancel()
        }} />
      {units && <span style={S.units}>{units}</span>}
      <button data-testid={`${testid}-set`} style={S.set}
        onClick={() => onCommit(draft)}>Set</button>
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  // THE dock's only scroll region — see PlotControlDock's `body`. Everything
  // else in the dock is pinned; this is the part that can be arbitrarily long.
  panel: {
    // SHRINK, don't grow (`0 1 auto`): the dock's only elastic child, so a long
    // metadata list gives way to the pinned sections and scrolls inside what is
    // left — but a short one keeps its natural height instead of absorbing all
    // the slack and opening a void above the Axes table. The minimum is sized to
    // the CURATED summary (chip strip + both field groups) so the default view
    // is never clipped mid-group; past that the dock as a whole scrolls, which
    // is the worse of the two.
    flex: '0 1 auto', minHeight: 112, overflowY: 'auto',
    padding: '6px 10px', borderBottom: '1px solid #1e1e2e',
  },
  strip: { display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 },
  chip: {
    fontSize: 9, color: '#a6adc8', background: '#1e1e2e',
    border: '1px solid #313244', borderRadius: 9, padding: '1px 6px',
    cursor: 'pointer', whiteSpace: 'nowrap',
  },
  group: { marginBottom: 4 },
  groupHead: { display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
  groupTitle: { fontSize: 10, color: '#a6adc8', fontWeight: 600 },
  more: {
    background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer',
    fontSize: 12, lineHeight: '10px', padding: '0 2px',
  },
  grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 8, rowGap: 1 },
  grid3: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', columnGap: 6, rowGap: 1 },
  grid4: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', columnGap: 4, rowGap: 1 },
  cell: { display: 'flex', flexDirection: 'column', minWidth: 0 },
  key: {
    color: '#6c7086', fontSize: 9, whiteSpace: 'nowrap',
    overflow: 'hidden', textOverflow: 'ellipsis',
  },
  val: {
    fontSize: 10, cursor: 'pointer', borderRadius: 3, padding: '0 2px',
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
  },
  popTitle: { fontSize: 11, fontWeight: 600, color: '#cdd6f4' },
  popValue: { fontSize: 13, color: '#89b4fa', wordBreak: 'break-word' },
  popDesc: { fontSize: 10, color: '#a6adc8', lineHeight: 1.35 },
  popNote: { fontSize: 9, color: '#6c7086', fontStyle: 'italic' },
  popKey: {
    fontSize: 9, color: '#45475a', fontFamily: 'ui-monospace, monospace',
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  editRow: { display: 'flex', alignItems: 'center', gap: 4 },
  input: {
    flex: 1, minWidth: 0, background: '#1e1e2e', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 3, padding: '2px 4px', fontSize: 11,
  },
  units: { fontSize: 9, color: '#6c7086' },
  set: {
    background: '#313244', color: '#cdd6f4', border: 'none', borderRadius: 3,
    padding: '2px 8px', fontSize: 10, cursor: 'pointer',
  },
  row: {
    display: 'flex', justifyContent: 'space-between', gap: 8, width: '100%',
    background: 'none', border: 'none', borderRadius: 3, padding: '2px 3px',
    cursor: 'pointer', textAlign: 'left',
  },
  rowKey: { fontSize: 10, color: '#6c7086', whiteSpace: 'nowrap' },
  rowVal: {
    fontSize: 10, minWidth: 0, overflow: 'hidden',
    textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
}
