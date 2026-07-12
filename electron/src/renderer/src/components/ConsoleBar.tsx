/**
 * ConsoleBar.tsx — the SpyDE math console: a single-line, Jupyter-flavoured
 * Python input docked at the bottom of the app, directly above StatusBar.
 *
 *   ┌───────────────────────────────────────────────────────────┐
 *   │ out3 = array(256, 256) f32   12 ms                        │ ← echo strip
 *   │ >>>  s1.data.mean()                    [s2 f32] [out3 f32]│ ← input + chips
 *   └───────────────────────────────────────────────────────────┘
 *
 * IPC: every command goes through the same `sendAction` channel every other
 * toolbar/menu action uses (SpyDEContext → window.electron.action →
 * ipcMain 'spyde:action' → runner.sendAction, which writes
 * `{type:"action", action, payload, window_id?}` to the Python subprocess's
 * stdin). `sendAction('console_exec', {code, exec_id})` therefore reaches the
 * backend as action="console_exec" with payload={code, exec_id} — the
 * logical {"command": "console_exec", ...} shape in the IPC contract, just
 * wrapped in the app's existing action envelope (see runner.ts sendAction).
 *
 * Results/vars/completions arrive as `console_result` / `console_vars` /
 * `console_completions` PLOTAPP messages, handled in SpyDEContext.tsx exactly
 * like `playback_state` was added.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import type { ConsoleVarEntry } from '../kernel/SpyDEContext'
import {
  SIGNAL_REF_DRAG_MIME, CONSOLE_VAR_DRAG_MIME, WORKFLOW_NODE_DRAG_MIME,
} from '../kernel/dnd'
import { ConsolePreviewPanel } from './ConsolePreviewPanel'

// One-time keyframes for the busy spinner (shared name/shape with StatusBar's;
// guarded so mounting both never double-injects).
if (typeof document !== 'undefined' && !document.getElementById('spyde-spin-kf')) {
  const el = document.createElement('style')
  el.id = 'spyde-spin-kf'
  el.textContent = '@keyframes spyde-spin { to { transform: rotate(360deg) } }'
  document.head.appendChild(el)
}

// One-time stylesheet for the overlay-pill input. The visible glyphs come from
// the aria-hidden overlay behind a TRANSPARENT-text input, so:
//   • ::selection must be TRANSLUCENT — a solid highlight would hide the
//     overlay text under it while dragging a selection.
//   • ::placeholder needs an EXPLICIT colour — `color:transparent` on the input
//     bleeds into the placeholder, making it invisible otherwise.
if (typeof document !== 'undefined' && !document.getElementById('spyde-console-overlay-css')) {
  const el = document.createElement('style')
  el.id = 'spyde-console-overlay-css'
  el.textContent =
    '.spyde-console-input::selection { background: rgba(137,180,250,0.30); }\n' +
    '.spyde-console-input::placeholder { color: #585b70; }\n' +
    // Chip remove (×): hidden until the chip is hovered, warms on its own hover.
    '.spyde-console-chip .spyde-chip-x { opacity: 0; transition: opacity 80ms; }\n' +
    '.spyde-console-chip:hover .spyde-chip-x { opacity: 1; }\n' +
    '.spyde-chip-x:hover { color: #f38ba8 !important; }'
  document.head.appendChild(el)
}

const HISTORY_KEY = 'spyde:console:history'
const HISTORY_CAP = 200

function loadHistory(): string[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (!raw) return []
    const arr = JSON.parse(raw)
    return Array.isArray(arr) ? arr.filter((x): x is string => typeof x === 'string') : []
  } catch {
    return []
  }
}

function saveHistory(h: string[]): void {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(-HISTORY_CAP)))
  } catch { /* localStorage unavailable/full — history just won't persist */ }
}

// identifier-ish token under the caret (identifier chars + dots), so Tab
// completes `s1.su` without offering to complete the whole line.
function tokenBeforeCaret(text: string, caret: number): string {
  let i = caret
  while (i > 0 && /[\w.]/.test(text[i - 1])) i--
  return text.slice(i, caret)
}

// ── Overlay tokenizer ───────────────────────────────────────────────────────
// Split `code` into a flat list of tokens for the pill overlay. We walk the
// identifier regex over the whole string (MAXIMAL MUNCH, so `s12` is one token,
// never `s1`+`2` — word-boundary correctness for free) and emit the plain-text
// GAPS between matches as literal tokens. A match is a PILL iff (a) it's a bound
// console var and (b) it isn't attribute access (`code[i-1] !== '.'`), so
// `s1.sum` pills only the `s1`.
interface OverlayToken {
  text: string
  start: number       // char index in `code`
  pill: boolean
  source?: 'signal' | 'assign' | 'out'
}
function tokenizeOverlay(
  code: string,
  varMap: Map<string, ConsoleVarEntry>,
): OverlayToken[] {
  const tokens: OverlayToken[] = []
  const re = /[A-Za-z_][A-Za-z0-9_]*/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(code)) !== null) {
    const start = m.index
    if (start > last) tokens.push({ text: code.slice(last, start), start: last, pill: false })
    const word = m[0]
    const v = varMap.get(word)
    const isAttr = start > 0 && code[start - 1] === '.'
    if (v && !isAttr) {
      tokens.push({ text: word, start, pill: true, source: v.source })
    } else {
      tokens.push({ text: word, start, pill: false })
    }
    last = start + word.length
  }
  if (last < code.length) tokens.push({ text: code.slice(last), start: last, pill: false })
  return tokens
}

// Module-cached monospace char advance for the overlay font. Measured once off
// an offscreen canvas (100 zeros / 100), keyed by the exact CSS font string so
// a different computed font re-measures. Used to hit-test which pill the cursor
// is over (the overlay is pointer-transparent, so we can't rely on the DOM).
let _charWidthCache: { font: string; width: number } | null = null
function monoCharWidth(font: string): number {
  if (_charWidthCache && _charWidthCache.font === font) return _charWidthCache.width
  let width = 7.5   // sane fallback if canvas is unavailable
  try {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.font = font
      const w = ctx.measureText('0'.repeat(100)).width / 100
      if (w > 0) width = w
    }
  } catch { /* no canvas — keep the fallback */ }
  _charWidthCache = { font, width }
  return width
}

function shapeDtypeBadge(v: { shape?: number[] | null; dtype?: string | null; lazy?: boolean }): string {
  const shape = v.shape && v.shape.length ? v.shape.join('×') : ''
  const dtype = v.dtype ?? ''
  const core = [shape, dtype].filter(Boolean).join(' ')
  return v.lazy ? `lazy ${core}`.trim() : core
}

export function ConsoleBar() {
  const { state, sendAction } = useSpyDE()
  const [code, setCode] = useState('')
  const [busy, setBusy] = useState(false)
  const [flash, setFlash] = useState(false)          // border flash: unresolvable signal-ref drop
  const [dropActive, setDropActive] = useState(false) // a droppable pill is hovering the bar
  const [tracebackOpen, setTracebackOpen] = useState(false)
  const [completions, setCompletions] = useState<string[] | null>(null)
  const [completionIdx, setCompletionIdx] = useState(0)

  // Overlay-pill state: the input scrollLeft (so the overlay tracks caret
  // autoscroll) and the hovered-pill tooltip.
  const [scrollLeft, setScrollLeft] = useState(0)
  const [pillHover, setPillHover] = useState<{ name: string; badge: string; left: number } | null>(null)

  // Live-preview state (the eye toggle + pop-out panel). The eye ALWAYS starts
  // off — deliberately NOT persisted: an auto-opening panel on launch reads as
  // noise (user call), and the preview is opt-in per session.
  const [previewOn, setPreviewOn] = useState(false)
  const [explicitPreview, setExplicitPreview] = useState(false)   // a one-shot (Ctrl+Enter) is showing
  const [previewContent, setPreviewContent] = useState<import('../kernel/SpyDEContext').ConsolePreviewResult | null>(null)

  const historyRef = useRef<string[]>(loadHistory())
  const historyPos = useRef<number>(-1)               // -1 = not navigating (draft is live)
  const draftRef = useRef('')                         // in-progress text, restored on ArrowDown past newest
  const execIdRef = useRef(0)
  const completeIdRef = useRef(0)
  const previewIdRef = useRef(0)                       // latest-wins gate for preview replies
  const previewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const flashTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingNodeBind = useRef(false)               // a workflow-node bind is awaiting its name

  const lastResult = state.consoleResult
  const chipVars = state.consoleVars.filter(v => v.source === 'assign' || v.source === 'out')

  // Fast name→entry lookup for the overlay tokenizer + tooltip.
  const varMap = React.useMemo(
    () => new Map(state.consoleVars.map(v => [v.name, v])),
    [state.consoleVars],
  )
  const overlayTokens = React.useMemo(() => tokenizeOverlay(code, varMap), [code, varMap])

  // A result_result for an exec_id we're waiting on clears the busy indicator.
  useEffect(() => {
    if (lastResult && lastResult.execId === execIdRef.current) setBusy(false)
  }, [lastResult])

  // New completions for a stale complete_id are ignored (debounce via id, not time).
  useEffect(() => {
    const c = state.consoleCompletions
    if (!c || c.completeId !== completeIdRef.current) return
    setCompletions(c.matches)
    setCompletionIdx(0)
  }, [state.consoleCompletions])

  // Collapse the traceback panel whenever a new result comes in.
  useEffect(() => { setTracebackOpen(false) }, [lastResult?.execId])

  // Keep the overlay's horizontal offset locked to the input's scrollLeft. Read
  // inside rAF because the browser autoscrolls the caret AFTER the event that
  // triggered it (setSelectionRange / typing past the edge). Wired to input
  // onScroll/onChange/onKeyUp/onClick and re-called after the setSelectionRange
  // rAF blocks in insertName / acceptCompletion.
  const syncScroll = useCallback(() => {
    requestAnimationFrame(() => {
      const el = inputRef.current
      if (el) setScrollLeft(el.scrollLeft)
    })
  }, [])

  // `sendAction` from context is recreated on EVERY provider render, so any
  // callback/effect that lists it as a dep re-runs on every state update — the
  // original debounce effect did exactly that and re-armed itself off its OWN
  // preview reply, an infinite request loop (the "flashing" preview). Route
  // sends through a ref so preview callbacks stay identity-stable and previews
  // fire ONLY on real code changes.
  const sendActionRef = useRef(sendAction)
  sendActionRef.current = sendAction

  // The live-preview request. Takes the code explicitly (identity-stable — no
  // closure over `code`); whitespace-only code clears the panel and tells the
  // backend to STOP (empty-code preview = clear the nav-refresh state).
  const requestPreview = useCallback((auto: boolean, codeArg: string) => {
    const id = ++previewIdRef.current
    if (!codeArg.trim()) {
      setPreviewContent(null)
      sendActionRef.current('console_preview', { code: '', preview_id: id, auto: true })
      return
    }
    sendActionRef.current('console_preview', { code: codeArg, preview_id: id, auto })
  }, [])

  // Debounced auto-preview: 400 ms after the code settles, while the eye is on.
  // Deps are ONLY [code, previewOn] (requestPreview is identity-stable) — the
  // effect must never re-run off unrelated state updates (see sendActionRef
  // above). The last content stays up at full opacity until the new reply
  // swaps it in place — no dim/blank between keystrokes (no flashing). The
  // backend re-runs the last auto preview by itself when the NAVIGATOR moves.
  useEffect(() => {
    if (previewTimerRef.current) { clearTimeout(previewTimerRef.current); previewTimerRef.current = null }
    if (!previewOn) return
    if (!code.trim()) { requestPreview(true, ''); return }
    previewTimerRef.current = setTimeout(() => requestPreview(true, code), 400)
    return () => {
      if (previewTimerRef.current) { clearTimeout(previewTimerRef.current); previewTimerRef.current = null }
    }
  }, [code, previewOn, requestPreview])

  // A one-shot explicit preview (Ctrl+Enter) only lives until the code changes
  // or a cell executes (cleared in runExec / here on code change).
  useEffect(() => { setExplicitPreview(false) }, [code])

  // Preview reply intake — newest wins (ignore a reply for a superseded id).
  // A backend NAV-REFRESH (navigator moved) re-emits with the SAME id, so it
  // passes this gate and swaps the panel content in place.
  useEffect(() => {
    const p = state.consolePreview
    if (!p || p.previewId !== previewIdRef.current) return
    setPreviewContent(p)
  }, [state.consolePreview])

  // A workflow-node bind's assigned name arrives asynchronously as the
  // `spyde:console_node_bound` CustomEvent (re-broadcast in SpyDEContext). Only
  // insert if WE initiated a node drop (pendingNodeBind), so an unrelated bind
  // never types into the input.
  useEffect(() => {
    const onBound = (e: Event) => {
      const detail = (e as CustomEvent).detail as { name?: string }
      if (!pendingNodeBind.current) return
      pendingNodeBind.current = false
      if (detail?.name) insertName(detail.name)
    }
    window.addEventListener('spyde:console_node_bound', onBound)
    return () => window.removeEventListener('spyde:console_node_bound', onBound)
    // insertName is stable enough (reads refs); intentionally not a dep.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const runExec = useCallback(() => {
    const trimmed = code
    if (!trimmed.trim()) return
    const id = ++execIdRef.current
    setBusy(true)
    setCompletions(null)
    // Exec wins over any in-flight preview: invalidate the pending reply and
    // cancel the debounce so a stale preview can't land after the cell ran
    // (matches the backend's exec-wins invalidation).
    previewIdRef.current++
    if (previewTimerRef.current) { clearTimeout(previewTimerRef.current); previewTimerRef.current = null }
    setExplicitPreview(false)
    sendAction('console_exec', { code: trimmed, exec_id: id })

    const h = historyRef.current
    if (h[h.length - 1] !== trimmed) {
      h.push(trimmed)
      if (h.length > HISTORY_CAP) h.splice(0, h.length - HISTORY_CAP)
      saveHistory(h)
    }
    historyPos.current = -1
    draftRef.current = ''
    setCode('')
  }, [code, sendAction])

  // Eye toggle. OFF sends an empty-code preview — the backend's STOP signal —
  // so navigator moves no longer re-run the last expression once the panel is
  // closed (and the pending-reply id is invalidated in the same call).
  const togglePreview = useCallback(() => {
    setPreviewOn(prev => {
      const next = !prev
      if (!next) requestPreview(true, '')
      return next
    })
  }, [requestPreview])

  const cycleHistory = useCallback((dir: -1 | 1) => {
    const h = historyRef.current
    if (h.length === 0) return
    if (historyPos.current === -1) {
      if (dir === -1) {
        draftRef.current = code
        historyPos.current = h.length - 1
        setCode(h[historyPos.current])
      }
      return
    }
    const next = historyPos.current + dir
    if (next < 0) return
    if (next >= h.length) {
      historyPos.current = -1
      setCode(draftRef.current)
      return
    }
    historyPos.current = next
    setCode(h[next])
  }, [code])

  const requestCompletions = useCallback(() => {
    const caret = inputRef.current?.selectionStart ?? code.length
    const prefix = tokenBeforeCaret(code, caret)
    const id = ++completeIdRef.current
    sendAction('console_complete', { prefix, complete_id: id })
  }, [code, sendAction])

  const acceptCompletion = useCallback((match: string) => {
    const el = inputRef.current
    const caret = el?.selectionStart ?? code.length
    const tok = tokenBeforeCaret(code, caret)
    const before = code.slice(0, caret - tok.length)
    const after = code.slice(caret)
    const next = before + match + after
    setCode(next)
    setCompletions(null)
    // Restore focus + caret just past the inserted text.
    requestAnimationFrame(() => {
      const pos = before.length + match.length
      el?.focus()
      el?.setSelectionRange(pos, pos)
      syncScroll()   // the browser may have autoscrolled the caret into view
    })
  }, [code, syncScroll])

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Ctrl/Cmd+Enter = one-shot live preview (works even with the eye off).
    // Checked FIRST so it wins over the completions/Enter blocks below.
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      setExplicitPreview(true)
      requestPreview(false, code)
      return
    }
    if (completions && completions.length > 0) {
      if (e.key === 'Tab' || e.key === 'ArrowDown') {
        e.preventDefault()
        setCompletionIdx(i => (i + 1) % completions.length)
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setCompletionIdx(i => (i - 1 + completions.length) % completions.length)
        return
      }
      if (e.key === 'Enter') {
        e.preventDefault()
        acceptCompletion(completions[completionIdx])
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        setCompletions(null)
        return
      }
    }
    if (e.key === 'Enter') {
      // Enter OR Shift+Enter both execute (Jupyter muscle memory) — neither
      // inserts a newline (this is a single-line console).
      e.preventDefault()
      runExec()
      return
    }
    if (e.key === 'ArrowUp') { e.preventDefault(); cycleHistory(-1); return }
    if (e.key === 'ArrowDown') { e.preventDefault(); cycleHistory(1); return }
    if (e.key === 'Tab') { e.preventDefault(); requestCompletions(); return }
    if (e.key === 'Escape') { setTracebackOpen(false) }
  }

  // ── Drag-IN: a dropped window/navigator/signal pill (or a Workflow node)
  //    binds into the console and inserts its variable name at the caret. The
  //    inserted name PILL-IFIES via the overlay (it matches a bound console var),
  //    so there's no separate cosmetic dropped-pill any more. Resolution:
  //      • WORKFLOW_NODE → backend console_bind_node (a mid-tree node); the
  //        assigned var name arrives async as the spyde:console_node_bound
  //        CustomEvent (see the effect above), and we insert it then — the
  //        pendingNodeBind flag gates it so only OUR drop types the name.
  //      • SIGNAL_REF (window pill) → resolve windowId → var via console_vars.
  const DROP_MIMES = [WORKFLOW_NODE_DRAG_MIME, SIGNAL_REF_DRAG_MIME, CONSOLE_VAR_DRAG_MIME]
  const onDragOver = (e: React.DragEvent) => {
    if (!DROP_MIMES.some(m => e.dataTransfer.types.includes(m))) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    setDropActive(true)
  }
  const insertName = (name: string) => {
    const el = inputRef.current
    // Read the current text off the DOM (the input is controlled, so
    // el.value === code) so this stays correct when called from a stale-closure
    // context (the console_node_bound effect / pendingInsert effect).
    const cur = el?.value ?? code
    const caret = el?.selectionStart ?? cur.length
    // Pad with a space so consecutive drops don't glue names together.
    const sep = caret > 0 && !/\s$/.test(cur.slice(0, caret)) ? ' ' : ''
    const ins = sep + name
    const next = cur.slice(0, caret) + ins + cur.slice(caret)
    setCode(next)
    requestAnimationFrame(() => {
      const pos = caret + ins.length
      el?.focus()
      el?.setSelectionRange(pos, pos)
      syncScroll()
    })
  }
  const flashUnresolved = () => {
    setFlash(true)
    if (flashTimer.current) clearTimeout(flashTimer.current)
    flashTimer.current = setTimeout(() => setFlash(false), 420)
  }
  const onDrop = (e: React.DragEvent) => {
    setDropActive(false)
    const dt = e.dataTransfer
    // Workflow node: bind a mid-tree node into the namespace (backend picks a
    // fresh var name and echoes it via console_node_bound). Flag the pending
    // bind so the CustomEvent handler inserts the assigned name when it lands —
    // the workflow branch itself inserts NO text.
    const wf = dt.getData(WORKFLOW_NODE_DRAG_MIME)
    if (wf) {
      e.preventDefault()
      try {
        const { windowId, signalId } = JSON.parse(wf) as
          { windowId: number; signalId: number; name: string }
        pendingNodeBind.current = true
        sendAction('console_bind_node', { signal_id: signalId }, windowId)
      } catch { flashUnresolved() }
      return
    }
    // Console result chip (out/assign): insert its name directly.
    const cv = dt.getData(CONSOLE_VAR_DRAG_MIME)
    if (cv) {
      e.preventDefault()
      try {
        const { name } = JSON.parse(cv) as { name: string }
        if (name) insertName(name)
      } catch { flashUnresolved() }
      return
    }
    // Window/navigator pill (signal ref): resolve windowId → console var name.
    const raw = dt.getData(SIGNAL_REF_DRAG_MIME)
    if (!raw) return
    e.preventDefault()
    let windowId: number | null = null
    try {
      windowId = (JSON.parse(raw) as { windowId?: number }).windowId ?? null
    } catch { /* malformed payload */ }
    if (windowId == null) { flashUnresolved(); return }
    const name = resolveSignalName(state.consoleVars, windowId)
    if (name == null) {
      // The console engine may not exist yet (never opened) → no bindings to
      // resolve. Force-create + bind, and insert the resolved name once
      // console_vars lands (pendingInsert).
      sendAction('console_bind_window', {}, windowId)
      pendingInsert.current = windowId
      return
    }
    insertName(name)
  }

  // Resolve a deferred signal-ref drop once the freshly-bound console_vars land.
  const pendingInsert = useRef<number | null>(null)
  useEffect(() => {
    const wid = pendingInsert.current
    if (wid == null) return
    const name = resolveSignalName(state.consoleVars, wid)
    if (name != null) {
      pendingInsert.current = null
      insertName(name)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.consoleVars])

  useEffect(() => () => {
    if (flashTimer.current) clearTimeout(flashTimer.current)
    if (previewTimerRef.current) clearTimeout(previewTimerRef.current)
  }, [])

  // Empty value_repr (assignment/None) reads as a subtle "ok" rather than blank.
  const displayRepr = lastResult ? (lastResult.valueRepr || (lastResult.ok ? 'ok' : '')) : ''

  // The preview pop-out only exists while the eye is on OR a one-shot is active.
  const showPreviewPanel = previewOn || explicitPreview

  // ── Pill hover tooltip. The overlay is pointer-transparent, so hit-test the
  //    hovered char against the pill token spans using the module-cached mono
  //    char width. charIdx = (mouseX − wrapLeft + scrollLeft) / charWidth.
  const onWrapMouseMove = (e: React.MouseEvent) => {
    const input = inputRef.current
    const wrap = wrapRef.current
    if (!input || !wrap) return
    const rect = wrap.getBoundingClientRect()
    const cw = monoCharWidth(getComputedStyle(input).font)
    if (cw <= 0) return
    const charIdx = Math.floor((e.clientX - rect.left + scrollLeft) / cw)
    for (const t of overlayTokens) {
      if (!t.pill) continue
      if (charIdx >= t.start && charIdx < t.start + t.text.length) {
        const v = varMap.get(t.text)
        setPillHover({
          name: t.text,
          badge: v ? shapeDtypeBadge(v) : '',
          left: t.start * cw - scrollLeft,
        })
        return
      }
    }
    setPillHover(null)
  }

  return (
    <div style={styles.root}>
      {/* Completion popup — anchored above the bar. */}
      {completions && completions.length > 0 && (
        <div data-testid="console-completions" style={styles.completionsPopup}>
          {completions.map((m, i) => (
            <div
              key={m}
              data-testid={`console-completion-${m}`}
              style={i === completionIdx ? styles.completionItemActive : styles.completionItem}
              onMouseEnter={() => setCompletionIdx(i)}
              onMouseDown={(e) => { e.preventDefault(); acceptCompletion(m) }}
            >
              {m}
            </div>
          ))}
        </div>
      )}

      {/* Echo strip: last result summary, or the expanded traceback panel. */}
      {lastResult && (
        <div style={styles.echoRow}>
          {!lastResult.ok ? (
            <button
              data-testid="console-error-toggle"
              style={styles.echoError}
              onClick={() => setTracebackOpen(v => !v)}
              title="Click to see the full traceback"
            >
              error{lastResult.error ? `: ${truncate(lastResult.error, 100)}` : ''}
            </button>
          ) : (
            <span data-testid="console-echo" style={styles.echoText}>
              {lastResult.stdout && (
                <span style={styles.echoStdout}>{truncate(lastResult.stdout, 160)}  </span>
              )}
              <span style={styles.echoValue}>{truncate(displayRepr, 160)}</span>
            </span>
          )}
          <span data-testid="console-duration" style={styles.echoDuration}>
            {Math.round(lastResult.durationMs)} ms
          </span>
        </div>
      )}

      {tracebackOpen && lastResult && !lastResult.ok && (
        <div data-testid="console-traceback" style={styles.tracebackPanel}>
          <pre style={styles.tracebackPre}>{lastResult.traceback || lastResult.error}</pre>
        </div>
      )}

      {/* Input row + result chips. */}
      <div
        data-testid="console-bar"
        style={{
          ...styles.inputRow,
          ...(flash ? styles.inputRowFlash : {}),
          ...(dropActive ? styles.inputRowDrop : {}),
        }}
        onDragOver={onDragOver}
        onDragLeave={() => setDropActive(false)}
        onDrop={onDrop}
      >
        <span style={styles.prompt}>&gt;&gt;&gt;</span>
        {/* Input + overlay: the input holds a TRANSPARENT-text value; the
            aria-hidden overlay BEHIND it (zIndex 0, pointer-transparent) draws
            the visible glyphs — plain text, plus pill-styled spans for bound
            console vars. Both share exact font metrics so glyphs align; the
            overlay is translated by −scrollLeft to track the caret autoscroll. */}
        <div
          ref={wrapRef}
          style={styles.inputWrap}
          onMouseMove={onWrapMouseMove}
          onMouseLeave={() => setPillHover(null)}
        >
          <div aria-hidden="true" style={styles.overlay}>
            <div style={{ ...styles.overlayInner, transform: `translateX(${-scrollLeft}px)` }}>
              {overlayTokens.map((t, i) => (
                t.pill
                  ? (
                    <span
                      key={i}
                      data-testid={`console-pill-${t.text}`}
                      style={t.source === 'signal' ? styles.pillSignal : styles.pillAssign}
                    >
                      {t.text}
                    </span>
                  )
                  : <span key={i}>{t.text}</span>
              ))}
            </div>
          </div>
          <input
            ref={inputRef}
            data-testid="console-input"
            className="spyde-console-input"
            style={styles.input}
            type="text"
            spellCheck={false}
            autoComplete="off"
            placeholder="Python — Enter to run · Tab to complete"
            value={code}
            onChange={(e) => { setCode(e.target.value); setCompletions(null); syncScroll() }}
            onKeyDown={onKeyDown}
            onKeyUp={syncScroll}
            onClick={syncScroll}
            onScroll={syncScroll}
          />
          {/* Pill hover tooltip — above the hovered pill, name + shape×dtype. */}
          {pillHover && (
            <div
              data-testid="console-pill-tooltip"
              style={{ ...styles.pillTooltip, left: Math.max(0, pillHover.left) }}
            >
              <span style={styles.pillTooltipName}>{pillHover.name}</span>
              {pillHover.badge && <span style={styles.pillTooltipBadge}>{pillHover.badge}</span>}
            </div>
          )}
        </div>
        {/* Live-preview eye toggle — on = accent, off = muted. One-shot via
            Ctrl+Enter regardless of this toggle. The pop-out panel is anchored
            to THIS wrapper (position:relative) so it opens directly above the
            eye, not at the far edge of the bar. */}
        <span style={styles.eyeWrap}>
          {showPreviewPanel && (
            <ConsolePreviewPanel preview={previewContent} />
          )}
          <button
            data-testid="console-preview-eye"
            title="Live preview (Ctrl+Enter for one-shot)"
            onClick={togglePreview}
            style={styles.eyeBtn}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" aria-hidden="true">
              <path
                fill={previewOn ? '#89b4fa' : '#585b70'}
                d="M12 5C5.6 5 1.7 11.1 1.5 11.4a1 1 0 0 0 0 1.2C1.7 12.9 5.6 19 12 19s10.3-6.1 10.5-6.4a1 1 0 0 0 0-1.2C22.3 11.1 18.4 5 12 5zm0 12c-4.3 0-7.4-3.6-8.4-5 1-1.4 4.1-5 8.4-5s7.4 3.6 8.4 5c-1 1.4-4.1 5-8.4 5zm0-8a3 3 0 1 0 0 6 3 3 0 0 0 0-6z"
              />
            </svg>
          </button>
        </span>
        {busy && <span data-testid="console-busy" style={styles.spinner} aria-label="Running" />}
        <div style={styles.chips}>
          {chipVars.map(v => (
            <ConsoleChip key={v.name} v={v} sendAction={sendAction} />
          ))}
        </div>
      </div>
    </div>
  )
}

function resolveSignalName(vars: ConsoleVarEntry[], windowId: number): string | null {
  for (const v of vars) {
    if (v.source === 'signal' && v.window_ids && v.window_ids.includes(windowId)) return v.name
  }
  return null
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '…' : s
}

function ConsoleChip({ v, sendAction }: {
  v: ConsoleVarEntry
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}) {
  const create = () => sendAction('console_create_window', { name: v.name })
  const remove = () => sendAction('console_remove_var', { name: v.name })
  const badge = shapeDtypeBadge(v)
  return (
    <button
      className="spyde-console-chip"
      data-testid={`console-chip-${v.name}`}
      title={`Drag onto the MDI area (or double-click) to open ${v.name} as a new window`}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(CONSOLE_VAR_DRAG_MIME, JSON.stringify({ name: v.name }))
        e.dataTransfer.effectAllowed = 'copy'
      }}
      onDoubleClick={create}
      style={styles.chip}
    >
      <span style={styles.chipCol}>
        <span style={styles.chipName}>{v.name}</span>
        {badge && <span style={styles.chipBadge}>{badge}</span>}
      </span>
      {/* Remove (×) — drops the result from the console namespace (the chip
          disappears when the refreshed console_vars lands). Hover-revealed via
          the injected .spyde-chip-x rules. */}
      <span
        className="spyde-chip-x"
        data-testid={`console-chip-remove-${v.name}`}
        title={`Remove ${v.name}`}
        onClick={(e) => { e.stopPropagation(); remove() }}
        style={styles.chipX}
      >
        ×
      </span>
    </button>
  )
}

const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace'

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: 'flex', flexDirection: 'column',
    background: '#181825', borderTop: '1px solid #313244',
    flexShrink: 0, position: 'relative',
  },
  echoRow: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '2px 12px', height: 20, overflow: 'hidden',
    borderBottom: '1px solid #1e1e2e',
  },
  echoText: {
    flex: 1, minWidth: 0, fontSize: 11, fontFamily: MONO,
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
    color: '#a6adc8',
  },
  echoStdout: { color: '#6c7086' },
  echoValue: { color: '#cdd6f4' },
  echoError: {
    flex: 1, minWidth: 0, textAlign: 'left',
    background: 'none', border: 'none', cursor: 'pointer', padding: 0,
    fontSize: 11, fontFamily: MONO, color: '#f38ba8',
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
  },
  echoDuration: { fontSize: 10, color: '#6c7086', flexShrink: 0, fontFamily: MONO },
  tracebackPanel: {
    maxHeight: 220, overflow: 'auto',
    borderBottom: '1px solid #313244', background: '#11111b',
    padding: '6px 12px',
  },
  tracebackPre: {
    margin: 0, fontSize: 11, fontFamily: MONO, color: '#f38ba8',
    whiteSpace: 'pre-wrap', wordBreak: 'break-word',
  },
  inputRow: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '4px 12px', minHeight: 28,
    transition: 'border-color 90ms ease',
    borderTop: '1px solid transparent', borderBottom: '1px solid transparent',
  },
  inputRowFlash: { borderColor: '#f38ba8' },
  inputRowDrop: { borderColor: '#89b4fa', background: 'rgba(137,180,250,0.06)' },
  prompt: {
    fontFamily: MONO, fontSize: 12, color: '#89b4fa', fontWeight: 700, flexShrink: 0,
    userSelect: 'none',
  },
  // The input's own box: fills the flex slot, relatively positioned so the
  // overlay can sit absolutely inside it.
  inputWrap: {
    position: 'relative', flex: 1, minWidth: 0,
    height: 18,   // matches lineHeight, so overlay + input agree vertically
  },
  // Behind the input (zIndex 0): the pill-styled visible text. Pointer-
  // transparent — hover hit-testing is done arithmetically in JS.
  overlay: {
    position: 'absolute', inset: 0, overflow: 'hidden',
    pointerEvents: 'none', zIndex: 0,
  },
  overlayInner: {
    whiteSpace: 'pre',
    fontFamily: MONO, fontSize: 12.5, lineHeight: '18px', letterSpacing: 'normal',
    color: '#cdd6f4',
  },
  input: {
    // TRANSPARENT text — the overlay supplies the visible glyphs; the caret
    // stays visible via caretColor. EXACT same font metrics as overlayInner.
    // Chromium's default input padding (1px 2px) MUST be zeroed or the overlay
    // text drifts out of alignment.
    position: 'relative', zIndex: 1,
    width: '100%', height: '100%', margin: 0, padding: 0,
    background: 'transparent', border: 'none', outline: 'none',
    color: 'transparent', caretColor: '#cdd6f4',
    fontFamily: MONO, fontSize: 12.5, lineHeight: '18px', letterSpacing: 'normal',
  },
  // Layout-NEUTRAL pill spans: background + inset box-shadow ring + radius only.
  // NO padding/margin/border/weight — anything that changes glyph advance would
  // shift the overlay out of sync with the input's caret.
  pillSignal: {
    color: '#bcd7ff', background: 'rgba(137,180,250,0.28)',
    boxShadow: '0 0 0 2px rgba(137,180,250,0.28)', borderRadius: 4,
  },
  pillAssign: {
    color: '#cdd6f4', background: 'rgba(166,173,200,0.16)',
    boxShadow: '0 0 0 2px rgba(166,173,200,0.16)', borderRadius: 4,
  },
  pillTooltip: {
    position: 'absolute', bottom: '100%', marginBottom: 6,
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 6,
    boxShadow: '0 -6px 20px rgba(0,0,0,0.45)',
    padding: '4px 8px', zIndex: 9200, pointerEvents: 'none',
    display: 'flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap',
  },
  pillTooltipName: { fontSize: 11, fontFamily: MONO, fontWeight: 600, color: '#89b4fa' },
  pillTooltipBadge: { fontSize: 9, fontFamily: MONO, color: '#a6adc8' },
  eyeWrap: { position: 'relative', display: 'inline-flex', flexShrink: 0 },
  eyeBtn: {
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    background: 'transparent', border: 'none', cursor: 'pointer',
    padding: 2, flexShrink: 0, lineHeight: 0,
  },
  spinner: {
    width: 11, height: 11, borderRadius: '50%', flexShrink: 0,
    border: '2px solid #45475a', borderTopColor: '#89b4fa',
    animation: 'spyde-spin 0.8s linear infinite',
  },
  chips: {
    display: 'flex', alignItems: 'center', gap: 5,
    overflowX: 'auto', maxWidth: '46%', flexShrink: 0,
  },
  chip: {
    display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 5,
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 6,
    color: '#cdd6f4', cursor: 'grab', padding: '2px 6px 2px 8px', flexShrink: 0,
    lineHeight: 1.25,
  },
  chipCol: { display: 'flex', flexDirection: 'column', alignItems: 'flex-start' },
  chipName: { fontSize: 11, fontFamily: MONO, fontWeight: 600, color: '#89b4fa' },
  chipBadge: { fontSize: 9, fontFamily: MONO, color: '#a6adc8' },
  chipX: {
    fontSize: 12, fontFamily: MONO, color: '#6c7086', cursor: 'pointer',
    lineHeight: 1, padding: '0 1px', userSelect: 'none',
  },
  completionsPopup: {
    position: 'absolute', left: 12, bottom: '100%', marginBottom: 4,
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 6,
    boxShadow: '0 -6px 20px rgba(0,0,0,0.45)',
    maxHeight: 220, overflowY: 'auto', minWidth: 200, zIndex: 9200,
  },
  completionItem: {
    padding: '4px 10px', fontSize: 12, fontFamily: MONO, color: '#cdd6f4', cursor: 'pointer',
  },
  completionItemActive: {
    padding: '4px 10px', fontSize: 12, fontFamily: MONO, color: '#11111b',
    background: '#89b4fa', cursor: 'pointer',
  },
}
