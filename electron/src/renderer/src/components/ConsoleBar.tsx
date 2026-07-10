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
import { SIGNAL_REF_DRAG_MIME, CONSOLE_VAR_DRAG_MIME } from '../kernel/dnd'

// One-time keyframes for the busy spinner (shared name/shape with StatusBar's;
// guarded so mounting both never double-injects).
if (typeof document !== 'undefined' && !document.getElementById('spyde-spin-kf')) {
  const el = document.createElement('style')
  el.id = 'spyde-spin-kf'
  el.textContent = '@keyframes spyde-spin { to { transform: rotate(360deg) } }'
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
  const [tracebackOpen, setTracebackOpen] = useState(false)
  const [completions, setCompletions] = useState<string[] | null>(null)
  const [completionIdx, setCompletionIdx] = useState(0)

  const historyRef = useRef<string[]>(loadHistory())
  const historyPos = useRef<number>(-1)               // -1 = not navigating (draft is live)
  const draftRef = useRef('')                         // in-progress text, restored on ArrowDown past newest
  const execIdRef = useRef(0)
  const completeIdRef = useRef(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const flashTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const lastResult = state.consoleResult
  const chipVars = state.consoleVars.filter(v => v.source === 'assign' || v.source === 'out')

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

  const runExec = useCallback(() => {
    const trimmed = code
    if (!trimmed.trim()) return
    const id = ++execIdRef.current
    setBusy(true)
    setCompletions(null)
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
    })
  }, [code])

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
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

  // ── Drag-IN: dropping a SubWindow's console-ref grip inserts its variable
  //    name at the caret. Resolved via the latest console_vars "signal" rows
  //    (window_ids → name). Unknown window → flash the border, insert nothing.
  const onDragOver = (e: React.DragEvent) => {
    if (!e.dataTransfer.types.includes(SIGNAL_REF_DRAG_MIME)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }
  const onDrop = (e: React.DragEvent) => {
    const raw = e.dataTransfer.getData(SIGNAL_REF_DRAG_MIME)
    if (!raw) return
    e.preventDefault()
    let windowId: number | null = null
    try {
      const parsed = JSON.parse(raw) as { windowId?: number }
      windowId = parsed.windowId ?? null
    } catch { /* malformed payload */ }
    const name = windowId != null ? resolveSignalName(state.consoleVars, windowId) : null
    if (name == null) {
      setFlash(true)
      if (flashTimer.current) clearTimeout(flashTimer.current)
      flashTimer.current = setTimeout(() => setFlash(false), 420)
      return
    }
    const el = inputRef.current
    const caret = el?.selectionStart ?? code.length
    const next = code.slice(0, caret) + name + code.slice(caret)
    setCode(next)
    requestAnimationFrame(() => {
      const pos = caret + name.length
      el?.focus()
      el?.setSelectionRange(pos, pos)
    })
  }

  useEffect(() => () => { if (flashTimer.current) clearTimeout(flashTimer.current) }, [])

  // Empty value_repr (assignment/None) reads as a subtle "ok" rather than blank.
  const displayRepr = lastResult ? (lastResult.valueRepr || (lastResult.ok ? 'ok' : '')) : ''

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
        style={{ ...styles.inputRow, ...(flash ? styles.inputRowFlash : {}) }}
        onDragOver={onDragOver}
        onDrop={onDrop}
      >
        <span style={styles.prompt}>&gt;&gt;&gt;</span>
        <input
          ref={inputRef}
          data-testid="console-input"
          style={styles.input}
          type="text"
          spellCheck={false}
          autoComplete="off"
          placeholder="Python — Enter to run · Tab to complete"
          value={code}
          onChange={(e) => { setCode(e.target.value); setCompletions(null) }}
          onKeyDown={onKeyDown}
        />
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
  const badge = shapeDtypeBadge(v)
  return (
    <button
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
      <span style={styles.chipName}>{v.name}</span>
      {badge && <span style={styles.chipBadge}>{badge}</span>}
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
  prompt: {
    fontFamily: MONO, fontSize: 12, color: '#89b4fa', fontWeight: 700, flexShrink: 0,
    userSelect: 'none',
  },
  input: {
    flex: 1, minWidth: 0,
    background: 'transparent', border: 'none', outline: 'none',
    color: '#cdd6f4', fontFamily: MONO, fontSize: 12.5,
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
    display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 6,
    color: '#cdd6f4', cursor: 'grab', padding: '2px 8px', flexShrink: 0,
    lineHeight: 1.25,
  },
  chipName: { fontSize: 11, fontFamily: MONO, fontWeight: 600, color: '#89b4fa' },
  chipBadge: { fontSize: 9, fontFamily: MONO, color: '#a6adc8' },
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
