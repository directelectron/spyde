/**
 * wizardHooks.tsx — the shared wizard lifecycle (the renderer side of the
 * staged-action protocol; backend contract in spyde/actions/README.md).
 *
 * A wizard caret and its backend controller speak `<key>_open / _close /
 * _tune / _run / _commit / _set_<param>` staged actions:
 *
 *   useWizardLifecycle — mount → open action, unmount → close action.
 *     StrictMode-safe: the open fire is deferred one tick and cancelled by
 *     StrictMode's synchronous mount→cleanup→remount, so exactly ONE open
 *     reaches the backend (whose run/stop generation guard remains as
 *     belt-and-braces); the close fires only if the open actually did.
 *   useDebouncedAction — the debounced live-tune sender (fv_tune /
 *     om_refine / vom_refine pattern); pending sends are cancelled on
 *     unmount so a tune can't fire at a torn-down preview.
 *   useWizardEvent — subscribe to a `spyde:<name>` CustomEvent re-broadcast
 *     by SpyDEContext, filtered to this window (events without a window_id
 *     pass through).
 *   CommitButton — the standard Commit affordance: promotes the wizard's
 *     live result into a NEW SignalTree by sending `<key>_commit` (backend:
 *     WizardController.commit() → commit_result_tree()).
 */
import React from 'react'
import { S } from './WizardShell'

export type SendAction = (
  action: string, payload?: Record<string, unknown>, windowId?: number,
) => void

export function useWizardLifecycle(opts: {
  windowId: number
  sendAction: SendAction
  /** staged action fired on mount (e.g. 'strain_open'); null → fire nothing */
  openAction: string | null
  /** payload builder, evaluated at fire time */
  openPayload?: () => Record<string, unknown>
  /** staged action fired on unmount (e.g. 'strain_close') */
  closeAction: string
  /** re-run the open/close pair when these change (e.g. the CZB tab) */
  deps?: React.DependencyList
}): void {
  const { windowId, sendAction, openAction, openPayload, closeAction } = opts
  React.useEffect(() => {
    let fired = false
    const t = setTimeout(() => {
      fired = true
      if (openAction) sendAction(openAction, openPayload ? openPayload() : {}, windowId)
    }, 0)
    return () => {
      clearTimeout(t)
      if (fired) sendAction(closeAction, {}, windowId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, opts.deps ?? [])
}

export function useDebouncedAction(
  sendAction: SendAction, action: string, windowId: number, ms = 120,
): (payload: () => Record<string, unknown>) => void {
  const timer = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  React.useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current)
  }, [])
  return React.useCallback((payload: () => Record<string, unknown>) => {
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => sendAction(action, payload(), windowId), ms)
  }, [sendAction, action, windowId, ms])
}

/**
 * useKeyedDebounce — a per-key debounced dispatcher. Two independent draggable
 * controls (e.g. two different layers' alpha sliders) each get their own timer
 * keyed by whatever string identifies them, so debouncing one never cancels or
 * delays the other (a single shared timer would make dragging layer B's slider
 * swallow layer A's pending send). Pending timers are cleared on unmount so a
 * fire can't land after the owning component (e.g. a closed wizard/editor) is
 * gone. Mirrors `useDebouncedAction` above, generalized to N independent keys.
 */
export function useKeyedDebounce(delay = 150): (key: string, fn: () => void) => void {
  const timers = React.useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  React.useEffect(() => {
    const t = timers.current
    return () => { t.forEach(clearTimeout); t.clear() }
  }, [])
  return React.useCallback((key: string, fn: () => void) => {
    const existing = timers.current.get(key)
    if (existing) clearTimeout(existing)
    timers.current.set(key, setTimeout(fn, delay))
  }, [delay])
}

export function useWizardEvent(
  name: string, windowId: number,
  handler: (detail: Record<string, unknown>) => void,
): void {
  const h = React.useRef(handler)
  h.current = handler
  React.useEffect(() => {
    const on = (e: Event) => {
      const d = (e as CustomEvent).detail as Record<string, unknown>
      if (d.window_id != null && d.window_id !== windowId) return
      h.current(d)
    }
    window.addEventListener(name, on)
    return () => window.removeEventListener(name, on)
  }, [name, windowId])
}

export function CommitButton({ wizardKey, windowId, sendAction, label, payload, onCommit }: {
  wizardKey: string
  windowId: number
  sendAction: SendAction
  label?: string
  payload?: () => Record<string, unknown>
  onCommit?: () => void
}): React.JSX.Element {
  return (
    <button data-testid={`${wizardKey}-commit`} style={S.primary}
      onClick={() => {
        sendAction(`${wizardKey}_commit`, payload ? payload() : {}, windowId)
        onCommit?.()
      }}>
      {label ?? 'Commit to New Tree'}
    </button>
  )
}
