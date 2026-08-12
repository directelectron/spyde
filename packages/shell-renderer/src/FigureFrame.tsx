/**
 * FigureFrame.tsx — an anyplotlib figure in an iframe, wired to the bridge.
 *
 * Handles the three things every host of a figure has to get right, and which
 * are easy to omit without any error appearing:
 *
 * * **Register with the bridge**, so state can be routed to this frame — and
 *   deregister on unmount, so a dead element is not held.
 * * **Replay on load**, passing THIS element. State that arrived before the
 *   frame mounted was posted into the void; replay is the only thing that
 *   recovers it, and a figure mounted twice must serve itself rather than
 *   whichever mount happens to hold the registry slot.
 * * **Report its size**, so the backend can lay the figure out to fit. Without
 *   it the figure renders at anyplotlib's default size and overflows its pane.
 *
 * `srcdoc` vs `src`: a backend that inlines the figure's ESM can be mounted
 * directly from `html`. One that swaps the bundle for a shared URL (SpyDE does,
 * so Chromium reuses the V8 code cache across many figure iframes) MUST be
 * served over a real origin and passes `fileUrl` instead — a srcdoc frame
 * cannot load it. Note that a srcdoc frame INHERITS the parent page's CSP, so
 * the host page needs `script-src … blob:` for anyplotlib's ESM boot.
 */
import React, { useEffect, useRef } from 'react'
import type { FigureBridge } from './figureBridge'

export interface FigureFrameProps {
  bridge: FigureBridge
  figId: string
  /** Inline figure HTML (mounted via srcdoc). Ignored when `fileUrl` is set. */
  html?: string
  /** Figure URL served over the app's own scheme. Takes precedence over `html`. */
  fileUrl?: string | null
  title?: string
  /** Called with the frame's pixel size whenever it changes. */
  onResize?: (width: number, height: number) => void
  className?: string
  style?: React.CSSProperties
  'data-testid'?: string
}

export function FigureFrame({
  bridge, figId, html, fileUrl, title, onResize, className, style,
  'data-testid': testId,
}: FigureFrameProps) {
  const ref = useRef<HTMLIFrameElement | null>(null)

  // Report size to the backend. Fires once on mount and on every resize; the
  // zero-size guard skips the frame's first layout pass, which would otherwise
  // tell the backend to lay the figure out at 0×0.
  useEffect(() => {
    const el = ref.current
    if (!el || !onResize) return
    const send = () => {
      const r = el.getBoundingClientRect()
      if (r.width > 0 && r.height > 0) {
        onResize(Math.round(r.width), Math.round(r.height))
      }
    }
    send()
    const ro = new ResizeObserver(send)
    ro.observe(el)
    return () => ro.disconnect()
  }, [figId, onResize])

  // Deregister on unmount so the bridge never holds a detached element.
  useEffect(() => () => bridge.registerIframe(figId, null), [bridge, figId])

  return (
    <iframe
      ref={(el) => { ref.current = el; bridge.registerIframe(figId, el) }}
      // Keyed by figId so React REUSES the element across repaints. Remounting
      // per frame would tear down the figure's WebGPU context and throw away
      // the user's zoom.
      key={figId}
      title={title ?? figId}
      {...(fileUrl ? { src: fileUrl } : { srcDoc: html })}
      className={className}
      style={style}
      data-testid={testId}
      // ITSELF, never whichever mount currently holds the registry slot.
      onLoad={() => bridge.replay(figId, ref.current ?? undefined)}
    />
  )
}
