/**
 * figureBridge.ts — routing backend state into figure iframes.
 *
 * An anyplotlib figure lives in an iframe and is fed by `postMessage`. The
 * backend sends only CHANGES, so a message that arrives before its iframe has
 * mounted a listener is gone for good — silently, because posting into an
 * unmounted frame is a no-op, not an error. That single fact is why this is a
 * stateful bridge rather than a forwarding function: every state is RETAINED so
 * a frame can be replayed when it loads.
 *
 * Lifted from SpyDE's implementation, which had already paid for the three
 * non-obvious parts:
 *
 * 1. **Replay takes an explicit target.** A figure can be mounted more than
 *    once under the same figId (SpyDE's report cell and its presented slide),
 *    and the registry holds ONE element per id. Resolving the target from the
 *    map means whichever mounted last wins, so a freshly-loaded frame would
 *    push its state into its SIBLING and receive nothing. Which mount won was a
 *    race, so a presented deck came up blank on some machines and not others.
 *    A frame that has just loaded passes ITSELF.
 *
 * 2. **Binary frames stash per PANEL, not per pixel field.** `key` is the pixel
 *    field ("image_b64" / "detail_b64" / …) and is identical across panels; the
 *    panel is identified by `header.geom`. Keying by `key` alone let each panel
 *    overwrite the previous one, so a multi-panel figure retained exactly one
 *    frame however many panels it had — and replayed as one drawn panel and the
 *    rest blank-but-for-their-scale-bars.
 *
 * 3. **Transfer the buffer, but replay a COPY.** Transferring detaches the
 *    original, which would empty the stash the first time it was replayed.
 */

export interface BinaryFrame {
  key: string
  header: unknown
  buffer: Uint8Array
}

/** A `{current}` box, so these drop straight into a React ref position. */
export interface RefLike<T> {
  current: T
}

export interface FigureBridge {
  /** figId → the iframe currently registered for it. */
  iframes: RefLike<Map<string, HTMLIFrameElement>>
  /** figId → (state key → latest value). */
  states: RefLike<Map<string, Map<string, unknown>>>
  /** figId → (`geom::pixelField` → latest binary frame). */
  binaryStates: RefLike<Map<string, Map<string, BinaryFrame>>>

  registerIframe(figId: string, el: HTMLIFrameElement | null): void
  applyState(figId: string, key: string, value: unknown): void
  applyBinary(figId: string, key: string, header: unknown, bytes: Uint8Array): void
  replay(figId: string, target?: HTMLIFrameElement): void
  evict(figId: string): void
  post(figId: string, message: Record<string, unknown>, target?: HTMLIFrameElement): boolean
  dump(classify?: (el: HTMLIFrameElement | undefined) => string): Record<string, unknown>[]
}

export function createFigureBridge(
  log: (label: string, detail: Record<string, unknown>) => void = () => {},
): FigureBridge {
  const iframes: RefLike<Map<string, HTMLIFrameElement>> = { current: new Map() }
  const states: RefLike<Map<string, Map<string, unknown>>> = { current: new Map() }
  const binaryStates: RefLike<Map<string, Map<string, BinaryFrame>>> = { current: new Map() }

  function post(figId: string, message: Record<string, unknown>,
                target?: HTMLIFrameElement): boolean {
    const iframe = target ?? iframes.current.get(figId)
    if (!iframe?.contentWindow) return false
    iframe.contentWindow.postMessage(message, '*')
    return true
  }

  function registerIframe(figId: string, el: HTMLIFrameElement | null): void {
    if (el) iframes.current.set(figId, el)
    else iframes.current.delete(figId)
  }

  function applyState(figId: string, key: string, value: unknown): void {
    let slot = states.current.get(figId)
    if (!slot) { slot = new Map(); states.current.set(figId, slot) }
    slot.set(key, value)
    post(figId, { type: 'awi_state', key, value })
  }

  function applyBinary(figId: string, key: string, header: unknown,
                       bytes: Uint8Array): void {
    if (bytes) {
      let slot = binaryStates.current.get(figId)
      if (!slot) { slot = new Map(); binaryStates.current.set(figId, slot) }
      const geom = (header as { geom?: string } | undefined)?.geom
      // Mirrors the figure ESM's own slot convention, `geom::pixelKey`.
      const stashKey = geom ? `${geom}::${key}` : key
      // slice() because the postMessage below TRANSFERS the buffer and detaches
      // the original — the stash has to own its own copy.
      slot.set(stashKey, { key, header, buffer: bytes.slice() })
    }
    const iframe = iframes.current.get(figId)
    iframe?.contentWindow?.postMessage(
      { type: 'awi_state_binary', key, header, buffer: bytes },
      '*',
      bytes?.buffer ? [bytes.buffer] : [],
    )
  }

  function replay(figId: string, target?: HTMLIFrameElement): void {
    const iframe = target ?? iframes.current.get(figId)
    if (!iframe?.contentWindow) return
    const slot = states.current.get(figId)
    if (slot) {
      for (const [key, value] of slot) {
        iframe.contentWindow.postMessage({ type: 'awi_state', key, value }, '*')
      }
    }
    const binSlot = binaryStates.current.get(figId)
    if (binSlot) {
      for (const { key, header, buffer } of binSlot.values()) {
        // A COPY: transfer detaches, and the stash must survive a later remount
        // (a re-tile, a dev StrictMode double-mount, a second presented copy).
        const copy = buffer.slice()
        iframe.contentWindow.postMessage(
          { type: 'awi_state_binary', key, header, buffer: copy }, '*', [copy.buffer],
        )
      }
    }
    log('replayState', {
      figId,
      jsonKeys: slot ? slot.size : 0,
      binaryKeys: binSlot ? binSlot.size : 0,
      target: iframe.getAttribute('data-testid'),
    })
  }

  function evict(figId: string): void {
    states.current.delete(figId)
    binaryStates.current.delete(figId)
    iframes.current.delete(figId)
  }

  function dump(classify?: (el: HTMLIFrameElement | undefined) => string) {
    const rows: Record<string, unknown>[] = []
    const figIds = new Set<string>([
      ...states.current.keys(),
      ...binaryStates.current.keys(),
      ...iframes.current.keys(),
    ])
    for (const figId of figIds) {
      const el = iframes.current.get(figId)
      const rect = el?.getBoundingClientRect()
      const bin = binaryStates.current.get(figId)
      rows.push({
        figId,
        jsonKeys: states.current.get(figId)?.size ?? 0,
        binaryKeys: bin?.size ?? 0,
        binaryKeyNames: bin ? Array.from(bin.keys()).join(',') : '',
        registeredIn: classify ? classify(el) : (el ? 'mounted' : 'NONE'),
        size: rect ? `${Math.round(rect.width)}x${Math.round(rect.height)}` : 'n/a',
      })
    }
    return rows
  }

  return {
    iframes, states, binaryStates,
    registerIframe, applyState, applyBinary, replay, evict, post, dump,
  }
}
