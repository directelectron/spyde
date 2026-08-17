/**
 * figureBridge.react.ts — the React binding for the figure bridge.
 *
 * Split from `figureBridge.ts` so the bridge itself stays plain TypeScript:
 * it is stateful but not reactive, and keeping it free of React means it can be
 * unit-tested without a renderer.
 */
import { useRef } from 'react'
import { createFigureBridge, type FigureBridge } from './figureBridge'

export { createFigureBridge }
export type { FigureBridge }

/**
 * One bridge per component tree, with a STABLE identity for the life of the
 * component.
 *
 * The stability matters: the bridge holds every figure's retained state, so a
 * bridge rebuilt on re-render would drop it, and any figure that had already
 * painted would go blank the next time its iframe reloaded. It is also
 * depended on by effects — a changing identity would re-run them every render.
 */
export function useFigureBridge(
  log?: (label: string, detail: Record<string, unknown>) => void,
): FigureBridge {
  const ref = useRef<FigureBridge | null>(null)
  if (ref.current === null) ref.current = createFigureBridge(log)
  return ref.current
}
