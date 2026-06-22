/**
 * guides/index.ts — the guide registry. Both the in-app tour launcher and the
 * docs website import from here so they stay in sync.
 */
import type { Guide } from './types'
import { findVectorsGuide } from './find-vectors'

export type { Guide, GuideStep, Placement } from './types'

export const GUIDES: Guide[] = [
  findVectorsGuide,
]

export function getGuide(id: string): Guide | undefined {
  return GUIDES.find((g) => g.id === id)
}
