/**
 * guides/index.ts — the guide registry. Both the in-app tour launcher and the
 * docs website import from here so they stay in sync.
 */
import type { Guide } from './types'
import { findVectorsGuide } from './find-vectors'
import { virtualImagingGuide } from './virtual-imaging'
import { orientationGuide } from './orientation'

export type { Guide, GuideStep, GuideDrive, Placement } from './types'

export const GUIDES: Guide[] = [
  findVectorsGuide,
  virtualImagingGuide,
  orientationGuide,
]

export function getGuide(id: string): Guide | undefined {
  return GUIDES.find((g) => g.id === id)
}
