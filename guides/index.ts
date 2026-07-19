/**
 * guides/index.ts — the guide registry. Both the in-app tour launcher and the
 * docs website import from here so they stay in sync.
 */
import type { Guide } from './types'
import { welcomeGuide } from './welcome'
import { findVectorsGuide } from './find-vectors'
import { virtualImagingGuide } from './virtual-imaging'
import { orientationGuide } from './orientation'
import { strainGuide } from './strain'
import { spectroscopyGuide } from './spectroscopy'

export type { Guide, GuideStep, GuideDrive, Placement } from './types'

export const GUIDES: Guide[] = [
  welcomeGuide,
  findVectorsGuide,
  virtualImagingGuide,
  orientationGuide,
  strainGuide,
  spectroscopyGuide,
]

export function getGuide(id: string): Guide | undefined {
  return GUIDES.find((g) => g.id === id)
}
