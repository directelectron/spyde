/**
 * report-catalogue/index.ts — the published-report registry.
 *
 * The docs site's Reports tab lists these; each entry points at a self-contained
 * .html file under docs-site/public/media/reports/. Same arrangement as
 * guides/index.ts: one registry, imported rather than duplicated.
 *
 * NB the directory is `report-catalogue/`, not `reports/`: a root `reports/` is
 * scratch output and is gitignored, so a registry there would never have been
 * committed.
 */
import type { Report } from './types'

export type { Report, ReportFact } from './types'

export const REPORTS: Report[] = [
  {
    id: 'pdcusi-crystallization',
    title: 'PdCuSi metallic glass — crystallization, in situ',
    summary:
      'A Pd–Cu–Si metallic glass crystallizing under the beam, recorded as a '
      + '4D-STEM series. 733,200 diffraction patterns reduced to 1.5 million '
      + 'diffraction vectors — and the crystallization onset falls out of the '
      + 'vector count alone. Includes a live explorer over the series.',
    file: 'pdcusi-crystallization.html',
    facts: [
      { label: 'Technique', value: '4D-STEM, in situ' },
      { label: 'Shape', value: '400 steps × 47 × 39 scan × 128² detector' },
      { label: 'Patterns', value: '733,200' },
      { label: 'Voltage', value: '200 kV' },
    ],
    source: {
      label: 'em-database — PdCuSiCrystallization (Carter Francis, UW–Madison)',
      url: 'https://pypi.org/project/em-database/',
    },
    build: [
      'python -m scripts.compute_pdcusi_vectors',
      'python -m scripts.gen_report_pdcusi',
    ],
  },
]

export function getReport(id: string): Report | undefined {
  return REPORTS.find((r) => r.id === id)
}
