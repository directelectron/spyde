/**
 * reports/types.ts — the shape of a published report.
 *
 * A report is an EXPORTED SpyDE report page: one self-contained .html file with
 * its figures baked in and its interactive panels running client-side. The docs
 * site does not re-render it, it hosts it — so what lives here is only the
 * catalogue entry (what the report is about and where the file sits), never the
 * content.
 */

/** One key/value fact shown in the report's header strip. */
export interface ReportFact {
  label: string
  value: string
}

export interface Report {
  /** Stable id — the sidebar test id and the selected-tab key. */
  id: string
  title: string
  /** One or two sentences: what the reader gets out of opening it. */
  summary: string
  /**
   * File name under `docs-site/public/media/reports/`. Built by a generator in
   * `scripts/` and NOT committed when it is large — the site degrades to a
   * "not built" note rather than a broken frame (see ReportView).
   */
  file: string
  /** Header strip: dataset size, technique, instrument — whatever orients. */
  facts: ReportFact[]
  /** Where the data came from, rendered under the facts. */
  source?: { label: string; url?: string }
  /** The commands that rebuild the report file, shown so it is reproducible. */
  build?: string[]
}
