/**
 * format.ts — the one place a number becomes text for a label.
 *
 * Three byte formatters grew in three panels, each spelling the units and the
 * rounding differently; a size should read the same wherever it is shown.
 */

/** "1.5 MB" — SI units, one decimal below ten, none above. *empty* is what
 *  an absent or nonsensical size renders as (a blank, or a dash). */
export function formatBytes(value: number | null | undefined, empty = ''): string {
  if (value == null || !Number.isFinite(value) || value <= 0) return empty
  const units = ['B', 'kB', 'MB', 'GB', 'TB']
  let scaled = value
  let unit = 0
  while (scaled >= 1000 && unit < units.length - 1) { scaled /= 1000; unit += 1 }
  return `${scaled < 10 && unit > 0 ? scaled.toFixed(1) : Math.round(scaled)} ${units[unit]}`
}
