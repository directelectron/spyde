/**
 * colormaps.ts — the shared colormap list offered by every colormap <select>
 * in the app (Plot Control dock's Layers section, Report figure-cell layer
 * editor, …). Single source of truth so the set can't drift between the two
 * pickers.
 */
export const COLORMAPS: string[] = [
  'gray', 'viridis', 'inferno', 'magma', 'plasma',
  'cividis', 'hot', 'jet', 'turbo', 'twilight',
]
