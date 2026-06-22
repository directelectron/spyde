import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// The renderer's coachmark tour renders the single-source guides from the
// repo-root guides/ dir (shared with the docs website so they never drift).
// @guides aliases that dir; fs.allow lets the dev server import from outside the
// renderer root.
const guidesDir = resolve(__dirname, '..', 'guides')

export default defineConfig({
  main: {
    build: { outDir: 'out/main', rollupOptions: { input: 'src/main/index.ts' } },
  },
  preload: {
    build: { outDir: 'out/preload', rollupOptions: { input: 'src/preload/index.ts' } },
  },
  renderer: {
    root: 'src/renderer',
    build: { outDir: 'out/renderer' },
    plugins: [react()],
    resolve: { alias: { '@guides': guidesDir } },
    server: { port: 5173, fs: { allow: [resolve(__dirname, '..')] } },
  },
})
