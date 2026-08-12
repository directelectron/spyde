import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// The shared main-process shell, consumed as raw TypeScript through an alias
// rather than a built artifact — same arrangement as SpyDE's config, and it has
// to agree with the `paths` entry in tsconfig.json or the editor and the bundler
// disagree about what resolves. Aliased in BOTH main and preload: each gets its
// own rollup pass.
const shellMain = resolve(__dirname, '..', '..', 'packages', 'shell-main', 'src', 'index.ts')
const shellRenderer = resolve(
  __dirname, '..', '..', 'packages', 'shell-renderer', 'src', 'index.ts')

export default defineConfig({
  main: {
    build: { outDir: 'out/main', rollupOptions: { input: 'src/main/index.ts' } },
    resolve: { alias: { '@de/shell-main': shellMain } },
  },
  preload: {
    build: { outDir: 'out/preload', rollupOptions: { input: 'src/preload/index.ts' } },
    resolve: { alias: { '@de/shell-main': shellMain } },
  },
  renderer: {
    root: 'src/renderer',
    build: { outDir: 'out/renderer' },
    plugins: [react()],
    resolve: { alias: { '@de/shell-renderer': shellRenderer } },
    server: {
      port: 5273,   // not SpyDE's 5173 — both dev servers may be up at once
      fs: { allow: [resolve(__dirname, '..', '..')] },
    },
  },
})
