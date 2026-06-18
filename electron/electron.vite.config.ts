import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'

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
    server: { port: 5173 },
  },
})
