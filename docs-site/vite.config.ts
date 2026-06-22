import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// The docs site renders the SAME guide modules the app uses (../guides), so the
// website and the in-app tour never drift. `fs.allow` lets Vite import from the
// repo-root guides/ dir which sits outside this package.
export default defineConfig({
  root: '.',
  plugins: [react()],
  resolve: {
    alias: { '@guides': resolve(__dirname, '..', 'guides') },
  },
  server: {
    fs: { allow: [resolve(__dirname, '..')] },
  },
  // Relative base so the built site works from any subpath (e.g. GitHub Pages).
  base: './',
})
