import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// The renderer's coachmark tour renders the single-source guides from the
// repo-root guides/ dir (shared with the docs website so they never drift).
// @guides aliases that dir; fs.allow lets the dev server import from outside the
// renderer root.
const guidesDir = resolve(__dirname, '..', 'guides')

// SPYDE_NO_HMR=1 — for a long dev session that has to survive the laptop
// sleeping.
//
// Vite's HMR client reloads the page whenever its websocket drops and comes
// back (`vite/dist/client/client.mjs`: on close it logs "server connection
// lost", polls for a successful ping, then calls `location.reload()`). A
// suspend does exactly that. The reload wipes the renderer's React state — which
// is where the ENTIRE workspace lives, since nothing persists or rebuilds the
// window list from the backend — so every plot window disappears while the
// Python process and its Dask cluster carry on untouched. Measured on a real
// lid-close: the dev build reloads (`navType: "reload"`, a fresh
// `performance.timeOrigin`, zero subwindows), the packaged build survives two
// sleep cycles with the workspace intact. Packaged builds load from a file and
// never fetch this client, so this is a dev-only failure.
//
// `ws: false` is the switch, NOT `hmr: false`. With `hmr: false` Vite still
// attaches the websocket server to the dev server (`createWebSocketServer`:
// `wsServer = hmrServer || portsAreCompatible && server`), so the socket still
// opens, still drops on suspend, and still reloads — the flag would look right
// and change nothing. With `ws: false` the endpoint doesn't exist, the client's
// socket never opens, and its close handler returns at the `!isOpened` guard
// before reaching `location.reload()`.
//
// The cost is real, hence opt-in: no hot reload, and the client logs a failed
// connection attempt to the renderer console.
const noHmr = process.env.SPYDE_NO_HMR === '1'

// The shared main-process shell (@de/shell-main) is consumed as RAW TypeScript
// via an alias rather than a built artifact, so editing the shell and running
// the app needs no intermediate build. It must be aliased in BOTH the main and
// preload configs — each gets its own rollup pass — and the tsconfig `paths`
// entry in tsconfig.node.json has to agree, or the editor and the bundler
// disagree about what resolves.
const shellMain = resolve(__dirname, '..', 'packages', 'shell-main', 'src', 'index.ts')
const shellPreload = resolve(
  __dirname, '..', 'packages', 'shell-preload', 'src', 'index.ts')
const shellRenderer = resolve(
  __dirname, '..', 'packages', 'shell-renderer', 'src', 'index.ts')

export default defineConfig({
  main: {
    build: { outDir: 'out/main', rollupOptions: { input: 'src/main/index.ts' } },
    resolve: { alias: { '@de/shell-main': shellMain } },
  },
  preload: {
    build: { outDir: 'out/preload', rollupOptions: { input: 'src/preload/index.ts' } },
    resolve: { alias: { '@de/shell-main': shellMain, '@de/shell-preload': shellPreload } },
  },
  renderer: {
    root: 'src/renderer',
    build: { outDir: 'out/renderer' },
    plugins: [react()],
    resolve: { alias: { '@guides': guidesDir, '@de/shell-renderer': shellRenderer } },
    server: {
      port: 5173,
      fs: { allow: [resolve(__dirname, '..')] },
      ...(noHmr ? { ws: false as const } : {}),
    },
  },
})
