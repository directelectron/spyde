// Types for shell-link.mjs — plain JavaScript because npm's postinstall runs it
// before any bundler is involved; this file is what lets the vite config import
// it under `strict`.

/** Where the installed de-shell keeps its TypeScript, by asking it. */
export function shellJsDir(electronDir?: string): string

/** Make `<electronDir>/shell` point at the installed shell's TypeScript and
 *  return the link path. Idempotent: an up-to-date link is left alone. */
export function ensureShellLink(electronDir?: string, options?: { quiet?: boolean }): string
