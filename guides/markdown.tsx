/**
 * markdown.tsx — a tiny markdown renderer shared by the in-app tour and the docs
 * website so guide text renders identically in both. Deliberately minimal:
 * paragraphs, **bold**, and `> ` blockquote callouts (used for "> 💡" tips). No
 * external dependency — the guide bodies only use these constructs.
 */
import React from 'react'

/** Render inline **bold** within a line. */
function inline(text: string, keyPrefix: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((p, i) => {
    if (p.startsWith('**') && p.endsWith('**')) {
      return <strong key={`${keyPrefix}-b${i}`}>{p.slice(2, -2)}</strong>
    }
    return <React.Fragment key={`${keyPrefix}-t${i}`}>{p}</React.Fragment>
  })
}

export interface MarkdownStyles {
  paragraph?: React.CSSProperties
  callout?: React.CSSProperties
}

/**
 * Render a small markdown string. Blank-line-separated blocks become paragraphs;
 * a block whose lines all start with `> ` becomes a callout box.
 */
export function Markdown({
  text,
  styles = {},
}: {
  text: string
  styles?: MarkdownStyles
}): React.ReactElement {
  const blocks = text.split(/\n\n+/)
  return (
    <>
      {blocks.map((block, i) => {
        const lines = block.split('\n')
        const isCallout = lines.every((l) => l.startsWith('>'))
        if (isCallout) {
          const inner = lines.map((l) => l.replace(/^>\s?/, '')).join(' ')
          return (
            <div key={`blk-${i}`} style={styles.callout}>
              {inline(inner, `cb-${i}`)}
            </div>
          )
        }
        return (
          <p key={`blk-${i}`} style={styles.paragraph}>
            {inline(block.replace(/\n/g, ' '), `p-${i}`)}
          </p>
        )
      })}
    </>
  )
}
