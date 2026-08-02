/**
 * ThemePanel.tsx — the deck THEME editor (presentations only).
 *
 * One modal covering everything that makes a deck look like YOURS: colours,
 * type, the footer bar (name / email / affiliation / slide numbers), and a
 * logo. Every edit dispatches `report_set_theme` with just the field that
 * changed — the backend MERGES, so a partial patch never resets the rest — and
 * the deck behind the modal restyles live.
 *
 * TWO SCOPES, deliberately separate:
 *   • The theme lives in the DOCUMENT, so a talk keeps its look when you reopen
 *     it or send the file to a colleague.
 *   • "Set as default" copies the current theme into settings.json, and every
 *     NEW deck starts from that. "Reset" goes back to the BUILT-IN look, not to
 *     your default — otherwise a customised default leaves no way back.
 *
 * The logo is embedded as a data: URL rather than a path, for the same reason:
 * the deck has to survive being handed to someone whose disk doesn't have your
 * logo file on it.
 */
import React, { useRef, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { DECK_THEME_DEFAULTS, type DeckTheme } from '../kernel/protocol'

/** Cap an embedded logo. It rides in the document AND in every report_state
 *  broadcast, so a 10 MB PNG would bloat both for a 30 px-tall mark. */
const LOGO_MAX_BYTES = 2 * 1024 * 1024

/** One-click starting points. Not a replacement for the controls — a way to be
 *  somewhere reasonable in one click rather than picking six colours cold. */
const PRESETS: { name: string; theme: Partial<DeckTheme> }[] = [
  { name: 'Midnight', theme: { bg: '#14141f', text: '#e8e8f0', muted: '#a6adc8', accent: '#89b4fa' } },
  { name: 'Ink', theme: { bg: '#0b0b0f', text: '#f2f2f5', muted: '#9a9aa8', accent: '#f5c2e7' } },
  { name: 'Paper', theme: { bg: '#f6f5f2', text: '#1b1b20', muted: '#5c5c68', accent: '#2f6fd0' } },
  { name: 'Slate', theme: { bg: '#1e2430', text: '#e6ecf5', muted: '#9fb0c6', accent: '#4fd6be' } },
]

export function ThemePanel({ theme, onClose }: {
  theme: DeckTheme
  onClose: () => void
}) {
  const { sendAction } = useSpyDE()
  const fileRef = useRef<HTMLInputElement>(null)
  const [note, setNote] = useState<string>('')

  const patch = (p: Partial<DeckTheme>) => sendAction('report_set_theme', { theme: p })

  // Escape closes — matches every other overlay in the app.
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  const pickLogo = async (file: File) => {
    if (file.size > LOGO_MAX_BYTES) {
      setNote(`That image is ${(file.size / 1e6).toFixed(1)} MB — keep a logo under 2 MB.`)
      return
    }
    const dataUrl = await new Promise<string>((resolve, reject) => {
      const r = new FileReader()
      r.onload = () => resolve(String(r.result ?? ''))
      r.onerror = () => reject(r.error)
      r.readAsDataURL(file)
    }).catch(() => '')
    if (!dataUrl) { setNote('Could not read that image.'); return }
    setNote('')
    patch({ logo: dataUrl })
  }

  const colorRow = (label: string, key: keyof DeckTheme, hint: string) => (
    <label style={styles.colorRow} key={key}>
      <input
        type="color"
        data-testid={`theme-color-${key}`}
        value={String(theme[key] ?? '#000000')}
        onChange={(e) => patch({ [key]: e.target.value } as Partial<DeckTheme>)}
        style={styles.swatch}
      />
      <span style={styles.colorLabel}>
        {label}
        <span style={styles.hint}>{hint}</span>
      </span>
      <input
        type="text"
        data-testid={`theme-hex-${key}`}
        value={String(theme[key] ?? '')}
        onChange={(e) => patch({ [key]: e.target.value } as Partial<DeckTheme>)}
        style={styles.hex}
        spellCheck={false}
      />
    </label>
  )

  return (
    <div style={styles.backdrop} data-testid="theme-panel-backdrop" onMouseDown={onClose}>
      <div
        style={styles.panel}
        data-testid="theme-panel"
        onMouseDown={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Deck theme"
      >
        <div style={styles.header}>
          <span style={styles.title}>Deck theme</span>
          <button style={styles.closeBtn} data-testid="theme-close" onClick={onClose}>✕</button>
        </div>

        <div style={styles.body}>
          {/* ── Presets ─────────────────────────────────────────────────────── */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Presets</div>
            <div style={styles.presetRow}>
              {PRESETS.map(p => (
                <button
                  key={p.name}
                  data-testid={`theme-preset-${p.name.toLowerCase()}`}
                  style={styles.preset}
                  title={`Apply the ${p.name} palette`}
                  onClick={() => patch(p.theme)}
                >
                  <span style={{ ...styles.presetChip, background: p.theme.bg }}>
                    <span style={{ ...styles.presetDot, background: p.theme.accent }} />
                  </span>
                  {p.name}
                </button>
              ))}
            </div>
          </div>

          {/* ── Colours ─────────────────────────────────────────────────────── */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Colours</div>
            {colorRow('Background', 'bg', 'the slide')}
            {colorRow('Text', 'text', 'body copy')}
            {colorRow('Muted', 'muted', 'subtitles, captions, footer')}
            {colorRow('Accent', 'accent', 'headings, rules, links')}
          </div>

          {/* ── Type ────────────────────────────────────────────────────────── */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Type</div>
            <input
              type="text"
              data-testid="theme-font"
              value={theme.font}
              placeholder="Font stack — blank for the app default"
              onChange={(e) => patch({ font: e.target.value })}
              style={styles.textInput}
              spellCheck={false}
            />
          </div>

          {/* ── Logo ────────────────────────────────────────────────────────── */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Logo</div>
            <div style={styles.logoRow}>
              <div style={styles.logoPreview}>
                {theme.logo
                  ? <img src={theme.logo} alt="" data-testid="theme-logo-preview"
                         style={{ maxHeight: 44, maxWidth: 150, objectFit: 'contain' }} />
                  : <span style={styles.hint}>none</span>}
              </div>
              <div style={styles.logoBtns}>
                <button style={styles.btn} data-testid="theme-logo-pick"
                        onClick={() => fileRef.current?.click()}>Choose image…</button>
                {theme.logo && (
                  <button style={styles.btnQuiet} data-testid="theme-logo-clear"
                          onClick={() => patch({ logo: '' })}>Remove</button>
                )}
              </div>
            </div>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              data-testid="theme-logo-input"
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void pickLogo(f)
                e.target.value = ''
              }}
            />
            {theme.logo && (
              <label style={styles.sliderRow}>
                <span style={styles.sliderLabel}>Height</span>
                <input
                  type="range" min={12} max={80} step={1}
                  data-testid="theme-logo-height"
                  value={theme.logo_height}
                  onChange={(e) => patch({ logo_height: Number(e.target.value) })}
                  style={{ flex: 1 }}
                />
                <span style={styles.sliderValue}>{theme.logo_height}px</span>
              </label>
            )}
          </div>

          {/* ── Footer ──────────────────────────────────────────────────────── */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>
              Footer
              <span style={styles.hint}>shown on every slide except the title</span>
            </div>
            <label style={styles.checkRow}>
              <input
                type="checkbox"
                data-testid="theme-footer-show"
                checked={theme.footer_show}
                onChange={(e) => patch({ footer_show: e.target.checked })}
              />
              Show the footer bar
            </label>
            <input type="text" data-testid="theme-footer-name" value={theme.footer_name}
                   placeholder="Name" style={styles.textInput}
                   onChange={(e) => patch({ footer_name: e.target.value })} />
            <input type="text" data-testid="theme-footer-email" value={theme.footer_email}
                   placeholder="Email" style={styles.textInput}
                   onChange={(e) => patch({ footer_email: e.target.value })} />
            <input type="text" data-testid="theme-footer-note" value={theme.footer_note}
                   placeholder="Affiliation, conference, date…" style={styles.textInput}
                   onChange={(e) => patch({ footer_note: e.target.value })} />
            <label style={styles.checkRow}>
              <input
                type="checkbox"
                data-testid="theme-slide-numbers"
                checked={theme.slide_numbers}
                onChange={(e) => patch({ slide_numbers: e.target.checked })}
              />
              Slide numbers
            </label>
          </div>

          {note && <div style={styles.note} data-testid="theme-note">{note}</div>}
        </div>

        {/* ── Scope actions ─────────────────────────────────────────────────── */}
        <div style={styles.footerBar}>
          <button
            style={styles.btnQuiet}
            data-testid="theme-reset"
            title="Back to SpyDE's built-in look (not your saved default)"
            onClick={() => sendAction('report_theme_reset', {})}
          >Reset</button>
          <button
            style={styles.btnQuiet}
            data-testid="theme-use-default"
            title="Apply your saved default theme to this deck"
            onClick={() => sendAction('report_theme_use_default', {})}
          >Use my default</button>
          <div style={{ flex: 1 }} />
          <button
            style={styles.btnPrimary}
            data-testid="theme-set-default"
            title="Every new deck will start from this theme"
            onClick={() => { sendAction('report_theme_set_default', {}); setNote('Saved as your default.') }}
          >Set as default</button>
        </div>
      </div>
    </div>
  )
}

/** Merge the backend theme over the defaults — one place, so callers don't each
 *  guard against an older backend shipping no theme. */
export function resolveTheme(raw: Partial<DeckTheme> | undefined | null): DeckTheme {
  return { ...DECK_THEME_DEFAULTS, ...(raw ?? {}) }
}

const styles: Record<string, React.CSSProperties> = {
  backdrop: {
    position: 'fixed', inset: 0, zIndex: 9600,
    background: 'rgba(8,8,12,0.55)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  panel: {
    width: 420, maxHeight: '86vh', display: 'flex', flexDirection: 'column',
    background: '#181825', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 12,
    boxShadow: '0 18px 50px rgba(0,0,0,0.6)',
    fontSize: 13,
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '12px 14px', borderBottom: '1px solid #313244',
  },
  title: { fontSize: 14, fontWeight: 700 },
  closeBtn: {
    background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer',
    fontSize: 15, width: 26, height: 26, borderRadius: 6,
  },
  body: { padding: '4px 14px 12px', overflowY: 'auto' },
  section: { padding: '10px 0', borderBottom: '1px solid #26263a' },
  sectionTitle: {
    fontSize: 11, fontWeight: 700, color: '#7f849c',
    textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8,
    display: 'flex', alignItems: 'baseline', gap: 8,
  },
  hint: { fontSize: 10.5, color: '#585b70', fontWeight: 500, textTransform: 'none', letterSpacing: 0 },
  presetRow: { display: 'flex', gap: 6 },
  preset: {
    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5,
    background: 'none', border: '1px solid #313244', borderRadius: 8,
    color: '#cdd6f4', fontSize: 11, padding: '7px 4px', cursor: 'pointer',
  },
  presetChip: {
    width: '100%', height: 22, borderRadius: 5, border: '1px solid #45475a',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  presetDot: { width: 10, height: 10, borderRadius: '50%' },
  colorRow: { display: 'flex', alignItems: 'center', gap: 9, margin: '6px 0' },
  swatch: {
    width: 30, height: 26, padding: 0, border: '1px solid #45475a',
    borderRadius: 5, background: 'none', cursor: 'pointer', flex: '0 0 auto',
  },
  colorLabel: { flex: 1, display: 'flex', flexDirection: 'column', lineHeight: 1.25 },
  hex: {
    width: 84, background: '#11111b', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 5, padding: '4px 6px',
    fontSize: 11.5, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  textInput: {
    width: '100%', boxSizing: 'border-box', margin: '4px 0',
    background: '#11111b', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 6, padding: '6px 8px', fontSize: 12.5,
  },
  checkRow: { display: 'flex', alignItems: 'center', gap: 7, margin: '6px 0', cursor: 'pointer' },
  logoRow: { display: 'flex', alignItems: 'center', gap: 12 },
  logoPreview: {
    flex: 1, minHeight: 52, display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: '#11111b', border: '1px dashed #45475a', borderRadius: 8, padding: 6,
  },
  logoBtns: { display: 'flex', flexDirection: 'column', gap: 5 },
  sliderRow: { display: 'flex', alignItems: 'center', gap: 9, marginTop: 9 },
  sliderLabel: { fontSize: 11.5, color: '#a6adc8', width: 44 },
  sliderValue: { fontSize: 11, color: '#7f849c', width: 38, textAlign: 'right' },
  note: { marginTop: 10, fontSize: 11.5, color: '#f9e2af' },
  footerBar: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '10px 14px', borderTop: '1px solid #313244',
  },
  btn: {
    background: 'rgba(137,180,250,0.14)', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 6,
    padding: '5px 10px', fontSize: 12, cursor: 'pointer',
  },
  btnQuiet: {
    background: 'none', color: '#a6adc8',
    border: '1px solid #313244', borderRadius: 6,
    padding: '5px 10px', fontSize: 12, cursor: 'pointer',
  },
  btnPrimary: {
    background: '#89b4fa', color: '#11111b', border: 'none', borderRadius: 6,
    padding: '6px 12px', fontSize: 12, fontWeight: 700, cursor: 'pointer',
  },
}
