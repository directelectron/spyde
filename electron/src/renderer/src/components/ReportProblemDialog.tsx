/**
 * ReportProblemDialog.tsx — Help -> Report a Problem…
 *
 * What the user writes is the smaller half of a report. The rest — OS, app and
 * runtime versions, GPU, the state of the managed Python environment, the last
 * thing the updater said, and the tail of the backend's output — is collected
 * by the main process (packages/shell-main/src/errorReport.ts) and shown here
 * BEFORE anything is sent, because a report that quietly ships a machine
 * description is a report people learn not to send.
 *
 * Nothing leaves the machine until Send is pressed, and a copy is always
 * written to disk so an offline instrument PC still produces something the user
 * can attach to an email.
 */
import React, { useEffect, useState } from 'react'

type Phase = 'writing' | 'sending' | 'done'

interface SubmitResult {
  sent: boolean
  eventId?: string
  bundlePath?: string
  error?: string
}

export function ReportProblemDialog({ onClose }: { onClose: () => void }) {
  const [message, setMessage] = useState('')
  const [contact, setContact] = useState('')
  const [canSend, setCanSend] = useState(false)
  const [diagnostics, setDiagnostics] = useState<Record<string, unknown> | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [phase, setPhase] = useState<Phase>('writing')
  const [result, setResult] = useState<SubmitResult | null>(null)

  useEffect(() => {
    let cancelled = false
    window.electron.reportDiagnostics().then((info) => {
      if (cancelled) return
      setCanSend(info.canSend)
      setDiagnostics(info.diagnostics)
    }).catch(() => { /* the dialog still works; details just stay empty */ })
    return () => { cancelled = true }
  }, [])

  const submit = async () => {
    setPhase('sending')
    try {
      setResult(await window.electron.submitReport({ message, contact }))
    } catch (err) {
      setResult({ sent: false, error: String(err) })
    }
    setPhase('done')
  }

  const problems = countProblems(diagnostics)

  return (
    <div style={styles.overlay} data-testid="report-problem-dialog">
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>Report a Problem</h3>

        {phase === 'done' && result ? (
          <Outcome result={result} canSend={canSend} onClose={onClose} />
        ) : (
          <>
            <p style={styles.sub}>
              {canSend
                ? 'This goes straight to the SpyDE maintainers, with the details below attached.'
                : 'This build has no reporting service configured, so the report will be saved '
                  + 'to your computer for you to send on.'}
            </p>

            <label style={styles.label} htmlFor="report-message">
              What happened?
            </label>
            <textarea
              id="report-message"
              data-testid="report-message"
              style={styles.textarea}
              rows={6}
              autoFocus
              placeholder={'What were you doing, and what did SpyDE do instead?\n\n'
                + 'If a file was involved, its format and rough size helps a lot.'}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
            />

            <label style={styles.label} htmlFor="report-contact">
              Email (optional, so we can ask a follow-up question)
            </label>
            <input
              id="report-contact"
              data-testid="report-contact"
              style={styles.input}
              type="email"
              placeholder="you@example.org"
              value={contact}
              onChange={(e) => setContact(e.target.value)}
            />

            <button
              style={styles.disclosure}
              data-testid="report-toggle-details"
              onClick={() => setShowDetails((v) => !v)}
            >
              {showDetails ? '▾' : '▸'} Details included with this report
              {problems > 0 && ` (${problems} recent error${problems === 1 ? '' : 's'})`}
            </button>
            {showDetails && (
              <pre style={styles.details} data-testid="report-details">
                {diagnostics ? JSON.stringify(diagnostics, null, 2) : 'Collecting…'}
              </pre>
            )}

            <div style={styles.footer}>
              <button style={styles.cancel} data-testid="report-cancel" onClick={onClose}>
                Cancel
              </button>
              <button
                style={{
                  ...styles.primary,
                  ...(message.trim() && phase === 'writing' ? {} : styles.primaryDisabled),
                }}
                data-testid="report-send"
                disabled={!message.trim() || phase !== 'writing'}
                onClick={submit}
              >
                {phase === 'sending' ? 'Sending…' : canSend ? 'Send Report' : 'Save Report'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

/**
 * What actually happened, including where the local copy went.
 *
 * `canSend` distinguishes the two ways a report ends up on disk: a build with
 * no reporting service was never going to send it, and saying "could not be
 * sent" there reads as a failure the user should chase. A build that CAN send
 * and did not has a real reason worth showing.
 */
function Outcome({ result, canSend, onClose }: {
  result: SubmitResult
  canSend: boolean
  onClose: () => void
}) {
  return (
    <div data-testid="report-outcome">
      <p style={{ ...styles.sub, color: result.sent ? '#a6e3a1' : '#f9e2af' }}>
        {result.sent
          ? 'Report sent. Thank you — this is genuinely how the Windows and GPU '
            + 'problems get found.'
          : canSend
            ? 'The report could not be sent, so it was saved on this computer.'
            : 'Report saved on this computer.'}
      </p>
      {result.error && !result.sent && canSend && (
        <p style={styles.sub} data-testid="report-error">{result.error}</p>
      )}
      {result.bundlePath && (
        <>
          <p style={styles.label}>Saved copy</p>
          <pre style={styles.details} data-testid="report-bundle-path">{result.bundlePath}</pre>
          <p style={styles.sub}>
            {result.sent
              ? 'Kept so you can see exactly what was sent.'
              : 'Attach this file to an email and we can pick it up from there.'}
          </p>
        </>
      )}
      {result.sent && result.eventId && (
        <>
          <p style={styles.label}>Reference</p>
          <pre style={styles.details} data-testid="report-event-id">{result.eventId}</pre>
        </>
      )}
      <div style={styles.footer}>
        <button style={styles.primary} data-testid="report-close" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  )
}

/** How many errors the app already noticed, so the user knows the report is not
 *  starting from nothing. Tolerant of a shape it does not recognise. */
function countProblems(diagnostics: Record<string, unknown> | null): number {
  const problems = (diagnostics as { problems?: unknown } | null)?.problems
  return Array.isArray(problems) ? problems.length : 0
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 9500,
    background: 'rgba(17,17,27,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  dialog: {
    width: 520, maxHeight: '80vh', overflowY: 'auto',
    display: 'flex', flexDirection: 'column',
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 10,
    padding: 18, color: '#cdd6f4', boxShadow: '0 16px 40px rgba(0,0,0,0.55)',
    fontSize: 13,
  },
  title: { margin: '0 0 12px', fontSize: 16, fontWeight: 600 },
  sub: { margin: '0 0 14px', fontSize: 12, color: '#a6adc8', lineHeight: 1.5 },
  label: { display: 'block', margin: '0 0 6px', fontSize: 12, color: '#a6adc8' },
  textarea: {
    width: '100%', boxSizing: 'border-box', marginBottom: 14, resize: 'vertical',
    background: '#11111b', border: '1px solid #313244', borderRadius: 6,
    padding: '8px 10px', color: '#cdd6f4', fontSize: 12.5,
    fontFamily: 'inherit', lineHeight: 1.5,
  },
  input: {
    width: '100%', boxSizing: 'border-box', marginBottom: 14,
    background: '#11111b', border: '1px solid #313244', borderRadius: 6,
    padding: '7px 10px', color: '#cdd6f4', fontSize: 12.5, fontFamily: 'inherit',
  },
  disclosure: {
    alignSelf: 'flex-start', background: 'transparent', border: 'none',
    color: '#a6adc8', cursor: 'pointer', fontSize: 12, padding: 0, marginBottom: 8,
  },
  details: {
    maxHeight: 220, overflow: 'auto', margin: '0 0 14px',
    background: '#11111b', border: '1px solid #313244', borderRadius: 6,
    padding: '10px 12px', fontSize: 11, lineHeight: 1.45,
    whiteSpace: 'pre-wrap', wordBreak: 'break-all',
  },
  footer: { display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 'auto' },
  cancel: {
    background: 'transparent', border: '1px solid #313244', color: '#cdd6f4',
    borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 12,
  },
  primary: {
    background: '#89b4fa', border: '1px solid #89b4fa', color: '#11111b',
    borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 12,
    fontWeight: 600,
  },
  primaryDisabled: {
    background: '#313244', border: '1px solid #313244', color: '#6c7086',
    cursor: 'default',
  },
}
