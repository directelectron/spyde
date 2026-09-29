/**
 * GpuStatusDialog.tsx — Help -> GPU Status…
 *
 * Surfaces spyde.torch_device's existing device diagnostics
 * (select_device / gpu_available / gpu_unavailable_reason / torch_available)
 * so a silent CPU fallback is never a mystery. Requests `get_gpu_status` on
 * open and listens for the `gpu_status_result` DOM CustomEvent the same way
 * the phase popout listens for `cod_results` (SpyDEContext.tsx
 * re-broadcasts wizard-scoped PLOTAPP messages as DOM events).
 */
import React, { useEffect, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { ModalDialog, S } from './WizardShell'

interface GpuStatusResult {
  torch_available: boolean
  torch_version: string | null
  device: string | null
  gpu_available: boolean
  reason: string
}

export function GpuStatusDialog({ onClose }: { onClose: () => void }) {
  const { sendAction } = useSpyDE()
  const [result, setResult] = useState<GpuStatusResult | null>(null)

  useEffect(() => {
    const onResult = (e: Event) => setResult((e as CustomEvent).detail as GpuStatusResult)
    window.addEventListener('spyde:gpu_status_result', onResult)
    sendAction('get_gpu_status')
    return () => window.removeEventListener('spyde:gpu_status_result', onResult)
  }, [])

  return (
    <ModalDialog testid="gpu-status-dialog" title="GPU Status" width={380} onClose={onClose}
      footer={
        <button data-testid="gpu-status-close" style={S.dialogCancel} onClick={onClose}>
          Close
        </button>
      }>
      {!result ? (
        <p style={styles.sub}>Checking…</p>
      ) : (
        <div style={styles.rows} data-testid="gpu-status-result">
          <Row label="Accelerated" value={result.gpu_available ? 'Yes' : 'No'}
               tone={result.gpu_available ? 'good' : 'warn'} />
          <Row label="Device" value={result.device ?? '—'} />
          <Row label="torch" value={result.torch_available
            ? `${result.torch_version ?? 'unknown version'}`
            : 'not installed'} />
          <Row label="Details" value={result.reason} multiline />
        </div>
      )}
    </ModalDialog>
  )
}

function Row({ label, value, tone, multiline }: {
  label: string
  value: string
  tone?: 'good' | 'warn'
  multiline?: boolean
}) {
  const color = tone === 'good' ? '#a6e3a1' : tone === 'warn' ? '#f9e2af' : '#cdd6f4'
  return (
    <div style={{ ...styles.row, alignItems: multiline ? 'flex-start' : 'center' }}>
      <span style={styles.rowLabel}>{label}</span>
      <span style={{ ...styles.rowValue, color, whiteSpace: multiline ? 'normal' : 'nowrap' }}>
        {value}
      </span>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  sub: { margin: '8px 0 14px', fontSize: 12, color: '#a6adc8' },
  rows: {
    display: 'flex', flexDirection: 'column', gap: 8, margin: '8px 0 14px',
    background: '#11111b', border: '1px solid #313244', borderRadius: 6,
    padding: '10px 12px',
  },
  row: { display: 'flex', gap: 12, fontSize: 12.5 },
  rowLabel: { minWidth: 84, color: '#a6adc8', flexShrink: 0 },
  rowValue: { flex: 1, lineHeight: 1.4 },
}
