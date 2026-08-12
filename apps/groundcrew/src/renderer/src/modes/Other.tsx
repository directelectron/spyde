/**
 * Other.tsx — Motion Correction and Calibration.
 *
 * Both are laid out but NOT wired: the compute behind them (drift estimation
 * across a movie's frames, the CTF fit, ring detection, `info.txt` parsing)
 * does not exist in this app yet. They are here because the rail is the app's
 * structure and hiding two of its four entries would misrepresent the shape of
 * the thing.
 *
 * They say so on the pane rather than presenting dead controls. A slider that
 * does nothing is worse than an honest empty state — it costs someone a
 * bug report and their trust in every other control.
 */
import React from 'react'
import { NotBuilt } from '../ui'

export function MotionMode() {
  return (
    <NotBuilt title="Motion correction">
      Drift correction across a movie's frames — with a frame slider, integration
      over a range of frames, and the CTF alongside. The estimator and the CTF
      fit are not ported into this app yet, so the controls are not drawn: they
      would have nothing behind them.
    </NotBuilt>
  )
}

export function CalibrateMode() {
  return (
    <NotBuilt title="Calibration">
      Real-space image and its FFT side by side, with the measurements table
      prefilled from <code>info.txt</code> — which wins over the live TEM
      channel, because it records the state the image was actually acquired
      under — and every field editable. Images can also be taken directly
      through the camera. Not wired yet.
    </NotBuilt>
  )
}
