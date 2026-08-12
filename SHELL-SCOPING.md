# Scoping the shell for Ground Crew and Autopilot

What each new app needs, what the shell already gives it, and what is left. The
Ground Crew column is measured against the real PySide6 app
(`~/PycharmProjects/de_ground_crew`, ~7,845 lines); Autopilot is greenfield, so
its rows are marked where they are inferred rather than observed.

The point of this document is to answer two questions before the real builds
start: **how much of each app is already done**, and **what should be shared but
is not yet**.

---

## 1. Ground Crew — the PySide6 app, module by module

| Module | Lines | Shell covers it? | Notes |
|---|---:|---|---|
| `ui/calibration_panel.py` | 1371 | ✗ app | Dark/gain reference workflows. Pure domain. |
| `ui/main_window.py` | 1214 | **~80%** | Window, menus, status, log docking, the busy gate, worker wiring. Most of this dissolves into `createShellWindow` + `SessionBase` + the chrome reducer. What remains is the busy gate (see §3). |
| `ui/widgets.py` | 854 | **partly** | Generic Qt widgets. In React most become app JSX; the reusable ones overlap the shell UI components not yet extracted (§4). |
| `workers/calibration_worker.py` | 681 | **pattern only** | `lifecycle.run_on_worker` replaces the QThread+Signal boilerplate; the calibration logic is domain. |
| `ui/image_viewer.py` | 598 | **✓ mostly** | `FigureView` + `FrameStream`. The FFT side-by-side needs a second FigureView (see §3). |
| `ui/control_panel.py` | 558 | ✗ app | Camera controls. The scaffold's sidebar is a sketch of this. |
| `ui/script_panel.py` | 473 | **candidate** | A scripting console. SpyDE has one (`backend/console.py` 951 + `ConsoleBar.tsx` 899) — see §4. |
| `ui/image_stats_panel.py` | 449 | **partly** | ADU stats strip + a 256-bin histogram. Stats strip exists in the scaffold; the histogram widget is a §4 candidate. |
| `ui/serialem_panel.py` + `serialem_worker.py` + `serialem_console.py` | 808 | ✗ app | SerialEM integration. Pure domain. |
| `workers/acquisition.py` | 441 | **pattern only** | Six workers, all the same shape. `run_on_worker` + `FrameStream.submit_future` is a direct translation. |
| `ui/log_panel.py` | 116 | **✓** | `de_shell.log_stream` + the shell's log ring. |
| `ui/camera_status_panel.py` | 111 | ✗ app (trivial) | 3-second property poll. |
| `ui/status_bar.py` | 95 | **✓** | Shell chrome state. |

**Rough split: ~40% of Ground Crew is plumbing the shell now provides, ~60% is
domain that has to be written either way.** The domain 60% is concentrated in
calibration, SerialEM and the control panel — none of which the scaffold has
started.

---

## 2. Autopilot — inferred, since it does not exist yet

| Need | Shell covers it? | Notes |
|---|---|---|
| Recipe model + runner | ✗ app | The scaffold has a working shape (`recipe.py`). |
| Step queue UI, progress | **✓** | `emit_progress` + the shell reducer; the scaffold uses both. |
| Last-acquisition view | **✓** | `FigureView` + `FrameStream`. |
| Driving the camera/stage | **shared with Ground Crew, not the shell** | See §3. |
| Scheduling / unattended runs | ✗ app | Not started. Cron-like triggers, overnight runs, failure policy. |
| Results storage + handoff to SpyDE | ✗ app, **and undecided** | The obvious integration: Autopilot writes what SpyDE opens. Worth settling the format early. |

Autopilot is the smaller app, and most of what it needs beyond the scaffold is
policy (what to do when a step fails at 3 a.m.) rather than plumbing.

---

## 3. The gap neither app can fill alone: the instrument layer

**Both new apps drive the DE Server SDK. Neither SpyDE nor the shell should.**

That is a fourth package, not a fifth shell module:

```
packages/de-instrument/     de_instrument
    client.py     DEAPI.Client lifecycle, connect/reconnect
    camera.py     Camera protocol; DEServerCamera + SimulatedCamera
    stage.py      Stage protocol
    serialization the shared-socket gate (below)
```

Ground Crew and Autopilot depend on it; SpyDE does not. Putting it in `de_shell`
would break the constraint the whole split rests on — the shell knows nothing
about instruments any more than it knows about diffraction.

### The one piece that IS a shell candidate

`DEAPI.Client` is **not thread-safe**: one socket shared between a 3-second
status poll on the main thread and every background worker. Ground Crew handles
it with a busy gate — `_set_busy(True)` → `cam_status.pause()` around every
worker — and its CLAUDE.md says "never bypass it".

That is structurally identical to SpyDE's `spyde/device_lock.py`: a process-wide
resource that corrupts under concurrent access, where **a lock only works if
every participant takes it**, and where the failure mode is severe (SpyDE: an
uncatchable native SIGSEGV; Ground Crew: a corrupted shared socket).

Two independent consumers, same shape, same "one missed call site re-opens it"
hazard → this clears the bar for extraction. A `de_shell.device` offering a
process-wide serialisation context, with SpyDE's `accelerator_lock` and Ground
Crew's SDK gate as the two implementations, is worth doing **before** Ground
Crew's port, not after — retrofitting a lock across an existing codebase is
exactly how SpyDE got the bug its module docstring describes.

---

## 4. Shell gaps with two consumers today

Justified now, by the standard used throughout this split (two real consumers,
not one plus a guess):

- **Shell UI components.** Ground Crew and Autopilot have near-identical
  `LogPanel` / `StatusBar` / `Stat`. Two copies already.
- **Histogram widget.** SpyDE has one; Ground Crew's `image_stats_panel` needs
  one (the SDK already returns 256 bins). Two consumers once the port starts.
- **Window registry + a `FixedLayout`.** Currently parked for want of a second
  consumer. Ground Crew's image + FFT side-by-side is that consumer: two figures
  in named panes. This unparks when the FFT view is built.

Still not justified: the toolbar/selector hooks. Neither app has toolbars.

---

## 5. Suggested order

1. **`de-instrument` + the shared device gate.** Blocks both apps; the gate is
   much cheaper to design in than to retrofit.
2. **Port Ground Crew's acquisition + control panel** onto it. That is the
   thinnest path to a Ground Crew that does something real, and it will exercise
   the window registry (image + FFT) and the histogram — unparking §4 with
   evidence rather than speculation.
3. **Shell UI components**, once the port shows which ones actually recur.
4. **Calibration and SerialEM.** The two biggest domain chunks, and the two least
   affected by any of this.

Autopilot can proceed in parallel after step 1, since its remaining work is
mostly policy.

---

## 6. What this does not cover

- **Windows.** Ground Crew ships as a PyInstaller single exe built on Windows
  against a conda env; the shell distributes via electron-builder + a uv-managed
  Python. That migration is not scoped here and is not trivial — see
  `pythonEnv.ts` for what first-run setup does today.
- **The DE SDK's own packaging.** `sdk/DEAPI.py` and `serialem.pyd` are vendored
  binaries in the PySide6 tree; how they reach a packaged Electron app is an open
  question.
- **Whether Ground Crew should be rewritten at all.** This document scopes the
  work assuming yes. The 60/40 split above is the input to that decision, not the
  answer to it.
