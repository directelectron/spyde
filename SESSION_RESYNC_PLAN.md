# Session resync — surviving a renderer loss

**Status:** proposal. Nothing implemented beyond the lifecycle diagnostics in
`electron/src/main/index.ts` (commit `4e1af8b`), which only log.

**Symptom.** Close a Mac laptop lid, reopen it, and every plot window is gone
from the workspace. The app still works — you just re-load the data.

---

## 1. What is already established

Two facts from the code rule out the obvious suspects before any reproduction.

**The Python backend is alive.** `runner.ts` never respawns it: on `close` it
sets `proc = null`, and every later `sendAction` silently no-ops. If the backend
had died, re-loading data afterwards would be impossible. It isn't — so Python
survived the sleep with its signal trees intact.

**The workspace is renderer-only state.** The window list lives in
`SpyDEContext`'s `windows: Map<number, SpyDEWindow>`, in memory. Nothing
persists it and nothing rebuilds it from the backend. So the backend can be
holding every signal while the workspace shows nothing — exactly the reported
symptom, including "just re-load the data".

**Therefore:** the renderer lost its state while Python kept running. Whether
the renderer *process* died (crash / GPU-process death on resume) or it survived
and something reset the state is what the diagnostics answer. **That distinction
changes the trigger, not this plan** — a workspace that cannot be rebuilt from
the backend is a latent bug under any renderer loss, including a devtools reload
or an `npm run dev` hot restart.

## 2. Why it happens: an asymmetry inside SpyDE

The Report Builder already solves this problem. `report_state` is a **full
document snapshot** — `emit_state()` is called 41 times across the report
handlers, and the renderer *mirrors* it wholesale rather than accumulating
deltas. That is why a report survives things the workspace doesn't.

The window layer never got that treatment. It is built up from incremental
messages (`figure`, `window_title`, `window_visibility`, `window_computing`,
`window_closed`) with **no snapshot** and **no way for the renderer to ask**.
The backend has `_reemit_signal_tree`, but it is per-tree, internally triggered,
and only re-sends the workflow tree — not the windows or their figures.

This is the standard split-process pattern (JupyterLab kernels outliving the
browser; VS Code's extension host surviving Reload Window). SpyDE has the right
architecture and implements the resync in one layer out of two.

## 3. Proposed design

**Make the backend authoritative for the workspace, and add a resync
handshake.** Not new architecture — the window layer behaving like the report
layer.

### 3.1 `session_state` — a snapshot verb

A new action, mirroring `report_state`:

```
renderer boots ──▶ sendAction('session_state')
                        │
backend ◀───────────────┘
  └─▶ for every live plot: re-emit the messages that construct it
```

The renderer already reconstructs a window purely from messages it receives, so
the backend does not need a new *description* format — it needs to **replay the
messages it already sends**. That is the cheap part, and it is why this is
tractable.

### 3.2 What must be replayed, per window

From `SpyDEWindow` / `SpyDEFigure`:

| field | source | difficulty |
|---|---|---|
| `windowId`, `title`, `isNavigator`, `visible` | `session._plots` | trivial |
| `toolbarActions` | toolbar YAML + gating | already re-derived per window |
| `figures[].figId`, `filePath`, `title` | `figure_registry` | **see 3.3** |
| `aspect` | plot's current data shape | trivial |
| `view` / `viewLabel` / `strainComponents` | plot state | moderate |
| signal tree | `_reemit_signal_tree` | **already exists** |

### 3.3 The part that is not free

A window is not a box. Each carries a live anyplotlib figure that the renderer
mounts as an **iframe pointing at an HTML file on disk** (`SpyDEFigure.filePath`,
set from the `figure` message's `file_url`).

Three questions decide the real cost, and I do not yet know the answers:

1. **Do those HTML files still exist after a resume?** If `figure_registry`'s
   keep-alive holds them, a resync is close to re-sending the same `figure`
   messages with the same paths. If they are temp files that get cleaned, every
   figure must be re-rendered.
2. **Is the figure's live state recoverable?** Contrast, colormap, zoom and
   selector positions ride in the iframe via `awi_state`. `replayState` /
   `latestStates` already exists to re-push state into a freshly-mounted iframe
   — that machinery is the reason this may be much cheaper than it looks, and it
   is the first thing to verify.
3. **What about in-flight compute?** A window mid-`window_computing` needs a
   defined resync state. Simplest defensible answer: resync shows the last good
   frame, and any genuinely in-flight future either completes into the rebuilt
   window or is dropped.

## 4. Phasing

**Phase 0 — confirm the trigger (blocked on you).** One lid-close with the
lifecycle logging, to see whether `render-process-gone` / `child-process-gone`
fires. Does not change the plan; does tell us whether a *recovery* path is also
needed (a dead GPU process may need a WebGPU re-init, not just a resync).

**Phase 1 — the probe.** Answer 3.3's three questions with a throwaway
experiment: reload the renderer (devtools reload) with data loaded, and see how
much of a window can be reconstructed from what the backend still holds. This is
the honest scoping step — everything after it is guesswork until it is done.

**Phase 2 — `session_state` for windows.** Backend replays construction
messages for every live plot; renderer requests it on boot when it has no
windows. Ship this alone: it fixes the reported symptom.

**Phase 3 — figure state.** Re-push `awi_state` per figure so contrast/zoom/
selectors come back, not just the windows.

**Phase 4 — geometry (optional, needs a decision).** See §5.

## 5. Decisions I need from you

1. **Geometry.** Should resync restore window *positions and sizes*, or
   re-create the windows and let the MDI lay them out fresh? Restoring geometry
   means the renderer must report layout back to the backend continuously (the
   backend does not know where windows are) — that is a real new channel, and I
   would default to **fresh layout** unless you want otherwise.
2. **Scope of "authoritative".** Phase 2 makes the backend able to *describe*
   the workspace. Should it also become the source of truth for window
   open/close, or stay a mirror the renderer can rebuild from? Mirror is much
   less invasive.
3. **Trigger.** Resync automatically on renderer boot, or a visible "Restore
   session" affordance? Automatic is better UX; explicit is safer while the path
   is new and cannot silently double-create windows.

## 6. Explicitly out of scope

- Persisting the workspace to disk across an app *quit* (that is session
  restore, a different feature — §2's pattern 2).
- Reconnecting or restarting the Dask cluster — but **not because it is
  unaffected**. See §8: it is a genuinely open question and probably a second,
  independent bug.
- Any change to the navigator read path, chunking, or signal-tree internals.

## 7. Cost

Phase 1 is under a day and is what makes the rest estimable. Phase 2 is the bulk
and is bounded by 3.3's answers — cheap if the figure HTML and `awi_state`
survive, considerably more if every figure must re-render. **I would not commit
to a Phase 2 estimate before Phase 1 runs.**

---

## 8. Correction: does the Dask cluster survive?

An earlier draft of this document asserted that the cluster survived, "which is
why re-loading data works". **That claim was wrong and is withdrawn.** Neither
observed symptom tells us anything about the cluster.

**Re-loading data working proves nothing.** `Session._await_dask()` gates every
load on `self._dask_ready`, which is a `threading.Event` — a **one-shot latch**.
It is `set()` once when the cluster first comes up and is cleared in exactly one
place in the codebase (`compute_config.py`, on an explicit user-driven cluster
restart). Nothing clears it when a cluster *dies*. So after a resume, a load
sails straight through the gate and proceeds against a **dead client**, exactly
as it would against a live one. The load succeeding is evidence about the latch,
not about the cluster.

**The dashboard disappearing proves nothing either.** `dashboardUrl` lives in
the *same* `SpyDEContext` React state as `windows`. Whatever loses the workspace
loses the dashboard link with it — one state object, one loss. (It is also the
field that commit `6731786` just fixed for an unrelated reason: a late `ready`
message with no dashboard field was overwriting it. Same field, different bug.)

So there are now **two independent unknowns**, and the two symptoms are
consistent with either or both:

| | cluster alive | cluster dead |
|---|---|---|
| **renderer state lost** | windows + dashboard link both gone; compute still fine | windows + dashboard gone; compute silently broken |
| **renderer state kept** | not what is observed | dashboard link would persist but be dead |

### 8.1 The latent bug, independent of sleep

**Nothing in SpyDE detects cluster death.** There is no liveness check, no
heartbeat consumed, and no path that re-arms `_dask_ready`. Whenever a cluster
dies for any reason, the app keeps accepting loads and computes against a dead
client instead of waiting, restarting, or saying so. Sleep may simply be the
easiest way to trigger it.

This is worth fixing on its own merits and is **separable** from the resync
work. It is also the more dangerous of the two: a vanished window is obvious,
whereas a silently dead compute backend is not.

### 8.2 How to actually tell

Cheap discriminators, in order of effort:

1. **Look for the workers.** After a resume, `ps` for the `dask-worker` /
   spawned Python children (macOS: `pgrep -fl distributed`). Present and
   parented correctly → the cluster lived.
2. **Ask the backend, not the UI.** The dashboard *link* is renderer state; the
   cluster is not. A backend-side probe (`client.scheduler_info()` in the
   console cell, or a log line on `power:resume`) answers the question directly
   without going through the state that we already know gets lost.
3. **Try a compute, not a load.** A load passes the latch either way. Something
   that actually round-trips through the scheduler is the real test.

### 8.3 Consequence for this plan

Phase 0 grows one question: *is the cluster alive after resume?* — answered by
8.2, not by the UI. If it is dead, that is a second workstream (detect death,
re-arm the gate, restart or report) that this document does **not** currently
scope, and it should not be folded into the resync work.

---

## 9. Second workstream: cluster liveness

Scoped here at your request. **Separate from the resync work above** — different
trigger, different code, different failure mode — and shipped independently.
Fold them together only if §8.2 shows they share a root cause.

### 9.1 The gap is DETECTION, not recovery

Recovery already exists and is proven in production code. `compute_config.py`'s
restart path does exactly the right three things:

```python
session._dask_ready.clear()                  # loads wait instead of racing a dead client
session.dask_manager.restart(n_workers, threads_per_worker)
# … _on_dask_ready re-opens the gate when the new cluster registers
```

That is the whole recovery primitive, already written, already used. What is
missing is **anything that calls it when a cluster dies on its own**. Today the
only trigger is a user changing compute settings.

So the work is: notice, then reuse.

### 9.2 Where detection could live

| option | how | cost | verdict |
|---|---|---|---|
| **Poll the scheduler** | periodic `client.scheduler_info()` on a worker thread | one timer, no new deps | plausible default; must not run on the asyncio main thread |
| **Consume what already exists** | `DaskStatsSampler` (backend/dask_stats.py) already samples worker CPU/mem for the StatusBar HUD — a sampler that starts failing IS the signal | near-zero: the loop exists | **preferred if its failure mode is distinguishable** — verify before committing |
| **distributed's own callbacks** | scheduler/client status hooks | least polling | needs a check that they fire for the death modes we care about (sleep, OOM-kill, worker loss ≠ scheduler loss) |
| **On `power:resume` only** | probe once after a wake | trivial | too narrow — this bug is not sleep-specific |

`DaskStatsSampler` is the interesting one: there is already a loop talking to the
cluster on a cadence, so a liveness signal may cost nothing beyond reading its
failures. **Check that first.**

### 9.3 Policy: what to do once it is noticed

Three defensible behaviours, in increasing ambition:

1. **Report.** Clear the gate, emit an error, let the user restart the cluster
   from the existing compute-settings UI. Smallest change; makes a silent
   failure loud, which is the actual harm.
2. **Restart automatically, once.** Call the §9.1 primitive; emit status while it
   comes back. Good UX, but see the traps.
3. **Restart and resubmit.** Re-run whatever was in flight. **Not recommended** —
   see 9.6.

I would ship (1), then (2) behind the same status-bar surface the HUD already
owns.

### 9.4 Phasing

- **9-A — measure.** Answer §8.2: does the cluster actually die on a Mac
  lid-close? If it does not, this workstream is still worth doing (nothing
  detects death from *any* cause) but drops in priority.
- **9-B — detect.** Whichever of 9.2 survives inspection. Ship it as a log line
  + status message only; no behaviour change. Confirms the detector fires when
  it should and, more importantly, does **not** fire when it shouldn't.
- **9-C — re-arm the gate.** On detection, `_dask_ready.clear()`. This alone
  converts "silently computes against a dead client" into "waits, then times out
  with a real message" — a large improvement for a tiny diff.
- **9-D — restart.** Wire the existing primitive to the detector.

### 9.5 Decisions needed

1. **Auto-restart, or report and let the user act?** (9.3 (1) vs (2).)
2. **What counts as "dead"?** A lost *worker* is not a lost *cluster* —
   `distributed` replaces workers routinely, and treating that as death would
   restart the cluster under normal operation. The detector must distinguish
   scheduler loss from worker churn, and that distinction is the whole risk.
3. **Behaviour mid-compute.** If detection fires while a long compute is in
   flight, does it cancel and restart, or wait?

### 9.6 Traps

- **A false positive is worse than the bug.** Restarting a healthy cluster
  cancels in-flight work. Whatever detector is chosen needs a consecutive-failure
  threshold, not a single miss — and 9.5(2) is why.
- **Do not probe from the asyncio main thread.** The existing code is careful
  about this (`_await_dask` documents "never call on the main asyncio thread");
  a liveness probe has the same constraint.
- **Do not auto-resubmit.** SpyDE's computes are not all idempotent, results
  land through `_dispatch_to_main` into plot state, and a resubmit racing a
  rebuilt window is a much harder bug than the one being fixed.
- **Threaded mode has no cluster.** `ComputeBackend` runs threaded by default in
  some configurations and `SPYDE_NO_DASK=1` sets the gate pre-opened; the
  detector must no-op there rather than reporting a permanently dead cluster.
- Nothing here touches the navigator read path, chunking, or the signal tree.
