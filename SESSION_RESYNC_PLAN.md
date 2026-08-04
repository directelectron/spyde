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
- Reconnecting or restarting the Dask cluster. Not implicated: the backend and
  its cluster survived, which is why re-loading data works.
- Any change to the navigator read path, chunking, or signal-tree internals.

## 7. Cost

Phase 1 is under a day and is what makes the rest estimable. Phase 2 is the bulk
and is bounded by 3.3's answers — cheap if the figure HTML and `awi_state`
survive, considerably more if every figure must re-render. **I would not commit
to a Phase 2 estimate before Phase 1 runs.**
