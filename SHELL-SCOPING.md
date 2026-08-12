# Ground Crew — decisions and open work

Supersedes the first version of this file, which measured the shell against the
existing PySide app as if it were a specification. It is not: the UI and code
are being replaced, and the numbers in that version were derived from a source
that does not hold. What follows is only what has actually been settled.

Answers below are from the `deapi` owner unless marked otherwise.

---

## 1. Connection model — settled

`deapi` is single-threaded per client: **one client ↔ one connection**, and most
calls block. Multiple clients are possible but each operates on its own port,
and more than one is a hazard rather than a feature.

**Two connections, one owner each:**

| Connection | `read_only` | Owns |
|---|---|---|
| Read | `True` | Status polling, `get_result` for rendering, histogram |
| Read/write | `False` | Acquisition, property writes, references, valve, temperature |

`Client.connect(host, port, read_only=…)` takes the flag directly, and
`set_client_read_only` exists for changing it later.

This is why **no process-wide device gate is needed** — the earlier proposal for
one came from the old app's docs and is withdrawn. Each connection has a single
owner, so there is nothing to serialise. Two writers on one socket would be the
bug; two connections with distinct roles avoids it by construction.

## 2. Rendering — settled, and it changes the viewer

**`get_result` IS the render backend.** Not "fetch a frame, then paint it" —
zoom and pan call back into `get_result`, and the server returns only the region
at the resolution asked for.

That maps onto anyplotlib's tile protocol almost exactly:

```
TileBackend.sample(x0, x1, y0, y1, out_w, out_h, method)   # anyplotlib asks
    ↓
client.get_result(centerX=(x0+x1)/2, centerY=(y0+y1)/2,
                  windowWidth=x1-x0, windowHeight=y1-y0,
                  zoom=out_w/(x1-x0), pixel_format=…)        # server crops + scales
```

`TileBackend` needs `full_shape`, `dtype`, `origin`, `extent()` and `sample()`;
`get_result` supplies all of it, **and returns the histogram from the same
call**, so Plot Control's histogram is server-side rather than recomputed in the
client.

Consequences:

- A `DeapiTileBackend` handed to `Plot2D.enable_tile(...)` is the whole viewer.
  No frame pushing, no shared-memory buffer, no client-side downsampling.
- Only the pixels on screen cross the wire — the thing that matters on a 4096²
  or 8192² detector.
- `FrameStream` still applies to the LIVE path (newest-wins painting of frames
  arriving off-thread), but the tile path is pull, not push. Those are two
  different mechanisms and both are needed.

## 3. Instrument properties — settled for now

`deapi` holds the instrument properties today, and that is fine for now; they
become settable through `deapi` or a separate channel later. Ground Crew reads
them from DE and does **not** build against the microscope-control package,
which is too early.

The `Instrument Project *` properties reading their unset sentinels (`-1`, `-1.0`)
means **the DE↔TEM channel is not reporting** — not that calibration is missing.
Those are separate cards on the status board.

## 4. Bad pixel correction — settled

It is an **instrument property holding a file reference**. If the file is set,
the correction has been applied; if it is blank, it has not. The status card
reads that property, and the rule is stated on the card rather than inferred.

## 5. Reference health — removed

Dark/gain staleness is not tracked, so the status card claiming it has been
deleted. The dark and gain **controls** remain; only the health claim went. To
be revisited with the owner's input later, along with sensor-health thresholds.

## 6. `info.txt` vs the live TEM channel — settled

**`info.txt` wins.** It records the state the image was actually acquired under;
the live channel describes now, which may be a different microscope state
entirely. Calibration prefills from it and every field stays editable, and the
measurements table records the provenance of each row (`info.txt` / `edited` /
`DEAPI`).

## 7. CTF — settled

A CTF finder has already been ported. No new dependency, no build-versus-buy
question.

## 8. Packaging — settled

Drop PyInstaller and conda; use **uv**, matching the rest of the family.
`deapi` is tested on macOS and Linux, so Windows is not a special case in the
way the old build script assumed.

---

## 9. `deapi` version — needs a decision

Verified end to end against the `FakeServer` (spawn
`simulated_server/initialize_server.py <port>`, wait for the `started` banner,
`Client()` with `usingMmf = False`):

| | published **5.2.2** | local **5.3b6** |
|---|---|---|
| connect, list cameras, properties | works | works |
| acquire + full `get_result` | **crashes** | works — (1024, 1024) uint16 |
| windowed `get_result` (the tile call) | unreachable | works — (256, 256) |

5.2.2 carries a stray unguarded `print("Response:", response.ByteSize())` at
`client.py:1450`, immediately before the `if response != False` check on the
next line — so it raises `AttributeError` on exactly the case the following line
exists to handle. It is gone in 5.3b6.

So the plan of record ("use the PyPI one") does not currently work for the call
the whole viewer is built on. Either 5.3 gets released, or `apps/groundcrew`
pins a git ref in the meantime. **Owner's call.**

Aside worth knowing: a stray `print()` in any dependency writes to stdout, which
is the PLOTAPP protocol channel. `de_shell.ipc.redirect_stray_stdout()` already
routes every `print()` to stderr for exactly this reason, so the shell is immune
— but it would corrupt a naive backend.

## 10. Scope — settled

SerialEM and the script panel are **out of v1**. No rail entry, no panel.

## Still open

- **`deapi` version** — see §9; blocks the dependency declaration only, not the
  work (the local checkout runs today).
- **Sensor health thresholds** — deferred by agreement; the owner will help set
  them.
- **Corrected movie output** — back to the server, or to a file only?
- **Retract while acquiring** — confirm dialog, or refuse?
- **Where a failed fit surfaces** — "no ring found" has no home louder than the
  status bar.
- **Re-refining a committed calibration row** — interaction undrawn.
- **Per-patch (5×5) shift display** in Motion.
- **Status polling cadence.** Now much less fraught: the read-only connection
  can poll without contending with acquisition. Still a choice.

## Tests

Judgement call, per the owner: keep what is genuinely valuable. The split is
roughly —

- **Keep**: anything exercising SDK interaction, calibration maths, or
  correction algorithms. These retarget onto `deapi`'s `FakeServer` (spawn
  `simulated_server/initialize_server.py <port>`, wait for `started`, connect
  with `usingMmf = False`; the fixture pattern is in `deapi/tests/conftest.py`).
- **Drop**: anything asserting on PySide widgets. It dies with the UI.

## What the scaffold gets wrong today

`apps/groundcrew/de_groundcrew/camera.py` invented a `Camera` protocol, a
`SimulatedCamera`, and a `DEServerCamera` stub pointing at the wrong SDK. All of
it should be deleted in favour of `deapi` plus its `FakeServer`. The Electron
side and everything in `packages/` stands.
