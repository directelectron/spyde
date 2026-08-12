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

## 9. `deapi` version — settled: pin the pre-release

`apps/groundcrew` depends on **`deapi>=5.3b6`**, from PyPI. Verified end to end
against the `FakeServer` (spawn `simulated_server/initialize_server.py <port>`,
wait for the `started` banner, `Client()` with `usingMmf = False`): connect,
acquire a (1024, 1024) uint16 frame, and a **windowed `get_result` returning
(256, 256)** with attributes populated — the tile contract the viewer is built
on.

**The specifier must name a pre-release.** pip and uv skip pre-releases unless
the constraint itself references one, so a bare `>=5.2` silently resolves to
5.2.2 — which carries a stray unguarded
`print("Response:", response.ByteSize())` at `client.py:1450`, immediately
before the `if response != False` check on the next line, and so raises
`AttributeError` on exactly the case that check exists to handle. Removed in
5.3b6. Keep a pre-release in the specifier until 5.3 final ships.

Workspace note: `uv sync` alone syncs only the ROOT project and will *uninstall*
the workspace members. Use **`uv sync --all-packages`**, or the apps and `deapi`
vanish from the shared environment and both live-app e2e suites break.

Aside worth knowing: a stray `print()` in any dependency writes to stdout, which
is the PLOTAPP protocol channel. `de_shell.ipc.redirect_stray_stdout()` already
routes every `print()` to stderr for exactly this reason, so the shell is immune
— but it would corrupt a naive backend.

## 10. Scope — settled

SerialEM and the script panel are **out of v1**. No rail entry, no panel.

## 11. What the simulator can and cannot support — measured

Built and verified end to end against `FakeServer`: connect → tiled
`get_result` → live acquisition → contrast → statistics, with the viewer
rendering the simulator's seven-disc scene.

Three findings from doing it, all of which constrain what can be built next.

**The simulator exposes 73 properties; a real server exposes 339.** The status
board and the top bar were designed against a dump of a real one, and **15 of
the 22 properties they need are absent** — every temperature, valve, vacuum,
`Instrument Project *` and image-correction property among them. `Instrument.
properties()` returns `None` for an unsupported name (checked against
`list_properties()`, not by inspecting the value — see below), so those
surfaces will render "unavailable" rather than lie. But they cannot be
*developed* against the fake server. Either the simulator grows those
properties or those cards wait for hardware.
`test_the_simulator_lacks_most_status_properties` fails the moment that
changes, so this note cannot go stale silently.

**`attrs.imageMin/imageMax/imageMean/imageStd` are not frame statistics on the
simulator.** They came back `0 / 32768 / 100 / 5` — exactly 2¹⁵ and two round
numbers — while the pixels returned by the *same* `get_result` ran 1…25 with a
mean of 1.98. Displaying them put "MAX 32768" under a picture whose brightest
pixel was 25. **The histogram, by contrast, is real** and agreed with the array
exactly, so both the stats strip and the display range are derived from it.
That is a better arrangement anyway — the histogram describes the whole frame,
so contrast does not shift as the user pans a tiled image — but if the
attributes are meant to be populated, this is a simulator bug worth knowing
about.

**`client[unknown_property]` returns `False`, not an exception.** Since `False`
is also a legitimate value for a boolean property, no amount of value
inspection can tell "unsupported" from "switched off". Unknown names are
therefore resolved against `list_properties()`.

## 12. `stop_acquisition()` is out-of-band UDP — do not put it on the io thread

**An earlier version of this section was wrong** and is corrected here. It
claimed `stop_acquisition()` "kills the connection". It does not. The owner
pushed back — *"it's fundamentally the same as the live acquisition, you're
doing something wrong which is killing the client, that shouldn't really even
be possible"* — and was right.

What the call actually does:

```python
sock = socket.socket(AF_INET, SOCK_DGRAM)      # UDP — not the client socket
sock.sendto(b"PyClientStopAcq", (host, port))
respond = sock.recv(32)                        # no timeout
```

It never touches the TCP client. deapi's own docstring says it may be called
from another thread while `get_result` blocks — being out-of-band is the point.

So there are two separate facts, and the earlier note conflated them:

1. **A real (small) deapi issue.** That `recv` has no timeout, so if nothing
   answers the datagram the call never returns. `FakeServer` listens only on
   TCP, so on the simulator it never returns, always. Real hardware answers.
2. **My bug, which caused everything that looked serious.** Routing it through
   the connection's single owning thread parked the one thread every other
   feature needed. That is what produced "the status board spins forever".

**The connection was healthy throughout.** With the stop blocked on its own
thread, property reads and a full `start_acquisition(1)` → `get_result` cycle
both still succeed — `TestStopIsOutOfBand` asserts exactly that, so the mistake
cannot come back.

`Instrument.stop_acquisition()` now runs it on a short-lived daemon thread,
with one outstanding at a time (they are idempotent, and spawning more threads
to park on an unanswered `recv` achieves nothing).

The stall watchdog from the earlier pass is **kept, with its rationale
corrected**: it was written for a failure I had caused, but a device call that
genuinely hangs is still possible, and a status board that reports "no response
within 6s" beats one that spins. It is ~30 lines and independently tested.

This remains an argument for the **two-connection** design in §1, though a
weaker one than claimed: the reads were never actually blocked by the server.

## 13. A separate bug this masked: the live loop out-ran the main thread

`_live_loop` posted a display refresh every 50 ms regardless of whether the
previous one had finished. Each refresh blocks the main thread for a full
server round trip, so whenever the server was slower than the tick the main
loop accumulated a backlog — which **outlived the acquisition**: after Stop,
status reports and property reads queued behind refreshes that were still
draining. The loop now waits for each refresh to land before requesting
another.

## Still open

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

## What the scaffold got wrong — fixed

`camera.py` invented a `Camera` protocol, a `SimulatedCamera` and a
`DEServerCamera` stub pointing at the wrong SDK. **Deleted**, and replaced by
`instrument.py` (one connection, one owning thread) and `tile.py`
(`get_result` as the render backend) against real `deapi` + its `FakeServer`.
The Electron side and everything in `packages/` stands.

**The UI is rebuilt** to the mode rail (Imaging / Motion / Calibrate / Status),
with an instrument sidebar of editable input boxes, a Plot Control strip on the
right, and temperature + camera position in the top bar. `theme.ts` holds the
tokens, `ui.tsx` the primitives, `modes/` the panes.

**Imaging and Status are wired; Motion and Calibrate are not.** The compute
behind them — drift estimation, the CTF fit, ring detection, `info.txt`
parsing — is not ported into this app yet, so those panes state that plainly
rather than presenting controls with nothing behind them.

### Running the e2e — a trap worth knowing

`apps/groundcrew`'s Playwright config has **no build step**: the harness
launches whatever is in `out/`, so a renderer edit is invisible until
`npm run build` runs. This cost a debugging cycle here — a stale bundle looked
exactly like a backend that was not sending state. Run `npm run build &&
npx playwright test --project=groundcrew`.
