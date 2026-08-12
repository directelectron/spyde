"""
timing.py — sleeping that actually wakes up in a shell backend.

`time.sleep` is not reliable in this process. The backend is a hidden child of
the Electron app, and the OS coalesces its timers hard enough that a
`time.sleep(0.05)` in a poll loop has been measured freezing for **15 seconds**,
waking only when process I/O arrived — while `threading.Event.wait(timeout)` in
the same process ticked exactly on schedule, 120 times out of 120.

So every poll loop in a shell app sleeps through `reliable_sleep`. This is the
same pathology the 0.5 Hz stdin tick in @de/shell-main's backendProcess exists
for, seen from the Python side.
"""
from __future__ import annotations

import threading

# Never set. It exists purely because `Event.wait(timeout)` is scheduled
# reliably where `time.sleep(timeout)` is not.
_WAKE = threading.Event()


def reliable_sleep(seconds: float) -> None:
    """Sleep that keeps ticking on the throttled, Electron-spawned backend.

    Use instead of ``time.sleep`` in any loop that must make progress.
    """
    _WAKE.wait(seconds)
