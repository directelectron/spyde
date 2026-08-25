"""Wall-clock ceilings for the tests that guard against a performance cliff.

These assertions exist to catch an ALGORITHMIC regression — a cache that stops
caching, an O(n) scan that becomes O(n²), a per-frame path that starts reading
the whole stack. Those show up as 10× to 1000×. They are not latency targets,
and a shared CI runner cannot measure one: it is virtualised, noisy-neighboured
and several times slower than a dev box, so a ceiling tight enough to be a
latency statement fails on scheduling noise instead.

A ceiling that fires on noise is worse than no ceiling. It trains everyone to
re-run the job, and then it does not catch the real regression either.

So: pick the budget from what the operation costs when it is working, leave
generous headroom, and widen it further where the timing is least trustworthy.
Real latency numbers belong in ``spyde/tests/benchmark_*.py``, which run
deliberately on known hardware and are not merge gates.
"""
from __future__ import annotations

import os

#: How much slower a shared CI runner is allowed to be before it is a
#: regression rather than the runner. Chosen against the failure that prompted
#: it: a 20 ms ceiling measured 26.5 ms on a macOS runner (1.3×), while the
#: regressions these tests exist to catch are an order of magnitude or more.
CI_SLACK = 3.0


def on_shared_runner() -> bool:
    """True when running somewhere with no timing guarantees. ``CI`` is set by
    GitHub Actions (and by most other providers)."""
    return bool(os.environ.get("CI"))


def budget_ms(local_ms: float) -> float:
    """Ceiling in milliseconds for a regression guard, widened on CI."""
    return local_ms * (CI_SLACK if on_shared_runner() else 1.0)


def budget_s(local_s: float) -> float:
    """Ceiling in seconds for a regression guard, widened on CI."""
    return local_s * (CI_SLACK if on_shared_runner() else 1.0)
