"""Playing the renderer's part in the report snapshot handshake.

Every report write path — save, HTML export, markdown export, slides — asks the
renderer for fresh figure PNGs before it writes, and falls back to baked ones if
the reply is slow (``handlers.harvest_snapshots``). Nothing in this suite ever
exercised that, because ``harvest_snapshots`` has a branch for it::

    if not live_cells or getattr(session, "_main_loop", None) is None:
        finish({})            # "no event loop (headless / tests)"
        return

No loop meant no handshake: the write happened synchronously, inline, and every
save/export test asserted on the result of a shortcut that only exists for
callers without a loop. The app always has one. So the handshake a user's every
save goes through — emit the request, wait for the reply, bake what is missing —
had no coverage at all, and the tests that looked like they covered it were
green on a path the app never takes.

Now that the fixtures register a loop (see ``conftest._attach_main_loop``) the
real handshake runs, and a test has to answer it. That is what this is for.
"""
from __future__ import annotations

from spyde.tests.migrated._async import quiesce, wait_until


def pending_harvest(messages) -> str | None:
    """The token of the most recent outstanding snapshot request, if any."""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("type") == "report_need_snapshots":
            return m.get("token")
    return None


def answer_harvest(session, messages, images: dict | None = None,
                   timeout: float = 10.0) -> bool:
    """Answer the pending snapshot request the way the renderer does.

    Passing no images reproduces exactly what the old no-loop shortcut did
    (``finish({})``), so an existing assertion still holds — it now just gets
    there through the path the app uses, with the missing pixels baked rather
    than skipped.

    Returns False if no request was made. That is not always a failure: a
    report with no LIVE figure cells still takes the synchronous branch, so a
    text-only export never asks. Call it unconditionally and ignore the result
    unless the test is about the handshake itself.
    """
    from spyde.actions.report import handlers as h

    # A direct read, not a poll: `harvest_snapshots` emits the request inline
    # and only then returns, so by the time the save/export call has returned
    # the message is already here. Polling for one that will never come cost a
    # full second on every text-only report.
    token = pending_harvest(messages)
    if token is None:
        return False
    h.report_snapshots(session, None,
                       {"token": token, "images": images or {}})
    quiesce(session, timeout=timeout)
    return True


def finished_export(session, messages, timeout: float = 10.0) -> list[dict]:
    """Answer the harvest if one is pending, then return the export replies.

    The one call a "did my export land?" assertion needs, whichever branch the
    report took.
    """
    answer_harvest(session, messages, timeout=timeout)
    wait_until(lambda: any(isinstance(m, dict)
                           and m.get("type") == "report_exported"
                           for m in messages), timeout=timeout)
    return [m for m in messages
            if isinstance(m, dict) and m.get("type") == "report_exported"]
