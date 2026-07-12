"""
Real-cluster verification that closing a tree cancels registered compute.

Spins a genuine LocalCluster (processes=True — the real production topology),
submits a slow lazy compute via client.compute, registers it on a real
SignalTree's cancellation registry, then closes the tree and asserts the
future is actually CANCELLED on the scheduler (not left running).

Also asserts a registered stopped_flag flips to True.

Must run as a plain-Python subprocess (real processes=True cluster won't run in
the agent sandbox / pytest). Prints a JSON result and os._exit(0).
"""
import json
import os
import sys
import time

import numpy as np


def main():
    from dask.distributed import LocalCluster, Client
    import dask.array as da

    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=True,
                           dashboard_address=None)
    client = Client(cluster)
    out = {"ok": False}
    try:
        # A deliberately slow lazy compute: a big array reduced through a
        # per-chunk sleep so it stays "processing" long enough to cancel.
        def _slow_block(b):
            time.sleep(3.0)
            return b.sum(keepdims=True)

        arr = da.ones((8, 4_000_000), chunks=(1, 4_000_000))
        lazy = arr.map_blocks(_slow_block, chunks=(1, 1), dtype=arr.dtype)

        # ── Build a minimal object exercising the REAL registry code path. ──
        # Import the actual methods off SignalTree so we test the shipped logic,
        # not a re-implementation. We bind them to a lightweight holder that has
        # the attributes _cancel_all_compute touches (client, _cancel_*).
        from spyde.signal_tree import BaseSignalTree

        class _Holder:
            pass

        h = _Holder()
        h.client = client
        h._cancel_flags = []
        h._cancel_futures = []
        h._spyde_closed = False
        # Bind the real methods.
        h.register_cancel = BaseSignalTree.register_cancel.__get__(h)
        h.unregister_cancel = BaseSignalTree.unregister_cancel.__get__(h)
        h._cancel_all_compute = BaseSignalTree._cancel_all_compute.__get__(h)

        flag = h.register_cancel()
        fut = client.compute(lazy)
        h.register_cancel(future=fut)

        # Let it actually reach the scheduler and start processing.
        time.sleep(1.0)
        status_before = fut.status

        # ── Simulate tree.close()'s first act. ──
        h._spyde_closed = True
        h._cancel_all_compute()

        # Give the scheduler a beat to register the cancel.
        time.sleep(1.0)
        status_after = fut.status

        out = {
            "ok": True,
            "flag_set": bool(flag[0]),
            "status_before": status_before,
            "status_after": status_after,
            "future_cancelled": fut.cancelled(),
            "registry_drained": (len(h._cancel_flags) == 0
                                 and len(h._cancel_futures) == 0),
        }
    except Exception as e:
        import traceback
        out = {"ok": False, "error": f"{type(e).__name__}: {e}",
               "tb": traceback.format_exc()}
    finally:
        try:
            client.close()
            cluster.close()
        except Exception:
            pass
    print("RESULT " + json.dumps(out))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
