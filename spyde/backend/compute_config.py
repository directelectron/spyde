"""compute_config.py — user-settable compute limits, applied by cluster restart.

The three knobs users actually reach for (max RAM, CPU use, GPU feeders) were
env-vars only (SPYDE_MEM_FRACTION / SPYDE_COMPUTE_FRACTION / SPYDE_FV_GPU) —
useless mid-session and easy to set in the wrong shell. This module makes them
first-class:

- ``compute_configure`` (staged action; the DaskMonitor popover's Apply
  button): clamps + writes the values into this process's environment (workers
  INHERIT env at spawn, and ``_gpu_task_allowed`` reads it inside each worker),
  persists them to ``~/.spyde/settings.json``, and RESTARTS the dask cluster
  with the recomputed worker plan — the knobs are all fixed at worker spawn,
  so a restart is the honest apply.
- ``apply_persisted_compute_env`` (startup, before the worker plan is
  computed): loads the persisted values into the environment — an explicitly
  set env var still wins (``setdefault``).
- ``current_config`` — the merged view the monitor UI displays.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

SETTINGS_KEY = "compute"

# knob -> (env var, clamp lo, clamp hi, default)
_KNOBS = {
    "mem_fraction":     ("SPYDE_MEM_FRACTION",     0.2, 0.8,  0.65),
    "compute_fraction": ("SPYDE_COMPUTE_FRACTION", 0.1, 1.0,  0.75),
}
GPU_ENV = "SPYDE_FV_GPU"
GPU_DEFAULT = "4"          # matches the neural lane default in orchestrate


def _settings_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".spyde", "settings.json")


def apply_persisted_compute_env() -> None:
    """Load persisted compute settings into the environment (startup, BEFORE
    the worker plan / cluster build). setdefault: a real env var wins."""
    try:
        with open(_settings_path(), encoding="utf-8") as fh:
            saved = (json.load(fh) or {}).get(SETTINGS_KEY) or {}
    except Exception:
        return
    for knob, (env, lo, hi, _default) in _KNOBS.items():
        if knob in saved:
            try:
                os.environ.setdefault(env, str(min(hi, max(lo, float(saved[knob])))))
            except (TypeError, ValueError):
                pass
    if saved.get("gpu_workers"):
        os.environ.setdefault(GPU_ENV, str(saved["gpu_workers"]))


def current_config() -> dict:
    """The effective knob values (env > persisted default) for the monitor UI."""
    out = {}
    for knob, (env, lo, hi, default) in _KNOBS.items():
        try:
            out[knob] = min(hi, max(lo, float(os.environ.get(env, default))))
        except ValueError:
            out[knob] = default
    out["gpu_workers"] = os.environ.get(GPU_ENV, GPU_DEFAULT)
    return out


def compute_configure(session, plot, payload) -> None:
    """Staged action: apply new compute limits and restart the cluster.

    Payload: ``{mem_fraction?, compute_fraction?, gpu_workers?}``. Values are
    clamped, written to the environment + settings.json, then the cluster is
    rebuilt with the recomputed (n_workers, threads) plan on a worker thread
    (the restart blocks for seconds). In-flight computes are cancelled by the
    teardown — the status line says so."""
    p = payload or {}
    changed = []
    for knob, (env, lo, hi, _default) in _KNOBS.items():
        if p.get(knob) is None:
            continue
        try:
            val = min(hi, max(lo, float(p[knob])))
        except (TypeError, ValueError):
            continue
        os.environ[env] = str(val)
        session._settings.setdefault(SETTINGS_KEY, {})[knob] = val
        changed.append(f"{knob}={val:g}")
    if p.get("gpu_workers"):
        gw = str(p["gpu_workers"]).lower()
        if gw not in ("one", "all", "off") and not gw.isdigit():
            gw = GPU_DEFAULT
        os.environ[GPU_ENV] = gw
        session._settings.setdefault(SETTINGS_KEY, {})["gpu_workers"] = gw
        changed.append(f"gpu_workers={gw}")
    if not changed:
        return
    try:
        session._save_settings()
    except Exception as e:
        log.debug("persisting compute settings failed: %s", e)

    from spyde.backend.ipc import emit_status

    def _work():
        from spyde.backend.app import _compute_worker_plan
        workers, threads = _compute_worker_plan(os.cpu_count() or 4)
        emit_status(f"Restarting compute cluster ({workers} workers × "
                    f"{threads} threads) — in-flight computes are cancelled…")
        # Close the dask gate so loads fired mid-restart wait instead of
        # racing a dead client; _on_dask_ready re-opens it.
        try:
            session._dask_ready.clear()
        except Exception:
            pass
        session.dask_manager.restart(n_workers=workers,
                                     threads_per_worker=threads)
        log.info("[compute-config] applied %s; cluster restarting with %d×%d",
                 ", ".join(changed), workers, threads)

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="compute-configure")
