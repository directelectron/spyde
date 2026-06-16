"""GPU readiness: detection, verification, and (uv-powered) wheel setup.

SpyDE's heavy compute (vector orientation mapping, peak finding) is torch-based
and runs CUDA → Apple-MPS → CPU. This module makes GPU readiness explicit so the
app never silently falls onto the slow CPU path:

- ``detect()``      — platform + NVIDIA driver + torch build summary.
- ``verify()``      — actually run a tiny torch op on the selected device.
- ``diagnostics()`` — everything above as one dict for the GPU Status dialog.
- ``ensure_backend()`` — re-install the GPU-correct torch wheel via
  ``uv ... --torch-backend=auto`` when an accelerator exists but torch is
  CPU-only (e.g. a portable build shipped the CPU wheel).

All functions are import-safe (no torch import at module load) and never raise —
they report problems as data.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Optional


# ── low-level probes ─────────────────────────────────────────────────────────

def _torch_summary() -> dict:
    out = {"installed": False, "version": None, "build": None,
           "cuda_available": False, "cuda_version": None,
           "mps_available": False, "device_name": None}
    try:
        import torch
    except Exception as e:
        out["error"] = f"torch not importable: {e}"
        return out
    out["installed"] = True
    out["version"] = getattr(torch, "__version__", "?")
    # wheel build flavour: "+cu129", "+cpu", or bare (mac/MPS)
    ver = out["version"] or ""
    out["build"] = ver.split("+", 1)[1] if "+" in ver else "cpu/mps"
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        mps = getattr(torch.backends, "mps", None)
        out["mps_available"] = bool(mps is not None and mps.is_available())
        if out["cuda_available"]:
            out["device_name"] = torch.cuda.get_device_name(0)
        elif out["mps_available"]:
            out["device_name"] = "Apple GPU (Metal)"
    except Exception as e:
        out["error"] = f"device probe failed: {e}"
    return out


def _nvidia_smi() -> Optional[dict]:
    """Driver/GPU info from nvidia-smi, or None if not present (no NVIDIA GPU)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        name, driver, mem = [s.strip() for s in
                             out.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "driver_version": driver,
                "memory_total_mb": int(float(mem))}
    except Exception:
        return None


# ── public API ───────────────────────────────────────────────────────────────

def detect() -> dict:
    """Platform + accelerator detection (no torch op run yet)."""
    is_mac_arm = sys.platform == "darwin" and platform.machine() in ("arm64",)
    nvidia = _nvidia_smi()
    torch_info = _torch_summary()
    # what device the app will actually use
    backend = "cpu"
    if torch_info.get("cuda_available"):
        backend = "cuda"
    elif torch_info.get("mps_available"):
        backend = "mps"
    # is acceleration *possible* on this hardware but not active in torch?
    accel_hw = nvidia is not None or is_mac_arm
    accelerated = backend in ("cuda", "mps")
    return {
        "platform": sys.platform,
        "machine": platform.machine(),
        "is_mac_arm": is_mac_arm,
        "nvidia": nvidia,
        "torch": torch_info,
        "backend": backend,
        "accelerated": accelerated,
        # accelerator hardware exists but torch isn't using it → fixable via uv
        "needs_gpu_wheel": bool(accel_hw and not accelerated
                                and torch_info.get("installed")),
    }


def verify() -> dict:
    """Run a tiny op on the selected device to confirm it actually works.

    Catches the case where torch *reports* CUDA but a driver/runtime mismatch
    makes real kernels fail."""
    res = {"ok": False, "device": None, "error": None}
    try:
        from spyde.actions.vector_orientation_gpu import select_device
        import torch
        dev = select_device() or torch.device("cpu")
        res["device"] = dev.type
        x = torch.ones(8, device=dev, requires_grad=True)
        y = (x * 2).sum()
        y.backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        res["ok"] = True
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def diagnostics() -> dict:
    """Everything the GPU Status dialog needs, in one call."""
    from spyde.actions.vector_orientation_gpu import gpu_unavailable_reason
    d = detect()
    d["verify"] = verify()
    d["reason"] = gpu_unavailable_reason()
    return d


def summary_lines() -> list[str]:
    """Human-readable diagnostics for a dialog / log."""
    d = diagnostics()
    t = d["torch"]
    lines = [
        f"Platform: {d['platform']} ({d['machine']})",
        f"torch: {'yes' if t['installed'] else 'NO'}"
        + (f"  {t['version']} (build {t['build']})" if t['installed'] else ""),
        f"Active backend: {d['backend'].upper()}"
        + ("  [accelerated]" if d["accelerated"] else "  [CPU only]"),
    ]
    if d["nvidia"]:
        n = d["nvidia"]
        lines.append(f"NVIDIA GPU: {n['name']}  "
                     f"driver {n['driver_version']}  {n['memory_total_mb']} MB")
    if t.get("device_name"):
        lines.append(f"Compute device: {t['device_name']}")
    v = d["verify"]
    lines.append("Verify op: " + ("ok" if v["ok"]
                 else f"FAILED {v['error']}") + f"  ({v['device']})")
    if d["needs_gpu_wheel"]:
        lines.append("! A GPU is present but torch is CPU-only - "
                     "run GPU setup to install the accelerated build.")
    if not d["accelerated"]:
        lines.append(f"Note: {d['reason']}")
    return lines


# ── uv-powered backend install ───────────────────────────────────────────────

def _uv_executable() -> Optional[str]:
    """Locate the uv binary: bundled next to the app, env override, or PATH."""
    override = os.environ.get("SPYDE_UV")
    if override and os.path.exists(override):
        return override
    # bundled (PyCrucible payload puts uv beside the app root)
    for base in (getattr(sys, "_MEIPASS", None),
                 os.path.dirname(sys.executable),
                 os.getcwd()):
        if not base:
            continue
        for name in ("uv.exe", "uv"):
            cand = os.path.join(base, name)
            if os.path.exists(cand):
                return cand
    return shutil.which("uv")


def ensure_backend(progress=None) -> dict:
    """Install the GPU-correct torch wheel via uv when needed.

    Runs ``uv pip install --torch-backend=auto torch`` into the running
    environment. uv picks the CUDA wheel on Win/Linux+NVIDIA, the MPS-capable
    wheel on mac arm64, CPU otherwise. ``progress(line)`` receives output lines.
    Returns {"ran", "ok", "message"}. Never raises.
    """
    d = detect()
    if not d["needs_gpu_wheel"]:
        return {"ran": False, "ok": d["accelerated"],
                "message": "GPU backend already correct"
                if d["accelerated"] else "No accelerator hardware detected"}
    uv = _uv_executable()
    if uv is None:
        return {"ran": False, "ok": False,
                "message": "uv not found — cannot reinstall the GPU wheel"}
    cmd = [uv, "pip", "install", "--python", sys.executable,
           "--torch-backend=auto", "--reinstall-package", "torch", "torch"]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:           # stream progress
            if progress is not None:
                progress(line.rstrip())
        proc.wait(timeout=1800)
        ok = proc.returncode == 0
        return {"ran": True, "ok": ok,
                "message": "GPU wheel installed — restart SpyDE"
                if ok else f"uv exited {proc.returncode}"}
    except Exception as e:
        return {"ran": True, "ok": False, "message": f"uv failed: {e}"}
