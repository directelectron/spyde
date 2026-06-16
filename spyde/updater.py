"""Update checking + (staged) applying against GitHub Releases.

Powered by the GitHub Releases API for *checking*, and uv for *applying* in the
uv-managed install (incremental ``uv sync``). The portable single-exe build
gets check-only behaviour (it points the user at the download).

Design notes:
- ``check()`` is network-only and side-effect free; safe to call on startup in a
  worker thread. It compares the running version to the latest release tag.
- Version comparison is a tolerant semver compare (handles ``v`` prefix and
  ``-rc.N`` pre-releases); on the ``stable`` channel pre-releases are ignored.
- Applying is gated behind an explicit user action and is implemented for the
  uv-managed layout (``apply_uv_sync``); the portable build returns a
  "manual download" result.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from typing import Optional

GITHUB_REPO = os.environ.get("SPYDE_UPDATE_REPO", "CSSFrancis/spyde")
_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
_HTML_RELEASES = f"https://github.com/{GITHUB_REPO}/releases"


# ── semver ───────────────────────────────────────────────────────────────────

_SEMVER = re.compile(
    r"^v?(\d+)\.(\d+)\.(\d+)(?:[-.]?(?:rc|a|b|beta|alpha)\.?(\d+))?")


def parse_version(s: str) -> Optional[tuple]:
    """(major, minor, patch, pre) where pre=inf for a final release (sorts
    after pre-releases). Returns None if unparseable."""
    if not s:
        return None
    m = _SEMVER.match(s.strip())
    if not m:
        return None
    major, minor, patch, pre = m.groups()
    pre_n = int(pre) if pre is not None else float("inf")
    return (int(major), int(minor), int(patch), pre_n)


def is_newer(candidate: str, current: str) -> bool:
    c, cur = parse_version(candidate), parse_version(current)
    if c is None or cur is None:
        return False
    return c > cur


def _is_prerelease(tag: str) -> bool:
    return bool(re.search(r"(rc|alpha|beta|[-.]a\.|[-.]b\.)", tag or "",
                          re.IGNORECASE))


# ── result type ──────────────────────────────────────────────────────────────

@dataclass
class UpdateInfo:
    available: bool
    current: str
    latest: Optional[str] = None
    url: Optional[str] = None
    notes: Optional[str] = None
    error: Optional[str] = None


# ── check ────────────────────────────────────────────────────────────────────

def _current_version() -> str:
    try:
        from spyde._build_info import build_info
        return build_info()["version"]
    except Exception:
        try:
            from spyde._version import __version__
            return __version__
        except Exception:
            return "0.0.0"


def _current_channel() -> str:
    try:
        from spyde._build_info import build_info
        return build_info()["channel"]
    except Exception:
        return "dev"


def check(channel: Optional[str] = None, timeout: float = 6.0) -> UpdateInfo:
    """Query GitHub Releases for a newer version. Network-only, never raises.

    channel: "stable" (only final releases) or "beta" (newest of stable +
    pre-releases). When None, falls back to the build channel.
    """
    current = _current_version()
    channel = channel or _current_channel()
    try:
        req = urllib.request.Request(
            _API_LATEST + "?per_page=15",
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "SpyDE-updater"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return UpdateInfo(available=False, current=current,
                          error=f"update check failed: {e}")

    best_tag = None
    best_rel = None
    for rel in releases:
        if rel.get("draft"):
            continue
        tag = rel.get("tag_name") or rel.get("name") or ""
        # on stable, skip pre-releases (GitHub flag OR tag heuristic)
        if channel == "stable" and (rel.get("prerelease") or _is_prerelease(tag)):
            continue
        if parse_version(tag) is None:
            continue
        if best_tag is None or is_newer(tag, best_tag):
            best_tag, best_rel = tag, rel

    if best_tag is None:
        return UpdateInfo(available=False, current=current,
                          error="no comparable release found")
    if is_newer(best_tag, current):
        return UpdateInfo(
            available=True, current=current, latest=best_tag,
            url=(best_rel.get("html_url") or _HTML_RELEASES),
            notes=(best_rel.get("body") or "")[:4000])
    return UpdateInfo(available=False, current=current, latest=best_tag)


# ── apply (uv-managed install) ───────────────────────────────────────────────

def is_uv_managed() -> bool:
    """True when SpyDE was installed in the uv-managed layout (a pyproject.toml +
    uv.lock sit next to the app), as opposed to the portable single-exe."""
    base = _install_root()
    return base is not None and os.path.exists(os.path.join(base, "uv.lock"))


def _install_root() -> Optional[str]:
    for base in (os.environ.get("SPYDE_HOME"),
                 os.path.dirname(sys.executable),
                 os.getcwd()):
        if base and os.path.exists(os.path.join(base, "pyproject.toml")):
            return base
    return None


def apply_uv_sync(progress=None) -> dict:
    """Update the uv-managed install in place: ``uv sync`` against the (already
    updated) pyproject/lock. Returns {"ok","message"}. The caller is expected to
    have fetched the new source/lock first (or this just re-syncs current)."""
    import shutil
    import subprocess
    base = _install_root()
    if base is None:
        return {"ok": False, "message": "not a uv-managed install"}
    uv = os.environ.get("SPYDE_UV") or shutil.which("uv")
    if uv is None:
        for name in ("uv.exe", "uv"):
            cand = os.path.join(base, name)
            if os.path.exists(cand):
                uv = cand
                break
    if uv is None:
        return {"ok": False, "message": "uv not found"}
    try:
        proc = subprocess.Popen(
            [uv, "sync", "--torch-backend=auto"], cwd=base,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:
            if progress is not None:
                progress(line.rstrip())
        proc.wait(timeout=3600)
        ok = proc.returncode == 0
        return {"ok": ok, "message": "Updated — restart SpyDE"
                if ok else f"uv sync exited {proc.returncode}"}
    except Exception as e:
        return {"ok": False, "message": f"uv sync failed: {e}"}
