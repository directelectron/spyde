"""gen_report_qr.py — poster QR codes for the published reports.

    uv run --with segno python -m scripts.gen_report_qr

A published report lives at a stable URL under the GitHub Pages docs
(``<base>/reports/<file>``), which makes it a good thing to put on a conference
poster: one scan and the reader is exploring the actual dataset on their phone
instead of looking at a static figure of it.

Writes, per report, into ``docs-site/public/media/reports/``:

  ``<id>-qr.svg``   vector — this is the one to place on a poster. Print it
                    at 3 cm or larger; below about 2 cm phone cameras start
                    struggling at typical poster-viewing distance.
  ``<id>-qr.png``   raster fallback for tools that will not take SVG.

``segno`` is a pure-Python, zero-dependency QR encoder and is NOT a project
dependency — run this script with ``uv run --with segno`` (as above) so it is
fetched only when a QR is actually being made.

Error correction is fixed at level H (~30% recoverable), because posters get
scuffed, curled, and photographed at an angle.
"""
from __future__ import annotations

import os
import sys

# The GitHub Pages root for this repo (repos/CSSFrancis/spyde/pages → html_url).
# Reports are published by doc/conf.py's `_stage_reports` + `html_extra_path`.
DOCS_BASE = "https://cssfrancis.github.io/spyde"
REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs-site", "public",
    "media", "reports")


def report_url(file_name: str) -> str:
    return f"{DOCS_BASE}/reports/{file_name}"


def _entries() -> list[tuple[str, str]]:
    """``(id, file)`` for every report in the registry — parsed out of
    ``report-catalogue/index.ts`` so this script and the docs site cannot drift.

    A tiny hand parse rather than a TS toolchain: the registry is a flat literal
    and the two fields we need are unambiguous."""
    import re
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "report-catalogue", "index.ts")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    ids = re.findall(r"^\s*id:\s*'([^']+)'", src, re.M)
    files = re.findall(r"^\s*file:\s*'([^']+)'", src, re.M)
    if len(ids) != len(files):
        raise SystemExit(
            f"report-catalogue/index.ts: {len(ids)} ids but {len(files)} files — the "
            "hand parse needs every entry to carry both")
    return list(zip(ids, files))


def main() -> int:
    try:
        import segno
    except ImportError:
        raise SystemExit(
            "segno is not installed — run:\n"
            "    uv run --with segno python -m scripts.gen_report_qr")

    out_dir = os.path.abspath(REPORTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    entries = _entries()
    if not entries:
        print("[qr] no reports registered", flush=True)
        return 0

    for rid, file_name in entries:
        url = report_url(file_name)
        qr = segno.make(url, error="h")
        svg = os.path.join(out_dir, f"{rid}-qr.svg")
        png = os.path.join(out_dir, f"{rid}-qr.png")
        # A quiet zone of 4 modules is the spec minimum and matters on a busy
        # poster; scale is in module-units for SVG (resolution-independent).
        qr.save(svg, scale=10, border=4, dark="#11111b", light="#ffffff")
        qr.save(png, scale=12, border=4, dark="#11111b", light="#ffffff")
        print(f"[qr] {rid}: {url}", flush=True)
        print(f"[qr]   {svg}", flush=True)
        print(f"[qr]   {png}", flush=True)
    print(f"[qr] {len(entries)} code(s); print the SVG at 3 cm or larger",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
