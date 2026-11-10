"""Regenerate SpyDE's copy of Autopilot's dataset reader.

    python -m spyde.autopilot._sync <de_autopilot checkout> [commit]

``dataset_reader.py`` here is Autopilot's ``de_automate/dataset_reader.py``
at the commit its first line names, byte for byte after that line, which
also records the SHA-256 of those bytes (``test_autopilot_run_folder``
checks the copy still matches it). It reads the record an Autopilot run
folder holds, and the two apps must not disagree about it, so a fix belongs
in Autopilot and comes over by running this again. The reader imports only
numpy and the standard library, which this checks before writing the copy.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess
import sys

SOURCE = "de_automate/dataset_reader.py"
HEADER = (
    "# Copied from directelectron/de_autopilot {source} at {commit} by _sync.py;"
    " edit it there. sha256 {digest}\n"
)
#: The header's record of what the copy must be.
HEADER_DIGEST = re.compile(rb"sha256 ([0-9a-f]{64})")
#: An import of anything but numpy, the standard library and __future__.
_FOREIGN_IMPORT = re.compile(rb"^\s*(from|import)\s+(de_\w+|spyde|pyxem\w*)\b", re.MULTILINE)
COPY = pathlib.Path(__file__).with_name("dataset_reader.py")


def sync(autopilot: str, commit: str = "origin/main") -> pathlib.Path:
    sha = subprocess.run(
        ["git", "-C", autopilot, "rev-parse", "--short", commit],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    # Bytes, not text: the copy is the file exactly, line endings and all.
    source = subprocess.run(
        ["git", "-C", autopilot, "show", f"{commit}:{SOURCE}"],
        capture_output=True, check=True,
    ).stdout
    if _FOREIGN_IMPORT.findall(source):
        raise SystemExit(f"{SOURCE}: imports more than numpy and the standard library")
    header = HEADER.format(source=SOURCE, commit=sha, digest=hashlib.sha256(source).hexdigest())
    COPY.write_bytes(header.encode("utf-8") + source)
    return COPY


def drifted() -> bool:
    """Whether the copy has been edited since it was synced."""
    header, _, body = COPY.read_bytes().partition(b"\n")
    recorded = HEADER_DIGEST.search(header)
    return recorded is None or recorded.group(1).decode() != hashlib.sha256(body).hexdigest()


if __name__ == "__main__":
    print(sync(*sys.argv[1:3]))
