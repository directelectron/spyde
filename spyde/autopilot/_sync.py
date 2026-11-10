"""Regenerate SpyDE's copy of Autopilot's dataset reader.

    python -m spyde.autopilot._sync <de_autopilot checkout> [commit]

``dataset_reader.py`` here is byte-identical to Autopilot's
``de_automate/dataset_reader.py`` at the commit its first line names, apart
from that line. It reads the record an Autopilot run folder holds, and the two
apps must not disagree about it, so a fix belongs in Autopilot and comes over
by running this again. The reader imports only numpy and the standard library,
which this checks before writing the copy.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

SOURCE = "de_automate/dataset_reader.py"
HEADER = "# Copied from directelectron/de_autopilot {source} at {commit} by _sync.py; edit it there.\n"
#: An import of anything but numpy, the standard library and __future__.
_FOREIGN_IMPORT = re.compile(r"^\s*(from|import)\s+(de_\w+|spyde|pyxem\w*)\b", re.MULTILINE)


def sync(autopilot: str, commit: str = "origin/main") -> pathlib.Path:
    sha = subprocess.run(
        ["git", "-C", autopilot, "rev-parse", "--short", commit],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    text = subprocess.run(
        ["git", "-C", autopilot, "show", f"{commit}:{SOURCE}"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    if _FOREIGN_IMPORT.findall(text):
        raise SystemExit(f"{SOURCE}: imports more than numpy and the standard library")
    target = pathlib.Path(__file__).with_name("dataset_reader.py")
    target.write_text(HEADER.format(source=SOURCE, commit=sha) + text, encoding="utf-8")
    return target


if __name__ == "__main__":
    print(sync(*sys.argv[1:3]))
