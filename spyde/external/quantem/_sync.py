"""Re-vendor the quantem orientation-matching subtree.

Run directly after changing ``COMMIT``::

    uv run python -m spyde.external.quantem._sync

Every vendored file is downloaded byte for byte from upstream and then passed
through :data:`REWRITES`, which only repoints absolute ``quantem.*`` imports at
this package. So a re-sync is this script plus ``git diff`` — read the diff to
see what upstream changed, never hand-merge. Anything that cannot be expressed
as a rewrite rule here belongs in the SpyDE-side adapter instead, so that the
vendored files stay a mechanical copy.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

REPO = "cophus/quantem"
BRANCH = "acom"
#: Pinned upstream commit. Bump deliberately — the branch is a long-lived fork
#: (hundreds of commits behind its own ``dev``), so "latest" is not a stable
#: target and the pin is what makes a re-sync reviewable.
COMMIT = "d50fec5f4bb8"

#: Upstream path → path under ``diffraction/``. This is the closure of
#: ``OrientationMap.match_orientations`` and ``.refine_orientations`` and
#: nothing more; see ``__init__.py`` for what is deliberately left out.
FILES = {
    "src/quantem/diffraction/orientation.py": "orientation.py",
    "src/quantem/diffraction/crystal.py": "crystal.py",
    "src/quantem/diffraction/rotations.py": "rotations.py",
    "src/quantem/diffraction/illumination.py": "illumination.py",
    "src/quantem/diffraction/defaults.py": "defaults.py",
    "src/quantem/diffraction/data/lobato.json": "data/lobato.json",
}

#: Applied in order to every vendored ``.py``. Ordinary string replacements,
#: so indentation of the function-local imports is preserved.
REWRITES = (
    ("from quantem.diffraction.", "from ."),
    ("from quantem.core.datastructures.vector import Vector",
     "from .._compat import Vector"),
    ("from quantem.core.io.serialize import AutoSerialize",
     "from .._compat import AutoSerialize"),
    ("from quantem.core.utils.utils import electron_wavelength_angstrom",
     "from .._compat import electron_wavelength_angstrom"),
    # The Lobato scattering-factor table is loaded by package name.
    ('resources.files("quantem.diffraction")', "resources.files(__package__)"),
)


def rewrite(source: str) -> str:
    """Repoint upstream's absolute imports at this package."""
    for old, new in REWRITES:
        source = source.replace(old, new)
    remaining = re.findall(r"^\s*(?:from|import)\s+quantem\S*", source, re.M)
    if remaining:
        raise RuntimeError(
            "unhandled quantem import — add a rule to REWRITES or drop the "
            f"file from FILES: {sorted(set(remaining))}"
        )
    return source


def main() -> None:
    here = Path(__file__).parent / "diffraction"
    for upstream_path, local_name in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{upstream_path}"
        with urllib.request.urlopen(url) as response:
            text = response.read().decode("utf-8")
        if local_name.endswith(".py"):
            text = rewrite(text)
        destination = here / local_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
        print(f"{destination.relative_to(here.parent)}  ({len(text):,} chars)")


if __name__ == "__main__":
    main()
