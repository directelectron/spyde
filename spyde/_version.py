"""Single source of truth for the SpyDE version.

Kept in its own tiny module (no heavy imports) so both ``spyde/__init__.py``
and ``pyproject.toml`` (via setuptools dynamic version) can read it without
importing the full package. Bump this one line to release.
"""

__version__ = "0.1.0"
