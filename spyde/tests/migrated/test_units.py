"""
Axis units must display as clean unicode, not raw LaTeX.

Calibrated diffraction axes often carry LaTeX-ish units like ``$A^{-1}$`` which
anyplotlib (no MathJax) would render literally on the axis labels / scale bar.
``_clean_units`` normalises them.
"""
from __future__ import annotations

from spyde.drawing.plots.plot import _clean_units


def test_reciprocal_angstrom_latex_becomes_unicode():
    assert _clean_units("$A^{-1}$") == "Å⁻¹"
    assert _clean_units("A^{-1}") == "Å⁻¹"


def test_pyxem_latex_angstrom_macro_becomes_unicode():
    # pyxem calibrates reciprocal axes as `$\AA^{-1}$` (LaTeX ångström macro) —
    # without expanding \AA the scale bar showed a literal backslash ("\AA⁻¹").
    assert _clean_units(r"$\AA^{-1}$") == "Å⁻¹"
    assert _clean_units(r"\AA") == "Å"
    assert "\\" not in _clean_units(r"$\AA^{-1}$")


def test_reciprocal_nm_and_plain_units():
    assert _clean_units("$nm^{-1}$") == "nm⁻¹"
    assert _clean_units("1/nm") == "1/nm"
    assert _clean_units("nm") == "nm"


def test_no_dollar_or_braces_remain():
    out = _clean_units("$A^{-1}$")
    assert "$" not in out and "{" not in out and "}" not in out
