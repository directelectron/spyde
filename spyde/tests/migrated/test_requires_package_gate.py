"""The `requires_package:` toolbar gate (0.3.0 Wave 0, #49).

Mirrors `requires_vectors`: an action declaring an optional extra stays hidden
until that extra is importable, so a user without `spyde[eels]` never sees a
button that would raise ImportError when clicked.

Both filter paths are covered. `get_toolbar_actions_for_plot` (which resolves
and imports the action function) and `get_toolbar_config_for_plot` /
`_action_matches_plot` (which must NOT import anything) apply the same gates,
and it is easy to add a gate to one and forget the other.
"""
from __future__ import annotations

import types

import pytest

from spyde.drawing.toolbars import plot_control_toolbar as pct
from spyde.drawing.toolbars.plot_control_toolbar import (
    _action_matches_plot,
    _packages_present,
    install_hint,
    package_available,
)


@pytest.fixture(autouse=True)
def _clear_package_cache():
    """package_available() memoises; a stale entry would leak between tests."""
    pct._PACKAGE_CACHE.clear()
    yield
    pct._PACKAGE_CACHE.clear()


class _FakePlot:
    """Minimal stand-in for the toolbar filter: it reads ``_signal_type`` off
    the current signal and ``diffraction_vectors`` off the tree."""

    def __init__(self, signal_type: str = ""):
        signal = types.SimpleNamespace(_signal_type=signal_type)
        self.plot_state = types.SimpleNamespace(
            current_signal=signal, dimensions=2, plot=self)
        self.signal_tree = types.SimpleNamespace(diffraction_vectors=None,
                                                 root=None)
        self.is_navigator = False


class TestPackageAvailable:
    def test_detects_an_installed_package(self):
        assert package_available("numpy") is True

    def test_detects_a_missing_package(self):
        assert package_available("spyde_definitely_not_installed") is False

    def test_result_is_cached(self):
        package_available("numpy")
        assert pct._PACKAGE_CACHE["numpy"] is True

    def test_does_not_import_the_package(self, monkeypatch):
        """find_spec, not import: resolving a toolbar must never pay a
        multi-second import or run a third-party package's side effects just to
        decide whether to draw a button."""
        import builtins
        real_import = builtins.__import__

        def _boom(name, *a, **k):
            if name == "numexpr":
                raise AssertionError("requires_package imported the package")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _boom)
        assert package_available("numexpr") in (True, False)

    def test_unresolvable_package_is_absent_not_an_error(self, monkeypatch):
        """A half-installed dist or a namespace-package parent makes find_spec
        raise; that must read as 'absent', not crash the whole toolbar."""
        def _raise(name):
            raise ValueError("mangled dist-info")

        monkeypatch.setattr(pct.importlib.util, "find_spec", _raise)
        assert package_available("anything") is False


class TestPackagesPresent:
    def test_no_declaration_is_always_allowed(self):
        assert _packages_present({}) is True
        assert _packages_present({"requires_package": None}) is True

    def test_single_name_as_a_string(self):
        assert _packages_present({"requires_package": "numpy"}) is True
        assert _packages_present({"requires_package": "spyde_nope"}) is False

    def test_list_requires_all_of_them(self):
        assert _packages_present({"requires_package": ["numpy", "scipy"]}) is True
        assert _packages_present(
            {"requires_package": ["numpy", "spyde_nope"]}) is False


class TestInstallHint:
    @pytest.mark.parametrize("pkg,extra", [("exspy", "eels"),
                                           ("kikuchipy", "ebsd"),
                                           ("atomap", "atoms")])
    def test_maps_each_gated_package_to_its_extra(self, pkg, extra):
        assert install_hint(pkg) == f'pip install "spyde[{extra}]"'

    def test_unknown_package_falls_back_to_a_plain_pip_line(self):
        assert install_hint("mystery") == "pip install mystery"

    def test_every_declared_extra_exists_in_pyproject(self):
        """The hint has to be runnable — an extra named here but absent from
        pyproject would tell the user to run a command that fails."""
        import re
        from pathlib import Path
        import spyde

        root = Path(spyde.__file__).resolve().parent.parent
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
        block = text.split("[project.optional-dependencies]", 1)[1]
        block = block.split("\n[", 1)[0]
        declared = set(re.findall(r"^(\w[\w-]*)\s*=\s*\[", block, re.M))
        for extra in pct._EXTRA_FOR_PACKAGE.values():
            assert extra in declared, f"extra {extra!r} missing from pyproject"


class TestFilterIntegration:
    """The gate is applied by BOTH filter paths, not just the one."""

    def test_action_matches_plot_hides_a_missing_package(self):
        plot = _FakePlot()
        meta = {"requires_package": "spyde_nope", "plot_dim": [1, 2]}
        assert _action_matches_plot("X", meta, plot.plot_state) is False

    def test_action_matches_plot_allows_a_present_package(self):
        plot = _FakePlot()
        meta = {"requires_package": "numpy", "plot_dim": [1, 2]}
        assert _action_matches_plot("X", meta, plot.plot_state) is True

    def test_gate_is_wired_into_both_filters(self):
        """Guards the real failure mode: adding the gate to one filter and
        forgetting the other, so the button renders but does not dispatch (or
        vice versa)."""
        import inspect
        src = inspect.getsource(pct)
        body = src.split("def get_toolbar_actions_for_plot", 1)[1]
        listing, matching = body.split("def _action_matches_plot", 1)
        assert "_packages_present" in listing, \
            "get_toolbar_actions_for_plot is missing the requires_package gate"
        assert "_packages_present" in matching, \
            "_action_matches_plot is missing the requires_package gate"


class TestRealToolbarUnaffected:
    def test_no_current_action_is_accidentally_gated(self):
        """No shipped action declares requires_package yet, so the whole
        existing toolbar must be untouched by this change."""
        from spyde import TOOLBAR_ACTIONS
        gated = {name for name, meta in TOOLBAR_ACTIONS["functions"].items()
                 if meta.get("requires_package")}
        assert gated == set()
