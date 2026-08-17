"""
registry.py — the staged-action registry and the window-controller protocol.

A shell app has exactly TWO dispatch paths for renderer→backend actions; do not
invent a third:

1. **Toolbar actions** — declared in the app's toolbar config, resolved and
   invoked by the session with an :class:`~de_shell.actions.context.ActionContext`.
2. **Staged actions** — the wizard/caret protocol: an action name maps to a
   ``"module.function"`` dotted path with the uniform ``fn(session, plot,
   payload)`` signature. Modules are imported LAZILY, so heavy dependencies load
   on first use rather than at startup — which is most of why the table is
   dotted strings instead of function references.

Staged-action NAMING CONVENTION (``<key>`` is the wizard's short prefix):

    <key>_open           wizard mounted → start live preview / controller
    <key>_close          wizard unmounted → tear everything down
    <key>_tune           debounced live re-tune of preview params
    <key>_set_<param>    discrete parameter change
    <key>_run            heavy compute stage (may open a result tree)
    <key>_commit         snapshot the live result into a new result tree

Wizard-specific extra stages are allowed but must keep the ``<key>_`` prefix.

**The tables live in the app, the mechanism lives here.** An app calls
:func:`register_staged` (or :func:`register_staged_table`) at import to populate
the registry, and :func:`register_wizard_schema` to say where each wizard's
parameter schema is declared. The shell has no business knowing that
``fit_open`` exists.

WindowController protocol
-------------------------
Windows that are NOT registered plots (bare ``figure`` emits) must register a
*controller* with ``session.register_window_controller(window_id, controller)``
so dispatch and teardown can reach them. A controller is duck-typed:

    window_id: int                     # the window it drives
    close() -> None                    # full teardown; called by the session's
                                       # _forget_window when the window goes
                                       # away for ANY reason
    handle_action(name, payload) -> bool   # optional: consume an action aimed
                                           # at this window; return True if
                                           # handled

:class:`de_shell.actions.wizard.WizardController` provides a base implementation.
"""
from __future__ import annotations

import importlib
from typing import Callable

#: action name → "module.function". Populated by the app.
STAGED_HANDLERS: dict[str, str] = {}


def register_staged(name: str, dotted_path: str) -> None:
    """Register a staged action (``fn(session, plot, payload)``) by dotted path."""
    STAGED_HANDLERS[name] = dotted_path


def register_staged_table(table: dict[str, str]) -> None:
    """Register a whole table at once. Later registrations win, so an app can
    override a default."""
    STAGED_HANDLERS.update(table)


def resolve_staged(name: str) -> Callable | None:
    """Lazily import and return the handler for a staged action name."""
    dotted = STAGED_HANDLERS.get(name)
    if dotted is None:
        return None
    mod, fn = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), fn)


# ─────────────────────────────────────────────────────────────────────────────
# Wizard parameter schemas — the single host-agnostic lookup
# ─────────────────────────────────────────────────────────────────────────────
#
# Every wizard declares its parameter schema in one place — a `parameters`
# classattr on its WizardController, or a module-level dict for controller-less
# wizards — in the same spec as the toolbar config's `parameters:`. This table
# maps a wizard key to wherever its schema lives, so ANY host (an Electron
# panel, a notebook form generator, a doc generator) resolves them uniformly.

#: wizard key → (module, attribute), or (`YAML_SCHEMA`, toolbar action title).
_WIZARD_SCHEMAS: dict[str, tuple[str, str]] = {}

#: Sentinel module name meaning "resolve this from the app's toolbar config".
YAML_SCHEMA = "__yaml__"

#: Set by `set_yaml_schema_resolver`. Takes a toolbar action title, returns its
#: `parameters` dict.
_yaml_resolver: Callable[[str], dict] | None = None


def register_wizard_schema(key: str, module: str, attr: str) -> None:
    """Declare where wizard *key*'s parameter schema lives.

    ``module=YAML_SCHEMA`` means "look *attr* up as a toolbar action title via
    the registered YAML resolver".
    """
    _WIZARD_SCHEMAS[key] = (module, attr)


def register_wizard_schemas(table: dict[str, tuple[str, str]]) -> None:
    """Register a whole schema table at once."""
    _WIZARD_SCHEMAS.update(table)


def set_yaml_schema_resolver(fn: Callable[[str], dict]) -> None:
    """Install the app's toolbar-config lookup for ``YAML_SCHEMA`` entries."""
    global _yaml_resolver
    _yaml_resolver = fn


def wizard_parameters(key: str) -> dict:
    """Return wizard ``key``'s declared parameter schema (a copy).

    The uniform entry point for rendering a wizard's controls in ANY host —
    same spec as the toolbar config's ``parameters:`` (type/name/default/min/
    max/step/choices/tab/extensions). Raises ``KeyError`` for unknown keys.
    """
    module, attr = _WIZARD_SCHEMAS[key]
    if module == YAML_SCHEMA:
        if _yaml_resolver is None:
            raise RuntimeError(
                f"wizard {key!r} declares a toolbar-config schema, but no "
                "resolver is installed — call set_yaml_schema_resolver() at "
                "app startup."
            )
        return _yaml_resolver(attr)
    obj = getattr(importlib.import_module(module), attr)
    schema = obj if isinstance(obj, dict) else getattr(obj, "parameters", {})
    return dict(schema)


def wizard_keys() -> list[str]:
    """All wizard keys with a declared schema."""
    return list(_WIZARD_SCHEMAS)
