from __future__ import annotations
import importlib
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from spyde.drawing.plots.plot_states import PlotState

from spyde import TOOLBAR_ACTIONS

from functools import partial


def resolve_icon_path(icon_value: str) -> str:
    """Resolve an icon specification into an absolute path or Qt resource path."""
    # Keep Qt resource paths (e.g., ":/icons/foo.png") as-is
    if isinstance(icon_value, str) and icon_value.startswith(":"):
        return icon_value

    p = Path(icon_value)
    if p.is_absolute():
        return str(p)

    # Resolve relative to the spyde package directory
    try:
        import spyde

        base = Path(spyde.__file__).resolve().parent
    except Exception:
        base = Path(__file__).resolve().parent

    return str((base / icon_value).resolve())


_SIGNAL_CLASS_CACHE: dict[str, type] = {}


def _resolve_signal_class(path: str) -> type:
    """Import and cache a signal class from a dotted path (e.g.
    ``pyxem.signals.Diffraction2D``)."""
    cls = _SIGNAL_CLASS_CACHE.get(path)
    if cls is None:
        module_path, _, attr = path.rpartition(".")
        cls = getattr(importlib.import_module(module_path), attr)
        _SIGNAL_CLASS_CACHE[path] = cls
    return cls


def _gate_signal_type(plot_state: "PlotState", navigation_only) -> str:
    """The ``_signal_type`` string a ``signal_types``/``exclude_signal_types``
    gate should compare against for *plot_state*.

    For a SIGNAL plot this is simply the displayed signal's own type. For a
    NAVIGATOR plot (``navigation_only`` truthy — the action is gated to
    navigator plots, e.g. Play/Fast Forward), the plot's *own* displayed
    signal is a DERIVED trace (``signal_tree._initialize_navigator`` sums the
    root over its signal axes to build the nav image/line) — it is never the
    tree root's class, so gating a navigator-only action on e.g.
    ``signal_types: [insitu]`` would never match if we looked at
    ``plot_state.current_signal``. Instead resolve against the TREE ROOT
    signal's type, which is what actually carries the user-facing signal_type
    (e.g. set via ``set_signal_type("insitu")`` on load).

    Non-navigator gates (the vector actions' ``exclude_signal_types``, dense
    diffraction gates, etc.) are unaffected — they keep reading
    ``current_signal`` as before.
    """
    signal = plot_state.current_signal
    if navigation_only:
        tree = getattr(plot_state.plot, "signal_tree", None)
        root = getattr(tree, "root", None)
        if root is not None:
            return root._signal_type
    return signal._signal_type


def get_toolbar_actions_for_plot(
    plot_state: "PlotState",
) -> tuple[
    list, list[str], list[str], list[str], list[bool], list[dict], list[list[tuple]], list
]:
    """
    Build the action metadata lists for a given PlotState by filtering TOOLBAR_ACTIONS.

    Returns
    -------
    (functions, icons, names, toolbar_sides, toggles, parameters, sub_toolbars, setup_functions)
      functions: list[Callable] (each partially applied with action_name)
      icons: list[str] (resolved icon paths)
      names: list[str] (action identifiers)
      toolbar_sides: list[str] ("left"/"right"/"top"/"bottom")
      toggles: list[bool] (whether action is checkable)
      parameters: list[dict] (parameter definitions for CaretParams popouts)
      sub_toolbars: list[list[tuple]] (each inner list holds sub-action tuples)
      setup_functions: list[Callable | None] (optional one-time first-show callbacks)
    """
    functions = []
    icons = []
    names = []
    toolbar_sides = []
    toggles = []
    parameters = []
    sub_toolbars = []  # Parallel list: each entry is a list of subfunction tuples
    setup_functions = []

    for action, meta in TOOLBAR_ACTIONS["functions"].items():
        signal_types = meta.get("signal_types")
        exclude_signal_types = meta.get("exclude_signal_types")
        signal_class = meta.get("signal_class")
        requires_vectors = meta.get("requires_vectors", False)
        plot_dim = meta.get("plot_dim", [1, 2])
        navigation_only = meta.get("navigation")
        params = meta.get("parameters", {})

        signal = plot_state.current_signal
        plot_signal_type = _gate_signal_type(plot_state, navigation_only)

        # requires_vectors: action only shows once the plot's signal tree has
        # diffraction_vectors attached (set after Find Vectors completes).
        # PlotState.rebuild_toolbars() re-runs this filter at that point.
        tree = getattr(plot_state.plot, "signal_tree", None)
        has_vectors = getattr(tree, "diffraction_vectors", None) is not None

        add_action = (
            (signal_types is None or plot_signal_type in signal_types)
            # exclude_signal_types: keep an action OFF a signal_type even though
            # it would match signal_class by isinstance. Matched on the
            # _signal_type STRING so it covers lazy+eager variants uniformly
            # (LazyDiffractionVectorsImage is NOT a subclass of the eager one,
            # but both share the signal_type) — the dense diffraction actions
            # use this to stay off the vectors-result image.
            and (
                exclude_signal_types is None
                or plot_signal_type not in exclude_signal_types
            )
            # signal_class gates by isinstance, so subclasses qualify too
            # (e.g. ElectronDiffraction2D passes a Diffraction2D gate)
            and (
                signal_class is None
                or isinstance(signal, _resolve_signal_class(signal_class))
            )
            and (not requires_vectors or has_vectors)
            and (plot_state.dimensions in plot_dim)
            and (
                navigation_only is None
                or navigation_only == plot_state.plot.is_navigator
            )
        )

        if not add_action:
            continue

        function_path = meta["function"]
        module_path, _, attr = function_path.rpartition(".")
        base_func = getattr(importlib.import_module(module_path), attr)
        wrapped_func = partial(base_func, action_name=action)
        functions.append(wrapped_func)
        icons.append(resolve_icon_path(meta["icon"]))
        names.append(action)
        toolbar_sides.append(meta.get("toolbar_side", "left"))
        toggles.append(meta.get("toggle", False))
        parameters.append(params)

        # Optional one-time setup function called on first show of the CaretParams.
        setup_fn = None
        setup_path = meta.get("setup_function")
        if setup_path:
            s_module, _, s_attr = setup_path.rpartition(".")
            setup_fn = getattr(importlib.import_module(s_module), s_attr)
        setup_functions.append(setup_fn)

        # Collect optional subfunctions
        sub_defs = meta.get("subfunctions", {})
        sub_entries = []
        for sub_meta in sub_defs:
            sub_function_path = sub_defs[sub_meta]["function"]
            sub_module_path, _, sub_attr = sub_function_path.rpartition(".")
            sub_func = getattr(importlib.import_module(sub_module_path), sub_attr)
            sub_func = partial(sub_func, action_name=sub_meta)
            sub_entries.append(
                (
                    sub_func,
                    resolve_icon_path(
                        sub_defs[sub_meta].get("icon", meta.get("icon", ""))
                    ),
                    sub_defs[sub_meta].get("name", sub_meta),
                    sub_defs[sub_meta].get("toggle", False),
                    sub_defs[sub_meta].get("parameters", {}),
                )
            )
        sub_toolbars.append(sub_entries)

    return functions, icons, names, toolbar_sides, toggles, parameters, sub_toolbars, setup_functions


def _action_matches_plot(action: str, meta: dict, plot_state: "PlotState") -> bool:
    """Apply the same signal_type / dimension / vectors filters used to decide
    whether an action is offered for a plot — WITHOUT importing the action's
    function module (so toolbar rendering never depends on heavy action code)."""
    signal_types = meta.get("signal_types")
    exclude_signal_types = meta.get("exclude_signal_types")
    signal_class = meta.get("signal_class")
    requires_vectors = meta.get("requires_vectors", False)
    plot_dim = meta.get("plot_dim", [1, 2])
    navigation_only = meta.get("navigation")

    signal = plot_state.current_signal
    plot_signal_type = _gate_signal_type(plot_state, navigation_only)

    tree = getattr(plot_state.plot, "signal_tree", None)
    has_vectors = getattr(tree, "diffraction_vectors", None) is not None

    return (
        (signal_types is None or plot_signal_type in signal_types)
        and (exclude_signal_types is None or plot_signal_type not in exclude_signal_types)
        and (
            signal_class is None
            or isinstance(signal, _resolve_signal_class(signal_class))
        )
        and (not requires_vectors or has_vectors)
        and (plot_state.dimensions in plot_dim)
        and (
            navigation_only is None
            or navigation_only == plot_state.plot.is_navigator
        )
    )


def get_toolbar_config_for_plot(plot_state: "PlotState") -> list[dict]:
    """
    Return a JSON-serialisable list of toolbar action descriptors for *plot_state*.
    Sent to Electron so it can render the per-plot toolbar buttons.

    This reads metadata from TOOLBAR_ACTIONS and applies the visibility filters
    directly — it never imports the action *function* modules, so rendering the
    toolbar is decoupled from the (heavy, possibly-Qt) action implementations.
    The function is resolved on demand when the action is actually invoked.
    """
    actions = []
    for action, meta in TOOLBAR_ACTIONS["functions"].items():
        try:
            if not _action_matches_plot(action, meta, plot_state):
                continue
        except Exception:
            continue

        sub_actions = []
        for sub_name, sub_meta in (meta.get("subfunctions", {}) or {}).items():
            sub_actions.append({
                "name": sub_name,
                "icon": resolve_icon_path(sub_meta.get("icon", meta.get("icon", ""))),
                "label": sub_meta.get("name", sub_name),
                "toggle": sub_meta.get("toggle", False),
                "parameters": sub_meta.get("parameters", {}),
            })

        actions.append({
            "name": action,
            "icon": resolve_icon_path(meta.get("icon", "")),
            "side": meta.get("toolbar_side", "left"),
            "toggle": meta.get("toggle", False),
            "parameters": meta.get("parameters", {}),
            "subfunctions": sub_actions,
        })
    return actions
