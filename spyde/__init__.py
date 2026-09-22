import logging
from importlib.resources import files
import yaml

log = logging.getLogger(__name__)

# Load the configuration .yaml files at package initialization

with files(__package__).joinpath("toolbars.yaml").open("r", encoding="utf-8") as f:
    TOOLBAR_ACTIONS = yaml.safe_load(f)

with files(__package__).joinpath("metadata_widget.yaml").open(
    "r", encoding="utf-8"
) as f:
    METADATA_WIDGET_CONFIG = yaml.safe_load(f)

__all__ = ["TOOLBAR_ACTIONS", "METADATA_WIDGET_CONFIG"]


# Map SpyDE's own loggers to the log panel's area filter. The shell only knows
# about itself and anyplotlib (de_shell.log_stream._SHELL_AREA_RULES); an app's
# subsystems are its own business, so SpyDE declares them here. Ordered
# most-specific prefix first — first match wins.
#
# Registered at package import rather than in backend/app.py so it is in effect
# for ANY spyde process, tests included: the handler drops sub-WARNING records
# from packages that aren't declared verbose, so a rule set that only landed on
# the app's startup path would silently swallow INFO logs everywhere else.

# A rule matches a logger name exactly or as a DOTTED prefix, so it has to name
# the module — "spyde.actions.orientation" never matched `orientation_action`,
# only a submodule of a package that does not exist. Every one of these was
# being tagged "actions" instead. `test_log_stream.TestOrientationArea` reads
# the directory, so a new orientation module fails the test rather than
# silently landing in the wrong filter.
_ORIENTATION_MODULES = (
    "orientation_action", "orientation_compute",
    "vector_orientation_om", "vector_orientation_quantem", "vector_refine_ipf",
    "ipf_density", "ipf_refine", "ipf_refine_render", "ipf_view", "ipf_window",
)

_LOG_AREA_RULES = (
    ("spyde.dask_manager", "dask"),
    ("spyde.compute_backend", "dask"),
    ("spyde.drawing.update_functions", "navigator"),
    ("spyde.drawing", "drawing"),
    ("spyde.signal_tree", "navigator"),
    ("spyde.array_cache", "navigator"),
    ("spyde.actions.find_vectors", "vectors"),
    *((f"spyde.actions.{name}", "orientation")
      for name in _ORIENTATION_MODULES),
    ("spyde.actions", "actions"),
    ("spyde.signals", "signals"),
    ("spyde.workers", "workers"),
    ("spyde.backend", "backend"),
    ("spyde.live", "instrument"),
    ("spyde", "spyde"),
    ("distributed", "dask"),
    ("dask", "dask"),
    ("hyperspy", "hyperspy"),
    ("rsciio", "io"),
    ("pyxem", "pyxem"),
)


def _register_log_areas() -> None:
    try:
        from de_shell.log_stream import register_area_rules
        register_area_rules(_LOG_AREA_RULES, verbose_packages=("spyde",))
    except Exception as exc:  # never block import on a logging-config hiccup
        log.warning("SpyDE log-area registration skipped: %s", exc)


_register_log_areas()


def _register_signal_extensions() -> None:
    """Register SpyDE's HyperSpy signal types in-process.

    The proper mechanism is the `hyperspy.extensions` entry point (declared in
    pyproject.toml + spyde/hyperspy_extension.yaml), which works on a normal
    install. But setuptools *editable* installs shadow the dist-info metadata so
    HyperSpy's entry-point discovery misses it during development. Inserting the
    entries into ALL_EXTENSIONS directly makes `set_signal_type` and isinstance
    gating work regardless of install mode. Idempotent.
    """
    try:
        import yaml
        from hyperspy.extensions import ALL_EXTENSIONS
        with files(__package__).joinpath("hyperspy_extension.yaml").open(
            "r", encoding="utf-8"
        ) as f:
            spec = yaml.safe_load(f) or {}
        for name, info in (spec.get("signals") or {}).items():
            ALL_EXTENSIONS["signals"].setdefault(name, info)
    except Exception as exc:  # never block import on a registration hiccup
        log.warning("SpyDE signal-extension registration skipped: %s", exc)


_register_signal_extensions()


from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spyde")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


# The script-parity layer (`from spyde import api` / `spyde.api.find_vectors`).
# Cheap: spyde.api defers every heavy import into its functions.
from spyde import api  # noqa: E402,F401

__all__ += ["api", "__version__"]
