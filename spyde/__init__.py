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


__version__ = "0.0.1"
