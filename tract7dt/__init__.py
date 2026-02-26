"""tract7dt package."""

from .config import load_config
from .pipeline import run_pipeline

try:
    from importlib.metadata import version as _meta_version
    __version__: str = _meta_version("tract7dt")
except Exception:
    __version__ = "0.0.0.dev"

__all__ = ["__version__", "load_config", "run_pipeline"]
