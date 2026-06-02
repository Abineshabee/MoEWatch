# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  __init__.py — Public API surface for the moewatch package
#
#  This is the only file researchers should ever need to import from.
#  All internal implementation details live in sub-modules; nothing that
#  is not listed in __all__ is considered a stable public API.
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#  Docs   : https://github.com/Abineshabee/moewatch
#
# =============================================================================

from __future__ import annotations

import logging
import os
import sys

from .config import (           # always fast — pure stdlib dataclass
    WatchConfig,
    OutputMode,
    AlertLevel,
)

from ._audit import audit       # fast import; torch/sub-module loads on call

from ._watcher import (
    MoEWatch,
    MoEWatchCallback,
    Alert,
)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__  = "0.1.0"
__author__   = "Abinesh"
__email__    = ""                            # fill in before PyPI publish
__license__  = "Apache 2.0"
__homepage__ = "https://github.com/Abineshabee/moewatch"

# ---------------------------------------------------------------------------
# Python version guard
# ---------------------------------------------------------------------------

if sys.version_info < (3, 8):
    raise RuntimeError(
        f"moewatch requires Python 3.8 or later. "
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}."
    )

# ---------------------------------------------------------------------------
# Logging setup
#
# moewatch uses the standard library ``logging`` module. By default, a
# NullHandler is attached so that the library emits no output unless the
# application configures logging. This is the recommended pattern for
# library code (see PEP 3122 / Python logging HOWTO).
#
# To see moewatch debug output in your application:
#
#   import logging
#   logging.getLogger("moewatch").setLevel(logging.DEBUG)
#   logging.basicConfig()
#
# ---------------------------------------------------------------------------

logging.getLogger(__name__).addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# NO_COLOR support
#
# moewatch respects the NO_COLOR standard (https://no-color.org/).
# When the NO_COLOR environment variable is set to any value, ANSI colour
# codes are suppressed globally. This is also overridable per-instance via
# WatchConfig(no_color=True).
#
# ---------------------------------------------------------------------------

_NO_COLOR_ENV: bool = "NO_COLOR" in os.environ

# ---------------------------------------------------------------------------
# Public API imports
#
# Only the symbols listed in __all__ are part of the stable public API.
# Import from sub-modules lazily where possible to keep ``import moewatch``
# fast — researchers often import at the top of long training scripts.
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# __all__ — explicit public surface
# ---------------------------------------------------------------------------

__all__ = [
    # -- Primary API --------------------------------------------------------
    "audit",            # offline diagnostic: audit(model, dataloader, steps)
    "MoEWatch",         # live training monitor: MoEWatch(model, config=...)
    # -- Configuration ------------------------------------------------------
    "WatchConfig",      # all thresholds and settings
    "OutputMode",       # CONSOLE | JSON | SILENT
    "AlertLevel",       # INFO | WARN | ERROR
    # -- HuggingFace integration --------------------------------------------
    "MoEWatchCallback", # TrainerCallback injected by MoEWatch.attach(trainer)
    # -- Alert data ---------------------------------------------------------
    "Alert",            # single diagnostic event; returned by MoEWatch.step()
    # -- Metadata -----------------------------------------------------------
    "__version__",
]

# ---------------------------------------------------------------------------
# Convenience: print version when run as a script
#   python -m moewatch  →  "moewatch 0.1.0"
# ---------------------------------------------------------------------------

def _cli_version() -> None:
    """Print the installed moewatch version and exit."""
    print(f"moewatch {__version__}")


# ---------------------------------------------------------------------------
# Torch availability check
#
# moewatch requires PyTorch. Provide a clear error message if it is missing
# rather than letting the user hit an obscure ImportError deep in _audit.py.
#
# ---------------------------------------------------------------------------

try:
    import torch as _torch  # noqa: F401  (existence check only)
except ImportError as _exc:
    raise ImportError(
        "moewatch requires PyTorch (torch). "
        "Install it from https://pytorch.org before using moewatch.\n"
        f"Original error: {_exc}"
    ) from _exc

# ---------------------------------------------------------------------------
# Transformers availability check (soft — warn, do not raise)
# ---------------------------------------------------------------------------

try:
    import transformers as _transformers  # noqa: F401
except ImportError:
    logging.getLogger(__name__).warning(
        "moewatch: 'transformers' package not found. "
        "HuggingFace model auto-detection and MoEWatch.attach(trainer) "
        "require 'transformers'. Install with: pip install transformers\n"
        "You can still use moewatch with custom PyTorch MoE models by "
        "specifying router_modules manually via WatchConfig."
    )
