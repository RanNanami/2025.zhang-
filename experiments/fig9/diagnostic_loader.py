"""Lazy imports for optional Fig.9 diagnostic families.

The strict runner imports this module unconditionally, but each diagnostic
module is imported only when its explicit loader is called. Loader functions
return the original module so call sites keep using the historical functions
and constants without a compatibility copy.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType


@lru_cache(maxsize=None)
def load_readout_dynamics() -> ModuleType:
    """Load the read-only column-level readout trace implementation."""

    return importlib.import_module(
        "experiments.diagnostics.fig9_readout_dynamics"
    )
