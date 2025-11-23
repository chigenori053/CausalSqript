"""Project-specific site customizations for optional dependencies."""

from __future__ import annotations

import sys
import types

try:  # pragma: no cover - only executed when PyYAML is installed.
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed in test envs without PyYAML.
    from causalscript_yaml import dump, safe_load

    fallback = types.ModuleType("yaml")
    fallback.safe_load = safe_load
    fallback.dump = dump
    fallback.FullLoader = object  # minimal compatibility shim
    fallback.__dict__["_CAUSALSCRIPT_SIMPLE"] = True
    sys.modules["yaml"] = fallback
