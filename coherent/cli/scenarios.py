"""Helpers for loading CLI scenario definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict


def _read_config(config_path: Path) -> Dict[str, dict[str, Any]]:
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    scenarios = data.get("scenarios")
    return scenarios if isinstance(scenarios, dict) else {}


def scenario_loader_from_file(config_path: Path) -> Callable[[str], dict[str, Any]]:
    """Return a loader that fetches scenario definitions from the given config path."""

    def _loader(name: str) -> dict[str, Any]:
        scenarios = _read_config(config_path)
        scenario = scenarios.get(name)
        if scenario is None:
            available = ", ".join(sorted(scenarios)) or "none"
            raise ValueError(f"Unknown scenario '{name}'. Available: {available}")
        # Return a shallow copy so callers can mutate the data safely.
        return dict(scenario)

    return _loader
