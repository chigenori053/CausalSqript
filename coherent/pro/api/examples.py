"""Utility helpers to locate pro sample programs."""

from __future__ import annotations

from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"


def list_examples() -> list[str]:
    return [path.name for path in sorted(EXAMPLES_ROOT.glob("*.mlang"))]


def load_example_source(name: str) -> str:
    file_path = EXAMPLES_ROOT / name
    if not file_path.exists():
        raise FileNotFoundError(f"Example '{name}' not found in {EXAMPLES_ROOT}.")
    return file_path.read_text(encoding="utf-8")
