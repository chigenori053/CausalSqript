"""Minimal renderer used by edu demos/tests."""

from __future__ import annotations

from typing import Iterable, Mapping


def render_step_log(records: Iterable[Mapping[str, object]]) -> str:
    """Build a textual summary for notebooks or CLI previews."""
    lines: list[str] = []
    for record in records:
        phase = record.get("phase")
        rendered = record.get("rendered") or record.get("expression") or ""
        status = record.get("status", "")
        lines.append(f"[{phase}] {rendered} ({status})")
    return "\n".join(lines)
