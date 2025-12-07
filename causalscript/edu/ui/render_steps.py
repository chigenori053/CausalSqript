"""Minimal renderer used by edu demos/tests."""

from __future__ import annotations

from typing import Iterable, Mapping


def render_step_log(records: Iterable[Mapping[str, object]]) -> str:
    """Build a textual summary for notebooks or CLI previews."""
    lines: list[str] = []
    for record in records:
        # Skip redundant or not-available entries unless they represent an error.
        if record.get("is_redundant"):
            continue
        rendered_text = record.get("rendered") or record.get("expression") or ""
        status = record.get("status", "")
        if isinstance(rendered_text, str) and rendered_text.strip().lower() == "not available" and status not in {"error", "mistake"}:
            continue

        phase = record.get("phase")
        depth = int(record.get("depth", 0) or 0)
        indent = "  " * depth
        scope = record.get("scope_id")
        scope_suffix = f" @{scope}" if scope else ""
        lines.append(f"{indent}[{phase}] {rendered_text} ({status}){scope_suffix}")
    return "\n".join(lines)
