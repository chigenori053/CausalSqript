"""Layout helpers for Edu notebooks."""

from __future__ import annotations

def two_column_layout(left: str, right: str) -> str:
    """Return a simple text layout placeholder."""
    divider = "-" * 40
    return f"{left}\n{divider}\n{right}"
