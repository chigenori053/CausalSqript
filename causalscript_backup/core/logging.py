"""Learning log data structures and helpers for MathLang."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Sequence


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LearningLogEntry:
    step_number: int
    phase: str  # problem, step, end, explain, show, assignment
    label: str | None
    expression: str
    rendered: str
    rule_id: str | None
    status: str  # e.g. "verified", "info"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)
    scope_id: str = "root"
    parent_scope_id: str | None = None
    depth: int = 0
    is_redundant: bool = False


class LearningLogger:
    """Collects learning log entries and can persist them as JSON."""

    def __init__(self) -> None:
        self._entries: List[LearningLogEntry] = []

    def record(
        self,
        *,
        step_number: int,
        phase: str,
        label: str | None,
        expression: str,
        rendered: str,
        rule_id: str | None,
        status: str,
        metadata: dict[str, Any] | None = None,
        scope_id: str = "root",
        parent_scope_id: str | None = None,
        depth: int = 0,
        is_redundant: bool = False,
    ) -> None:
        entry = LearningLogEntry(
            step_number=step_number,
            phase=phase,
            label=label,
            expression=expression,
            rendered=rendered,
            rule_id=rule_id,
            status=status,
            metadata=metadata or {},
            scope_id=scope_id,
            parent_scope_id=parent_scope_id,
            depth=depth,
            is_redundant=is_redundant,
        )
        self._entries.append(entry)

    def entries(self) -> Sequence[LearningLogEntry]:
        return tuple(self._entries)

    def to_dict(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self._entries]

    def write(self, path: Path) -> None:
        import json

        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
