"""Learning log helpers for MathLang."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class LearningLogger:
    """Collects and serializes step-by-step execution logs (v2 format)."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._step_index = 0

    def record(
        self,
        *,
        phase: str,
        expression: str | None,
        rendered: str | None,
        status: str,
        rule_id: str | None = None,
        meta: Optional[Dict[str, Any]] = None,
        step_index: int | None = None,
    ) -> None:
        idx = step_index if step_index is not None else self._step_index
        if step_index is None:
            self._step_index += 1
        self.records.append(
            {
                "step_index": idx,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "expression": expression,
                "rendered": rendered,
                "status": status,
                "rule_id": rule_id,
                "meta": meta or {},
            }
        )

    def to_list(self) -> List[dict[str, Any]]:
        return list(self.records)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
