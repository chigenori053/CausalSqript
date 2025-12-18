"""Log inspection helpers for pro workflows."""

from __future__ import annotations

from typing import Iterable, Mapping, List


def filter_by_phase(records: Iterable[Mapping[str, object]], phase: str) -> List[Mapping[str, object]]:
    phase_lower = phase.lower()
    return [record for record in records if str(record.get("phase", "")).lower() == phase_lower]
