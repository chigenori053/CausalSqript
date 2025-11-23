"""Knowledge rule loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .errors import InvalidExprError
from .symbolic_engine import SymbolicEngine


@dataclass
class KnowledgeNode:
    id: str
    domain: str
    category: str
    pattern_before: str
    pattern_after: str
    description: str
    extra: Dict[str, Any] | None = None

    def to_metadata(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "domain": self.domain,
            "category": self.category,
            "pattern_before": self.pattern_before,
            "pattern_after": self.pattern_after,
            "description": self.description,
        }
        if self.extra:
            data.update(self.extra)
        return data


class KnowledgeRegistry:
    """Loads YAML rule files and performs basic matching."""

    def __init__(self, base_path: Path, engine: SymbolicEngine) -> None:
        self.base_path = base_path
        self.engine = engine
        self.nodes: List[KnowledgeNode] = self._load_all(base_path)

    def _load_all(self, base_path: Path) -> List[KnowledgeNode]:
        nodes: List[KnowledgeNode] = []
        for path in sorted(base_path.rglob("*.yaml")):
            text = path.read_text(encoding="utf-8")
            if yaml is not None:
                data = yaml.safe_load(text) or []
            else:
                data = self._parse_simple_yaml(text)
            if not isinstance(data, list):
                continue
            for entry in data:
                node = KnowledgeNode(
                    id=entry["id"],
                    domain=entry.get("domain", ""),
                    category=entry.get("category", ""),
                    pattern_before=entry.get("pattern_before", ""),
                    pattern_after=entry.get("pattern_after", ""),
                    description=entry.get("description", ""),
                    extra={k: v for k, v in entry.items() if k not in {"id", "domain", "category", "pattern_before", "pattern_after", "description"}},
                )
                nodes.append(node)
        return nodes

    def _parse_simple_yaml(self, text: str) -> List[dict]:
        """Fallback parser for the limited YAML subset used by rule files."""
        nodes: List[dict] = []
        current: dict[str, str] | None = None
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- "):
                if current:
                    nodes.append(current)
                current = {}
                stripped = stripped[2:]
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"')
            if current is None:
                current = {}
            current[key] = value
        if current:
            nodes.append(current)
        return nodes

    def match(self, before: str, after: str) -> Optional[KnowledgeNode]:
        try:
            normalized_before = self.engine.simplify(before)
            normalized_after = self.engine.simplify(after)
        except InvalidExprError:
            return None

        for node in self.nodes:
            try:
                if (
                    self.engine.simplify(node.pattern_before) == normalized_before
                    and self.engine.simplify(node.pattern_after) == normalized_after
                ):
                    return node
            except InvalidExprError:
                continue
        return None
