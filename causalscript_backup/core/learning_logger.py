"""Learning log helpers for MathLang."""

from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LearningLogEntry:
    step_number: int
    phase: str
    expression: str | None
    rendered: str | None
    status: str
    rule_id: str | None
    meta: Dict[str, Any]
    timestamp: str
    scope_id: str = "main"
    parent_scope_id: str | None = None
    depth: int = 0
    context_type: str = "main"
    is_redundant: bool = False

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access for compatibility with tests and formatters."""
        if hasattr(self, key):
            return getattr(self, key)
        data = self.to_dict()
        if key in data:
            return data[key]
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_number,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "expression": self.expression,
            "rendered": self.rendered,
            "status": self.status,
            "rule_id": self.rule_id,
            "meta": self.meta,
            "scope_id": self.scope_id,
            "parent_scope_id": self.parent_scope_id,
            "depth": self.depth,
            "is_redundant": self.is_redundant,
        }


class LearningLogger:
    """Collects and serializes step-by-step execution logs (v2 format)."""

    def __init__(self) -> None:
        self.records: List[LearningLogEntry] = []
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
        scope_id: str = "main",
        parent_scope_id: str | None = None,
        depth: int = 0,
        context_type: str = "main",
        is_redundant: bool = False,
    ) -> None:
        idx = step_index if step_index is not None else self._step_index
        if step_index is None:
            self._step_index += 1
        
        meta_dict = meta or {}
        timestamp = datetime.now(timezone.utc).isoformat()
        
        entry = LearningLogEntry(
            step_number=idx,
            phase=phase,
            expression=expression,
            rendered=rendered,
            status=status,
            rule_id=rule_id,
            meta=meta_dict,
            timestamp=timestamp,
            scope_id=scope_id,
            parent_scope_id=parent_scope_id,
            depth=depth,
            context_type=context_type,
            is_redundant=is_redundant,
        )
        self.records.append(entry)
        
        # --- Memory Integration ---
        # Only record meaningful steps (successful rule applications or explicit failures)
        # Avoid recording overhead for every small internal step unless meaningful
        if status in ("ok", "fuzzy_ok", "error", "contradiction") and expression and phase == "step":
            try:
                # Lazy import to avoid circular dependencies and startup cost
                from .memory.factory import get_vector_store, get_embedder
                from .memory.schema import ExperienceEntry
                
                store = get_vector_store()
                embedder = get_embedder()
                
                # Determine result label
                if status == "ok":
                    result_label = "EXACT"
                elif status == "fuzzy_ok":
                    result_label = "ANALOGOUS"
                elif status == "contradiction":
                    result_label = "CONTRADICT"
                else: 
                    # For error/mistakes, we label as MISTAKE if it was a valid attempt 
                    # (which it is if we are here and have a rule_id usually, or just a failed step)
                    result_label = "MISTAKE"
                
                # Construct vector representation
                # "Problem: {expression}. Action: {rule_id}. Result: {result_label}"
                text_to_embed = f"Problem: {expression}. Action: {rule_id or 'unknown'}. Result: {result_label}"
                vector = embedder.embed_text(text_to_embed)
                
                # Store in DB
                exp_entry = ExperienceEntry(
                    id=f"exp_{timestamp}_{idx}",
                    original_expr=expression,
                    next_expr=str(rendered) if rendered else "",
                    rule_id=rule_id or "",
                    result_label=result_label,
                    category=meta_dict.get("category", "universal"),
                    score=float(meta_dict.get("score", 1.0)),
                    vector=vector,  # type: ignore
                    metadata={
                        "timestamp": timestamp,
                        "phase": phase
                    }
                )
                
                # Using add via wrapper
                # Ideally we check check if the collection 'experience' exists or create it
                # The adapter handles get_or_create
                store.add(
                    collection_name="experience",
                    vectors=[vector],
                    metadatas=[exp_entry.to_metadata()],
                    ids=[exp_entry.id]
                )
            except ImportError:
                # If memory dependencies are missing, skip quietly
                pass
            except Exception as e:
                # Don't crash the main loop for logging errors, but log it
                print(f"Warning: Failed to save experience to memory: {e}")

    def to_list(self) -> List[dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_list()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
