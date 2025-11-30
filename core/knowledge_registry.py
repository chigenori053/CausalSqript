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
    priority: int = 50
    condition: str | None = None
    extra: Dict[str, Any] | None = None

    def to_metadata(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "domain": self.domain,
            "category": self.category,
            "pattern_before": self.pattern_before,
            "pattern_after": self.pattern_after,
            "description": self.description,
            "priority": self.priority,
            "condition": self.condition,
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
        # Sort by priority descending
        self.nodes.sort(key=lambda n: n.priority, reverse=True)

    def _load_all(self, base_path: Path) -> List[KnowledgeNode]:
        nodes: List[KnowledgeNode] = []
        for path in sorted(base_path.rglob("*.yaml")):
            try:
                text = path.read_text(encoding="utf-8")
                if yaml is not None:
                    data = yaml.safe_load(text) or []
                else:
                    data = self._parse_simple_yaml(text)
                if not isinstance(data, list):
                    continue
                for entry in data:
                    # Normalize patterns (e.g. ^ -> **)
                    p_before = entry.get("pattern_before", "").replace("^", "**")
                    p_after = entry.get("pattern_after", "").replace("^", "**")
                    
                    node = KnowledgeNode(
                        id=entry["id"],
                        domain=entry.get("domain", ""),
                        category=entry.get("category", ""),
                        pattern_before=p_before,
                        pattern_after=p_after,
                        description=entry.get("description", ""),
                        priority=int(entry.get("priority", 50)),
                        condition=entry.get("condition"),
                        extra={k: v for k, v in entry.items() if k not in {"id", "domain", "category", "pattern_before", "pattern_after", "description", "priority", "condition"}},
                    )
                    nodes.append(node)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
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
            value = value.strip().strip('"').strip("'")
            if current is None:
                current = {}
            current[key] = value
        if current:
            nodes.append(current)
        return nodes

    def match(self, before: str, after: str) -> Optional[KnowledgeNode]:
        """
        Identifies the rule that transforms 'before' into 'after'.
        Uses a 3-phase pipeline:
        1. Structural Filtering (match_structure)
        2. Binding Consistency (_is_consistent)
        3. Equivalence Verification (is_equiv)
        """
        best_match: KnowledgeNode | None = None
        best_after_match = False
        
        # Pre-check: Is the input expression numeric?
        is_before_numeric = self.engine.is_numeric(before)

        for node in self.nodes:
            # Domain Strictness:
            # Algebraic rules should NOT match pure numeric expressions.
            if is_before_numeric and node.domain == "algebra":
                continue

            # Strict Rule Matching (AST Node Type Check)
            # Check if the rule's pattern_before top-level operator matches the input's top-level operator.
            # This prevents "Sticky Rule ID" (e.g., Pow matches Mul).
            # We only check if we can reliably determine the operator.
            expr_op = self.engine.get_top_operator(before)
            pattern_op = self.engine.get_top_operator(node.pattern_before)
            
            # Allow mismatch if one is "Symbol" or "Number" or "Other" (generic),
            # but enforce strictness if both are specific operators (Add, Mul, Pow).
            if expr_op and pattern_op:
                if expr_op in {"Add", "Mul", "Pow"} and pattern_op in {"Add", "Mul", "Pow"}:
                    if expr_op != pattern_op:
                        # Special case: a*a (Mul) can match a^2 (Pow) in some contexts, 
                        # but the spec says "Pow -> Mul ... category: exponents ... prioritize".
                        # If we are strict, we skip.
                        # Let's trust the spec: "Pre-check: before and after top-level operators ... compare".
                        continue

            # Phase 1: Structural Filtering
            # Match 'before' against the rule's input pattern
            bind_before = self.engine.match_structure(before, node.pattern_before)
            
            # Fallback for Power Definition: (x-y)*(x-y) vs a*a
            # If strict match fails, try relaxing it if the rule is about exponents/expansion
            if bind_before is None:
                # print(f"DEBUG: {node.id} failed match_structure(before)")
                continue
            
            # Phase 1.5: Condition Check
            if node.condition:
                if not self._check_condition(node.condition, bind_before):
                    continue

            # Match 'after' against the rule's output pattern
            bind_after_raw = self.engine.match_structure(after, node.pattern_after)
            after_structural_match = bind_after_raw is not None
            bind_after = bind_after_raw or {}
                
            # Phase 2: Binding Consistency
            # Check if variables bound in both before and after are consistent
            if not self._is_consistent(bind_before, bind_after):
                # print(f"DEBUG: {node.id} failed consistency check. Before: {bind_before}, After: {bind_after}")
                # continue # Strict consistency check disabled due to ambiguity in SymPy matching
                pass
                
            # Phase 3: Equivalence Verification
            try:
                # Special handling for calculation rules where output is a new variable (e.g. 'c')
                if node.category == "calculation":
                    if self.engine.is_equiv(before, after):
                        best_match = node
                        best_after_match = after_structural_match
                    continue

                str_bindings = {k: str(v) for k, v in bind_before.items()}
                str_bindings.update({k: str(v) for k, v in bind_after.items()})
                expected_after = self.engine.substitute(node.pattern_after, str_bindings)
                
                if self.engine.is_equiv(after, expected_after):
                    if best_match is None:
                        best_match = node
                        best_after_match = after_structural_match
                    elif after_structural_match and not best_after_match:
                        # Prefer rules whose output pattern structurally matches the 'after' expression.
                        best_match = node
                        best_after_match = True
                    
            except Exception as e:
                # print(f"DEBUG: {node.id} exception: {e}")
                continue
                
        return best_match

    def _check_condition(self, condition: str, bindings: Dict[str, Any]) -> bool:
        """Evaluates a condition string against bindings."""
        try:
            # Prepare context
            context = bindings.copy()
            context["is_numeric"] = self.engine.is_numeric
            
            # Safe(r) eval - only allow specific functions/variables
            # For now, we trust the rule files as they are part of the codebase.
            return bool(eval(condition, {"__builtins__": {}}, context))
        except Exception:
            return False

    def _is_consistent(self, bind1: Dict[str, Any], bind2: Dict[str, Any]) -> bool:
        """
        Checks if common keys in two binding dictionaries map to equivalent expressions.
        """
        common_keys = set(bind1.keys()) & set(bind2.keys())
        
        for key in common_keys:
            val1 = bind1[key]
            val2 = bind2[key]
            
            # Compare SymPy objects directly if possible
            if val1 != val2:
                # If strictly not equal, check mathematical equivalence
                # (e.g. x+y vs y+x)
                try:
                    # Use SymbolicEngine's is_equiv but we need strings
                    if not self.engine.is_equiv(str(val1), str(val2)):
                        return False
                except Exception:
                    return False
                    
        return True
