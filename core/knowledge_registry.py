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


@dataclass
class RuleMap:
    id: str
    name: str
    description: str
    priority: int
    rules: List[str]


class KnowledgeRegistry:
    """Loads YAML rule files and performs basic matching."""

    def __init__(self, base_path: Path, engine: SymbolicEngine) -> None:
        self.base_path = base_path
        self.engine = engine
        self.nodes: List[KnowledgeNode] = self._load_all(base_path)
        # Sort by priority descending
        self.nodes.sort(key=lambda n: n.priority, reverse=True)
        
        # Index rules by ID
        self.rules_by_id: Dict[str, KnowledgeNode] = {node.id: node for node in self.nodes}
        
        # Load maps
        self.maps: List[RuleMap] = self._load_maps(base_path / "maps")
        self.maps.sort(key=lambda m: m.priority, reverse=True)
        
        # Sort rules within each map by priority
        for r_map in self.maps:
            r_map.rules.sort(
                key=lambda rid: self.rules_by_id[rid].priority if rid in self.rules_by_id else -1,
                reverse=True
            )

    def _load_all(self, base_path: Path) -> List[KnowledgeNode]:
        nodes: List[KnowledgeNode] = []
        # Exclude maps directory from rule loading
        for path in sorted(base_path.rglob("*.yaml")):
            if "maps" in path.parts:
                continue
                
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

    def _load_maps(self, maps_path: Path) -> List[RuleMap]:
        maps: List[RuleMap] = []
        if not maps_path.exists():
            return maps
            
        for path in sorted(maps_path.glob("*.yaml")):
            try:
                text = path.read_text(encoding="utf-8")
                if yaml is not None:
                    data = yaml.safe_load(text)
                else:
                    # Simple parser might fail on lists, but let's try or assume yaml is present
                    # For now assuming yaml is present as it is a dependency
                    data = yaml.safe_load(text)
                
                if not data:
                    continue
                    
                rule_map = RuleMap(
                    id=data.get("id", path.stem),
                    name=data.get("name", ""),
                    description=data.get("description", ""),
                    priority=int(data.get("priority", 50)),
                    rules=data.get("rules", [])
                )
                maps.append(rule_map)
            except Exception as e:
                print(f"Error loading map {path}: {e}")
                continue
        return maps

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

    def match(self, before: str, after: str, context_domains: List[str] | None = None) -> Optional[KnowledgeNode]:
        """
        Identifies the rule that transforms 'before' into 'after'.
        Uses a RuleMap-based priority search, filtered/prioritized by context_domains.
        """
        # Pre-check: Is the input expression numeric?
        is_before_numeric = self.engine.is_numeric(before)
        
        # Determine maps to search
        maps_to_search = []
        
        if context_domains:
            # 1. Prioritize maps matching the context domains
            # We search maps whose ID (or name?) matches the domain.
            # Map IDs: arithmetic, algebra_basic, algebra_advanced, calculus, etc.
            # Context Domains: arithmetic, algebra, calculus
            
            # Strategy:
            # - Iterate context_domains in order.
            # - For each domain, find matching maps.
            # - Add them to search list.
            # - Then add remaining maps (as fallback)? Or strict filtering?
            # User request implies "Mapped Symbolic Engine -> Mapped Calculation Rules".
            # So we should probably prioritize context maps.
            
            seen_map_ids = set()
            
            for domain in context_domains:
                # Find maps that belong to this domain
                # We can check if map.id starts with domain or equals it.
                # e.g. domain="algebra" matches "algebra_basic", "algebra_advanced"
                
                for m in self.maps:
                    if m.id == domain or m.id.startswith(domain + "_"):
                        if m.id not in seen_map_ids:
                            maps_to_search.append(m)
                            seen_map_ids.add(m.id)
            
            # Add remaining maps as fallback (optional, but good for safety)
            # If we want strict mode, we might skip this.
            # But "arithmetic" is often needed for "algebra".
            # Our classifier adds "arithmetic" to "algebra" context, so it should be fine.
            # Let's add remaining maps at the end just in case.
            for m in self.maps:
                if m.id not in seen_map_ids:
                    maps_to_search.append(m)
        else:
            # Default: Search all maps in priority order
            maps_to_search = self.maps

        # 1. Search through Maps
        for rule_map in maps_to_search:
            for rule_id in rule_map.rules:
                node = self.rules_by_id.get(rule_id)
                if not node:
                    continue
                
                match = self._match_node(node, before, after, is_before_numeric)
                if match:
                    return match

        # 2. Fallback: Search remaining rules (if any not in maps)
        # For now, we can iterate all nodes again, but skip those we already checked?
        # Or just iterate all nodes as a fallback with lower priority?
        # To be safe and simple, let's iterate all nodes that were NOT in any map.
        mapped_ids = set()
        for m in self.maps:
            mapped_ids.update(m.rules)
            
        for node in self.nodes:
            if node.id in mapped_ids:
                continue
            match = self._match_node(node, before, after, is_before_numeric)
            if match:
                return match
                
        return None

    def _match_node(self, node: KnowledgeNode, before: str, after: str, is_before_numeric: bool) -> Optional[KnowledgeNode]:
        # Domain Strictness:
        # Algebraic rules should NOT match pure numeric expressions.
        if is_before_numeric and node.domain == "algebra":
            return None

        # Fraction-specific patterns should not match when the input has no '/'.
        if "/" in node.pattern_before and "/" not in before:
            return None

        # Strict Rule Matching (AST Node Type Check)
        expr_op = self.engine.get_top_operator(before)
        pattern_op = self.engine.get_top_operator(node.pattern_before)
        
        if expr_op and pattern_op:
            # Strict operators that must match exactly if present in the pattern
            strict_ops = {"Add", "Sub", "Mul", "Mult", "Div", "Pow", "Integral", "Derivative", "Limit", "Subs"}
            
            # If the pattern expects a specific structure, the expression must match it.
            # This prevents "Integral(...)" from matching "a + b" (Add).
            if pattern_op in strict_ops:
                if expr_op != pattern_op:
                    return None
            
            # Also, if the expression is a strict op, but the pattern expects a different strict op
            # (e.g. expr is Add, pattern is Mul), reject.
            if expr_op in strict_ops and pattern_op in strict_ops:
                if expr_op != pattern_op:
                    return None

        expr_for_match = before
        # Preserve subtraction structure when pattern expects '-'
        if "-" in node.pattern_before and "+ -" in before:
            expr_for_match = before.replace("+ -", "- ")

        # Phase 1: Structural Filtering
        bind_before = self.engine.match_structure(expr_for_match, node.pattern_before)
        
        if bind_before is None:
            return None
        
        # Phase 1.5: Condition Check
        if node.condition:
            if not self._check_condition(node.condition, bind_before):
                return None

        # Match 'after' against the rule's output pattern
        bind_after_raw = self.engine.match_structure(after, node.pattern_after)
        bind_after = bind_after_raw or {}
            
        # Phase 2: Binding Consistency
        if not self._is_consistent(bind_before, bind_after):
            pass
            
        # Phase 3: Equivalence Verification
        try:
            # Special handling for calculation rules
            if node.category == "calculation":
                if self.engine.is_equiv(before, after):
                    return node
                return None

            str_bindings = {k: str(v) for k, v in bind_before.items()}
            str_bindings.update({k: str(v) for k, v in bind_after.items()})
            expected_after = self.engine.substitute(node.pattern_after, str_bindings)
            
            if self.engine.is_equiv(after, expected_after):
                return node
        except Exception:
            pass
            
        return None

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
