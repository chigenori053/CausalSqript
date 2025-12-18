"""Knowledge rule loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

import logging
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
    concept: str | None = None
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
            "concept": self.concept,
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

    def __init__(self, root_path: Path, engine: SymbolicEngine, custom_rules_path: Optional[Path] = None) -> None:
        self.root_path = root_path
        self.engine = engine
        self.logger = logging.getLogger(__name__)

        self._all_nodes: List[KnowledgeNode] = [] # Temporary list to collect all nodes
        
        # Load core knowledge
        self._load_nodes_from_dir(root_path)

        # Index rules by ID (Initialize before custom loading)
        self.rules_by_id: Dict[str, KnowledgeNode] = {node.id: node for node in self._all_nodes}
        
        # Load custom rules if path is provided
        if custom_rules_path:
            self.load_custom_rules(custom_rules_path)
            # Re-sort after adding custom rules
            self._all_nodes.sort(key=lambda n: n.priority, reverse=True)

        # Legacy support
        self.nodes = self._all_nodes 
        
        # Load maps
        self.maps: List[RuleMap] = self._load_maps(root_path / "maps")
        self.maps.sort(key=lambda m: m.priority, reverse=True)
        
        # Sort rules within each map by priority
        for r_map in self.maps:
            r_map.rules.sort(
                key=lambda rid: self.rules_by_id[rid].priority if rid in self.rules_by_id else -1,
                reverse=True
            )

    def load_custom_rules(self, custom_path: Path):
        """Loads additional rules from a user-specified directory."""
        if not custom_path.exists():
            self.logger.warning(f"Custom rule path not found: {custom_path}")
            return
            
        self.logger.info(f"Loading custom rules from: {custom_path}")
        new_nodes = self._load_nodes_from_dir(custom_path)
        
        # Add to collection and update index
        self._all_nodes.extend(new_nodes)
        self._all_nodes.sort(key=lambda n: n.priority, reverse=True)
        
        # Re-index
        for node in new_nodes:
            self.rules_by_id[node.id] = node
            
        # Update self.nodes (legacy access, if any)
        # Note: self.nodes was renamed to _all_nodes in __init__ but we should keep it for compatibility if needed, 
        # or just make self.nodes a property. 
        # For now, let's just make sure we populate it if the legacy code expects it. 
        # In __init__ we removed `self.nodes = ...` line, so let's alias it.
        self.nodes = self._all_nodes

    def _load_nodes_from_dir(self, dir_path: Path) -> List[KnowledgeNode]:
        """Recursive loader helper."""
        nodes: List[KnowledgeNode] = []
        if not dir_path.exists():
            return nodes
            
        # Exclude maps directory from rule loading
        for path in sorted(dir_path.rglob("*.yaml")):
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
                    # Normalize patterns
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
                        concept=entry.get("concept"),
                        extra={k: v for k, v in entry.items() if k not in {"id", "domain", "category", "pattern_before", "pattern_after", "description", "priority", "condition", "concept"}},
                    )
                    nodes.append(node)
                    self._all_nodes.append(node) # Also add to main list during init load
            except Exception as e:
                self.logger.error(f"Error loading {path}: {e}")
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

    def match(self, before: str, after: str, category: str | None = None) -> Optional[KnowledgeNode]:
        """
        Identifies the rule that transforms 'before' into 'after'.
        Optimized by category filtering.
        Returns the matching rule with the highest priority.
        """
        # Pre-check: Is the input expression numeric?
        is_before_numeric = self.engine.is_numeric(before)
        
        # Determine allowed domains based on category
        allowed_domains = self._resolve_domains(category)
        
        best_match: Optional[KnowledgeNode] = None

        def check_node(node: KnowledgeNode):
            nonlocal best_match
            # Optimization: if we already have a match with strictly higher priority, 
            # we can skip checking this node IF we assume we want the highest priority.
            # But we need to be careful. If priorities are equal, first one wins?
            # Let's just check priority after matching to be safe, or check before to save time.
            if best_match and best_match.priority > node.priority:
                return

            # --- Filtering ---
            if allowed_domains and node.domain not in allowed_domains:
                return
            # -----------------

            match = self._match_node(node, before, after, is_before_numeric)
            if match:
                if best_match is None or match.priority > best_match.priority:
                    best_match = match

        # 1. Search through Maps
        for rule_map in self.maps:
            for rule_id in rule_map.rules:
                node = self.rules_by_id.get(rule_id)
                if node:
                    check_node(node)

        # 2. Fallback (Unmapped rules)
        mapped_ids = set()
        for m in self.maps:
            mapped_ids.update(m.rules)
            
        for node in self.nodes:
            if node.id in mapped_ids:
                continue
            check_node(node)
            
        return best_match

    def _resolve_domains(self, category: str | None) -> Set[str] | None:
        """Returns the set of domains to search based on MathCategory."""
        if not category:
            return None
            
        # Always allow basic arithmetic and algebra
        commons = {"universal", "arithmetic", "algebra"}
        
        if category == "geometry":
            return commons | {"geometry"}
        elif category == "calculus":
            return commons | {"calculus", "analysis"}
        elif category == "linear_algebra":
            return commons | {"linear_algebra", "matrix"}
        elif category == "statistics":
            return commons | {"statistics", "probability"}
        
        return commons

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
            pass # We allow relaxed consistency for now as per previous sessions?
            
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

    def suggest_outcomes(self, before: str, category: str | None = None) -> List[str]:
        """
        Generate potential next steps by applying known rules.
        Uses fast structural matching to skip expensive symbolic validation.
        """
        suggestions = []
        is_before_numeric = self.engine.is_numeric(before)
        allowed_domains = self._resolve_domains(category)
        
        seen_rules = set()

        def check_node(node: KnowledgeNode):
            if node.id in seen_rules:
                return
            seen_rules.add(node.id)
            if allowed_domains and node.domain not in allowed_domains:
                return
            
            # Domain and Operator checks (fast)
            if is_before_numeric and node.domain == "algebra":
                return
            if "/" in node.pattern_before and "/" not in before:
                return
            
            # Structural Match (fast)
            bind_before = self.engine.match_structure(before, node.pattern_before)
            if bind_before is None:
                return
                
            if node.condition:
                if not self._check_condition(node.condition, bind_before):
                    return

            try:
                # Generate 'after' candidate
                str_bindings = {k: str(v) for k, v in bind_before.items()}
                candidate = self.engine.substitute(node.pattern_after, str_bindings)
                suggestions.append(candidate)
            except Exception:
                pass

        # Check Maps and Fallbacks
        for rule_map in self.maps:
            for rule_id in rule_map.rules:
                node = self.rules_by_id.get(rule_id)
                if node:
                    check_node(node)
                    
        for node in self.nodes:
            check_node(node)
            
        # Ensure all suggestions are strings to avoid unhashable type errors (e.g. lists for matrices)
        str_suggestions = [str(s) for s in suggestions]
        return list(dict.fromkeys(str_suggestions))



    def match_rules(self, before: str, category: str | None = None) -> List[Tuple[KnowledgeNode, str]]:
        """
        Finds all rules that match the 'before' expression and returns the rule + resulting expression.
        """
        matches: List[Tuple[KnowledgeNode, str]] = []
        is_before_numeric = self.engine.is_numeric(before)
        allowed_domains = self._resolve_domains(category)
        
        seen_rules = set()

        def check_node(node: KnowledgeNode):
            if node.id in seen_rules:
                return
            seen_rules.add(node.id)

            if allowed_domains and node.domain not in allowed_domains:
                return
            
            # Domain and Operator checks (fast)
            if is_before_numeric and node.domain == "algebra":
                return
            if "/" in node.pattern_before and "/" not in before:
                return
            
            # Structural Match (fast)
            bind_before = self.engine.match_structure(before, node.pattern_before)
            if bind_before is None:
                return
                
            if node.condition:
                if not self._check_condition(node.condition, bind_before):
                    return

            try:
                # Generate 'after' candidate
                str_bindings = {k: str(v) for k, v in bind_before.items()}
                candidate = self.engine.substitute(node.pattern_after, str_bindings)
                matches.append((node, candidate))
            except Exception:
                pass

        # Check Maps and Fallbacks
        for rule_map in self.maps:
            for rule_id in rule_map.rules:
                node = self.rules_by_id.get(rule_id)
                if node:
                    check_node(node)
                    
        for node in self.nodes:
            check_node(node)
            
        return matches

    def apply_rule(self, expr: str, rule_id: str) -> Optional[str]:
        """
        Applies a specific rule to the expression.
        Returns the transformed expression or None if the rule cannot be applied.
        """
        node = self.rules_by_id.get(rule_id)
        if not node:
            return None
            
        try:
            # 1. Match Structure
            bind_before = self.engine.match_structure(expr, node.pattern_before)
            if bind_before is None:
                return None
                
            # 2. Check Condition
            if node.condition:
                if not self._check_condition(node.condition, bind_before):
                    return None
            
            # 3. Substitute
            str_bindings = {k: str(v) for k, v in bind_before.items()}
            candidate = self.engine.substitute(node.pattern_after, str_bindings)
            return candidate
        except Exception:
            return None

    def semantic_search(self, query: str, category: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search for rules relevant to the query.
        """
        try:
            from .memory import MemoryRetriever
            retriever = MemoryRetriever()
            return retriever.search_rules(query, category=category, top_k=top_k)
        except ImportError:
            print("Warning: Semantic search unavailable (dependencies missing).")
            return []
        except Exception as e:
            print(f"Warning: Semantic search failed: {e}")
            return []

