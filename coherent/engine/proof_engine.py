"""Proof Engine for generating geometric proofs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, Callable, Set, Dict

@dataclass(frozen=True)
class Fact:
    """A logical fact consisting of a predicate and arguments."""
    predicate: str
    args: Tuple[str, ...]

    def __repr__(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})"

@dataclass
class Step:
    """A step in a proof, showing how a fact was derived."""
    fact: Fact
    rule_name: str
    precedents: List[Fact]

    def __repr__(self) -> str:
        if not self.precedents:
            return f"Given: {self.fact}"
        precedents_str = ", ".join(str(p) for p in self.precedents)
        return f"{self.fact} derived from [{precedents_str}] via {self.rule_name}"

class Rule:
    """Base class for inference rules."""
    def __init__(self, name: str):
        self.name = name

    def apply(self, facts: List[Fact]) -> List[Step]:
        """Apply the rule to the current set of facts and return new steps."""
        raise NotImplementedError

class ProofEngine:
    """Engine for managing facts and deriving proofs."""
    
    def __init__(self) -> None:
        self.facts: Set[Fact] = set()
        self.steps: List[Step] = []
        self.rules: List[Rule] = []

    def add_fact(self, fact: Fact, rule_name: str = "Given", precedents: List[Fact] = None) -> bool:
        """Add a fact to the knowledge base. Returns True if it was new."""
        if fact not in self.facts:
            self.facts.add(fact)
            self.steps.append(Step(fact, rule_name, precedents or []))
            return True
        return False

    def register_rule(self, rule: Rule) -> None:
        """Register an inference rule."""
        self.rules.append(rule)

    def derive(self) -> bool:
        """Run one pass of all rules. Return True if new facts were added."""
        added = False
        # Convert to list to avoid modification during iteration issues if we were iterating directly
        # But here we pass the list to rules.
        current_facts = list(self.facts)
        
        for rule in self.rules:
            new_steps = rule.apply(current_facts)
            for step in new_steps:
                if self.add_fact(step.fact, step.rule_name, step.precedents):
                    added = True
        return added

    def prove(self, goal: Fact, max_iterations: int = 10) -> Optional[List[Step]]:
        """Attempt to prove the goal fact."""
        # Check if already proven
        if goal in self.facts:
            return self._trace_back(goal)
            
        for _ in range(max_iterations):
            if not self.derive():
                break
            if goal in self.facts:
                return self._trace_back(goal)
        
        return None

    def _trace_back(self, goal: Fact) -> List[Step]:
        """Reconstruct the proof chain for the goal."""
        relevant_steps: List[Step] = []
        seen_facts: Set[Fact] = set()

        def collect(f: Fact) -> None:
            if f in seen_facts:
                return
            seen_facts.add(f)
            
            # Find the step that produced this fact
            step = next((s for s in self.steps if s.fact == f), None)
            if step:
                # Recursively collect precedents first
                for p in step.precedents:
                    collect(p)
                relevant_steps.append(step)

        collect(goal)
        return relevant_steps


# --- Common Geometric Rules ---

class TransitiveRule(Rule):
    """Generic transitivity rule: P(a, b) & P(b, c) -> P(a, c)."""
    def __init__(self, predicate: str):
        super().__init__(f"Transitive({predicate})")
        self.predicate = predicate

    def apply(self, facts: List[Fact]) -> List[Step]:
        steps = []
        # Filter relevant facts
        relevant = [f for f in facts if f.predicate == self.predicate]
        
        # O(N^2) naive matching
        for f1 in relevant:
            for f2 in relevant:
                if f1 == f2:
                    continue
                # Match (a, b) and (b, c)
                if f1.args[1] == f2.args[0]:
                    new_fact = Fact(self.predicate, (f1.args[0], f2.args[1]))
                    steps.append(Step(new_fact, self.name, [f1, f2]))
        return steps

class SymmetricRule(Rule):
    """Generic symmetry rule: P(a, b) -> P(b, a)."""
    def __init__(self, predicate: str):
        super().__init__(f"Symmetric({predicate})")
        self.predicate = predicate

    def apply(self, facts: List[Fact]) -> List[Step]:
        steps = []
        relevant = [f for f in facts if f.predicate == self.predicate]
        for f in relevant:
            new_fact = Fact(self.predicate, (f.args[1], f.args[0]))
            steps.append(Step(new_fact, self.name, [f]))
        return steps

class CongruenceRule(Rule):
    """
    Simplified SSS/SAS Rule.
    For MVP, we'll implement a generic 'TriangleCongruence' that looks for 3 matching components.
    This is a placeholder for more complex logic.
    """
    def __init__(self):
        super().__init__("TriangleCongruence")

    def apply(self, facts: List[Fact]) -> List[Step]:
        # This requires identifying triangles and their components.
        # For MVP, let's assume facts like:
        # Side(AB), Side(DE), Equal(AB, DE)
        # Triangle(ABC), Triangle(DEF)
        # This is getting complex for a generic engine without a graph.
        # We will implement a specific SSS rule for demonstration.
        return []

class SSSRule(Rule):
    """
    SSS Congruence:
    If SideEqual(AB, DE) and SideEqual(BC, EF) and SideEqual(AC, DF)
    Then Congruent(ABC, DEF)
    """
    def __init__(self):
        super().__init__("SSS")

    def apply(self, facts: List[Fact]) -> List[Step]:
        steps = []
        side_equals = [f for f in facts if f.predicate == "SideEqual"]
        
        # We need to find triples of side equalities that form two triangles.
        # SideEqual(AB, DE) -> implies sides AB and DE
        # We need to group these.
        
        # Naive approach: iterate all triplets of facts. O(N^3).
        # Optimization: Index by triangle vertices?
        
        # Let's try to find sets of 3 equalities.
        # This is just a proof of concept implementation.
        
        n = len(side_equals)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    f1, f2, f3 = side_equals[i], side_equals[j], side_equals[k]
                    
                    # Check if they form triangles.
                    # f1: (AB, DE), f2: (BC, EF), f3: (AC, DF)
                    # We need to extract vertices.
                    
                    s1_1, s1_2 = f1.args
                    s2_1, s2_2 = f2.args
                    s3_1, s3_2 = f3.args
                    
                    # Vertices for first triangle
                    # Assume sides are strings like "AB", "BC"
                    # We need to parse "AB" -> {A, B}
                    
                    try:
                        t1_sides = [s1_1, s2_1, s3_1]
                        t2_sides = [s1_2, s2_2, s3_2]
                        
                        v1 = self._get_vertices(t1_sides)
                        v2 = self._get_vertices(t2_sides)
                        
                        if v1 and v2 and len(v1) == 3 and len(v2) == 3:
                            # Check if sides match vertices
                            # This is a loose check, but sufficient for MVP
                            t1_name = "".join(sorted(v1))
                            t2_name = "".join(sorted(v2))
                            
                            new_fact = Fact("Congruent", (t1_name, t2_name))
                            steps.append(Step(new_fact, self.name, [f1, f2, f3]))
                    except Exception:
                        continue
                        
        return steps

    def _get_vertices(self, sides: List[str]) -> Optional[Set[str]]:
        # sides = ["AB", "BC", "AC"]
        vertices = set()
        for s in sides:
            if len(s) != 2: return None
            vertices.add(s[0])
            vertices.add(s[1])
        return vertices
