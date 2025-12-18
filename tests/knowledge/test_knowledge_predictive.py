import unittest
import pytest
pytest.importorskip("sympy")
from pathlib import Path
from coherent.engine.knowledge_registry import KnowledgeRegistry
from coherent.engine.symbolic_engine import SymbolicEngine

class TestKnowledgePredictive(unittest.TestCase):
    def setUp(self):
        self.engine = SymbolicEngine()
        base_path = Path("coherent/engine/knowledge")
        if not base_path.exists():
             base_path = Path("../coherent/engine/knowledge")
        self.registry = KnowledgeRegistry(base_path, self.engine)

    def test_match_rules_arithmetic(self):
        # x + y -> y + x (Commutative)
        expr = "x + y"
        matches = self.registry.match_rules(expr)
        
        # Expect finding ARITH-ADD-001 (Commutative)
        found = False
        for rule, next_expr in matches:
            if rule.id == "ARITH-ADD-001":
                if "y" in next_expr and "x" in next_expr and "+" in next_expr:
                    found = True
                    break
        self.assertTrue(found, f"Could not find ARITH-ADD-001 for {expr}")

    def test_apply_rule_specific(self):
        # Test applying a specific rule ID
        # Let's use ARITH-ADD-001: pattern_before: "a + b", pattern_after: "a + b" (eval)
        # Or ALG-DIST-001: a*(b+c) -> a*b + a*c
        
        # Check if ALG-DIST-001 exists (Distributive Property)
        # Note: Rule IDs depend on YAML files. Let's assume standard set or check registry first.
        # Find a rule to test
        dist_rule = next((r for r in self.registry.nodes if "distributive" in r.description.lower()), None)
        
        if dist_rule:
            expr = "2 * (x + 3)"
            result = self.registry.apply_rule(expr, dist_rule.id)
            self.assertIsNotNone(result)
            # 2x + 6
            self.assertTrue("2*x" in result and "6" in result)
        else:
            print("Skipping apply_rule test (Distributive rule not found)")

    def test_match_rules_equation(self):
        # Eq(2*x, 6) -> Eq(x, 3)
        expr = "Eq(2*x, 6)"
        matches = self.registry.match_rules(expr)
        
        found = False
        for rule, next_expr in matches:
            if "Eq(x, 3)" in next_expr or "Eq(x, 6 / 2)" in next_expr:
                found = True
                break
        self.assertTrue(found, f"Could not find rule for {expr}")

if __name__ == "__main__":
    unittest.main()
