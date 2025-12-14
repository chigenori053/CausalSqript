
import unittest
import pytest
pytest.importorskip("sympy")
from pathlib import Path
from unittest.mock import MagicMock

from causalscript.core.core_runtime import CoreRuntime
from causalscript.core.computation_engine import ComputationEngine
from causalscript.core.validation_engine import ValidationEngine
from causalscript.core.hint_engine import HintEngine
from causalscript.core.knowledge_registry import KnowledgeRegistry
from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.reasoning.agent import ReasoningAgent
from causalscript.core.reasoning.generator import HypothesisGenerator
from causalscript.core.reasoning.goal import GoalScanner
from causalscript.core.reasoning.types import Hypothesis

class TestReasoningComponents(unittest.TestCase):
    def setUp(self):
        self.sym_engine = SymbolicEngine()
        self.comp_engine = ComputationEngine(self.sym_engine)
        
        # Mock registry for unit tests to avoid dependency on filesystem rules
        self.mock_registry = MagicMock(spec=KnowledgeRegistry)
        self.mock_registry.nodes = [] # Needed for Optical Layer init
        self.mock_registry.match_rules.return_value = []
        
        self.generator = HypothesisGenerator(self.mock_registry, self.sym_engine)
        self.goal_scanner = GoalScanner(self.sym_engine)

    def test_generator_equation_normalization(self):
        # Test that "a = b" is normalized to "Eq(a, b)" for internal processing
        # We can't easily check internal state, but we can check if it calls registry with normalized form
        
        expr = "x + 2 = 5"
        self.generator.generate(expr)
        
        # Verify registry was called with Eq(x + 2, 5) or similar
        args, _ = self.mock_registry.match_rules.call_args
        called_expr = args[0]
        self.assertIn("Eq(", called_expr)
        self.assertIn("x", called_expr) # simplistic check

    def test_generator_format_output(self):
        # Test that internal Eq output is formatted back to "="
        # Inject a fake match
        rule_stub = MagicMock()
        rule_stub.id = "TEST-RULE"
        rule_stub.priority = 50
        
        self.mock_registry.match_rules.return_value = [
            (rule_stub, "Eq(x, 3)")
        ]
        
        candidates = self.generator.generate("x = y")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].next_expr, "x = 3")

    def test_goal_scanner_is_solved(self):
        self.assertTrue(self.goal_scanner._is_solved_form("x = 3"))
        self.assertTrue(self.goal_scanner._is_solved_form("y = -5.5"))
        self.assertFalse(self.goal_scanner._is_solved_form("x + 1 = 3"))
        self.assertFalse(self.goal_scanner._is_solved_form("3 = x")) # Strict left-side var? Implementation says left is symbol.
        self.assertFalse(self.goal_scanner._is_solved_form("x = y")) # y is not numeric

class TestReasoningIntegration(unittest.TestCase):
    def setUp(self):
        self.sym_engine = SymbolicEngine()
        self.comp_engine = ComputationEngine(self.sym_engine)
        self.val_engine = ValidationEngine(self.comp_engine)
        self.hint_engine = HintEngine(self.comp_engine)
        
        base_path = Path("causalscript/core/knowledge")
        if not base_path.exists():
             base_path = Path("../causalscript/core/knowledge")

        self.registry = KnowledgeRegistry(base_path, self.sym_engine)
        
        self.runtime = CoreRuntime(
            computation_engine=self.comp_engine,
            validation_engine=self.val_engine,
            hint_engine=self.hint_engine,
            knowledge_registry=self.registry
        )
        self.agent = ReasoningAgent(self.runtime)

    def test_solve_linear_equation_flow(self):
        # Integration test (same as reproduction)
        expr = "2*x + 4 = 10"
        
        # Step 1: Move Constant
        hyp1 = self.agent.think(expr)
        self.assertIsNotNone(hyp1)
        # We accept multiple valid paths, but with current priority, it should be Move Add
        # 2x = 6
        target1 = "2*x = 6"
        # Determine equivalence via engine to avoid whitespace issues
        is_equiv1 = self.sym_engine.is_equiv(hyp1.next_expr.replace("=", "-(")+")", target1.replace("=", "-(")+")")
        # Just simple string check for now as is_equiv on logic expressions is tricky
        self.assertTrue("2*x=6" in hyp1.next_expr.replace(" ", "") or "2x=6" in hyp1.next_expr.replace(" ",""))

        # Step 2: Divide
        hyp2 = self.agent.think(hyp1.next_expr)
        self.assertIsNotNone(hyp2)
        self.assertTrue("x=3" in hyp2.next_expr.replace(" ", ""))

if __name__ == "__main__":
    unittest.main()
