
import unittest
from unittest.mock import MagicMock
from core.core_runtime import CoreRuntime
from core.computation_engine import ComputationEngine
from core.validation_engine import ValidationEngine
from core.hint_engine import HintEngine
from core.symbolic_engine import SymbolicEngine
from core.fuzzy.types import FuzzyResult, FuzzyLabel, FuzzyScore

class TestGradingImprovement(unittest.TestCase):
    def setUp(self):
        self.symbolic = SymbolicEngine()
        self.computation = ComputationEngine(self.symbolic)
        self.validation = ValidationEngine(self.computation)
        # Mock FuzzyJudge
        self.validation.fuzzy_judge = MagicMock()
        self.validation.fuzzy_judge.encoder.normalize.side_effect = lambda x: x # Identity for simplification
        
        self.hint = HintEngine(self.computation)
        self.runtime = CoreRuntime(self.computation, self.validation, self.hint)

    def test_fuzzy_pass_symbolic_fail(self):
        """
        Test case: User enters an incorrect step that is accepted by FuzzyJudge.
        Expectation: Valid=True, but 'mathematical_error' is True and 'correction_notice' is present.
        """
        # Problem: x + 1
        self.runtime.set("x + 1")
        
        # User Input: x + 1.1 (Mathematically incorrect, but maybe fuzzy approx equal)
        user_input = "x + 1.1"
        
        # Configure Mock FuzzyJudge to return APPROX_EQ
        mock_result = FuzzyResult(
            label=FuzzyLabel.APPROX_EQ,
            score=FuzzyScore(combined_score=0.9, expr_similarity=0.9, rule_similarity=0.9, text_similarity=0.9), # filled dummy values for TypedDict
            reason="Approximate match",
            debug={"decision_action": "approve"}
        )
        self.validation.fuzzy_judge.judge_step.return_value = mock_result

        # Execute check_step
        result = self.runtime.check_step(user_input)
        
        print("DEBUG: Result Details:", result["details"])

        # Assertions
        self.assertTrue(result["valid"], "Step should be valid due to FuzzyJudge")
        self.assertTrue(result["details"].get("mathematical_error"), "Should flag mathematical error")
        self.assertIn("correction_notice", result["details"], "Should contain correction notice")
        self.assertEqual(result["details"]["expected_formula"], "x + 1", "Expected formula should be x + 1")
        self.assertEqual(result["details"]["evaluation_note"], "Accepted by fuzzy match, but contains symbolic error.")

    def test_symbolic_fail_fuzzy_fail(self):
        """
        Test case: User enters a completely wrong step.
        Expectation: Valid=False, 'mathematical_error' is True, standard hint present.
        """
        self.runtime.set("x + 1")
        user_input = "x + 100"
        
        # Configure Mock FuzzyJudge to return WRONG
        mock_result = FuzzyResult(
            label=FuzzyLabel.CONTRADICT,
            score=FuzzyScore(combined_score=0.1, expr_similarity=0.1, rule_similarity=0.1, text_similarity=0.1),
            reason="Wrong",
            debug={}
        )
        self.validation.fuzzy_judge.judge_step.return_value = mock_result

        result = self.runtime.check_step(user_input)

        self.assertFalse(result["valid"], "Step should be invalid")
        self.assertTrue(result["details"].get("mathematical_error"), "Should flag mathematical error")
        # 'hint' is present (standard behavior), 'correction_notice' is NOT strictly required if it failed,
        # but our logic puts the feedback in 'hint' in this case.
        self.assertIn("hint", result["details"])
        self.assertNotIn("correction_notice", result["details"]) # Logic puts it in hint for fail case

if __name__ == "__main__":
    unittest.main()
