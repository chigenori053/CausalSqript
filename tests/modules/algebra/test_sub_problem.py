import unittest
import pytest
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.parser import Parser
sympy = pytest.importorskip("sympy")
from coherent.engine.evaluator import Evaluator
from coherent.engine.learning_logger import LearningLogger

class TestSubProblem(unittest.TestCase):
    def setUp(self):
        self.sym_engine = SymbolicEngine()
        self.comp_engine = ComputationEngine(self.sym_engine)
        self.val_engine = ValidationEngine(self.comp_engine)
        self.hint_engine = HintEngine(self.comp_engine)
        self.logger = LearningLogger()
        self.runtime = CoreRuntime(
            self.comp_engine, 
            self.val_engine, 
            self.hint_engine, 
            learning_logger=self.logger
        )

    def run_script(self, script):
        parser = Parser(script)
        program = parser.parse()
        evaluator = Evaluator(program, self.runtime, learning_logger=self.logger)
        return evaluator.run()

    def test_user_example(self):
        """Test the user's example with sub_problem."""
        script = """
        problem: sin(theta)^2 + cos(theta)^2
        prepare:
          - theta = pi / 3
        sub_problem: sin(pi / 3)^2
        step: (sqrt(3) / 2)^2
        step: 3 / 4
        end: done
        sub_problem: cos(pi / 3)^2
        step: (1 / 2)^2
        step: 1 / 4
        end: done
        step: (3 / 4) + (1 / 4)
        step: 1
        end: done
        """
        # Note: I replaced unicode chars with ascii for safety in this test file first,
        # but the parser should handle unicode if normalize is good.
        # Let's try to stick to standard ascii names for the test to isolate logic from unicode issues first.
        
        success = self.run_script(script)
        self.assertTrue(success)
        
        logs = self.logger.to_list()
        
        # Verify sub-problem phases
        sub_probs = [l for l in logs if l['phase'] == 'sub_problem']
        self.assertEqual(len(sub_probs), 2)
        self.assertEqual(sub_probs[0]['expression'], "sin(pi / 3)^2")
        self.assertEqual(sub_probs[1]['expression'], "cos(pi / 3)^2")
        
        sub_ends = [l for l in logs if l['phase'] == 'sub_problem_end']
        self.assertEqual(len(sub_ends), 2)
        
        # Check return to parent
        # After first sub-problem: sin(pi/3)^2 -> 3/4
        # Parent becomes: 3/4 + cos(pi/3)^2
        self.assertIn("3/4", sub_ends[0]['rendered'])
        
        # After second sub-problem: cos(pi/3)^2 -> 1/4
        # Parent becomes: 3/4 + 1/4
        self.assertIn("1/4", sub_ends[1]['rendered'])

    def test_nested_sub_problem(self):
        """Test nested sub-problems."""
        script = """
        problem: (1 + 2) * (3 + 4)
        sub_problem: 1 + 2
        step: 3
        end: done
        sub_problem: 3 + 4
        step: 7
        end: done
        step: 3 * 7
        step: 21
        end: done
        """
        success = self.run_script(script)
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()
