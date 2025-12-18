import unittest
import pytest
sympy = pytest.importorskip("sympy")
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.parser import Parser
from coherent.engine.evaluator import Evaluator
from coherent.engine.learning_logger import LearningLogger

class TestPartialCalculation(unittest.TestCase):
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

    def test_partial_trig_calculation(self):
        """Test the user's example with partial trig calculations."""
        script = """
        problem: sin(pi / 3)^2 + cos(pi / 3)^2
        step: sin(pi / 3)^2 = (3 / 4)
        step: cos(pi / 3)^2 = (1 / 4)
        step: (3 / 4) + (1 / 4) = 1
        end: 1
        """
        success = self.run_script(script)
        self.assertTrue(success)
        
        logs = self.logger.to_list()
        # Verify steps are marked as partial where appropriate
        step_logs = [l for l in logs if l['phase'] == 'step']
        self.assertEqual(len(step_logs), 3)
        
        # Step 1: sin^2 = 3/4 (Partial)
        self.assertTrue(step_logs[0]['meta'].get('partial'))
        self.assertEqual(step_logs[0]['status'], 'ok')
        
        # Step 2: cos^2 = 1/4 (Partial)
        self.assertTrue(step_logs[1]['meta'].get('partial'))
        self.assertEqual(step_logs[1]['status'], 'ok')
        
        # Step 3: 3/4 + 1/4 = 1 (State Update)
        # This is NOT partial because it updates the state (LHS matches Before)
        self.assertFalse(step_logs[2]['meta'].get('partial'))
        self.assertEqual(step_logs[2]['status'], 'ok')

    def test_partial_algebraic_calculation(self):
        """Test partial calculation with algebraic expressions."""
        # problem: (x + 1)^2 + (y - 1)^2
        # step: (x + 1)^2 = x^2 + 2x + 1
        # step: (y - 1)^2 = y^2 - 2y + 1
        # step: (x^2 + 2x + 1) + (y^2 - 2y + 1) = x^2 + y^2 + 2x - 2y + 2
        # end: x^2 + y^2 + 2x - 2y + 2
        
        script = """
        problem: (x + 1)^2 + (y - 1)^2
        step: (x + 1)^2 = x^2 + 2*x + 1
        step: (y - 1)^2 = y^2 - 2*y + 1
        step: (x^2 + 2*x + 1) + (y^2 - 2*y + 1) = x^2 + y^2 + 2*x - 2*y + 2
        end: x^2 + y^2 + 2*x - 2*y + 2
        """
        success = self.run_script(script)
        self.assertTrue(success)
        
        logs = self.logger.to_list()
        step_logs = [l for l in logs if l['phase'] == 'step']
        
        # Step 1: Partial
        self.assertTrue(step_logs[0]['meta'].get('partial'))
        
        # Step 2: Partial
        self.assertTrue(step_logs[1]['meta'].get('partial'))
        
        # Step 3: State Update
        self.assertFalse(step_logs[2]['meta'].get('partial'))

    def test_invalid_partial_step(self):
        """Test that an invalid partial step fails."""
        script = """
        problem: sin(pi / 3)^2 + cos(pi / 3)^2
        step: sin(pi / 3)^2 = (1 / 2)
        end: 1
        """
        # sin(pi/3)^2 is 3/4, not 1/2.
        success = self.run_script(script)
        self.assertFalse(success)
        
        logs = self.logger.to_list()
        step_logs = [l for l in logs if l['phase'] == 'step']
        self.assertEqual(step_logs[0]['status'], 'mistake')

if __name__ == '__main__':
    unittest.main()
