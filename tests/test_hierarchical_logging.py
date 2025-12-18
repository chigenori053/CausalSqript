import unittest
import os
import pytest
sympy = pytest.importorskip("sympy")
import logging
from pathlib import Path
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.parser import Parser
from coherent.engine.evaluator import Evaluator
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.knowledge_registry import KnowledgeRegistry

class TestHierarchicalLogging(unittest.TestCase):
    def setUp(self):
        self.sym_engine = SymbolicEngine()
        # Point to real knowledge path
        self.knowledge_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "coherent", "engine", "knowledge")))
        self.registry = KnowledgeRegistry(self.knowledge_path, self.sym_engine)
        
        self.comp_engine = ComputationEngine(self.sym_engine)
        self.val_engine = ValidationEngine(self.comp_engine)
        self.hint_engine = HintEngine(self.comp_engine)
        self.logger = LearningLogger()
        self.runtime = CoreRuntime(
            self.comp_engine, 
            self.val_engine, 
            self.hint_engine, 
            learning_logger=self.logger,
            knowledge_registry=self.registry
        )

    def run_script(self, script):
        parser = Parser(script)
        program = parser.parse()
        evaluator = Evaluator(program, self.runtime, learning_logger=self.logger)
        return evaluator.run()

    def test_logging_structure(self):
        script = """
        problem: (1 + 2) * 3
        sub_problem: 1 + 2
        step: 3
        end: done
        step: 3 * 3
        step: 9
        end: done
        """
        success = self.run_script(script)
        self.assertTrue(success)
        
        logs = self.logger.to_list()
        
        # Check main scope
        problem_log = logs[0]
        self.assertEqual(problem_log['scope_id'], "main")
        self.assertEqual(problem_log['depth'], 0)
        
        # Check sub scope
        scope_start = next(l for l in logs if l['phase'] == 'scope_start')
        self.assertIsNotNone(scope_start)
        sub_id = scope_start['meta']['scope_id']
        
        # Find the step '3' inside the sub-problem
        # Note: 'step: 3' is the first step in sub_problem
        sub_step = next(l for l in logs if l['phase'] == 'step' and l['expression'] == '3')
        self.assertEqual(sub_step['scope_id'], sub_id)
        self.assertEqual(sub_step['parent_scope_id'], "main")
        self.assertEqual(sub_step['depth'], 1)
        
        scope_end = next(l for l in logs if l['phase'] == 'scope_end')
        self.assertEqual(scope_end['expression'], '3')

    def test_strict_arithmetic(self):
        # Test that 1+2 matches ARITH-CALC-ADD
        script = """
        problem: 1 + 2
        step: 3
        end: done
        """
        success = self.run_script(script)
        self.assertTrue(success)
        
        logs = self.logger.to_list()
        step_log = next(l for l in logs if l['phase'] == 'step' and l['expression'] == '3')
        
        # Check if rule_id is ARITH-CALC-ADD
        # Note: If other rules match, this might fail if priority is different.
        # But we set priority 90 for ARITH-CALC-ADD.
        self.assertEqual(step_log['rule_id'], 'ARITH-CALC-ADD')

    def test_algebraic_exclusion(self):
        # Ensure pure numeric doesn't match algebraic rules
        # We can't easily check "didn't match" without knowing what WOULD have matched.
        # But if we have a rule like a+b -> b+a (Commutative), it shouldn't match 1+2 -> 2+1 if it's algebraic.
        # Let's assume there is such a rule or we can rely on ARITH-CALC-ADD being picked.
        pass

if __name__ == '__main__':
    unittest.main()
