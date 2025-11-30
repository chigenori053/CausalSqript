import unittest
from core.symbolic_engine import SymbolicEngine
from core.evaluator import Evaluator, Engine
from core.learning_logger import LearningLogger
from core.parser import Parser

class MockEngine(Engine):
    def __init__(self, sym_engine):
        self.symbolic_engine = sym_engine
        self._current_expr = None

    def set(self, expr):
        self._current_expr = expr

    def check_step(self, expr):
        return {
            "valid": True,
            "before": self._current_expr,
            "after": expr,
            "rule_id": None,
            "details": {}
        }
    
    def finalize(self, expr):
        return {
            "valid": True,
            "before": self._current_expr,
            "after": expr,
            "details": {}
        }

class TestV1_2(unittest.TestCase):
    def setUp(self):
        self.sym_engine = SymbolicEngine()
        self.logger = LearningLogger()
        self.engine = MockEngine(self.sym_engine)
        
    def test_fraction_rendering(self):
        # Test 3/4 rendering
        expr = "3/4"
        simplified = self.sym_engine.simplify(expr)
        print(f"Simplified '3/4': '{simplified}'")
        self.assertEqual(simplified, "3/4")
        
        # Test 1/2 + 1/4 -> 3/4
        expr2 = "1/2 + 1/4"
        simplified2 = self.sym_engine.simplify(expr2)
        print(f"Simplified '1/2 + 1/4': '{simplified2}'")
        self.assertEqual(simplified2, "3/4")

    def test_redundancy_logging(self):
        script = """
        problem: x
        step: x
        step: x
        end: done
        """
        parser = Parser(script)
        program = parser.parse()
        evaluator = Evaluator(program, self.engine, learning_logger=self.logger)
        evaluator.run()
        
        logs = self.logger.to_list()
        
        # Find step logs
        step_logs = [l for l in logs if l['phase'] == 'step']
        self.assertEqual(len(step_logs), 2)
        
        first = step_logs[0]
        second = step_logs[1]
        
        print(f"First step: {first}")
        print(f"Second step: {second}")
        
        self.assertFalse(first.get('is_redundant', False))
        self.assertTrue(second.get('is_redundant', False))

if __name__ == '__main__':
    unittest.main()
