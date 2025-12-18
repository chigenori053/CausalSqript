
import unittest
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry
from pathlib import Path

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.symbolic_engine = SymbolicEngine()
        self.computation_engine = ComputationEngine(self.symbolic_engine)
        self.validation_engine = ValidationEngine(self.computation_engine)
        self.hint_engine = HintEngine(self.computation_engine)
        self.knowledge_registry = KnowledgeRegistry(Path("core/knowledge"), self.symbolic_engine)
        
        self.runtime = CoreRuntime(
            self.computation_engine,
            self.validation_engine,
            self.hint_engine,
            knowledge_registry=self.knowledge_registry
        )

    def test_calculus_classification(self):
        expr = "Integral(x**2, x)"
        self.runtime.set(expr)
        report = self.runtime.generate_optimization_report()
        
        self.assertIn("calculus", report["classification"])
        self.assertEqual(report["symbolic_engine_mode"], "optimized")
        self.assertEqual(report["report_rendering_strategy"], "latex_enhanced")

    def test_linear_algebra_classification(self):
        expr = "Vector([1, 2]).dot(Vector([3, 4]))"
        self.runtime.set(expr)
        report = self.runtime.generate_optimization_report()
        
        self.assertIn("linear_algebra", report["classification"])
        self.assertEqual(report["report_rendering_strategy"], "latex_enhanced")

    def test_statistics_classification(self):
        expr = "normal_pdf(x, mean=0, std=1)" # Assuming normal_pdf or similar keyword
        # Using keyword 'normal' from my list
        expr = "normal(0, 1)"
        self.runtime.set(expr)
        report = self.runtime.generate_optimization_report()
        
        self.assertIn("statistics", report["classification"])

    def test_algebra_classification(self):
        expr = "x**2 + 2*x + 1"
        self.runtime.set(expr)
        report = self.runtime.generate_optimization_report()
        
        self.assertIn("algebra", report["classification"])
        self.assertEqual(report["symbolic_engine_mode"], "standard")

if __name__ == "__main__":
    unittest.main()
