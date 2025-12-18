
import unittest
from coherent.engine.category_analyzer import CategoryAnalyzer
from coherent.engine.math_category import MathCategory
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.symbolic_engine import SymbolicEngine

class TestCategoryOptimization(unittest.TestCase):
    def setUp(self):
        self.symbolic_engine = SymbolicEngine()
        self.comp_engine = ComputationEngine(self.symbolic_engine)

    def verify_case(self, expr, expected_category, description):
        print(f"\n--- Verifying {description} ---")
        print(f"Expression: {expr}")
        
        # 1. Detect Category
        category = self.comp_engine.detect_category(expr)
        print(f"Detected Category: {category}")
        self.assertEqual(category, expected_category, f"Failed to identify {description}")
        
        # 2. Compute Optimized
        # We just want to ensure it runs without error and returns something valid
        try:
            result = self.comp_engine.compute_optimized(expr)
            print(f"Computation Result: {result}")
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Computation failed for {description}: {e}")

    def test_arithmetic(self):
        self.verify_case("1 + 2 * 3", MathCategory.ARITHMETIC, "Arithmetic")

    def test_algebra(self):
        self.verify_case("x**2 + 2*x + 1", MathCategory.ALGEBRA, "Algebra")
        self.verify_case("Eq(x + 1, 5)", MathCategory.ALGEBRA, "Algebra Equation")

    def test_calculus(self):
        self.verify_case("diff(x**2, x)", MathCategory.CALCULUS, "Calculus Derivative")
        self.verify_case("integrate(x, x)", MathCategory.CALCULUS, "Calculus Integral")

    def test_linear_algebra(self):
        # We rely on 'Matrix' keyword being present (as produced by parser)
        expr = "Matrix([[1, 2], [3, 4]])"
        self.verify_case(expr, MathCategory.LINEAR_ALGEBRA, "Linear Algebra Matrix")
        
        expr_op = "Matrix([[1, 0], [0, 1]]) * Matrix([[1], [2]])"
        self.verify_case(expr_op, MathCategory.LINEAR_ALGEBRA, "Matrix Multiplication")

    def test_statistics(self):
        self.verify_case("mean([1, 2, 3])", MathCategory.STATISTICS, "Statistics Mean")

if __name__ == "__main__":
    unittest.main()
