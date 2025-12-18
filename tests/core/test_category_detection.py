import pytest
from coherent.engine.category_analyzer import CategoryAnalyzer
from coherent.engine.math_category import MathCategory
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.symbolic_engine import SymbolicEngine

try:
    import sympy
except ImportError:
    sympy = None

class TestCategoryDetection:
    def test_category_analyzer_detect(self):
        assert CategoryAnalyzer.detect("1 + 2") == MathCategory.ARITHMETIC
        assert CategoryAnalyzer.detect("x^2 + 2*x + 1") == MathCategory.ALGEBRA
        assert CategoryAnalyzer.detect("diff(x^2, x)") == MathCategory.CALCULUS
        assert CategoryAnalyzer.detect("Triangle(p1, p2, p3)") == MathCategory.GEOMETRY
        assert CategoryAnalyzer.detect("mean([1, 2, 3])") == MathCategory.STATISTICS
        
        # Test new keywords
        assert CategoryAnalyzer.detect("d/dx(x^2)") == MathCategory.CALCULUS
        assert CategoryAnalyzer.detect("Matrix([[1, 2], [3, 4]])") == MathCategory.LINEAR_ALGEBRA

    def test_computation_engine_integration(self):
        symbolic_engine = SymbolicEngine()
        comp_engine = ComputationEngine(symbolic_engine)
        
        assert comp_engine.detect_category("integrate(x, x)") == MathCategory.CALCULUS
        
        # Test compute_optimized dispatch (mocking behavior via return value check or side effect if possible)
        # For now, just check it runs without error and returns simplified result
        result = comp_engine.compute_optimized("1 + 1")
        assert result == "2"

if __name__ == "__main__":
    pytest.main([__file__])
