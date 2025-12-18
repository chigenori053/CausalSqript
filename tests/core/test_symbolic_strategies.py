import pytest
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.math_category import MathCategory
from coherent.engine.symbolic_strategies import (
    ArithmeticStrategy,
    AlgebraStrategy,
    CalculusStrategy
)

try:
    import sympy
except ImportError:
    sympy = None

class TestSymbolicStrategies:
    def setup_method(self):
        self.engine = SymbolicEngine()

    def test_arithmetic_strategy(self):
        # Set context to Arithmetic
        self.engine.set_context([MathCategory.ARITHMETIC])
        
        # Test equivalence
        assert self.engine.is_equiv("1 + 2", "3")
        assert self.engine.is_equiv("2 * 3", "6")
        assert not self.engine.is_equiv("1 + 2", "4")
        
        # Test simplification
        assert self.engine.simplify("1 + 2") == "3"

    def test_algebra_strategy(self):
        if sympy is None:
            pytest.skip("SymPy not installed")
            
        # Set context to Algebra
        self.engine.set_context([MathCategory.ALGEBRA])
        
        # Test equivalence
        assert self.engine.is_equiv("x + x", "2*x")
        assert self.engine.is_equiv("(x + 1)**2", "x**2 + 2*x + 1")
        
        # Test simplification
        assert self.engine.simplify("x + x") == "2*x"
        
        # Test LaTeX
        # Algebra strategy should use implicit multiplication
        latex = self.engine.to_latex("2*x")
        assert "cdot" not in latex

    def test_calculus_strategy(self):
        if sympy is None:
            pytest.skip("SymPy not installed")
            
        # Set context to Calculus
        self.engine.set_context([MathCategory.CALCULUS])
        
        # Test equivalence with pending operations
        # Integral(2*x, x) -> x**2
        assert self.engine.is_equiv("Integral(2*x, x)", "x**2")
        
        # Derivative(x**2, x) -> 2*x
        assert self.engine.is_equiv("Derivative(x**2, x)", "2*x")

    def test_strategy_fallback(self):
        # Set context to Arithmetic but give algebraic expression
        self.engine.set_context([MathCategory.ARITHMETIC])
        
        # Arithmetic strategy should return None for "x + x", falling back to default (Algebra/Fallback)
        # If SymPy is missing, fallback logic (numeric sampling) should still handle basic algebra if variables are standard
        # But "x + x" vs "2*x" might fail in numeric sampling if not robust?
        # Actually _numeric_sampling_equiv uses _SAMPLE_ASSIGNMENTS which has 'x'.
        # So it should pass even without SymPy!
        assert self.engine.is_equiv("x + x", "2*x")

    def test_mixed_context(self):
        if sympy is None:
            pytest.skip("SymPy not installed")

        # Set context to [Calculus, Algebra]
        self.engine.set_context([MathCategory.CALCULUS, MathCategory.ALGEBRA])
        
        # Should handle calculus
        assert self.engine.is_equiv("Derivative(x**2, x)", "2*x")
        
        # Should handle algebra
        assert self.engine.is_equiv("x + x", "2*x")

if __name__ == "__main__":
    pytest.main([__file__])
