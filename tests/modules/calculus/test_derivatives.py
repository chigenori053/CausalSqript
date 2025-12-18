import pytest
from coherent.engine.input_parser import CoherentInputParser

class TestDerivatives:
    def test_leibniz_simple(self):
        # d/dx x^2 -> diff(x**2, x)
        expr = "d/dx x^2"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "diff(x**2,x)"

    def test_leibniz_parens(self):
        # d/dx (x^2 + 1) -> diff((x**2 + 1), x)
        expr = "d/dx (x^2 + 1)"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "diff((x**2 + 1),x)"

    def test_leibniz_trig(self):
        # d/dt sin(t) -> diff(sin(t), t)
        expr = "d/dt sin(t)"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "diff(sin(t),t)"

    def test_leibniz_separated(self):
        # d/d x x^2 -> diff(x**2, x)
        expr = "d/d x x^2"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "diff(x**2,x)"

    def test_lagrange_simple(self):
        # y' -> diff(y, x)
        expr = "y'"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "diff(y,x)"

    def test_lagrange_func(self):
        # sin(x)' -> diff(sin(x), x)
        expr = "sin(x)'"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "diff(sin(x),x)"

    def test_lagrange_func_var(self):
        # sin(t)' -> diff(sin(t), t)
        expr = "sin(t)'"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "diff(sin(t),t)"

    def test_lagrange_parens(self):
        # (x^2)' -> diff((x**2), x)
        expr = "(x^2)'"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "diff((x**2),x)"

    def test_nested_derivatives(self):
        # d/dx (d/dx x^3) -> diff((diff(x**3, x)), x)
        # Extra parens are preserved from input (d/dx (...))
        expr = "d/dx (d/dx x^3)"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "diff((diff(x**3,x)),x)"

    def test_mixed_notation(self):
        # d/dx y' -> diff(diff(y, x), x)
        expr = "d/dx y'"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "diff(diff(y,x),x)"

if __name__ == "__main__":
    pytest.main([__file__])
