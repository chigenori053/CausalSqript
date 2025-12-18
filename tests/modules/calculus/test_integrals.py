import pytest
from coherent.engine.input_parser import CoherentInputParser

class TestIntegrals:
    def test_indefinite_integral(self):
        # ∫ x dx -> integrate(x,x)
        expr = "∫ x dx"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "integrate(x,x)"

    def test_definite_integral(self):
        # ∫_0^1 x dx -> integrate(x,(x,0,1))
        expr = "∫_0^1 x dx"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "integrate(x,(x,0,1))"

    def test_definite_integral_with_vars(self):
        # ∫_a^b x^2 dx -> integrate(x**2,(x,a,b))
        expr = "∫_a^b x^2 dx"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "integrate(x**2,(x,a,b))"

    def test_nested_integral(self):
        # ∫ (∫ x dy) dx -> integrate((integrate(x,y)),x)
        # Parens from input are preserved
        expr = "∫ (∫ x dy) dx"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "integrate((integrate(x,y)),x)"

    def test_trig_integral(self):
        # ∫ sin(x) dx -> integrate(sin(x),x)
        expr = "∫ sin(x) dx"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "integrate(sin(x),x)"

    def test_complex_integrand(self):
        # ∫ (x^2 + 1) dx -> integrate((x**2 + 1),x)
        expr = "∫ (x^2 + 1) dx"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "integrate((x**2 + 1),x)"

    def test_separated_differential(self):
        # ∫ x d x -> integrate(x,x)
        expr = "∫ x d x"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "integrate(x,x)"

    def test_integral_with_implicit_mult(self):
        # ∫ 2x dx -> integrate(2*x,x)
        expr = "∫ 2x dx"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "integrate(2*x,x)"

    def test_mixed_bounds_order(self):
        # ∫^b _a x dx -> integrate(x,(x,a,b))
        # Note: Space required before _a to separate it from b
        expr = "∫^b _a x dx"
        normalized = CoherentInputParser.normalize(expr)
        assert normalized == "integrate(x,(x,a,b))"

if __name__ == "__main__":
    pytest.main([__file__])
