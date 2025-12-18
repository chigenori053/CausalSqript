import pytest
sympy = pytest.importorskip("sympy")
from coherent.engine.input_parser import CoherentInputParser
from coherent.engine.symbolic_engine import SymbolicEngine

class TestIntegralSteps:
    def test_brackets_parsing(self):
        # [x^3]_0^2 -> (Subs(x**3, x, 2) - Subs(x**3, x, 0))
        expr = "[x^3]_0^2"
        normalized = CoherentInputParser.normalize(expr)
        print(f"Normalized: {normalized}")
        
        # Check structure (roughly)
        assert "Subs(x**3,x,2)" in normalized.replace(" ", "")
        assert "Subs(x**3,x,0)" in normalized.replace(" ", "")
        assert "-" in normalized

    def test_brackets_evaluation(self):
        expr = "[x^3]_0^2"
        normalized = CoherentInputParser.normalize(expr)
        
        engine = SymbolicEngine()
        result = engine.evaluate(normalized, {})
        assert result == 8

    def test_brackets_variable_inference(self):
        # [t^2]_1^2 -> ((t**2).subs(t, 2) - (t**2).subs(t, 1))
        expr = "[t^2]_1^2"
        normalized = CoherentInputParser.normalize(expr)
        
        engine = SymbolicEngine()
        result = engine.evaluate(normalized, {})
        # 2^2 - 1^2 = 4 - 1 = 3
        assert result == 3

if __name__ == "__main__":
    pytest.main([__file__])
