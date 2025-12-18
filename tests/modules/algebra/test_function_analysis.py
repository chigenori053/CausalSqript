import pytest
sympy = pytest.importorskip("sympy")
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.function_analysis import FunctionAnalyzer


@pytest.fixture
def analyzer():
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    return FunctionAnalyzer(computation)


def test_analyze_polynomial(analyzer):
    result = analyzer.analyze("x**2 - 4")
    assert result.domain["type"] == "all_real"
    assert result.range["min"]["y"] <= -4
    assert result.intercepts["y"] == pytest.approx(-4.0)


def test_analyze_rational_function_domain(analyzer):
    result = analyzer.analyze("1 / (x - 2)")
    assert result.domain["type"] == "restricted"
    restrictions = result.domain["restrictions"]
    assert restrictions, "Expected domain restrictions for rational function."
    assert any(2.0 in restriction["values"] for restriction in restrictions)


def test_generate_plot_data(analyzer):
    data = analyzer.generate_plot_data("x", start=-1, end=1, num_points=5)
    assert len(data["x"]) == 5
    assert len(data["y"]) == 5
    assert data["x"][0] == -1
    assert data["x"][-1] == 1
    assert data["y"][2] == pytest.approx(0.0)
