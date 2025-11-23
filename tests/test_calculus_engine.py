import pytest

from core.symbolic_engine import SymbolicEngine
from core.computation_engine import ComputationEngine
from core.calculus_engine import CalculusEngine


@pytest.fixture
def calc_engine():
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    return CalculusEngine(computation)


def test_derivative_expression(calc_engine):
    result = calc_engine.derivative("x**2 + 3*x", variable="x")
    assert "2*x" in result["expression"]
    numeric = calc_engine.derivative("x**2", variable="x", at=2)
    assert numeric["value"] == pytest.approx(4.0, rel=1e-3)


def test_integral_value(calc_engine):
    result = calc_engine.integral("x", variable="x", lower=0, upper=2)
    assert result["value"] == pytest.approx(2.0, rel=1e-3)


def test_slope_and_area(calc_engine):
    slope = calc_engine.slope_of_tangent("x**2", "x", at=1)
    assert slope == pytest.approx(2.0, rel=1e-2)
    area = calc_engine.area_under_curve("2*x", "x", 0, 1)
    assert area == pytest.approx(1.0, rel=1e-2)
