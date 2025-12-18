import pytest

from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.errors import InvalidExprError


def test_is_equiv_checks_math_equivalence():
    engine = SymbolicEngine()
    assert engine.is_equiv("3 + 5", "8")
    assert engine.is_equiv("(x + 1) * (x + 2)", "x**2 + 3*x + 2")
    assert not engine.is_equiv("x + 1", "x + 2")


def test_simplify_and_explain():
    engine = SymbolicEngine()
    assert engine.simplify("(3 + 5) * 1") == "8"
    message = engine.explain("x + 0", "x")
    assert "equivalent" in message.lower()


def test_invalid_expression_raises():
    engine = SymbolicEngine()
    with pytest.raises(InvalidExprError):
        engine.to_internal("???")
