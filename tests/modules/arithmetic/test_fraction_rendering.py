from fractions import Fraction

from coherent.engine.symbolic_engine import SymbolicEngine


def test_fraction_to_string_plain():
    engine = SymbolicEngine()
    assert engine.to_string(Fraction(3, 4)) == "3/4"
    assert engine.to_string(Fraction(6, 2)) == "3"


def test_fraction_to_string_latex():
    engine = SymbolicEngine()
    assert engine.to_string(Fraction(3, 4), latex=True) == "\\frac{3}{4}"
