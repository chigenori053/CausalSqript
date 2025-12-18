import pytest
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.unit_engine import UnitEngine

@pytest.fixture
def unit_engine():
    symbolic = SymbolicEngine()
    return UnitEngine(symbolic)

def test_convert_length(unit_engine):
    result = unit_engine.convert("10 * meter", "centimeter")
    # 10 m = 1000 cm
    assert result == "1000*centimeter"

def test_convert_time(unit_engine):
    result = unit_engine.convert("1 * hour", "second")
    # 1 hr = 3600 s
    assert result == "3600*second"

def test_compound_units(unit_engine):
    # 1 km/h to m/s
    # 1 km = 1000 m, 1 h = 3600 s => 1000/3600 = 5/18 m/s
    result = unit_engine.convert("36 * kilometer / hour", "meter / second")
    assert result == "10*meter/second"

def test_simplify(unit_engine):
    result = unit_engine.simplify("1000 * meter / kilometer")
    # Should be dimensionless 1
    assert result == "1"

def test_check_consistency_valid(unit_engine):
    assert unit_engine.check_consistency("1 * meter + 20 * centimeter") is True

def test_check_consistency_invalid(unit_engine):
    assert unit_engine.check_consistency("1 * meter + 1 * second") is False

def test_get_si_units(unit_engine):
    result = unit_engine.get_si_units("1 * kilometer")
    assert result == "1000*meter"
