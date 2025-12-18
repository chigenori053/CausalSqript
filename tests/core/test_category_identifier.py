import pytest
from coherent.engine.category_identifier import CategoryIdentifier, MathCategory, CategoryResult
from coherent.engine.symbolic_engine import SymbolicEngine

@pytest.fixture
def identifier():
    engine = SymbolicEngine()
    return CategoryIdentifier(engine)

def test_identify_arithmetic(identifier):
    result = identifier.identify("1 + 1")
    assert result.primary_category == MathCategory.ARITHMETIC
    assert result.confidence == 1.0

def test_identify_algebra(identifier):
    result = identifier.identify("x + y")
    assert result.primary_category == MathCategory.ALGEBRA
    assert MathCategory.ARITHMETIC in result.related_categories

def test_identify_calculus(identifier):
    # Depending on classifier implementation, it might need specific keywords
    result = identifier.identify("Integral(x^2, x)")
    assert result.primary_category == MathCategory.CALCULUS

def test_identify_linear_algebra(identifier):
    result = identifier.identify("Vector([1, 2])")
    assert result.primary_category == MathCategory.LINEAR_ALGEBRA

def test_identify_statistics(identifier):
    result = identifier.identify("mean([1, 2, 3])")
    assert result.primary_category == MathCategory.STATISTICS

def test_identify_unknown(identifier):
    # Empty string or weird input might result in unknown or fallback
    # But current classifier usually defaults to arithmetic or algebra
    # Let's try something that might fail classification if empty
    pass
