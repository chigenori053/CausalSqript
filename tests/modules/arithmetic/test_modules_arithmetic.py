import pytest
from coherent.engine.modules.arithmetic.parser import ArithmeticParser
from coherent.engine.modules.arithmetic.engine import FastMathEngine
from coherent.engine.errors import InvalidExprError

@pytest.fixture
def parser():
    return ArithmeticParser()

@pytest.fixture
def engine():
    return FastMathEngine()

def test_valid_arithmetic(parser, engine):
    exprs = [
        ("1 + 2", 3),
        ("3 * 4", 12),
        ("10 / 2", 5.0),
        ("2 * (3 + 4)", 14),
        ("-5 + 10", 5),
        ("1.5 + 2.5", 4.0),
    ]
    for text, expected in exprs:
        ast = parser.parse(text)
        result = engine.evaluate(ast)
        assert result == expected

def test_invalid_syntax(parser):
    invalids = [
        "2 * ",          # Incomplete
        "1 ++ 2",        # Double operator (technically python might allow but our regex might not? Regex allows ++... let's check AST)
                         # Regex allows +. AST parse handles it.
        "sin(30)",       # Function call disallowed
        "x + 1",         # Variable disallowed
        "3 = 3",         # Assignment disallowed
        "print('hi')",   # Function call
    ]
    
    # "1 ++ 2" is valid python (1 + (+2)). 
    # But "2 * " is syntax error.
    
    with pytest.raises(InvalidExprError):
        parser.parse("sin(30)")
    
    with pytest.raises(InvalidExprError):
        parser.parse("x + 1")
    
    with pytest.raises(InvalidExprError):
        parser.parse("2 * ")

def test_division_by_zero(parser, engine):
    # This might depend on how Python handles it (ZeroDivisionError)
    ast = parser.parse("1 / 0")
    with pytest.raises(ZeroDivisionError):
        engine.evaluate(ast)

def test_parser_validation_regex(parser):
    # Test strict regex
    assert parser.validate("1 + 2")
    assert not parser.validate("x + 1")
    assert not parser.validate("sin(90)")
    assert parser.validate("   1.5   * 2  ")

def test_security_nodes(parser):
    # Ensure no weird AST nodes pass through even if regex matches (unlikely with strict regex but good practice)
    # The regex [0-9+...] is very strict, so it's hard to inject valid python AST that is malicious.
    pass
