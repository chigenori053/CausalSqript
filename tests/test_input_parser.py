from __future__ import annotations

import pytest

from core.input_parser import CausalScriptInputParser


@pytest.mark.parametrize(
    "input_expr, expected_output",
    [
        # Test Case 1: From user's bug report
        ("x^2 +2xy +y^2", "x**2 + 2*x*y + y**2"),
        # Test Case 2: Implicit multiplication with parentheses
        ("5(a+b)c", "5*(a + b)*c"),
        # Test Case 3: Implicit multiplication between functions
        ("sin(theta)cos(phi)", "sin(theta)*cos(phi)"),
        # Test Case 4: Implicit multiplication with multi-digit numbers
        ("12x", "12*x"),
        # Test Case 5: Implicit multiplication between identifiers (sanity check)
        ("abc", "a*b*c"),
    ],
)
def test_normalize_implicit_multiplication(input_expr: str, expected_output: str):
    """Tests normalization of expressions with implicit multiplication."""
    assert CausalScriptInputParser.normalize(input_expr) == expected_output