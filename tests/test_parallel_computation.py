"""Tests for Parallel Computation in Scenarios."""

import pytest
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.symbolic_engine import SymbolicEngine

@pytest.fixture
def engine():
    return ComputationEngine(SymbolicEngine())

def test_parallel_scenario_evaluation(engine):
    expr = "x**2 + y"
    scenarios = {
        "case1": {"x": 1, "y": 2}, # 1 + 2 = 3
        "case2": {"x": 2, "y": 3}, # 4 + 3 = 7
        "case3": {"x": 3, "y": 4}, # 9 + 4 = 13
        "case4": {"x": 4, "y": 5}, # 16 + 5 = 21
        "case5": {"x": 5, "y": 6}, # 25 + 6 = 31
    }
    
    results = engine.evaluate_in_scenarios(expr, scenarios)
    
    assert results["case1"] == 3
    assert results["case2"] == 7
    assert results["case3"] == 13
    assert results["case4"] == 21
    assert results["case5"] == 31

def test_parallel_equivalence_check(engine):
    expr1 = "(x + 1)**2"
    expr2 = "x**2 + 2*x + 1"
    expr3 = "x**2 + 2*x + 2" # Not equivalent
    
    scenarios = {
        "s1": {"x": 1},
        "s2": {"x": 10},
        "s3": {"x": -5},
    }
    
    # Should be equivalent in all scenarios
    results_equiv = engine.check_equivalence_in_scenarios(expr1, expr2, scenarios)
    assert all(results_equiv.values())
    
    # Should fail in all scenarios
    results_diff = engine.check_equivalence_in_scenarios(expr1, expr3, scenarios)
    assert not any(results_diff.values())
