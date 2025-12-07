from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.fuzzy.judge import FuzzyJudge
from causalscript.core.fuzzy.encoder import ExpressionEncoder
from causalscript.core.fuzzy.metric import SimilarityMetric
from causalscript.core.fuzzy.config import FuzzyThresholdConfig
from causalscript.core.decision_theory import DecisionConfig

def test_power_rule_error():
    engine = SymbolicEngine()
    if not engine.has_sympy():
        print("Skipping test: SymPy not available")
        return

    judge = FuzzyJudge(
        encoder=ExpressionEncoder(),
        metric=SimilarityMetric(),
        thresholds=FuzzyThresholdConfig(),
        decision_config=DecisionConfig(),
        symbolic_engine=engine
    )
    
    # Case 1: Missing Division
    # Problem: Integral(3*x^2) -> x^3
    # User writes: 3*x^3 (Forgot to divide by 3, or kept the constant 3 and multiplied by x^3?)
    # Correct: x^3
    # Candidate: 3*x^3
    # Ratio: 3
    
    problem_raw = "Integral(3*x**2, (x, 0, 2))"
    candidate_raw = "3*x**3"
    
    print(f"Testing Case 1: Problem={problem_raw}, Candidate={candidate_raw}")
    
    result = judge.judge_step(
        problem_expr={"raw": problem_raw, "sympy": problem_raw, "tokens": []},
        previous_expr={"raw": problem_raw, "sympy": problem_raw, "tokens": []}, # Dummy
        candidate_expr={"raw": candidate_raw, "sympy": candidate_raw, "tokens": []}
    )
    
    print(f"Result Label: {result['label']}")
    print(f"Result Reason: {result['reason']}")
    
    assert "divide by the new exponent" in result['reason']
    assert "Ratio: 3" in result['reason']
    
    # Case 2: Forgot to Integrate
    # Problem: Integral(x^2)
    # User writes: x^2
    
    problem_raw_2 = "Integral(x**2, x)"
    candidate_raw_2 = "x**2"
    
    print(f"\nTesting Case 2: Problem={problem_raw_2}, Candidate={candidate_raw_2}")
    
    result_2 = judge.judge_step(
        problem_expr={"raw": problem_raw_2, "sympy": problem_raw_2, "tokens": []},
        previous_expr={"raw": problem_raw_2, "sympy": problem_raw_2, "tokens": []},
        candidate_expr={"raw": candidate_raw_2, "sympy": candidate_raw_2, "tokens": []}
    )
    
    print(f"Result Label: {result_2['label']}")
    print(f"Result Reason: {result_2['reason']}")
    
    assert "wrote the integrand but didn't integrate it" in result_2['reason']

if __name__ == "__main__":
    try:
        test_power_rule_error()
        print("\nTest Passed!")
    except AssertionError as e:
        print(f"\nTest Failed: {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
