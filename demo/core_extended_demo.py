#!/usr/bin/env python3
"""
MathLang Core Extended Demo

Demonstrates the new Core Extended features:
- ComputationEngine: Symbolic and numeric computation
- ValidationEngine: Answer validation with multiple modes
- HintEngine: Intelligent hint generation
- CoreRuntime: Orchestrated problem-solving workflow
"""

from pathlib import Path
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.exercise_spec import ExerciseSpec, load_exercise_spec


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_computation_engine():
    """Demonstrate ComputationEngine capabilities."""
    print_header("1. ComputationEngine Demo")
    
    symbolic = SymbolicEngine()
    engine = ComputationEngine(symbolic)
    
    # Simplification
    print("Simplification:")
    expr = "x + x + x"
    result = engine.simplify(expr)
    print(f"  {expr} → {result}")
    
    expr = "x**2 - 2*x + 1"
    result = engine.simplify(expr)
    print(f"  {expr} → {result}")
    
    # Expansion
    print("\nExpansion:")
    expr = "(x + 1)**2"
    result = engine.expand(expr)
    print(f"  {expr} → {result}")
    
    expr = "(x - 3)*(x + 3)"
    result = engine.expand(expr)
    print(f"  {expr} → {result}")
    
    # Factoring
    print("\nFactoring:")
    expr = "x**2 - 9"
    result = engine.factor(expr)
    print(f"  {expr} → {result}")
    
    # Numeric evaluation
    print("\nNumeric Evaluation:")
    engine.bind("x", 5)
    expr = "x**2 + 2*x + 1"
    result = engine.numeric_eval(expr)
    print(f"  {expr} (with x=5) → {result}")


def demo_validation_engine():
    """Demonstrate ValidationEngine with different modes."""
    print_header("2. ValidationEngine Demo")
    
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    validator = ValidationEngine(computation)
    
    # Symbolic equivalence
    print("Mode: Symbolic Equivalence")
    spec = ExerciseSpec(
        id="test1",
        target_expression="x**2 + 2*x + 1",
        validation_mode="symbolic_equiv"
    )
    
    result = validator.check_answer("(x + 1)**2", spec)
    print(f"  Answer: (x + 1)**2")
    print(f"  Result: {'✓ Correct' if result.is_correct else '✗ Incorrect'}")
    print(f"  Message: {result.message}")
    
    result = validator.check_answer("x**2 + x + 1", spec)
    print(f"\n  Answer: x**2 + x + 1")
    print(f"  Result: {'✓ Correct' if result.is_correct else '✗ Incorrect'}")
    print(f"  Message: {result.message}")
    
    # Canonical form
    print("\nMode: Canonical Form")
    spec = ExerciseSpec(
        id="test2",
        target_expression="(x - 3)*(x + 3)",
        validation_mode="canonical_form",
        canonical_form="(x - 3)*(x + 3)"
    )
    
    result = validator.check_answer("x**2 - 9", spec)
    print(f"  Answer: x**2 - 9")
    print(f"  Result: {'✓ Correct' if result.is_correct else '✗ Incorrect'}")
    print(f"  Message: {result.message}")


def demo_hint_engine():
    """Demonstrate HintEngine with pattern matching and heuristics."""
    print_header("3. HintEngine Demo")
    
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    hint = HintEngine(computation)
    
    # Pattern matching
    print("Pattern Matching:")
    spec = ExerciseSpec(
        id="test1",
        target_expression="x**2 + 2*x + 1",
        hint_rules={
            "x**2 + 1": "Missing the middle term!",
            "x**2 + x + 1": "The coefficient of x should be 2"
        }
    )
    
    result = hint.generate_hint("x**2 + 1", spec)
    print(f"  User answer: x**2 + 1")
    print(f"  Hint: {result.message}")
    print(f"  Type: {result.hint_type}")
    
    # Sign error detection
    print("\nHeuristic: Sign Error Detection")
    spec = ExerciseSpec(
        id="test2",
        target_expression="x - 5"
    )
    
    result = hint.generate_hint("-x + 5", spec)
    print(f"  User answer: -x + 5")
    print(f"  Hint: {result.message}")
    print(f"  Type: {result.hint_type}")


def demo_core_runtime():
    """Demonstrate full problem lifecycle with CoreRuntime."""
    print_header("4. CoreRuntime Demo - Full Problem Lifecycle")
    
    # Setup engines
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    validation = ValidationEngine(computation)
    hint_engine = HintEngine(computation)
    
    # Load exercise from file
    exercises_dir = Path(__file__).parent / "exercises"
    spec_path = exercises_dir / "quadratic_expansion.yaml"
    
    print(f"Loading exercise from: {spec_path.name}")
    spec = load_exercise_spec(spec_path)
    print(f"  Problem: Expand (x + 1)^2")
    print(f"  Target: {spec.target_expression}")
    
    # Create runtime
    runtime = CoreRuntime(computation, validation, hint_engine, spec)
    
    # Simulate student attempts
    print("\nStudent Attempts:")
    
    attempts = [
        ("x**2 + 1", "Forgetting middle term"),
        ("x**2 + x + 1", "Wrong coefficient"),
        ("x**2 + 2*x + 1", "Correct answer")
    ]
    
    for i, (answer, description) in enumerate(attempts, 1):
        print(f"\n  Attempt {i}: {answer} ({description})")
        
        # Set initial state
        runtime.set("(x + 1)**2")
        
        # Validate answer
        result = runtime.finalize(answer)
        
        if result["valid"]:
            print(f"    ✓ Correct!")
        else:
            print(f"    ✗ Incorrect")
            if "hint" in result["details"]:
                print(f"    💡 Hint: {result['details']['hint']['message']}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  MathLang Core Extended Feature Demonstrations")
    print("=" * 70)
    
    demo_computation_engine()
    demo_validation_engine()
    demo_hint_engine()
    demo_core_runtime()
    
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
