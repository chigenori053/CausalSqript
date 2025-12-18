"""Demo script for ValidationEngine."""

from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.exercise_spec import ExerciseSpec


def main():
    """Demonstrate ValidationEngine functionality."""
    print("=" * 60)
    print("ValidationEngine Demo")
    print("=" * 60)
    
    # Initialize engines
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    validator = ValidationEngine(computation)
    
    # Example 1: Symbolic Equivalence
    print("\n1. Symbolic Equivalence Validation")
    print("-" * 60)
    spec1 = ExerciseSpec(
        id="algebra-01",
        target_expression="x**2 - 2*x*y + y**2",
        validation_mode="symbolic_equiv"
    )
    
    # Correct answer
    result = validator.check_answer("(x - y)**2", spec1)
    print(f"User answer: (x - y)**2")
    print(f"Target: {spec1.target_expression}")
    print(f"Correct: {result.is_correct}")
    print(f"Message: {result.message}")
    
    # Incorrect answer
    result = validator.check_answer("x**2 + y**2", spec1)
    print(f"\nUser answer: x**2 + y**2")
    print(f"Target: {spec1.target_expression}")
    print(f"Correct: {result.is_correct}")
    print(f"Message: {result.message}")
    if result.details:
        print(f"Details: {result.details}")
    
    # Example 2: Exact Form Validation
    print("\n\n2. Exact Form Validation")
    print("-" * 60)
    spec2 = ExerciseSpec(
        id="algebra-02",
        target_expression="x**2 + 2*x + 1",
        validation_mode="exact_form"
    )
    
    result = validator.check_answer("x**2 + 2*x + 1", spec2)
    print(f"User answer: x**2 + 2*x + 1")
    print(f"Target: {spec2.target_expression}")
    print(f"Correct: {result.is_correct}")
    print(f"Message: {result.message}")
    
    # Example 3: Canonical Form Validation
    print("\n\n3. Canonical Form Validation")
    print("-" * 60)
    spec3 = ExerciseSpec(
        id="algebra-03",
        target_expression="x**2 - y**2",
        validation_mode="canonical_form",
        canonical_form="(x - y)*(x + y)"
    )
    
    # Correct form
    result = validator.check_answer("(x - y)*(x + y)", spec3)
    print(f"User answer: (x - y)*(x + y)")
    print(f"Target: {spec3.target_expression}")
    print(f"Canonical form: {spec3.canonical_form}")
    print(f"Correct: {result.is_correct}")
    print(f"Message: {result.message}")
    
    # Wrong form but mathematically correct
    result = validator.check_answer("x**2 - y**2", spec3)
    print(f"\nUser answer: x**2 - y**2")
    print(f"Target: {spec3.target_expression}")
    print(f"Canonical form: {spec3.canonical_form}")
    print(f"Correct: {result.is_correct}")
    print(f"Message: {result.message}")
    
    # Example 4: Multiple equivalent forms
    print("\n\n4. Multiple Equivalent Forms")
    print("-" * 60)
    spec4 = ExerciseSpec(
        id="algebra-04",
        target_expression="2*x + 4",
        validation_mode="symbolic_equiv"
    )
    
    equivalent_forms = ["4 + 2*x", "2*(x + 2)", "x + x + 4"]
    for form in equivalent_forms:
        result = validator.check_answer(form, spec4)
        print(f"User answer: {form:15} -> Correct: {result.is_correct}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
