
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.classifier import ExpressionClassifier
from causalscript.core.latex_formatter import LaTeXFormatter
from causalscript.core.knowledge_registry import KnowledgeRegistry
from causalscript.core.core_runtime import CoreRuntime
from causalscript.core.computation_engine import ComputationEngine
from causalscript.core.validation_engine import ValidationEngine
from causalscript.core.hint_engine import HintEngine

def verify_rendering_optimization():
    print("\n--- Verifying Rendering Optimization ---")
    engine = SymbolicEngine()
    classifier = ExpressionClassifier(engine)
    formatter = LaTeXFormatter(engine, classifier)
    
    test_cases = [
        ("2 * 3", "Arithmetic (Explicit Mul)"),
        ("2 * x", "Algebra (Implicit Mul)"),
        ("integrate(x**2, x)", "Calculus (Integral)")
    ]
    
    for expr, desc in test_cases:
        latex = formatter.format_expression(expr)
        domains = classifier.classify(expr)
        print(f"Expr: {expr} ({desc})")
        print(f"  Domains: {domains}")
        print(f"  LaTeX: {latex}")
        
        # Verification logic
        if "algebra" in domains and "2 * x" in expr:
            if "\\cdot" not in latex and "2x" in latex.replace(" ", ""):
                print("  [PASS] Implicit multiplication used for Algebra.")
            else:
                print("  [FAIL] Implicit multiplication NOT used for Algebra.")
        elif "arithmetic" in domains and "2 * 3" in expr:
            if "\\cdot" in latex or "\\times" in latex: # Default mul symbol
                 print("  [PASS] Explicit multiplication used for Arithmetic.")
            else:
                 # Fallback might be just "6" if simplified, or "2*3"
                 pass

def verify_rule_optimization_potential():
    print("\n--- Verifying Rule Optimization Potential ---")
    engine = SymbolicEngine()
    base_path = Path(__file__).parent / "core" / "knowledge"
    registry = KnowledgeRegistry(base_path, engine)
    
    # We need a rule that is domain-specific or has different behavior based on domain.
    # Or simply check if we can pass domains.
    
    # Let's try to match a calculus rule
    before = "integrate(x**2, (x, 0, 2))"
    after = "Subs(integrate(x**2, x), x, 2) - Subs(integrate(x**2, x), x, 0)"
    
    # Without domain
    match_no_domain = registry.match(before, after)
    print(f"Match without domain: {match_no_domain.id if match_no_domain else 'None'}")
    
    # With domain
    match_with_domain = registry.match(before, after, context_domains=["calculus"])
    print(f"Match with domain=['calculus']: {match_with_domain.id if match_with_domain else 'None'}")
    
    if match_no_domain and match_with_domain:
        print("  [INFO] Rule matched in both cases (optimization might be priority-based or filtering).")
        
def check_core_runtime_integration():
    print("\n--- Checking CoreRuntime Integration ---")
    # Instantiate CoreRuntime and check if it has a classifier
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    val_engine = ValidationEngine(comp_engine)
    hint_engine = HintEngine(comp_engine)
    
    runtime = CoreRuntime(comp_engine, val_engine, hint_engine)
    
    if hasattr(runtime, "classifier"):
        print("  [PASS] CoreRuntime has 'classifier' attribute.")
    else:
        print("  [FAIL] CoreRuntime does NOT have 'classifier' attribute.")
        
    # Check if check_step uses classifier (static analysis of source code is better, but runtime check:)
    # We can't easily check internal method calls without mocking.
    # Rely on the attribute check for now.

if __name__ == "__main__":
    verify_rendering_optimization()
    verify_rule_optimization_potential()
    check_core_runtime_integration()
