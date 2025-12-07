from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.classifier import ExpressionClassifier
from causalscript.core.latex_formatter import LaTeXFormatter

def verify_rendering():
    engine = SymbolicEngine()
    classifier = ExpressionClassifier(engine)
    formatter = LaTeXFormatter(engine, classifier)
    
    test_cases = [
        ("2 * 3", "Arithmetic", r" \cdot "),
        ("2 * x", "Algebra", "2x"), # Implicit
        ("x * y", "Algebra", "xy"), # Implicit
        ("a * (b + c)", "Algebra", r"a \left(b + c\right)")
    ]
    
    print(f"{'Expression':<15} | {'Domain':<10} | {'LaTeX':<30}")
    print("-" * 60)
    
    for expr, expected_domain, expected_part in test_cases:
        latex = formatter.format_expression(expr)
        domains = classifier.classify(expr)
        domain_str = domains[0] if domains else "None"
        
        print(f"{expr:<15} | {domain_str:<10} | {latex:<30}")
        
        # Basic assertions
        if expected_domain == "Algebra":
            if r"\cdot" in latex:
                print(f"FAIL: Found dot in Algebra expression '{expr}' -> '{latex}'")
            else:
                print("PASS")
        elif expected_domain == "Arithmetic":
            if r"\cdot" not in latex:
                 print(f"FAIL: Missing dot in Arithmetic expression '{expr}' -> '{latex}'")
            else:
                print("PASS")

if __name__ == "__main__":
    verify_rendering()
