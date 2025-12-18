import pytest
sympy = pytest.importorskip("sympy")
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.latex_formatter import LaTeXFormatter
from coherent.engine.proof_engine import Fact, Step

def test_format_expression():
    sym_engine = SymbolicEngine()
    formatter = LaTeXFormatter(sym_engine)
    
    assert formatter.format_expression("x^2") == "x^{2}"
    if sym_engine.has_sympy():
        assert formatter.format_expression("1/2") == r"\frac{1}{2}"
        # Ensure exponents are preserved and not concatenated (e.g., 2**3 -> 2^{3}, not "23").
        assert formatter.format_expression("2**3") == "2^{3}"
        # Negative multiplication should keep the sign visible.
        latex_neg = formatter.format_expression("-2*x")
        assert "-" in latex_neg and ("2" in latex_neg or r"2" in latex_neg)

def test_format_step():
    sym_engine = SymbolicEngine()
    formatter = LaTeXFormatter(sym_engine)
    
    output = formatter.format_step(1, "x + 5 = 10", "Subtract 5")
    assert "Step 1" in output
    assert "x + 5 = 10" in output or "x + 5 = 10" in output # SymPy might reorder
    assert "Subtract 5" in output

def test_format_proof():
    sym_engine = SymbolicEngine()
    formatter = LaTeXFormatter(sym_engine)
    
    # Mock steps
    f1 = Fact("Equal", ("A", "B"))
    s1 = Step(f1, "Given", [])
    
    f2 = Fact("Equal", ("B", "C"))
    s2 = Step(f2, "Given", [])
    
    f3 = Fact("Equal", ("A", "C"))
    s3 = Step(f3, "Transitive", [f1, f2])
    
    proof = [s1, s2, s3]
    
    latex = formatter.format_proof(proof)
    
    assert "\\begin{itemize}" in latex
    assert "\\end{itemize}" in latex
    assert "\\text{Equal}(A, B)" in latex
    assert "Given" in latex
    assert "Derived via \\textbf{Transitive}" in latex
