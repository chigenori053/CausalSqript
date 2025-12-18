"""LaTeX Formatter for Coherent output."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .symbolic_engine import SymbolicEngine
    from .proof_engine import Step

    from .classifier import ExpressionClassifier


class LaTeXFormatter:
    """
    Formats mathematical expressions and proofs into LaTeX.
    """

    def __init__(self, symbolic_engine: SymbolicEngine, classifier: ExpressionClassifier | None = None):
        self.symbolic_engine = symbolic_engine
        self.classifier = classifier

    def format_expression(self, expr: str) -> str:
        """
        Convert a single expression string to LaTeX.
        """
        domains = []
        if self.classifier:
            domains = self.classifier.classify(expr)
        return self.symbolic_engine.to_latex(expr, context_domains=domains)

    def format_step(self, step_index: int, expr: str, explanation: str) -> str:
        """
        Format a single solution step.
        
        Returns:
            A LaTeX string suitable for inclusion in an itemize/enumerate list or standalone.
        """
        latex_expr = self.format_expression(expr)
        # Using a generic format that can be adjusted based on UI needs
        return f"\\textbf{{Step {step_index}}}: ${latex_expr}$ \\\\ \\textit{{{explanation}}}"

    def format_proof(self, proof_steps: List[Step]) -> str:
        """
        Format a geometric proof into a LaTeX list.
        """
        lines = ["\\begin{itemize}"]
        for step in proof_steps:
            # Format the fact
            # Assuming Fact args are simple strings for now.
            # If predicates are like "SideEqual", we might want "SideEqual(A, B)" -> "AB = DE" eventually.
            # For now, generic representation.
            args_str = ", ".join(step.fact.args)
            fact_latex = f"\\text{{{step.fact.predicate}}}({args_str})"
            
            # Format the reason
            if not step.precedents:
                reason = "Given"
            else:
                reason = f"Derived via \\textbf{{{step.rule_name}}}"
            
            lines.append(f"  \\item ${fact_latex}$ \\\\ \\textit{{{reason}}}")
        lines.append("\\end{itemize}")
        return "\n".join(lines)
