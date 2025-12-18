"""Arithmetic Engine for symbolic manipulation of non-fractional expressions."""

from __future__ import annotations

from typing import Dict, List

from . import ast_nodes as ast

class ArithmeticEngine:
    """Handles symbolic manipulation of arithmetic expressions."""

    def normalize(self, expression: ast.Expr) -> ast.Expr:
        """
        Normalizes an arithmetic expression based on v2.3 rules.
        - Flattens nested Add/Mul
        - Folds constants (e.g., 2+3=5)
        - Removes identities (e.g., a+0=a, a*1=a)
        - Handles zero (e.g., a*0=0)
        - Combines like terms (e.g., x+x=2*x)
        - Sorts terms and factors
        - Converts division to multiplication by a reciprocal
        """
        if isinstance(expression, (ast.Int, ast.Sym)):
            return expression

        if isinstance(expression, ast.Neg):
            expr = self.normalize(expression.expr)
            if isinstance(expr, ast.Neg): # -(-a) -> a
                return expr.expr
            if isinstance(expr, ast.Int): # -(5) -> -5
                return ast.Int(value=-expr.value)
            return ast.Neg(expr=expr)

        if isinstance(expression, ast.Div):
            # a/b -> a * b^-1
            left = self.normalize(expression.left)
            right = self.normalize(expression.right)
            return self.normalize(ast.Mul(factors=[left, ast.Pow(base=right, exp=ast.Int(-1))]))

        if isinstance(expression, ast.Add):
            # Flatten and normalize terms
            terms: List[ast.Expr] = []
            for term in expression.terms:
                normalized_term = self.normalize(term)
                if isinstance(normalized_term, ast.Add):
                    terms.extend(normalized_term.terms)
                else:
                    terms.append(normalized_term)

            # Remove identity element +0
            terms = [t for t in terms if not (isinstance(t, ast.Int) and t.value == 0)]

            # Fold constants
            constants = [t.value for t in terms if isinstance(t, ast.Int)]
            non_constants = [t for t in terms if not isinstance(t, ast.Int)]
            
            const_sum = sum(constants)
            
            # Combine like terms (e.g., x + x -> 2*x)
            term_counts: Dict[str, int] = {}
            processed_terms: Dict[str, ast.Expr] = {}
            for term in non_constants:
                # Handle Neg terms, e.g., a + (-b)
                is_neg = isinstance(term, ast.Neg)
                base_term = term.expr if is_neg else term
                key = str(base_term)
                term_counts[key] = term_counts.get(key, 0) + (-1 if is_neg else 1)
                processed_terms[key] = base_term

            new_terms: List[ast.Expr] = []
            for key, count in term_counts.items():
                if count == 0:
                    continue
                term = processed_terms[key]
                if count == 1:
                    new_terms.append(term)
                elif count == -1:
                    new_terms.append(ast.Neg(term))
                else:
                    new_terms.append(self.normalize(ast.Mul(factors=[ast.Int(count), term])))

            if const_sum != 0 or not new_terms:
                new_terms.append(ast.Int(const_sum))

            if not new_terms:
                return ast.Int(0)
            if len(new_terms) == 1:
                return new_terms[0]

            new_terms.sort(key=_get_sort_key)
            return ast.Add(terms=new_terms)

        if isinstance(expression, ast.Mul):
            # Flatten and normalize factors
            factors: List[ast.Expr] = []
            for factor in expression.factors:
                normalized_factor = self.normalize(factor)
                if isinstance(normalized_factor, ast.Mul):
                    factors.extend(normalized_factor.factors)
                else:
                    factors.append(normalized_factor)

            # Handle zero element a*0 -> 0
            if any(isinstance(f, ast.Int) and f.value == 0 for f in factors):
                return ast.Int(0)

            # Remove identity element *1
            factors = [f for f in factors if not (isinstance(f, ast.Int) and f.value == 1)]

            # Fold constants
            constants = [f.value for f in factors if isinstance(f, ast.Int)]
            new_factors = [f for f in factors if not isinstance(f, ast.Int)]
            
            product = 1
            for c in constants:
                product *= c
            
            if product != 1 or not new_factors:
                new_factors.append(ast.Int(product))

            if not new_factors:
                return ast.Int(1)
            if len(new_factors) == 1:
                return new_factors[0]

            new_factors.sort(key=_get_sort_key)
            return ast.Mul(factors=new_factors)

        if isinstance(expression, ast.Pow):
            base = self.normalize(expression.base)
            exp = self.normalize(expression.exp)
            if isinstance(base, ast.Int) and isinstance(exp, ast.Int):
                if exp.value >= 0:
                    return ast.Int(value=base.value ** exp.value)
                # Negative exponents are handled by FractionEngine, but we can simplify if base is 1
                if base.value == 1:
                    return ast.Int(1)
            if isinstance(exp, ast.Int) and exp.value == 1:
                return base
            if isinstance(exp, ast.Int) and exp.value == 0:
                return ast.Int(1)
            return ast.Pow(base=base, exp=exp)

        if isinstance(expression, ast.RationalNode):
            # This engine does not handle rationals, but must recurse.
            return ast.RationalNode(
                numerator=self.normalize(expression.numerator),
                denominator=self.normalize(expression.denominator)
            )

        return expression

def _get_sort_key(expr: ast.Expr) -> tuple[int, int | str]:
    """Returns a sort key for an expression. Constants, then symbols."""
    if isinstance(expr, ast.Int):
        return (0, expr.value)
    if isinstance(expr, ast.Sym):
        return (1, expr.name)
    return (2, "")
