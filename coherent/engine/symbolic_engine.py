"""Symbolic manipulation helpers built on top of SymPy (with a fallback)."""

from __future__ import annotations

import ast as py_ast
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Sequence, Set, Optional

from .errors import InvalidExprError, EvaluationError
from .simple_algebra import SimpleAlgebra, _Polynomial
from .linear_algebra_engine import LinearAlgebraEngine

import math
import statistics

try:  # pragma: no cover - SymPy is an optional dependency at import time.
    import sympy as _sympy
except Exception:  # pragma: no cover
    _sympy = None


_SAMPLE_ASSIGNMENTS: Sequence[Dict[str, int]] = (
    {"x": -2, "y": 1, "z": 3, "a": 1},
    {"x": 0, "y": 0, "z": 0, "a": 2, "b": -1},
    {"x": 1, "y": 2, "z": -1, "b": 3, "c": 4},
    {"a": 2, "b": 5, "c": -3},
    {},
)

def _symbolic_mean(data):
    return sum(data) / len(data)

def _symbolic_median(data):
    sorted_data = sorted(data)
    n = len(data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2

def _symbolic_mode(data):
    # simplistic mode for verification
    return max(set(data), key=data.count)

class _FallbackEvaluator:
    """Very small arithmetic evaluator used when SymPy is unavailable."""

    def parse(self, expr: str) -> py_ast.AST:
        expr = expr.replace("^", "**")
        try:
            tree = py_ast.parse(expr, mode="eval")
        except SyntaxError as exc:  # pragma: no cover - Python syntax errors.
            raise InvalidExprError(str(exc)) from exc
        return tree

    def evaluate(self, expr: str | py_ast.AST, values: Dict[str, Any]) -> Any:
        if isinstance(expr, py_ast.AST):
             tree = expr
             # If it's a Module/Expression wrapper, get body
             if isinstance(tree, py_ast.Expression):
                 return self._eval_node(tree.body, values)
             if isinstance(tree, py_ast.Module):
                 # Assume single expression or last? Usually Module.body is list
                 # Fallback parser uses mode='eval' -> Expression
                 pass
             # Otherwise assume it's a node
             return self._eval_node(tree, values)
             
        tree = self.parse(expr)
        return self._eval_node(tree.body, values)

    def _is_matrix(self, val: Any) -> bool:
        return isinstance(val, list) and len(val) > 0 and isinstance(val[0], list)

    def _is_vector(self, val: Any) -> bool:
        return isinstance(val, list) and (not val or not isinstance(val[0], list))

    def _eval_node(self, node: py_ast.AST, values: Dict[str, Any]) -> Any:
        if isinstance(node, py_ast.List):
             return [self._eval_node(elt, values) for elt in node.elts]

        if isinstance(node, py_ast.Constant):
            return Fraction(node.value) if isinstance(node.value, int) else node.value
        if isinstance(node, py_ast.Name):
            if node.id == "pi":
                return math.pi
            if node.id == "e":
                return math.e
            val = values.get(node.id)
            if val is None:
                raise EvaluationError(f"Symbol '{node.id}' not defined.")
            return Fraction(val) if isinstance(val, int) else val
        if isinstance(node, py_ast.Call):
            if isinstance(node.func, py_ast.Name):
                arg = self._eval_node(node.args[0], values)
                if node.func.id == "sin":
                    return math.sin(arg)
                if node.func.id == "cos":
                    return math.cos(arg)
                if node.func.id == "tan":
                    return math.tan(arg)
                if node.func.id == "sqrt":
                    return math.sqrt(arg)
            raise InvalidExprError(f"Unsupported function call: {py_ast.dump(node)}")
        if isinstance(node, py_ast.BinOp):
            left = self._eval_node(node.left, values)
            right = self._eval_node(node.right, values)
            if isinstance(node.op, py_ast.Add):
                if isinstance(left, list) and isinstance(right, list):
                     la = LinearAlgebraEngine()
                     if self._is_matrix(left): return la.matrix_add(left, right)
                     return la.vector_add(left, right)
                return left + right
            if isinstance(node.op, py_ast.Sub):
                if isinstance(left, list) and isinstance(right, list):
                     la = LinearAlgebraEngine()
                     # Reuse matrix_add logic for sub? LinearAlgebraEngine has vector_subtract but maybe not matrix_subtract?
                     # It has vector_subtract. Let's check matrix_subtract.
                     # It does not have matrix_subtract explicitly in the file view I saw?
                     # Checked file ID 9: matrix_add exists. matrix_subtract DOES NOT exist.
                     # Implementation plan didn't catch this. I should implement matrix subtraction using scalar mult -1 + add, or update LA engine.
                     # Updating LA engine is better, but for now I can do element-wise sub here or use scalar mul.
                     if self._is_matrix(left):
                         neg_right = la.scalar_multiply(-1, [val for row in right for val in row]) # Flatten/Unflatten tricky
                         # Easier: manual loop for fallback
                         return [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(left, right)]
                     return la.vector_subtract(left, right)
                return left - right
            if isinstance(node.op, py_ast.Mult):
                if isinstance(left, list) and isinstance(right, list):
                     la = LinearAlgebraEngine()
                     if self._is_matrix(left) and self._is_matrix(right):
                         return la.matrix_multiply(left, right)
                     # Vector dot product? Or elementwise?
                     # Standard math: dot product for vectors often uses dot() function, * might be ambiguous.
                     # But typically v * w is not valid unless definition exists. 
                     # Let's assume dot product for vectors if 1D? Or error?
                     # The user prompt A * B implies matrix multiplication.
                     pass 
                
                # Scalar multiplication
                if isinstance(left, (int, float, Fraction)) and isinstance(right, list):
                     la = LinearAlgebraEngine()
                     if self._is_matrix(right):
                         return [[float(left) * val for val in row] for row in right]
                     return la.scalar_multiply(float(left), right)
                if isinstance(right, (int, float, Fraction)) and isinstance(left, list):
                     la = LinearAlgebraEngine()
                     if self._is_matrix(left):
                         return [[val * float(right) for val in row] for row in left]
                     return la.scalar_multiply(float(right), left)

                return left * right
            if isinstance(node.op, py_ast.Div):
                if right == 0:
                    raise InvalidExprError("Division by zero")
                return left / right
            if isinstance(node.op, py_ast.Pow):
                # Handle Fraction ** Fraction if possible, else float
                try:
                    return left ** right
                except Exception:
                    return float(left) ** float(right)
        if isinstance(node, py_ast.UnaryOp):
            operand = self._eval_node(node.operand, values)
            if isinstance(node.op, py_ast.UAdd):
                return operand
            if isinstance(node.op, py_ast.USub):
                return -operand
        raise InvalidExprError(f"Unsupported expression: {py_ast.dump(node)}")

    def symbols(self, expr: str) -> Set[str]:
        tree = self.parse(expr)
        return {node.id for node in py_ast.walk(tree) if isinstance(node, py_ast.Name) and node.id not in ("pi", "e")}

    def is_subexpression(self, sub_expr: str, full_expr: str) -> bool:
        """Check if sub_expr AST is contained within full_expr AST."""
        try:
            sub_tree = self.parse(sub_expr)
            full_tree = self.parse(full_expr)
            
            # Extract the expression part from the 'eval' mode wrapper
            sub_node = sub_tree.body
            
            # Simple recursive check
            for node in py_ast.walk(full_tree):
                if self._nodes_equal(sub_node, node):
                    return True
            return False
        except Exception:
            return False

    def _nodes_equal(self, node1: py_ast.AST, node2: py_ast.AST) -> bool:
        if type(node1) is not type(node2):
            return False
        if isinstance(node1, py_ast.Name):
            return node1.id == node2.id # type: ignore
        if isinstance(node1, py_ast.Constant):
            return node1.value == node2.value # type: ignore
        if isinstance(node1, py_ast.BinOp):
            return (self._nodes_equal(node1.left, node2.left) and # type: ignore
                    self._nodes_equal(node1.right, node2.right) and # type: ignore
                    type(node1.op) is type(node2.op)) # type: ignore
        if isinstance(node1, py_ast.UnaryOp):
            return (self._nodes_equal(node1.operand, node2.operand) and # type: ignore
                    type(node1.op) is type(node2.op)) # type: ignore
        if isinstance(node1, py_ast.Call):
             # Simplified call check
             if not self._nodes_equal(node1.func, node2.func): # type: ignore
                 return False
             if len(node1.args) != len(node2.args): # type: ignore
                 return False
             return all(self._nodes_equal(a1, a2) for a1, a2 in zip(node1.args, node2.args)) # type: ignore
        return False


from .math_category import MathCategory
from .symbolic_strategies import (
    SymbolicStrategy,
    ArithmeticStrategy,
    AlgebraStrategy,
    CalculusStrategy,
    LinearAlgebraStrategy,
    StatisticsStrategy,
    GeometryStrategy
)

@dataclass
class SymbolicEngine:
    """Thin wrapper providing equivalence and simplification utilities."""

    def __post_init__(self) -> None:
        self._fallback = _FallbackEvaluator() if _sympy is None else None
        
        # Initialize strategies
        self.strategies: Dict[MathCategory, SymbolicStrategy] = {
            MathCategory.ARITHMETIC: ArithmeticStrategy(self._fallback),
            MathCategory.ALGEBRA: AlgebraStrategy(self._fallback),
            MathCategory.CALCULUS: CalculusStrategy(self._fallback),
            MathCategory.LINEAR_ALGEBRA: LinearAlgebraStrategy(self._fallback),
            MathCategory.STATISTICS: StatisticsStrategy(self._fallback),
            MathCategory.GEOMETRY: GeometryStrategy(self._fallback),
        }
        self.active_categories: List[MathCategory] = []

    def has_sympy(self) -> bool:
        return self._fallback is None

    def to_string(self, value: Any, *, latex: bool = False) -> str:
        """
        Normalize values (especially fractions) into a consistent string form.
        - Fraction(3, 4) -> "3/4" or "\\frac{3}{4}" when latex=True
        - SymPy Rational -> same as above
        """
        try:
            if isinstance(value, Fraction):
                if latex:
                    return r"\frac{{{}}}{{{}}}".format(value.numerator, value.denominator)
                return f"{value.numerator}/{value.denominator}" if value.denominator != 1 else str(value.numerator)
            if _sympy and isinstance(value, _sympy.Rational):
                if latex:
                    return _sympy.latex(value)
                return f"{value.p}/{value.q}" if value.q != 1 else str(value.p)
        except Exception:
            pass
        return str(value)

    def is_numeric(self, expr: Any) -> bool:
        """Checks if the expression is purely numeric (no free symbols)."""
        if self._fallback is not None:
            # Fallback: check if symbols() returns empty set
            if isinstance(expr, str):
                return not bool(self._fallback.symbols(expr))
            return True # Assume non-string is numeric value
            
        try:
            # If it's a string, convert to internal first
            if isinstance(expr, str):
                # Avoid eager simplification which may remove symbols (e.g., -x*y + x*y -> 0).
                local_dict = {"e": _sympy.E, "pi": _sympy.pi, "integrate": _sympy.Integral}
                internal = _sympy.sympify(expr, locals=local_dict, evaluate=False)
            else:
                internal = expr
                
            # Check free symbols
            if hasattr(internal, "free_symbols"):
                return not bool(internal.free_symbols)
            return True # Constants/Numbers have no free symbols
        except Exception:
            return False

    def to_internal(self, expr: Any, *, extra_locals: Optional[Dict[str, Any]] = None) -> Any:
        # If it's already an internal type (SymPy object or AST), return it.
        if self._fallback is not None:
             if isinstance(expr, (py_ast.AST, list)): # list for matrix AST?
                 return expr
             if not isinstance(expr, str):
                 return expr # Assume it's a number or compatible type
             return self._fallback.parse(expr)
        
        # If we have SymPy
        if not isinstance(expr, str):
            # Assume it's already a SymPy object or compatible (int, float)
            return expr

        try:
            # Ensure 'e' is treated as Euler's number and 'pi' as pi
            # Map 'integrate' to 'Integral' to prevent eager evaluation (Late Evaluation Mode)
            # Map 'diff' to 'Derivative' for same reason
            local_dict = {
                "e": _sympy.E, 
                "pi": _sympy.pi, 
                "integrate": _sympy.Integral, 
                "Integral": _sympy.Integral, 
                "diff": _sympy.Derivative,
                "Derivative": _sympy.Derivative,
                "Subs": _sympy.Subs,
                "System": _sympy.FiniteSet, # Map System to FiniteSet
                "Eq": _sympy.Eq,
                "Matrix": _sympy.Matrix,
                "mean": _symbolic_mean,
                "median": _symbolic_median,
                "mode": _symbolic_mode
            }
            
            # Add Geometry classes if available
            if hasattr(_sympy, 'geometry'):
                geo_classes = ['Point', 'Line', 'Circle', 'Polygon', 'Segment', 'Ray', 'Triangle']
                for cls_name in geo_classes:
                    if hasattr(_sympy.geometry, cls_name):
                        local_dict[cls_name] = getattr(_sympy.geometry, cls_name)

            if extra_locals:
                # Safe conversion of context values to compatible types for SymPy
                safe_locals = {}
                for k, v in extra_locals.items():
                    if isinstance(v, str):
                        # Attempt to convert numeric strings to numbers to avoid SymPy eval errors
                        # e.g. "1" * x -> "1"*x which SymPy dislikes in eval if treating "1" as str
                        if v.isdigit():
                            safe_locals[k] = int(v)
                        else:
                            try:
                                safe_locals[k] = float(v)
                            except ValueError:
                                safe_locals[k] = v
                    else:
                        safe_locals[k] = v
                local_dict.update(safe_locals)
            
            # Normalize power symbol
            expr = expr.replace("^", "**")
            return _sympy.sympify(expr, locals=local_dict)
        except Exception as exc:  # pragma: no cover - SymPy provides details.
            raise InvalidExprError(str(exc)) from exc

    def set_context(self, categories: List[MathCategory]) -> None:
        """Set the active mathematical context to prioritize strategies."""
        self.active_categories = categories

    def is_equiv(self, expr1: str, expr2: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if two expressions are symbolically equivalent.
        """
        
        # 1. Try active strategies first
        for category in self.active_categories:
            strategy = self.strategies.get(category)
            if strategy:
                # Strategies typically handle bare strings, but we might want to pass context down?
                # For now, keep existing interface for strategies, they might simple-check
                result = strategy.is_equiv(expr1, expr2, self)
                if result is not None:
                    return result
                    
        # 2. Fallback to default logic (existing implementation)
        if self._fallback is not None:
            return self._fallback_is_equiv(expr1, expr2)
            
        try:
            # Pass context as locals to handle Matrix multiplication correctly
            internal1 = self.to_internal(expr1, extra_locals=context)
            internal2 = self.to_internal(expr2, extra_locals=context)
            
            if context and _sympy is not None:
                # Still check for symbols that might remain (if not in context)
                # But to_internal with extra_locals handles the Matrix case.
                pass

            if _sympy:
                # Automatic List-to-Matrix conversion if comparing against a Matrix
                is_mat1 = getattr(internal1, "is_Matrix", False)
                is_mat2 = getattr(internal2, "is_Matrix", False)

                if is_mat1 and isinstance(internal2, list):
                    try:
                        internal2 = _sympy.Matrix(internal2)
                    except Exception:
                        pass
                elif is_mat2 and isinstance(internal1, list):
                    try:
                        internal1 = _sympy.Matrix(internal1)
                    except Exception:
                        pass

            diff = _sympy.simplify(internal1 - internal2)
            
            # Check for zero matrix/vector
            if hasattr(diff, 'is_zero'): # Matrix/Vector
                if diff.is_zero:
                    return True
            elif diff == 0:
                return True
                
            # If simplify didn't strictly reduce to 0, try evalf if numeric?
            # Or Norm if matrix?
            if hasattr(diff, 'norm'):
                if diff.norm() == 0:
                    return True
                    
        except (TypeError, ValueError, AttributeError, Exception) as e:
            # If subtraction fails (e.g. FiniteSet - Equality), they are not equivalent in the standard sense.
            return False
            
        # Sampling fallback - typically for numeric scalars
        return self._numeric_sampling_equiv(expr1, expr2)
    
    def is_subexpression(self, sub_expr: str, full_expr: str) -> bool:
        """Check if sub_expr is mathematically contained in full_expr."""
        if self._fallback is not None:
            return self._fallback.is_subexpression(sub_expr, full_expr)

        # Avoid SymPy eagerly simplifying away structure (e.g., sin(pi/3)**2 -> 3/4)
        # by parsing with evaluate=False first. Fall back to the standard parser and
        # a lightweight AST check if needed.
        try:
            from sympy.parsing.sympy_parser import parse_expr  # type: ignore

            local_dict = {"e": _sympy.E, "pi": _sympy.pi}
            internal_sub = parse_expr(sub_expr, evaluate=False, local_dict=local_dict)
            internal_full = parse_expr(full_expr, evaluate=False, local_dict=local_dict)
            if internal_full.has(internal_sub):
                return True
        except Exception:
            pass

        try:
            internal_sub = self.to_internal(sub_expr)
            internal_full = self.to_internal(full_expr)
            if internal_full.has(internal_sub):
                return True
        except Exception:
            pass

        try:
            return _FallbackEvaluator().is_subexpression(sub_expr, full_expr)
        except Exception:
            return False

    def replace(self, full_expr: str, target: str, replacement: str) -> str:
        """Replace target sub-expression with replacement in full_expr."""
        # Prefer structural string replacement to avoid SymPy auto-simplifying
        # the entire expression (e.g., collapsing (1+2)*(3+4) to 21).
        if full_expr.strip() == target.strip():
            return replacement

        # Naive textual replacement first; validated by attempting to parse later.
        candidate = full_expr.replace(target, f"({replacement})")
        try:
            self.to_internal(candidate)
            return candidate
        except Exception:
            pass

        if self._fallback is not None:
            return candidate

        try:
            internal_full = self.to_internal(full_expr)
            internal_target = self.to_internal(target)
            internal_replacement = self.to_internal(replacement)
            new_internal = internal_full.xreplace({internal_target: internal_replacement})
            return str(new_internal)
        except Exception:
            return candidate

    def _fallback_is_equiv(self, expr1: str, expr2: str) -> bool:
        assert self._fallback is not None
        symbols = self._fallback.symbols(expr1) | self._fallback.symbols(expr2)
        success = False
        for assignment in _SAMPLE_ASSIGNMENTS:
            subset = {name: assignment.get(name, 1) for name in symbols}
            try:
                left = self._fallback.evaluate(expr1, subset)
                right = self._fallback.evaluate(expr2, subset)
            except (InvalidExprError, EvaluationError):
                continue
            
            # Handle float comparison for trig functions
            if isinstance(left, float) or isinstance(right, float):
                if abs(float(left) - float(right)) > 1e-9:
                    return False
            elif left != right:
                return False
            success = True
        if not success:
            # If we couldn't evaluate (maybe due to missing symbols in sample), 
            # we can't be sure. But for now let's assume if it fails all samples it's bad.
            # Or maybe we just need more robust sampling.
            # For the user's case (sin(pi/3)), there are no variables, so it should run once and succeed.
            pass
            
        return success

    def _numeric_sampling_equiv(self, expr1: str, expr2: str) -> bool:
        try:
            internal1 = self.to_internal(expr1)
            internal2 = self.to_internal(expr2)
        except InvalidExprError:
            raise
        success = False
        for assignment in _SAMPLE_ASSIGNMENTS:
            try:
                subs1 = {sym: assignment.get(str(sym), 1) for sym in internal1.free_symbols}
                subs2 = {sym: assignment.get(str(sym), 1) for sym in internal2.free_symbols}
                val1 = internal1.subs(subs1)
                val2 = internal2.subs(subs2)
                if val1.free_symbols or val2.free_symbols:
                    continue
                if val1.evalf() != val2.evalf():
                    return False
                success = True
            except Exception:
                continue
        return success

    def simplify(self, expr: str) -> str:
        # 1. Try active strategies
        for category in self.active_categories:
            strategy = self.strategies.get(category)
            if strategy:
                result = strategy.simplify(expr, self)
                if result is not None:
                    return result

        # 2. Fallback
        if self._fallback is not None:
            try:
                return SimpleAlgebra.simplify(expr)
            except InvalidExprError:
                pass
            try:
                value = self._fallback.evaluate(expr, {})
                if isinstance(value, Fraction):
                    if value.denominator == 1:
                        return str(value.numerator)
                    return f"{value.numerator}/{value.denominator}"
                return str(value)
            except (InvalidExprError, EvaluationError):
                return expr
        internal = self.to_internal(expr)
        return self.to_string(_sympy.simplify(internal))

    def to_latex(self, expr: str, context_domains: list[str] | None = None) -> str:
        """
        Convert an expression to its LaTeX representation.
        
        Args:
            expr: The expression string to convert.
            context_domains: List of domains (e.g., ["algebra", "arithmetic"]) to guide rendering.
            
        Returns:
            LaTeX string representation.
        """
        # 1. Try active strategies (or context_domains if provided)
        # We prioritize active_categories if context_domains is not passed, 
        # or we can merge them. For now, let's use active_categories if context_domains is None.
        
        # Note: context_domains is legacy string list, active_categories is Enum list.
        # We might want to align them.
        
        for category in self.active_categories:
            strategy = self.strategies.get(category)
            if strategy:
                result = strategy.to_latex(expr, self)
                if result is not None:
                    return result

        # Determine multiplication symbol based on context
        mul_symbol = r" \cdot " # Default for arithmetic or unknown
        if context_domains and "algebra" in context_domains:
            # Implicit multiplication for algebra
            mul_symbol = ""
            
        if self._fallback is not None:
            # Fallback: basic conversion (e.g., * to \cdot, / to \frac)
            # This is a very simple approximation
            latex = expr.replace("*", mul_symbol if mul_symbol else "").replace("pi", r"\pi")
            # Handle simple fractions if possible, but regex is tricky.
            # For now, return the expression with minor tweaks.
            return latex
        
        try:
            from sympy.parsing.sympy_parser import parse_expr
            local_dict = {"e": _sympy.E, "pi": _sympy.pi, "integrate": _sympy.Integral}
            # Normalize power symbol
            expr_norm = expr.replace("^", "**")
            internal = parse_expr(expr_norm, evaluate=False, local_dict=local_dict)
            latex = _sympy.latex(internal, mul_symbol=mul_symbol)
            
            # Clean up artifacts from evaluate=False
            # 1. "1 \cdot " (e.g. 1/2 -> 1 * 1/2)
            if mul_symbol.strip():
                latex = latex.replace(f"1{mul_symbol}", "")
            else:
                # If implicit, it might look like "1 x" -> "x"
                # But sympy usually handles 1*x -> x well.
                # Check for "1 " at start?
                pass
            
            # 2. Replace "\left(-1\right) \cdot" with "-" (e.g. (-1)*0 -> -0)
            if mul_symbol.strip():
                 latex = latex.replace(rf"\left(-1\right){mul_symbol}", "-")
            
            # 3. Handle "+ -" -> "-" (e.g. + -0 -> - 0)
            latex = latex.replace(r"+ -", "- ")
            
            return latex
        except Exception:
            return expr

    def evaluate(self, expr: Any, context: Dict[str, Any]) -> Any:
        print(f"DEBUG: evaluate called for {expr} with context {context}, fallback={self._fallback}")
        if self._fallback is not None:
            symbols = self._fallback.symbols(expr)
            if not context and symbols:
                return {"not_evaluatable": True}
            return self._fallback.evaluate(expr, context)

        internal_expr = self.to_internal(expr, extra_locals=context)

        # Handle primitives (int, float) or other objects without free_symbols
        if not hasattr(internal_expr, 'free_symbols'):
             # It's likely a constant or fully evaluated result
             return internal_expr

        free_symbols = internal_expr.free_symbols
        undefined_symbols = [s for s in free_symbols if str(s) not in context]

        # MODIFIED: Try symbolic evaluation (doit) even if context is missing
        # This allows computing derivatives/integrals etc.
        subs = {s: context.get(str(s)) for s in free_symbols if str(s) in context}

        try:
            result = internal_expr.subs(subs)
            
            # If we still have free symbols, try explicit evaluation (doit)
            # e.g. Derivative(x**2, x) -> 2*x
            if hasattr(result, "doit"):
                result = result.doit()
                
            # If it's still containing symbols but that's expected (symbolic calc), return it.
            # But the contract might be "return number if possible".
            # If result still has integrals/derivatives that couldn't be done, maybe we keep it.
            
            # For consistency with existing tests that might expect numbers:
            # The failing test expects '3*x**2'. So returning symbolic is DESIRED.
            
            # We return Python numbers if it fully simplifies to one.
            if not result.free_symbols:
                simplified = _sympy.simplify(result)
                python_val = self._to_python_number(simplified)
                # If it's a symbolic constant (pi, E), _to_python_number keeps it. good.
                return python_val
            
            # If context was provided but we still have symbols (because we didn't provide all?)
            # The original logic blocked this.
            # "undefined_symbols" check was strict.
            # But for calculus, we might only provide SOME symbols or NONE.
            
            # Let's return the symbolic result if it looks significantly evaluated?
            # Or just return it always?
            # The previous logic returned {"not_evaluatable": True}.
            # Let's return the internal result (Symbolic object), and let to_python_number handle it if needed?
            # But normally evaluate returns primitives or SymPy objects.
            
            return result

        except Exception as exc:
            raise EvaluationError(f"Failed to evaluate expression: {exc}")

    def substitute(self, expr: str, context: Dict[str, Any]) -> str:
        """Substitute variables in expr using context, without full evaluation."""
        if self._fallback is not None:
             # Fallback: naive string replacement for now
             # This is risky but better than nothing for fallback
             result = expr
             for k, v in context.items():
                 result = result.replace(k, str(v))
             return result

        try:
            # Use parse_expr with evaluate=False to prevent auto-simplification
            from sympy.parsing.sympy_parser import parse_expr # type: ignore
            local_dict = {"e": _sympy.E, "pi": _sympy.pi, "integrate": _sympy.Integral}
            
            # Parse expr
            internal = parse_expr(expr, evaluate=False, local_dict=local_dict)
            
            # Parse context values
            subs = {}
            for k, v in context.items():
                val_internal = parse_expr(str(v), evaluate=False, local_dict=local_dict)
                subs[k] = val_internal
                # Also handle symbol objects if keys are strings
                subs[_sympy.Symbol(k)] = val_internal

            new_internal = internal.xreplace(subs)
            return str(new_internal)
        except Exception:
            return expr


    def evaluate_numeric(self, expr: str, assignment: Dict[str, int]) -> Any:
        if self._fallback is not None:
            return self._fallback.evaluate(expr, assignment)
        internal = self.to_internal(expr)
        subs = {symbol: assignment.get(str(symbol), 0) for symbol in internal.free_symbols}
        return internal.subs(subs)

    def explain(self, before: str, after: str) -> str:
        try:
            if self.is_equiv(before, after):
                return "Expressions are equivalent."
        except EvaluationError:
            pass # Fallback to simplification comparison
            
        simplified_before = self.simplify(before)
        simplified_after = self.simplify(after)
        hint = "sympy" if self._fallback is None else "numeric sampling"
        return f"Compared via {hint}: {simplified_before} → {simplified_after}."

    def _to_python_number(self, value: Any) -> Any:
        print(f"DEBUG: _to_python_number called with {value} type {type(value)}")
        if isinstance(value, int):
            return value

        is_integer_attr = getattr(value, "is_integer", None)
        is_integer = False
        if callable(is_integer_attr):
            try:
                is_integer = bool(is_integer_attr())
            except Exception:
                is_integer = False
        elif is_integer_attr is not None:
            is_integer = bool(is_integer_attr)
        if is_integer:
            try:
                return int(value)
            except Exception:
                pass

        is_rational_attr = getattr(value, "is_rational", None)
        is_rational = False
        if callable(is_rational_attr):
            try:
                is_rational = bool(is_rational_attr())
            except Exception:
                is_rational = False
        elif is_rational_attr is not None:
            is_rational = bool(is_rational_attr)
        if is_rational:
            try:
                numer, denom = value.as_numer_denom()
                numer = int(numer)
                denom = int(denom)
                return Fraction(numer, denom)
            except Exception:
                pass

        if _sympy is not None:
             # print(f"DEBUG: checking value {value} type {type(value)}")
             if value.has(_sympy.pi) or value.has(_sympy.E):
                 print(f"DEBUG: Keeping symbolic value: {value}")
                 return value
        else:
             print("DEBUG: _sympy is None")

        try:
            return float(value)
        except Exception:
            return value

    def match_structure(self, expression: str, pattern: str) -> Dict[str, Any] | None:
        """
        Matches an expression against a pattern using SymPy's unification.
        Returns a dictionary of bindings if the structure matches, or None.
        """
        if self._fallback is not None:
            return None 

        try:
            from sympy import Wild
            from sympy.parsing.sympy_parser import parse_expr
            
            local_dict = {"e": _sympy.E, "pi": _sympy.pi, "integrate": _sympy.Integral, "Integral": _sympy.Integral, "Subs": _sympy.Subs, "Eq": _sympy.Eq}
            
            # 1. Parse the concrete expression
            # Use evaluate=True to allow basic simplification like (x-y)*(x-y) -> (x-y)**2
            # This is crucial for matching a*a against (x-y)^2 if pattern is a**2,
            # OR matching a**2 against (x-y)*(x-y).
            # However, we want to preserve structure.
            # The issue is that (x-y)*(x-y) IS (x-y)**2 mathematically, but structurally it's Mul vs Pow.
            # If the rule is a**2 -> a*a, we expect:
            # Before: (x-y)**2 (Pow) matches a**2 (Pow) -> a=(x-y)
            # After: (x-y)*(x-y) (Mul) matches a*a (Mul) -> a=(x-y)
            
            # But SymPy parses (x-y)*(x-y) as (x-y)**2 automatically even with evaluate=False?
            # No, evaluate=False should preserve it.
            # Let's try parsing with evaluate=False.
            # Normalize power symbol
            expression = expression.replace("^", "**")
            pattern = pattern.replace("^", "**")
            expr_internal = parse_expr(expression, evaluate=False, local_dict=local_dict)
            
            # 2. Prepare the pattern with Wild symbols
            wild_names = ['a', 'b', 'c', 'd', 'f', 'g', 'h', 'n', 'm', 'x', 'y', 'z']
            pattern_locals = local_dict.copy()
            
            for name in wild_names:
                # exclude=[] allows matching anything
                w = Wild(name, exclude=[]) 
                pattern_locals[name] = w
                
            pattern_internal = parse_expr(pattern, evaluate=False, local_dict=pattern_locals)
            
            # 3. Perform matching
            
            # Strict Structure Check:
            # If pattern is a specific operation (Add, Mul, Pow), the expression MUST be compatible.
            # Incompatible pairs: (Mul, Add), (Add, Mul), (Pow, Add), (Add, Pow).
            # Compatible: Same type, or one is Atom (Symbol, Number), or (Mul, Pow) pair.
            if not isinstance(pattern_internal, Wild):
                from sympy import Add, Mul, Pow, Atom
                
                p_type = type(pattern_internal)
                e_type = type(expr_internal)
                
                # If expression is Atom (e.g. Symbol, Number), it can match any operation (simplification)
                if isinstance(expr_internal, Atom):
                    pass
                elif p_type != e_type:
                    # Check for incompatible pairs
                    is_incompatible = False
                    
                    if isinstance(pattern_internal, Mul) and isinstance(expr_internal, Add):
                        is_incompatible = True
                    elif isinstance(pattern_internal, Add) and isinstance(expr_internal, Mul):
                        is_incompatible = True
                    elif isinstance(pattern_internal, Pow) and isinstance(expr_internal, Add):
                        is_incompatible = True
                    elif isinstance(pattern_internal, Add) and isinstance(expr_internal, Pow):
                        is_incompatible = True
                        
                    if is_incompatible:
                        # print(f"DEBUG: Structure mismatch. Expr: {e_type}, Pattern: {p_type}")
                        return None

            matches = expr_internal.match(pattern_internal)
            
            if matches is not None:
                # print(f"DEBUG: match_structure success. Expr: {expr_internal} ({type(expr_internal)}), Pattern: {pattern_internal} ({type(pattern_internal)})")
                return {k.name: v for k, v in matches.items()}
            
            # Fallback: Argument-wise matching for Mul
            # If both are Mul and have same number of args (e.g. 2), try matching args individually.
            # This handles (x+3)(x-3) vs (a+b)(a-b).
            from sympy import Mul
            if isinstance(expr_internal, Mul) and isinstance(pattern_internal, Mul):
                if len(expr_internal.args) == len(pattern_internal.args) == 2:
                    e1, e2 = expr_internal.args
                    p1, p2 = pattern_internal.args
                    
                    # Helper to merge bindings
                    def merge(b1, b2):
                        if b1 is None or b2 is None: return None
                        merged = b1.copy()
                        for k, v in b2.items():
                            if k in merged:
                                if merged[k] != v: return None # Conflict
                            else:
                                merged[k] = v
                        return merged

                    # Helper to validate binding against other pair
                    def validate(binding, expr, pattern):
                        if binding is None: return False
                        try:
                            subbed = pattern.subs(binding)
                            # print(f"DEBUG: Validate subbed={subbed}, expr={expr}, eq={subbed.equals(expr)}")
                            return subbed.equals(expr)
                        except Exception as e:
                            # print(f"DEBUG: Validate exception: {e}")
                            return False

                    # Try Permutation 1: e1->p1, e2->p2
                    m1 = e1.match(p1)
                    m2 = e2.match(p2)
                    merged1 = merge(m1, m2)
                    
                    if merged1 is not None:
                        return {k.name: v for k, v in merged1.items()}
                    
                    # Cross-validation for Permutation 1
                    # If merge failed, check if m1 works for (e2, p2) or m2 works for (e1, p1)
                    if m1 and validate(m1, e2, p2):
                        return {k.name: v for k, v in m1.items()}
                    if m2 and validate(m2, e1, p1):
                        return {k.name: v for k, v in m2.items()}
                        
                    # Try Permutation 2: e1->p2, e2->p1
                    m3 = e1.match(p2)
                    m4 = e2.match(p1)
                    merged2 = merge(m3, m4)
                    
                    if merged2 is not None:
                        return {k.name: v for k, v in merged2.items()}
                        
                    # Cross-validation for Permutation 2
                    if m3 and validate(m3, e2, p1):
                        return {k.name: v for k, v in m3.items()}
                    if m4 and validate(m4, e1, p2):
                        return {k.name: v for k, v in m4.items()}

            # Special handling for a*a pattern (Mul(a, a)) matching Pow(b, 2) or Mul(b, b)
            # This is needed because SymPy's match is strict about structure but we want to allow
            # a*a to match a^2 or (x-y)(x-y).
            from sympy import Pow
            if isinstance(pattern_internal, Mul) and len(pattern_internal.args) == 2 and pattern_internal.args[0] == pattern_internal.args[1]:
                # Pattern is a * a
                wild_a = pattern_internal.args[0]
                if isinstance(wild_a, Wild):
                    # Check if expression is Pow(b, 2)
                    if isinstance(expr_internal, Pow) and expr_internal.args[1] == 2:
                        return {wild_a.name: expr_internal.args[0]}
                    # Check if expression is Mul(b, b)
                    if isinstance(expr_internal, Mul) and len(expr_internal.args) == 2 and expr_internal.args[0] == expr_internal.args[1]:
                        return {wild_a.name: expr_internal.args[0]}
                        
            # Special handling for a**2 pattern matching Mul(b, b)
            if isinstance(pattern_internal, Pow) and pattern_internal.args[1] == 2:
                wild_a = pattern_internal.args[0]
                if isinstance(wild_a, Wild):
                     if isinstance(expr_internal, Mul) and len(expr_internal.args) == 2 and expr_internal.args[0] == expr_internal.args[1]:
                        return {wild_a.name: expr_internal.args[0]}

            return None
            
        except Exception:
            return None

    def get_top_operator(self, expr: str) -> str | None:
        """Returns the name of the top-level operator (Add, Mul, Pow, etc.)."""
        try:
            if self._fallback is not None:
                tree = self._fallback.parse(expr)
                node = tree.body
            else:
                # Use Python's AST for consistency and speed, as we just want structure
                tree = py_ast.parse(expr.replace("^", "**"), mode="eval")
                node = tree.body
                
            if isinstance(node, py_ast.BinOp):
                return type(node.op).__name__
            if isinstance(node, py_ast.UnaryOp):
                return type(node.op).__name__
            if isinstance(node, py_ast.Call):
                if isinstance(node.func, py_ast.Name):
                    name = node.func.id
                    if name == "integrate":
                        return "Integral"
                    return name
                return "Call"
            if isinstance(node, py_ast.Num) or isinstance(node, py_ast.Constant):
                return "Number"
            if isinstance(node, py_ast.Name):
                return "Symbol"
            return "Other"
        except Exception:
            return None

    def is_antiderivative(self, func_expr: str, integral_expr: str) -> bool:
        """
        Checks if func_expr is an antiderivative of the integrand in integral_expr.
        integral_expr must be a Definite Integral string, e.g., "Integral(f(x), (x, a, b))".
        """
        if self._fallback is not None:
             return False

        try:
             # Just checking if d(func_expr)/dx == integrand
             # But we need parsing.
             # This is a stub or simple implementation.
             pass
        except Exception:
             pass
        return False
        
    def is_implied_by_system(self, target: str, source: str) -> bool:
        """
        Check if the 'target' equation/expression is logically implied by the 'source'.
        Useful for checking if a solution (e.g. x=4) is valid given a system (e.g. {3x+2y=14, x-2y=2}).
        
        Args:
            target: The derived step (e.g. "x=4")
            source: The original system/expression (e.g. "System(Eq(...), Eq(...))")
            
        Returns:
            True if target is a necessary consequence of source.
        """
        # Delegate to LinearAlgebraStrategy if available
        # We try Linear Algebra strategy specifically as it handles systems.
        # Ideally we should detect category, but Implication is primarily a system property.
        
        la_strategy = self.strategies.get(MathCategory.LINEAR_ALGEBRA)
        if la_strategy:
             # This works for both SymPy and Fallback modes (if strategy supports fallback)
             result = la_strategy.check_implication(source, target, self)
             if result is not None:
                 return result
        
        # If strategy returns None or not found, try generic logic (if SymPy available)
        if self._fallback is not None:
            return False

        try:
            target_eqn = self.to_internal(target)
            source_sys = self.to_internal(source)
            
            # If source isn't a collection, wrap it or treat as single
            if not isinstance(source_sys, (_sympy.FiniteSet, list, tuple, set)):
                 source_sys = [source_sys]
            
            # Solve the source system
            solutions = _sympy.solve(source_sys, dict=True)
            
            if not solutions:
                return False

            for sol in solutions:
                if isinstance(target_eqn, _sympy.Eq):
                    lhs = target_eqn.lhs.subs(sol)
                    rhs = target_eqn.rhs.subs(sol)
                    if not _sympy.simplify(lhs - rhs) == 0:
                        return False
                else:
                    return False
                    
            return True
        except Exception:
            return False
        if self._fallback is not None:
            return False

        try:
            from sympy.parsing.sympy_parser import parse_expr
            local_dict = {"e": _sympy.E, "pi": _sympy.pi, "integrate": _sympy.Integral, "Integral": _sympy.Integral}
            
            # Parse expressions
            func = parse_expr(func_expr, local_dict=local_dict)
            integral = parse_expr(integral_expr, local_dict=local_dict)
            
            if not isinstance(integral, _sympy.Integral):
                return False
                
            # Extract integrand and variable
            # Integral(f, (x, a, b)) -> args[0] is f, args[1] is (x, a, b)
            integrand = integral.function
            limits = integral.limits
            
            if not limits:
                return False
                
            var = limits[0][0]
            
            # Differentiate the candidate function
            deriv = _sympy.diff(func, var)
            
            # Check equivalence
            # simplify(deriv - integrand) == 0
            if _sympy.simplify(deriv - integrand) == 0:
                return True
                
            return False
        except Exception:
            return False

    def is_system(self, expr: str) -> bool:
        """Check if the expression represents a system of equations."""
        try:
            internal = self.to_internal(expr)
            
            if self._fallback is not None:
                # Fallback: internal is py_ast.AST (Expression)
                if isinstance(internal, py_ast.Expression):
                    body = internal.body
                    if isinstance(body, py_ast.Call) and isinstance(body.func, py_ast.Name):
                        return body.func.id == "System"
                return False
                
            return isinstance(internal, _sympy.FiniteSet)
        except Exception:
            return False

    def solve_system(self, expr: str) -> Any:
        """Solve a system of equations."""
        # Try strategies
        for category in self.active_categories:
            strategy = self.strategies.get(category)
            if strategy and hasattr(strategy, 'solve_system'):
                result = strategy.solve_system(expr, self)
                if result is not None:
                    return result
        
        # Also try LinearAlgebraStrategy explicitly if not active
        from .math_category import MathCategory
        if MathCategory.LINEAR_ALGEBRA not in self.active_categories:
             strategy = self.strategies.get(MathCategory.LINEAR_ALGEBRA)
             if strategy and hasattr(strategy, 'solve_system'):
                 result = strategy.solve_system(expr, self)
                 if result is not None:
                     return result
        return None

    def check_implication(self, system_expr: str, step_expr: str) -> bool:
        """Check if system_expr implies step_expr (solutions of system satisfy step)."""
        # Try strategies
        for category in self.active_categories:
            strategy = self.strategies.get(category)
            if strategy and hasattr(strategy, 'check_implication'):
                if strategy.check_implication(system_expr, step_expr, self):
                    return True
        
        # Try LinearAlgebra explicitly
        from .math_category import MathCategory
        if MathCategory.LINEAR_ALGEBRA not in self.active_categories:
             strategy = self.strategies.get(MathCategory.LINEAR_ALGEBRA)
             if strategy and hasattr(strategy, 'check_implication'):
                 if strategy.check_implication(system_expr, step_expr, self):
                     return True
                     
        return False
