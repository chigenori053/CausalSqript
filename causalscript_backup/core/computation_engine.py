"""Computation Engine for MathLang Core."""

from __future__ import annotations

from typing import Any, Dict, Optional
from contextlib import contextmanager
import concurrent.futures

from .symbolic_engine import SymbolicEngine
from .geometry_engine import GeometryEngine
from .errors import EvaluationError, InvalidExprError
from .ast_nodes import Node
from .simple_algebra import SimpleAlgebra
from .category_analyzer import CategoryAnalyzer
from .math_category import MathCategory

try:
    import sympy
except ImportError:
    sympy = None


def _eval_scenario_task(name: str, expr: str, context: Dict[str, Any], symbolic_engine: SymbolicEngine) -> tuple[str, Any]:
    """Helper function for process-based evaluation."""
    try:
        try:
            return name, symbolic_engine.evaluate(expr, context)
        except EvaluationError:
            return name, {"error": "Evaluation failed"}
    except Exception as e:
        return name, {"error": str(e)}

def _check_scenario_task(name: str, expr1: str, expr2: str, context: Dict[str, Any], symbolic_engine: SymbolicEngine) -> tuple[str, bool]:
    """Helper function for process-based equivalence check."""
    try:
        val1 = symbolic_engine.evaluate(expr1, context)
        val2 = symbolic_engine.evaluate(expr2, context)
        return name, (val1 == val2)
    except Exception:
        return name, False


class ComputationEngine:
    """
    Provides symbolic and numeric evaluation using SymbolicEngine and SymPy.
    
    The ComputationEngine wraps the SymbolicEngine to provide a higher-level
    interface for mathematical computations, including simplification, expansion,
    factoring, and numeric evaluation.
    Handles core mathematical computations and symbolic manipulations.
    Delegates to specialized engines (Geometry, Calculus, etc.) as needed.
    """

    def __init__(self, symbolic_engine: SymbolicEngine):
        """
        Initialize the computation engine.
        
        Args:
            symbolic_engine: SymbolicEngine instance for symbolic operations
        """

        self.symbolic_engine = symbolic_engine
        self.variables: Dict[str, Any] = {}
        self._shared_executor: concurrent.futures.Executor | None = None
        self._shared_executor_kind: str | None = None
        try:
            self.geometry = GeometryEngine()
        except ImportError:
            self.geometry = None

    def detect_category(self, expr: str) -> MathCategory:
        """Detect the mathematical category of an expression."""
        return CategoryAnalyzer.detect(expr)

    def compute_optimized(self, expr: str, category: MathCategory | None = None) -> Any:
        """
        Perform computation using the engine optimized for the detected category.
        Returns the simplified string or computed value.
        """
        target_category = category or self.detect_category(expr)

        # Dispatch to specialized engines
        if target_category == MathCategory.CALCULUS:
            # Attempt to solve derivatives/integrals if explicitly requested in syntax
            if "diff" in expr or "d/dx" in expr or "Derivative" in expr:
                # Fallback to symbolic simplify which handles 'diff(x**2, x)' via SymPy
                pass 

        elif target_category == MathCategory.GEOMETRY and self.geometry:
            # If the expression evaluates to a Geometric entity, return its properties
            pass
            
        # Default fallback: Symbolic simplification
        return self.simplify(expr)

    def _get_executor(self) -> concurrent.futures.Executor:
        """
        Lazily initialize and cache an executor to avoid per-call overhead.
        Prefers processes but falls back to threads if processes are unavailable.
        """
        if self._shared_executor is not None:
            return self._shared_executor
        try:
            self._shared_executor = concurrent.futures.ProcessPoolExecutor()
            self._shared_executor_kind = "process"
        except (PermissionError, NotImplementedError, OSError):
            self._shared_executor = concurrent.futures.ThreadPoolExecutor()
            self._shared_executor_kind = "thread"
        return self._shared_executor

    def _shutdown_executor(self) -> None:
        if self._shared_executor is not None:
            self._shared_executor.shutdown(wait=False)
            self._shared_executor = None
            self._shared_executor_kind = None

    def to_sympy(self, expr: str | Node) -> Any:
        """
        Converts an expression string or ASTNode to a SymPy expression.
        
        Args:
            expr: Expression string or ASTNode to convert
            
        Returns:
            SymPy expression object (or fallback AST if SymPy not available)
            
        Raises:
            NotImplementedError: If expr is an ASTNode (not yet implemented)
            InvalidExprError: If the expression is invalid
        """
        if isinstance(expr, str):
            return self.symbolic_engine.to_internal(expr)
        # If it's an ASTNode, we might need to convert it to string first or handle it
        # For now, assuming string input as primary interface
        raise NotImplementedError("ASTNode conversion not yet implemented")

    def simplify(self, expr: str) -> str:
        """
        Simplifies the given expression string.
        
        Uses SymPy's simplify function when available, otherwise attempts
        numeric evaluation for constant expressions.
        
        Args:
            expr: Expression string to simplify
            
        Returns:
            Simplified expression as a string
            
        Examples:
            >>> engine.simplify("2 + 2")
            "4"
            >>> engine.simplify("x + x")  # With SymPy
            "2*x"
        """
        if not self.symbolic_engine.has_sympy():
            try:
                return SimpleAlgebra.simplify(expr)
            except InvalidExprError:
                return self.symbolic_engine.simplify(expr)
        return self.symbolic_engine.simplify(expr)

    def to_latex(self, expr: str) -> str:
        """
        Converts the given expression to LaTeX format.
        
        Args:
            expr: Expression string to convert
            
        Returns:
            LaTeX representation of the expression
        """
        return self.symbolic_engine.to_latex(expr)

    def expand(self, expr: str) -> str:
        """
        Expands the given algebraic expression.
        
        Args:
            expr: Expression string to expand
            
        Returns:
            Expanded expression as a string
            
        Raises:
            InvalidExprError: If the expression is invalid
            
        Examples:
            >>> engine.expand("(x + y)**2")
            "x**2 + 2*x*y + y**2"
            >>> engine.expand("(a + b)*(c + d)")
            "a*c + a*d + b*c + b*d"
        """
        if not self.symbolic_engine.has_sympy():
            try:
                return SimpleAlgebra.expand(expr)
            except InvalidExprError:
                return expr
        
        try:
            internal = self.symbolic_engine.to_internal(expr)
            if sympy is not None:
                expanded = sympy.expand(internal)
                return str(expanded)
            return expr
        except Exception as exc:
            raise InvalidExprError(f"Failed to expand expression: {exc}") from exc

    def factor(self, expr: str) -> str:
        """
        Factors the given algebraic expression.
        
        Args:
            expr: Expression string to factor
            
        Returns:
            Factored expression as a string
            
        Raises:
            InvalidExprError: If the expression is invalid
            
        Examples:
            >>> engine.factor("x**2 - y**2")
            "(x - y)*(x + y)"
            >>> engine.factor("x**2 + 2*x + 1")
            "(x + 1)**2"
        """
        if not self.symbolic_engine.has_sympy():
            try:
                return SimpleAlgebra.factor(expr)
            except InvalidExprError:
                return expr
        
        try:
            internal = self.symbolic_engine.to_internal(expr)
            if sympy is not None:
                factored = sympy.factor(internal)
                return str(factored)
            return expr
        except Exception as exc:
            raise InvalidExprError(f"Failed to factor expression: {exc}") from exc

    def substitute(self, expr: str, substitutions: Dict[str, Any]) -> str:
        """
        Substitutes variables in an expression with given values.
        
        Args:
            expr: Expression string containing variables
            substitutions: Dictionary mapping variable names to values
            
        Returns:
            Expression with substitutions applied as a string
            
        Raises:
            InvalidExprError: If the expression is invalid
            
        Examples:
            >>> engine.substitute("x + y", {"x": 2, "y": 3})
            "5"
            >>> engine.substitute("a*x + b", {"x": 5})
            "5*a + b"
        """
        try:
            internal = self.symbolic_engine.to_internal(expr, extra_locals=substitutions)
            
            if self.symbolic_engine.has_sympy() and sympy is not None:
                # result = internal.subs(subs_dict) # Redundant now
                result = sympy.simplify(internal)
                return str(result)

            try:
                return SimpleAlgebra.substitute(expr, substitutions)
            except InvalidExprError:
                return expr
        except Exception as exc:
            raise InvalidExprError(f"Failed to substitute in expression: {exc}") from exc

    def numeric_eval(self, expr: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Evaluates the expression numerically.
        
        Combines the engine's bound variables with the provided context,
        then evaluates the expression to a numeric value.
        
        Args:
            expr: Expression string to evaluate
            context: Optional dictionary of variable values for this evaluation
            
        Returns:
            Numeric result of evaluation
            
        Raises:
            EvaluationError: If evaluation fails
            
        Examples:
            >>> engine.numeric_eval("3 * 4")
            12
            >>> engine.bind("x", 10)
            >>> engine.numeric_eval("x + 5")
            15
            >>> engine.numeric_eval("y * 2", context={"y": 3})
            6
        """
        eval_context = self.variables.copy()
        if context:
            eval_context.update(context)
        
        try:
            return self.symbolic_engine.evaluate(expr, eval_context)
        except EvaluationError:
            # If symbolic engine fails, we might want to try direct numeric evaluation if possible
            # But SymbolicEngine.evaluate already handles numeric evaluation if variables are present
            raise


    def evaluate_in_scenarios(self, expr: str, scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate an expression in multiple scenarios in parallel using processes.
        
        Args:
            expr: The expression to evaluate.
            scenarios: A dictionary mapping scenario names to their contexts.
            
        Returns:
            A dictionary mapping scenario names to evaluation results.
        """
        results = {}
        
        executor = self._get_executor()
        future_to_name = {
            executor.submit(
                _eval_scenario_task, 
                name, 
                expr, 
                {**self.variables, **context}, 
                self.symbolic_engine
            ): name 
            for name, context in scenarios.items()
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name, result = future.result()
            results[name] = result
                
        return results

    def check_equivalence_in_scenarios(self, expr1: str, expr2: str, scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """
        Check if two expressions are equivalent in multiple scenarios in parallel using processes.
        
        Args:
            expr1: First expression.
            expr2: Second expression.
            scenarios: A dictionary mapping scenario names to their contexts.
            
        Returns:
            A dictionary mapping scenario names to boolean equivalence results.
        """
        results = {}
        
        executor = self._get_executor()
        future_to_name = {
            executor.submit(
                _check_scenario_task, 
                name, 
                expr1, 
                expr2, 
                {**self.variables, **context}, 
                self.symbolic_engine
            ): name 
            for name, context in scenarios.items()
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name, result = future.result()
            results[name] = result
                
        return results

    def bind(self, name: str, value: Any) -> None:
        """
        Binds a variable to a value in the engine's context.
        
        The bound variable will be available for all subsequent evaluations
        until explicitly unbound or overwritten.
        
        Args:
            name: Variable name to bind
            value: Value to bind to the variable
            
        Examples:
            >>> engine.bind("a", 100)
            >>> engine.numeric_eval("a + 10")
            110
        """
        self.variables[name] = value
