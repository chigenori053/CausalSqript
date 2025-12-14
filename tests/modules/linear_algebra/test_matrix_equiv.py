
import pytest
from causalscript.core.symbolic_engine import SymbolicEngine

try:
    import sympy
    from sympy import Matrix
except ImportError:
    sympy = None

@pytest.mark.skipif(sympy is None, reason="SymPy not available")
class TestMatrixEquivalence:
    def setup_method(self):
        self.engine = SymbolicEngine()

    def test_matrix_vs_list_equiv(self):
        # Context with Matrices
        ctx = {
            "A": Matrix([[1, 2], [3, 4]]),
            "B": Matrix([[2, 0], [1, 2]])
        }
        
        # A*B = [[4, 4], [10, 8]]
        
        # Target expression (Matrix)
        expr1 = "A*B"
        
        # List representation (valid calculation result)
        expr2 = "[[1*2 + 2*1, 1*0 + 2*2], [3*2 + 4*1, 3*0 + 4*2]]"
        
        assert self.engine.is_equiv(expr1, expr2, context=ctx)
        
        # Already evaluated list
        expr3 = "[[4, 4], [10, 8]]"
        assert self.engine.is_equiv(expr1, expr3, context=ctx)

    def test_matrix_vs_list_not_equiv(self):
        ctx = {
            "A": Matrix([[1, 2], [3, 4]])
        }
        
        expr1 = "A"
        expr2 = "[[1, 2], [3, 5]]" # Different element
        
        assert not self.engine.is_equiv(expr1, expr2, context=ctx)

    def test_matrix_vs_matrix(self):
        ctx = {
            "A": Matrix([[1, 2]])
        }
        assert self.engine.is_equiv("A", "Matrix([[1, 2]])", context=ctx)

    def test_vector_vs_list(self):
        # SymPy treats 1D list as .. column vector? No, Matrix([1,2]) is usually col unless specified.
        # But Matrix([[1, 2]]) is row.
        # Let's check 1D list [1, 2] vs Matrix([1, 2]) (col vector)
        
        ctx = {
            "v": Matrix([1, 2]) # Column vector
        }
        
        # [1, 2] usually means list of scalars. SymPy Matrix([1, 2]) creates column vector.
        # Matrix conversion of [1, 2] creates column vector.
        assert self.engine.is_equiv("v", "[1, 2]", context=ctx)

    def test_nested_list_evaluation(self):
        # Ensure evaluate returns Matrix-like object if upgraded?
        # SymbolicEngine.evaluate usually returns internal type.
        pass
