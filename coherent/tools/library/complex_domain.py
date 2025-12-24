from typing import Any, Union, Optional
import sympy
from sympy import I, re, im, Abs, arg, exp, pi, Symbol, Expr, sympify

# Define a simple BaseDomain interface if it doesn't exist globally
class BaseDomain:
    def contains(self, expr: Any) -> bool:
        """Check if the expression belongs to this domain."""
        raise NotImplementedError

    def canonicalize(self, expr: Any) -> Any:
        """Convert expression to canonical form."""
        raise NotImplementedError

class ComplexDomain(BaseDomain):
    """
    ComplexDomain v1.0 (Adapted)
    
    Handlers for Complex Numbers (C) in the COHERENT system.
    Adapts standard SymPy complex arithmetic for the optical memory architecture.
    """
    
    def __init__(self):
        self._imaginary_unit_symbols = {Symbol('i'), Symbol('j'), Symbol('I')}

    def contains(self, expr: Any) -> bool:
        """
        Check if the expression is a complex number.
        
        Args:
            expr: The expression to check (str, SymPy Expr, or numeric).
            
        Returns:
            bool: True if it is a complex number (including reals as subset), False otherwise.
        """
        try:
            val = self._ensure_sympy(expr)
            # In SymPy, real numbers are also complex. 
            # We want to check if it's a number (scalar), finite, etc.
            # But specifically, we might want to know if it involves I explicitly if we are strict,
            # but mathematically R \subset C.
            # Here we assume standard mathematical definition: if it's a scalar number, it's in ComplexDomain.
            if val.is_number:
                return True
            # If it has symbols, check if those symbols are declared simple scalars? 
            # For now, if it evaluates to a number with substitution, it might be.
            # But purely symbolic:
            return False
        except:
            return False

    def is_strictly_imaginary(self, expr: Any) -> bool:
        """Check if expression has a non-zero imaginary part."""
        try:
            val = self._ensure_sympy(expr)
            return not im(val).equals(0)
        except:
            return False

    def distance(self, z1: Any, z2: Any) -> float:
        """
        Compute the Euclidean / Modulus distance between two complex numbers.
        d(z1, z2) = |z1 - z2|
        """
        try:
            v1 = self._ensure_sympy(z1)
            v2 = self._ensure_sympy(z2)
            return float(Abs(v1 - v2))
        except Exception:
            return float('inf')

    def canonicalize(self, expr: Any) -> str:
        """
        Convert to 'a + b*i' format (Cartesian form).
        Also ensures 'j' or 'I' are standardized to 'i' for display if requested,
        but SymPy uses 'I'. We will map 'I' to 'i' for string output.
        """
        val = self._ensure_sympy(expr)
        # Simplify first
        simplified = val.simplify()
        # Expand complexity to standard form a + bi
        expanded = simplified.expand(complex=True)
        
        # Convert to string and replace I with i
        s = str(expanded).replace('I', 'i')
        return s

    def to_polar(self, expr: Any) -> str:
        """Convert to Euler/Polar form: r*exp(i*theta)"""
        val = self._ensure_sympy(expr)
        r = Abs(val)
        theta = arg(val)
        polar = r * exp(I * theta)
        return str(polar).replace('I', 'i')

    def parse_with_i(self, text: str) -> Expr:
        """
        Parse text treating 'i', 'I', 'j' as imaginary unit.
        """
        # 1. Normalize input (handle implicit multiplication, etc.)
        from coherent.core.input_parser import CausalScriptInputParser
        normalized = CausalScriptInputParser.normalize(text)
        
        # 2. Replace 'i' or 'j' with 'I' for SymPy
        # Use simple string replacement on normalized string which should have spaces/operators
        # We need to be careful about not replacing inside function names (like 'sin' -> 'sIn'?)
        # But normalize adds spaces.
        # It's safer to use regex tokens. Or just regex on word boundaries.
        import re
        # Pattern: word boundary, i/j, word boundary
        # i.e. " i " or "i " or " i" or just "i"
        # Since normalize separates tokens mostly
        
        clean_text = re.sub(r'\b[ij]\b', 'I', normalized)
        
        return sympify(clean_text)

    def _ensure_sympy(self, expr: Any) -> Expr:
        if isinstance(expr, Expr):
            return expr
        # If string, use our parse_with_i to handle i/j and multiplication
        if isinstance(expr, str):
            # Try parsing
            try:
                return self.parse_with_i(expr)
            except Exception:
                # If specialized parsing fails, try raw sympify but that likely won't work for "2i"
                return sympify(expr)
        return sympify(expr)
