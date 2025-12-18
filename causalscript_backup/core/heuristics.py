"""
Heuristics for common mathematical errors and rule misuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

from .symbolic_engine import SymbolicEngine


@dataclass
class MisusePattern:
    name: str
    description: str
    before_pattern: str  # Pattern to match the previous step
    after_pattern: str   # Pattern to match the current step (the incorrect result)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None


class MisusePatternDetector:
    """
    Detects specific patterns of mathematical rule misuse (e.g., Freshman's Dream).
    """

    def __init__(self, symbolic_engine: SymbolicEngine):
        self.symbolic_engine = symbolic_engine
        self.patterns: List[MisusePattern] = []
        self._register_default_patterns()
        
    def _register_default_patterns(self) -> None:
        """Register common misuse patterns."""
        
        # Helper for conditions
        def basic_non_trivial(binds):
            # Check a, b are not 0 (additive identity)
            for k in ['a', 'b']:
                val = binds.get(k)
                if val is not None:
                     # Check if val is effectively zero
                     try:
                         if val == 0 or (hasattr(val, 'is_zero') and val.is_zero):
                             return False
                     except: pass
            return True

        # 1. Freshman's Dream: (a + b)^n -> a^n + b^n
        self.patterns.append(MisusePattern(
            name="Freshman's Dream",
            description="Incorrectly distributing exponent over addition: (a+b)^n is not a^n + b^n.",
            before_pattern="(a + b)**n",
            after_pattern="a**n + b**n",
            condition=lambda binds: self._check_freshman_condition(binds) and basic_non_trivial(binds)
        ))

        # 2. Linear Trigonometry: sin(a + b) -> sin(a) + sin(b)
        self.patterns.append(MisusePattern(
            name="Linear Sine",
            description="Sine is not distributive: sin(a + b) is not sin(a) + sin(b).",
            before_pattern="sin(a + b)",
            after_pattern="sin(a) + sin(b)",
            condition=basic_non_trivial
        ))
        
        # 3. Linear Cosine: cos(a + b) -> cos(a) + cos(b) (or -cos(b))
        self.patterns.append(MisusePattern(
            name="Linear Cosine",
            description="Cosine is not distributive: cos(a + b) is not cos(a) + cos(b).",
            before_pattern="cos(a + b)",
            after_pattern="cos(a) + cos(b)",
            condition=basic_non_trivial
        ))
        
        # 4. Incorrect Sqrt Dist: sqrt(a + b) -> sqrt(a) + sqrt(b)
        # Often covered by Freshman's Dream (n=1/2), but explicit handling might be robust
        self.patterns.append(MisusePattern(
            name="Sqrt Distribution",
            description="Square root does not distribute over addition: sqrt(a + b) is not sqrt(a) + sqrt(b).",
            before_pattern="sqrt(a + b)",
            after_pattern="sqrt(a) + sqrt(b)",
            condition=basic_non_trivial
        ))

    def _check_freshman_condition(self, binds: Dict[str, Any]) -> bool:
        """Ensure n != 1 for Freshman's Dream."""
        n = binds.get('n')
        if n is None:
            return True
        try:
            # Check if n is structurally '1'
            if str(n) == '1': 
                return False
            # Or evaluate
            # We assume symbolic_engine has methods, but 'n' is a sympy object here usually
            # if we use match_structure from symbolic_engine which uses sympy active objects.
             
            # If n is a float/int
            if hasattr(n, "is_Number") and n.is_Number and n == 1:
                return False
        except Exception:
            pass
        return True

    def detect_misuse(self, before_expr: str, after_expr: str) -> Optional[str]:
        """
        Check if the transition from before_expr to after_expr matches any known misuse pattern.
        Returns the name of the matched pattern, or None.
        """
        if not self.symbolic_engine.has_sympy():
            # Most matching requires SymPy
            return None

        # Optimization: Early exit if expressions look nothing like patterns?
        # Maybe not worth pre-optimization.

        for pattern in self.patterns:
            # 1. Match 'before'
            binds = self.symbolic_engine.match_structure(before_expr, pattern.before_pattern)
            if binds is None:
                continue

            # 2. Check condition if any
            if pattern.condition and not pattern.condition(binds):
                continue

            # 3. Construct expected incorrect 'after'
            try:
                # We need to substitute bindings into pattern.after_pattern
                # self.symbolic_engine.substitute_pattern(pattern.after_pattern, binds)
                # But substitute_pattern isn't exposed perfectly.
                # Let's try to parse the pattern and sub.
                
                # We can't use basic string replace for Wilds easily unless we map them.
                # But binds keys are strings (names of wilds).
                
                # Let's assume pattern.after_pattern uses same wild names (a, b, n).
                # We can reconstruct a dict {str: val} and use symbolic_engine to substitute?
                # But binds values are SymPy objects.
                
                # Better approach: Use symbolic_engine helper? 
                # Or just implement a small helper here using sympy if available.
                
                # Since we checked has_sympy(), we can import
                from sympy import sympify
                from sympy.parsing.sympy_parser import parse_expr
                
                # We need to parse pattern.after_pattern with Wilds...
                # Actually, simpler: define 'after_pattern' string, replace wild names with their values.
                # This breaks if wild names are substrings of other things, but our patterns are simple (a, b, n).
                # To be safe, use SymPy subs.
                
                local_dict = {"e": sympify("E"), "pi": sympify("pi")}
                wild_names = ['a', 'b', 'c', 'd', 'n', 'x', 'y', 'z'] # Must match match_structure
                # We know match_structure uses specific wilds.
                # However, our pattern strings use 'a', 'b', 'n' directly.
                
                # We need to convert binds (dict of str->Expr) back to something we can use.
                
                # Let's try to parse after_pattern into a sympy expr
                # Then sub the binds.
                # But the pattern has symbols 'a', 'b'. The binds has keys 'a', 'b'.
                
                # Be careful: `symbolic_engine.match_structure` returns { "a": ..., "b": ... }
                
                # Parse pattern.after_pattern
                target_template = parse_expr(pattern.after_pattern, evaluate=False)
                
                # Perform substitution
                # binds keys are correct names.
                # target_template symbols need to match these names.
                subs_dict = {}
                for sym in target_template.free_symbols:
                    if sym.name in binds:
                        subs_dict[sym] = binds[sym.name]
                        
                expected_incorrect = target_template.subs(subs_dict)
                
                # 4. Comparision
                # We want to check if user's 'after_expr' is equivalent to 'expected_incorrect'
                # Convert expected_incorrect to string to use is_equiv? 
                # Or convert after_expr to internal and compare?
                
                # expected_incorrect is a sympy object.
                # Let's convert to string to use the robust is_equiv machinery.
                expected_str = str(expected_incorrect)
                
                if self.symbolic_engine.is_equiv(after_expr, expected_str):
                    return pattern.name

            except Exception as e:
                # print(f"DEBUG: Error checking pattern {pattern.name}: {e}")
                continue
                
        return None
