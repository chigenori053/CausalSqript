from typing import Any
from causalscript.core.interfaces import BaseParser
from causalscript.core.input_parser import CausalScriptInputParser
from causalscript.core.symbolic_engine import SymbolicEngine

class GeometryParser(BaseParser):
    """
    Parser for Geometry mode.
    Leverages CausalScriptInputParser and SymbolicEngine.
    """
    
    def __init__(self):
        self.symbolic_engine = SymbolicEngine()

    def parse(self, text: str) -> Any:
        # Use SymbolicEngine to convert to SymPy/Geom objects
        normalized = CausalScriptInputParser.normalize(text)
        return self.symbolic_engine.to_internal(normalized)

    def validate(self, text: str) -> bool:
        try:
            self.parse(text)
            return True
        except Exception:
            return False
