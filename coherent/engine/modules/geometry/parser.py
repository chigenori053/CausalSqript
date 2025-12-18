from typing import Any
from coherent.engine.interfaces import BaseParser
from coherent.engine.input_parser import CoherentInputParser
from coherent.engine.symbolic_engine import SymbolicEngine

class GeometryParser(BaseParser):
    """
    Parser for Geometry mode.
    Leverages CoherentInputParser and SymbolicEngine.
    """
    
    def __init__(self):
        self.symbolic_engine = SymbolicEngine()

    def parse(self, text: str) -> Any:
        # Use SymbolicEngine to convert to SymPy/Geom objects
        normalized = CoherentInputParser.normalize(text)
        return self.symbolic_engine.to_internal(normalized)

    def validate(self, text: str) -> bool:
        try:
            self.parse(text)
            return True
        except Exception:
            return False
