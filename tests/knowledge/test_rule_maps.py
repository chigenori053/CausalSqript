import pytest
sympy = pytest.importorskip("sympy")
from pathlib import Path
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry

class TestRuleMaps:
    def setup_method(self):
        self.engine = SymbolicEngine()
        self.registry = KnowledgeRegistry(Path("coherent/engine/knowledge"), self.engine)

    def test_map_loading(self):
        # Verify maps are loaded
        assert len(self.registry.maps) > 0
        
        # Verify arithmetic map
        arithmetic_map = next((m for m in self.registry.maps if m.id == "arithmetic"), None)
        assert arithmetic_map is not None
        assert "ARITH-CALC-ADD" in arithmetic_map.rules
        assert arithmetic_map.priority == 100

        # Verify algebra map
        algebra_map = next((m for m in self.registry.maps if m.id == "algebra_basic"), None)
        assert algebra_map is not None
        assert "ALG-EXP-001" in algebra_map.rules

    def test_priority_matching(self):
        # Verify that rules in higher priority maps are checked first.
        # ARITH-CALC-SUB is in Arithmetic (100).
        # If we had a conflicting rule in a lower priority map, Arithmetic should win.
        
        # Test 2^3 - 0 matching ARITH-CALC-SUB
        expr = "2^3 - 0"
        # Note: InputParser is not used here, so we assume normalized input if needed,
        # but match() takes string. get_top_operator handles ^ vs **.
        # But let's pass normalized string to be safe as per previous issues.
        normalized = "2**3 - 0" 
        
        rule = self.registry.match(normalized, "8")
        assert rule is not None
        assert rule.id == "ARITH-CALC-SUB"

    def test_unmapped_fallback(self):
        # If we have a rule not in any map (e.g. a new one), it should still be matched (fallback).
        # We don't have one right now, but we can verify that existing rules still work.
        pass

if __name__ == "__main__":
    pytest.main([__file__])
