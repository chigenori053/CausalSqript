import pytest
sympy = pytest.importorskip("sympy")
from pathlib import Path
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry
from coherent.engine.input_parser import CoherentInputParser

class TestStrictRules:
    def setup_method(self):
        self.engine = SymbolicEngine()
        self.registry = KnowledgeRegistry(Path("coherent/engine/knowledge"), self.engine)

    def test_subtraction_match(self):
        # 2^3 - 0 should match ARITH-CALC-SUB, not ARITH-CALC-ADD
        expr = "2^3 - 0"
        normalized = CoherentInputParser.normalize(expr)
        # normalized is "2**3 - 0"
        
        # We need to ensure the result is 8 for the rule to match "after"
        # pattern_after is "c", which matches anything if it's equivalent.
        
        rule = self.registry.match(normalized, "8")
        assert rule is not None
        # It might match ARITH-CALC-DONE if it thinks it's done? No, 8 is different from 2^3-0.
        
        # Ideally it matches ARITH-CALC-SUB
        print(f"Matched: {rule.id}")
        assert rule.id == "ARITH-CALC-SUB"

    def test_add_mismatch(self):
        # 2^3 - 0 should NOT match ARITH-CALC-ADD
        # We can check this by inspecting the registry logic or just asserting the result above.
        pass

if __name__ == "__main__":
    pytest.main([__file__])
