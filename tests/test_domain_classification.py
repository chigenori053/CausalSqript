import pytest
sympy = pytest.importorskip("sympy")
from pathlib import Path
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry
from coherent.engine.classifier import ExpressionClassifier

class TestDomainClassification:
    def setup_method(self):
        self.engine = SymbolicEngine()
        self.registry = KnowledgeRegistry(Path("coherent/engine/knowledge"), self.engine)
        self.classifier = ExpressionClassifier(self.engine)

    def test_classify_arithmetic(self):
        assert self.classifier.classify("1 + 2") == ["arithmetic"]
        assert self.classifier.classify("2^3 - 0") == ["arithmetic"]

    def test_classify_algebra(self):
        # Algebra usually implies arithmetic fallback
        domains = self.classifier.classify("x + 1")
        assert "algebra" in domains
        assert "arithmetic" in domains
        assert domains.index("algebra") < domains.index("arithmetic")

    def test_classify_calculus(self):
        domains = self.classifier.classify("Integral(x, x)")
        assert "calculus" in domains
        assert "algebra" in domains # Calculus implies algebra
        
    def test_context_aware_matching(self):
        # Verify that context affects map search order
        
        # Case: Arithmetic expression
        # Should prioritize Arithmetic map
        expr = "2 + 3"
        domains = ["arithmetic"]
        
        # We can inspect the order of maps checked if we mock or debug, 
        # but here we can verify that it finds the rule.
        match = self.registry.match("2 + 3", "5", category="arithmetic")
        assert match is not None
        assert match.id == "ARITH-CALC-ADD"
        
        # Case: Algebra expression
        # Should prioritize Algebra map
        expr = "x + x"
        domains = ["algebra", "arithmetic"]
        match = self.registry.match("x + x", "2*x", category="algebra")
        assert match is not None
        assert match.id == "ALG-OP-002"

    def test_evaluator_integration(self):
        # Verify that SymbolicEvaluationEngine uses the classifier
        from coherent.engine.evaluator import SymbolicEvaluationEngine
        
        engine = SymbolicEvaluationEngine(self.engine, self.registry)
        engine.set("x + x")
        
        # This step should trigger classification of "x + x" -> ["algebra", ...]
        # And thus find ALG-OP-002
        result = engine.check_step("2*x")
        
        assert result["valid"] is True
        assert result["rule_id"] == "ALG-OP-002"
        # If classifier wasn't used, it might still find it if default order works.
        # But we know default order (without priority fix) found FAC-001.
        # With priority fix, default order also finds OP-002.
        # So this test doesn't strictly prove classifier usage unless we have a conflict
        # that ONLY context can resolve.
        # But it proves it works end-to-end.


if __name__ == "__main__":
    pytest.main([__file__])
