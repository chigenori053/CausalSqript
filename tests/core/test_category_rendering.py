import pytest
from coherent.engine.renderers import ContentRenderer, RenderContext
from coherent.engine.knowledge_registry import KnowledgeRegistry, KnowledgeNode
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.math_category import MathCategory
from unittest.mock import MagicMock

class TestCategoryRendering:
    def test_render_algebra(self):
        ctx = RenderContext(
            expression="x**2 + 2*x + 1",
            category="algebra",
            metadata={}
        )
        assert ContentRenderer.render_step(ctx) == "x**2 + 2*x + 1"

    def test_render_calculus(self):
        ctx = RenderContext(
            expression="diff(x**2, x)",
            category="calculus",
            metadata={}
        )
        assert ContentRenderer.render_step(ctx) == r"\frac{d}{dx}(x**2, x)" # Improved LaTeX rendering

    def test_render_geometry(self):
        ctx = RenderContext(
            expression="Triangle(p1, p2, p3)",
            category="geometry",
            metadata={"description": "Equilateral Triangle"}
        )
        assert ContentRenderer.render_step(ctx) == "Equilateral Triangle (Triangle(p1, p2, p3))"

class TestKnowledgeRegistryFiltering:
    def setup_method(self):
        self.engine = MagicMock(spec=SymbolicEngine)
        # Mock match_structure to always return a match if not filtered
        self.engine.match_structure.return_value = {"x": "x"}
        self.engine.is_numeric.return_value = False
        self.engine.get_top_operator.return_value = "Add" # Dummy
        self.engine.is_equiv.return_value = True # Assume valid for this test
        self.engine.substitute.return_value = "2*x" # Dummy

        # Mock registry with manual nodes
        self.registry = KnowledgeRegistry(MagicMock(), self.engine)
        self.registry.nodes = [
            KnowledgeNode(
                id="geo_rule",
                domain="geometry",
                category="property",
                pattern_before="Area(Square(s))",
                pattern_after="s**2",
                description="Area of square",
                priority=100 # Higher priority
            ),
            KnowledgeNode(
                id="alg_rule",
                domain="algebra",
                category="simplification",
                pattern_before="x + x",
                pattern_after="2*x",
                description="Combine like terms",
                priority=50 # Lower priority
            )
        ]
        # Sort by priority descending (simulating __init__)
        self.registry.nodes.sort(key=lambda n: n.priority, reverse=True)
        self.registry.rules_by_id = {n.id: n for n in self.registry.nodes}
        self.registry.maps = [] # No maps for this test

    def test_match_filtering(self):
        # Algebra context should match algebra rule
        # Even though geo_rule is higher priority, it should be filtered out
        match = self.registry.match("x + x", "2*x", category="algebra")
        assert match is not None
        assert match.id == "alg_rule"

        # Geometry context should match geometry rule (higher priority)
        match = self.registry.match("Area(Square(s))", "s**2", category="geometry")
        assert match is not None
        assert match.id == "geo_rule"

if __name__ == "__main__":
    pytest.main([__file__])
