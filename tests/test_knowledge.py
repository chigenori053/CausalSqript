from pathlib import Path

from core.knowledge_registry import KnowledgeRegistry
from core.symbolic_engine import SymbolicEngine


def test_knowledge_registry_loads_rules_and_matches():
    engine = SymbolicEngine()
    registry = KnowledgeRegistry(Path("core/knowledge"), engine)
    assert any(node.id == "ARITH-ADD-001" for node in registry.nodes)
    match = registry.match("a + b", "b + a")
    assert match is not None
