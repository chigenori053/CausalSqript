
import pytest
pytest.importorskip("sympy")
from pathlib import Path
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry

def test_knowledge_concepts_loading():
    # Setup
    base_path = Path("coherent/engine/knowledge")
    engine = SymbolicEngine()
    registry = KnowledgeRegistry(base_path, engine)
    
    # 1. Verify CALC-DIFF-POW has concept 'power_rule'
    node = registry.rules_by_id.get("CALC-DIFF-POW")
    assert node is not None
    assert node.concept == "power_rule"
    
    # 2. Verify New Rule CALC-DIFF-SIN
    node_sin = registry.rules_by_id.get("CALC-DIFF-SIN")
    assert node_sin is not None
    assert node_sin.concept == "trig_derivative"
    
    # 3. Match Logic Verification
    # Match Derivative(sin(x), x) -> cos(x)
    match = registry.match("Derivative(sin(x), x)", "cos(x)", category="calculus")
    assert match is not None
    assert match.id == "CALC-DIFF-SIN"
    assert match.concept == "trig_derivative"
    
    # 4. Match Derivative(tan(x), x) -> sec(x)**2
    match_tan = registry.match("Derivative(tan(x), x)", "sec(x)**2", category="calculus")
    assert match_tan is not None
    assert match_tan.id == "CALC-DIFF-TAN"
    
    # 5. Verify Integration Concept
    node_int = registry.rules_by_id.get("CALC-DEF-INT")
    assert node_int is not None
    assert node_int.concept == "fundamental_theorem_of_calculus"

def test_causal_engine_produces_concept_payload():
    # Helper to simulate CausalEngine behavior if possible, 
    # but CausalEngine requires complex setup. 
    # We can check if KnowledgeNode.to_metadata() includes concept.
    
    base_path = Path("coherent/engine/knowledge")
    engine = SymbolicEngine()
    registry = KnowledgeRegistry(base_path, engine)
    node = registry.rules_by_id.get("CALC-DIFF-POW")
    
    meta = node.to_metadata()
    assert "concept" in meta
    assert meta["concept"] == "power_rule"
