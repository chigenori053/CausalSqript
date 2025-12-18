
import pytest
from unittest.mock import MagicMock
from coherent.memory.ast_generalizer import ASTGeneralizer
from coherent.memory.experience_manager import ExperienceManager

def test_ast_generalization():
    gen = ASTGeneralizer()
    
    # Should replace numbers and variable names with generic placeholders
    # Note: Our regex is simple, might keep numbers or structure.
    # Implementation replaces \b[a-zA-Z_]\w*\b with _v if not keyword.
    
    assert "_v" in gen.generalize("x + y")
    assert gen.generalize("sin(x)") == "sin(_v)"
    # Ensure structure is preserved
    assert gen.generalize("x^2 + 2*x") == "_v^2 + 2*_v" 

def test_experience_manager():
    mock_store = MagicMock()
    manager = ExperienceManager(mock_store)
    
    # Test Save
    manager.save_edge("src_gen", "dst_gen", "rule_1", [0.1, 0.2])
    mock_store.add.assert_called_once()
    
    # Test Query
    mock_store.query.return_value = [{
        "id": "e1",
        "metadata": {
            "original_expr": "src_gen",
            "next_expr": "dst_gen",
            "rule_id": "rule_1",
            "result_label": "EXACT"
        }
    }]
    
    results = manager.find_similar_edges([0.1, 0.2])
    assert len(results) == 1
    assert results[0].rule_id == "rule_1"
    assert results[0].next_expr == "dst_gen"
