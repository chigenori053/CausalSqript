
import pytest
import shutil
from pathlib import Path
from coherent.engine.knowledge_registry import KnowledgeRegistry, KnowledgeNode
from coherent.engine.symbolic_engine import SymbolicEngine

@pytest.fixture
def temp_rule_dir(tmp_path):
    # Create a temporary directory for custom rules
    rule_dir = tmp_path / "custom_rules"
    rule_dir.mkdir()
    
    # Create a dummy rule file
    rule_content = """
- id: custom_rule_1
  domain: algebra
  category: simplification
  pattern_before: "custom(x)"
  pattern_after: "x + 100"
  description: "A custom test rule"
  priority: 999
"""
    (rule_dir / "my_rules.yaml").write_text(rule_content)
    return rule_dir

def test_custom_rule_loading(temp_rule_dir):
    engine = SymbolicEngine()
    
    # Initialize registry pointing to correct core knowledge path
    # We use the real core path + our custom path
    core_path = Path("coherent/engine/knowledge").resolve()
    
    registry = KnowledgeRegistry(
        root_path=core_path, 
        engine=engine,
        custom_rules_path=temp_rule_dir
    )
    
    # Check if custom rule is loaded
    assert "custom_rule_1" in registry.rules_by_id
    rule = registry.rules_by_id["custom_rule_1"]
    assert rule.priority == 999
    assert rule.description == "A custom test rule"
    
    # Check if matching works with custom rule
    # Note: custom(x) might not parse if 'custom' is not a known function in parser,
    # but SymbolicEngine often handles unknown funcs as Function('custom').
    # Let's try matching via registry logic directly if possible, or mocked.
    
    # For this test, we verify loading presence primarily.
    assert registry.match_rules("custom(x)") != []

