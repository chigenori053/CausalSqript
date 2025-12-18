import pytest
from coherent.engine.context_manager import ContextManager

def test_context_manager_basic_ops():
    cm = ContextManager()
    
    # 1. Set/Get
    cm.set_variable("x", 10)
    assert cm.get_variable("x") == 10
    assert cm.current_context["x"] == 10
    
    # 2. Missing
    assert cm.get_variable("y") is None
    
    # 3. Clear
    cm.clear()
    assert cm.get_variable("x") is None

def test_context_manager_mode_switch():
    cm = ContextManager()
    cm.switch_context("Arithmetic")
    
    # Currently context is global, so it should persist
    cm.set_variable("val", 42)
    
    cm.switch_context("Algebra")
    assert cm.get_variable("val") == 42
    
    # Future note: If we implement isolation, this test would change to assert None
