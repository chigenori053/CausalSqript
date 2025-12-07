"""Tests for the CausalScript Core DSL v2.5 evaluator."""

import pytest
from causalscript.core.parser import Parser
from causalscript.core.evaluator import Evaluator, SymbolicEvaluationEngine
from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.learning_logger import LearningLogger

def create_evaluator(source: str) -> Evaluator:
    parser = Parser(source)
    program = parser.parse()
    symbolic_engine = SymbolicEngine()
    engine = SymbolicEvaluationEngine(symbolic_engine)
    logger = LearningLogger()
    return Evaluator(program, engine, logger)

def test_evaluator_prepare_block():
    source = """
problem: x + y
prepare:
    - x = 10
    - y = 20
step:
    after: 30
end: 30
"""
    evaluator = create_evaluator(source)
    assert evaluator.run()
    assert evaluator.engine._context == {"x": 10, "y": 20}


def test_evaluator_prepare_auto_and_directive():
    auto_source = """
problem: 1
prepare: auto
step: 1
end: 1
"""
    evaluator = create_evaluator(auto_source)
    assert evaluator.run()
    log = evaluator.learning_logger.to_list()
    prepare_log = next(item for item in log if item["phase"] == "prepare")
    assert "auto" in prepare_log["rendered"]

    directive_source = """
problem: 1
prepare: normalize(mode=strict)
step: 1
end: 1
"""
    evaluator = create_evaluator(directive_source)
    assert evaluator.run()
    log = evaluator.learning_logger.to_list()
    prepare_log = next(item for item in log if item["phase"] == "prepare")
    assert "normalize" in prepare_log["rendered"]

def test_evaluator_counterfactual_block():
    source = """
problem: 3 * y
prepare:
    - y = 2
step: 3 * y
end: 6
counterfactual:
    assume:
        y: 5
    expect: 3 * y
"""
    evaluator = create_evaluator(source)
    assert evaluator.run()
    
    # Check the learning log for the counterfactual result
    log = evaluator.learning_logger.to_list()
    cf_log = next(item for item in log if item["phase"] == "counterfactual")
    assert cf_log["status"] == "ok"
    assert float(cf_log["meta"]["result"]) == 15.0

def test_evaluator_mode_block():
    source = """
mode: fuzzy
problem: 1+1
step: 2
end: 2
"""
    evaluator = create_evaluator(source)
    assert evaluator.run()
    assert evaluator._mode == "fuzzy"

def test_backward_compatibility():
    source = """
problem: 2 + 2
step1: 4
end: 4
"""
    evaluator = create_evaluator(source)
    assert evaluator.run()
    log = evaluator.learning_logger.to_list()
    step_log = next(item for item in log if item["phase"] == "step")
    assert step_log["status"] == "ok"

def test_full_v25_evaluation():
    source = """
meta:
    id: sample_01
    topic: arithmetic
config:
    causal: true
    fuzzy-threshold: 0.5
mode: causal
problem: (x + y) * 4
prepare:
    - x = 3
    - y = 5
step:
    before: (x + y) * 4
    after: 8 * 4
    note: "simplify addition"
step:
    before: 8 * 4
    after: 32
    note: "multiplication"
end: 32
counterfactual:
    assume:
        x: 10
    expect: 3*x + 2
"""
    evaluator = create_evaluator(source)
    assert evaluator.run()
    
    log = evaluator.learning_logger.to_list()
    
    # Check meta
    meta_log = next(item for item in log if item["phase"] == "meta")
    assert meta_log["rendered"] == "Meta: {'id': 'sample_01', 'topic': 'arithmetic'}"
    
    # Check config
    config_log = next(item for item in log if item["phase"] == "config")
    assert config_log["rendered"] == "Config: {'causal': True, 'fuzzy-threshold': 0.5}"

    # Check mode
    mode_log = next(item for item in log if item["phase"] == "mode")
    assert mode_log["rendered"] == "Mode: causal"

    # Check prepare
    prepare_logs = [item for item in log if item["phase"] == "prepare"]
    assert len(prepare_logs) == 2
    assert prepare_logs[0]["expression"] == "x = 3"
    assert prepare_logs[1]["expression"] == "y = 5"

    # Check steps
    step_logs = [item for item in log if item["phase"] == "step"]
    assert len(step_logs) == 2
    assert step_logs[0]["status"] == "ok"
    assert step_logs[1]["status"] == "ok"

    # Check counterfactual
    cf_log = next(item for item in log if item["phase"] == "counterfactual")
    assert cf_log["status"] == "ok"
    assert float(cf_log["meta"]["result"]) == 32.0
