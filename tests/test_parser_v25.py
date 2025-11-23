"""Tests for the CausalScript Core DSL v2.5 parser."""

import pytest
from core.parser import Parser
from core.ast_nodes import (
    MetaNode,
    ConfigNode,
    ModeNode,
    PrepareNode,
    CounterfactualNode,
    ProblemNode,
    StepNode,
    EndNode,
)

def test_parse_meta_block():
    source = """
meta:
    id: test_01
    topic: algebra
problem: x
step: x
end: x
"""
    parser = Parser(source)
    program = parser.parse()
    assert any(isinstance(node, MetaNode) for node in program.body)
    meta_node = next(node for node in program.body if isinstance(node, MetaNode))
    assert meta_node.data == {"id": "test_01", "topic": "algebra"}

def test_parse_config_block():
    source = """
config:
    causal: true
    fuzzy-threshold: 0.8
problem: x
step: x
end: x
"""
    parser = Parser(source)
    program = parser.parse()
    assert any(isinstance(node, ConfigNode) for node in program.body)
    config_node = next(node for node in program.body if isinstance(node, ConfigNode))
    assert config_node.options == {"causal": True, "fuzzy-threshold": 0.8}

def test_parse_mode_block():
    source = """
mode: fuzzy
problem: x
step: x
end: x
"""
    parser = Parser(source)
    program = parser.parse()
    assert any(isinstance(node, ModeNode) for node in program.body)
    mode_node = next(node for node in program.body if isinstance(node, ModeNode))
    assert mode_node.mode == "fuzzy"

def test_parse_prepare_block():
    source = """
problem: x + y
prepare:
    - x = 10
    - y = 20
step: x + y
end: 30
"""
    parser = Parser(source)
    program = parser.parse()
    assert any(isinstance(node, PrepareNode) for node in program.body)
    prepare_node = next(node for node in program.body if isinstance(node, PrepareNode))
    assert prepare_node.statements == ["x = 10", "y = 20"]


def test_parse_prepare_inline_expr_and_auto():
    source = """
problem: x
prepare: temp = 5
step: x
end: x
"""
    program = Parser(source).parse()
    prepare_node = next(node for node in program.body if isinstance(node, PrepareNode))
    assert prepare_node.kind == "expr"
    assert prepare_node.expr == "temp = 5"

    auto_source = """
problem: x
prepare: auto
step: x
end: x
"""
    program = Parser(auto_source).parse()
    prepare_node = next(node for node in program.body if isinstance(node, PrepareNode))
    assert prepare_node.kind == "auto"


def test_parse_prepare_directive():
    source = """
problem: x
prepare: normalize(mode=strict)
step: x
end: x
"""
    program = Parser(source).parse()
    prepare_node = next(node for node in program.body if isinstance(node, PrepareNode))
    assert prepare_node.kind == "directive"
    assert prepare_node.directive == "normalize(mode=strict)"

def test_parse_counterfactual_block():
    source = """
problem: x * y
step: x * y
end: 1
counterfactual:
    assume:
        x: 5
    expect: x * y
"""
    parser = Parser(source)
    program = parser.parse()
    assert any(isinstance(node, CounterfactualNode) for node in program.body)
    cf_node = next(node for node in program.body if isinstance(node, CounterfactualNode))
    assert cf_node.assume == {"x": "5"}
    assert cf_node.expect == "x*y"

def test_full_v25_example():
    source = """
meta:
    id: sample_01
    topic: arithmetic

config:
    causal: true
    fuzzy-threshold: 0.5

mode: causal

problem: (3 + 5) * 4

step:
    before: (3+5)*4
    after: 8*4
    note: "simplify addition"

step:
    before: 8*4
    after: 32
    note: "multiplication"

end: 32

counterfactual:
    assume:
        x: 10
    expect: 3*x + 2
"""
    parser = Parser(source)
    program = parser.parse()
    assert len(program.body) == 8
    assert isinstance(program.body[0], MetaNode)
    assert isinstance(program.body[1], ConfigNode)
    assert isinstance(program.body[2], ModeNode)
    assert isinstance(program.body[3], ProblemNode)
    assert isinstance(program.body[4], StepNode)
    assert isinstance(program.body[5], StepNode)
    assert isinstance(program.body[6], EndNode)
    assert isinstance(program.body[7], CounterfactualNode)
