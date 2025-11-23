import pytest

from core.parser import Parser
from core import ast_nodes as ast
from core.errors import SyntaxError

def test_parser_builds_nodes_with_line_numbers():
    source = """# comment

problem: (x + 1) * (x + 2)
step1: x^2 + 3*x + 2
explain: "expansion"
end: x^2 + 3*x + 2
"""
    program = Parser(source).parse()
    assert isinstance(program, ast.ProgramNode)
    assert len(program.body) == 4

    problem = program.body[0]
    assert isinstance(problem, ast.ProblemNode)
    assert problem.expr == "(x + 1)*(x + 2)"
    assert problem.line == 3

    step = program.body[1]
    assert isinstance(step, ast.StepNode)
    assert step.step_id == "1"
    assert step.expr == "x**2 + 3*x + 2"

    explain = program.body[2]
    assert isinstance(explain, ast.ExplainNode)
    assert explain.text == "expansion"

    end = program.body[3]
    assert isinstance(end, ast.EndNode)
    assert end.expr == "x**2 + 3*x + 2"


def test_parser_requires_problem_and_end():
    with pytest.raises(SyntaxError):
        Parser("step: 1 = 1").parse()
    with pytest.raises(SyntaxError):
        Parser("problem: 1 = 1").parse()


def test_parser_negative_coefficient_syntax_error():
    source = """
problem: 1
step: x^2-2xy+y^2
end: 1
"""
    program = Parser(source).parse()
    step_node = program.body[1]
    print(f"Parsed expression: {step_node.expr}")


def test_parser_handles_negative_coefficients():
    source = "problem: 1\nstep: x^2 - 2*x*y + y^2\nend: 1"
    program = Parser(source).parse()
    step_node = program.body[1]
    assert step_node.expr == "x**2 - 2*x*y + y**2"
