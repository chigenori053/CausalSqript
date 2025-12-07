import pytest

from causalscript.core import ast_nodes as ast
from causalscript.core.parser import Parser
from causalscript.core.core_runtime import CoreRuntime
from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.computation_engine import ComputationEngine
from causalscript.core.validation_engine import ValidationEngine
from causalscript.core.hint_engine import HintEngine
from causalscript.core.evaluator import Evaluator
from causalscript.core.learning_logger import LearningLogger


def test_parser_supports_named_problem_and_variable_sub_problem():
    source = """
    problem: SolverA = (x + 1)^2
    sub_problem: disc = b^2 - 4*a*c
    step: x^2 + 2*x + 1
    end: done
    """
    program = Parser(source).parse()
    problem = program.body[0]
    assert isinstance(problem, ast.ProblemNode)
    assert problem.name == "SolverA"
    assert problem.expr == "(x + 1)**2"

    sub = program.body[1]
    assert isinstance(sub, ast.SubProblemNode)
    assert sub.target_variable == "disc"
    assert sub.expr == "b**2 - 4*a*c"


def test_parser_ignores_nested_equals():
    program = Parser("problem: (x=1) + 2\nstep: 1\nend: done").parse()
    problem = program.body[0]
    assert problem.name is None
    assert "(x" in problem.expr and "=" in problem.expr


def test_variable_binding_sub_problem_sets_context():
    source = """
    problem: MainExpr = temp + b
    prepare:
      - b = 2
    sub_problem: temp = 1 + 2
    step: 1 + 2
    end: 3
    step: temp + b
    step: 3 + 2
    end: done
    """
    sym = SymbolicEngine()
    comp = ComputationEngine(sym)
    val = ValidationEngine(comp)
    hint = HintEngine(comp)
    logger = LearningLogger()
    runtime = CoreRuntime(comp, val, hint, learning_logger=logger)

    program = Parser(source).parse()
    evaluator = Evaluator(program, runtime, learning_logger=logger)
    assert evaluator.run() is True

    # temp should be available in context for later steps
    assert runtime._context.get("temp") == "3"
