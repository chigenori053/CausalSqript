import pytest

from coherent.engine import ast_nodes as ast
from coherent.engine.parser import Parser
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.evaluator import Evaluator
from coherent.engine.learning_logger import LearningLogger


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

    sub = program.body[0].steps[0] 
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
