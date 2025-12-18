import pytest
from coherent.engine.orchestrator import CoreOrchestrator
from coherent.engine.parser import Parser
from coherent.engine.errors import EvaluationError

def test_orchestrator_arithmetic_mode():
    orchestrator = CoreOrchestrator()
    
    # 1. Switch to Arithmetic
    orchestrator.set_mode("Arithmetic")
    assert orchestrator.current_mode == "Arithmetic"
    assert "Arithmetic" in orchestrator.modules
    
    # 2. Execute calculation
    res = orchestrator.execute_step("1 + 2 * 3")
    assert res == 7
    
    # 3. Validation Logic (should fail)
    with pytest.raises(Exception): # InvalidExprError from parser
        orchestrator.execute_step("x + 1")

def test_dsl_parsing_with_mode():
    script = """
    problem: MyCalc(Arithmetic) = 5 * 5
    step: 25
    """
    parser = Parser(script)
    prog = parser.parse()
    
    problem = prog.body[0]
    assert problem.name == "MyCalc"
    assert problem.mode == "Arithmetic"
    assert problem.expr.replace(" ", "") == "5*5"

def test_orchestrator_script_execution():
    orchestrator = CoreOrchestrator()
    
    # Script that defines a problem in Arithmetic mode
    script = """
    problem: MyCalc(Arithmetic) = 5 * 5
    step: 25
    """
    
    orchestrator.execute_script(script)
    assert orchestrator.current_mode == "Arithmetic"
    
    # Now interactively solve
    res = orchestrator.execute_step("5 * 5")
    assert res == 25

def test_loading_unknown_mode():
    orchestrator = CoreOrchestrator()
    with pytest.raises(ValueError):
        orchestrator.set_mode("UnknownMode")
