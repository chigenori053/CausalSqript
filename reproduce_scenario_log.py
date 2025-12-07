from causalscript.core.parser import Parser
from causalscript.core.evaluator import Evaluator, SymbolicEvaluationEngine
from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.learning_logger import LearningLogger

def test_scenario_logging():
    script = """
problem: x + y
scenario "Case A":
    x = 1
    y = 2
step: x + y
end: done
"""
    parser = Parser(script)
    program = parser.parse()
    
    sym_engine = SymbolicEngine()
    engine = SymbolicEvaluationEngine(sym_engine)
    logger = LearningLogger()
    
    evaluator = Evaluator(program, engine, learning_logger=logger)
    evaluator.run()
    
    print("--- Logs ---")
    found_scenario = False
    for record in logger.records:
        print(f"Phase: {record.phase}, Meta: {record.meta}")
        if record.phase == "scenario":
            found_scenario = True
            if "context" in record.meta:
                print(f"Scenario Context: {record.meta['context']}")
                
    if found_scenario:
        print("\nSUCCESS: Scenario logged.")
    else:
        print("\nFAILURE: Scenario not logged.")

if __name__ == "__main__":
    test_scenario_logging()
