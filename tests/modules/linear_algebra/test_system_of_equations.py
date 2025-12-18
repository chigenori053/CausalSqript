
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coherent.engine.parser import Parser
from coherent.engine.evaluator import Evaluator, SymbolicEvaluationEngine
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.learning_logger import LearningLogger

def test_system_of_equations_flow():
    script = """
problem: 
 y = 3x - 2
 2x + y = 13
step: 2x + (3x - 2) = 13
step: 2x + 3x - 2 = 13
step: 5x - 2 = 13
step: 5x = 15
step: x = 3

step: y = 3(3) - 2
step: y = 9 - 2
step: y = 7

step: x = 3
      y = 7
end: done
"""
    print("Parsing script...")
    # 1. Parse
    try:
        parser = Parser(script)
        program = parser.parse()
    except Exception as e:
        print(f"Parsing failed as expected: {e}")
        return

    print("Parsing succeeded (unexpectedly?)")

    # 2. Evaluate
    sym_engine = SymbolicEngine()
    engine = SymbolicEvaluationEngine(sym_engine)
    logger = LearningLogger()
    evaluator = Evaluator(program, engine, learning_logger=logger)
    
    success = evaluator.run()
    
    # Check logs
    logs = logger.to_list()
    for log in logs:
        print(f"{log['phase']}: {log['status']} - {log['expression']}")
        
    if not success:
        print("Evaluation failed as expected (if parsing succeeded)")
    else:
        print("Evaluation succeeded (unexpectedly)")

if __name__ == "__main__":
    test_system_of_equations_flow()
