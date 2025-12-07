import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.computation_engine import ComputationEngine
from causalscript.core.validation_engine import ValidationEngine
from causalscript.core.hint_engine import HintEngine
from causalscript.core.core_runtime import CoreRuntime
from causalscript.core.parser import Parser
from causalscript.core.evaluator import Evaluator
from causalscript.core.learning_logger import LearningLogger

# Note: Using 'pi' instead of 'π' for simplicity in python string, 
# but the parser handles unicode. Let's use unicode to match user example exactly.
script = """
problem: sin(pi / 3)^2 + cos(pi / 3)^2
step: sin(pi / 3)^2 = (3 / 4)
step: cos(pi / 3)^2 = (1 / 4)
step: (3 / 4) + (1 / 4) = 1
end: 1
"""

def run_test():
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    val_engine = ValidationEngine(comp_engine)
    hint_engine = HintEngine(comp_engine)
    logger = LearningLogger()
    runtime = CoreRuntime(comp_engine, val_engine, hint_engine, learning_logger=logger)
    
    parser = Parser(script)
    program = parser.parse()
    evaluator = Evaluator(program, runtime, learning_logger=logger)
    
    try:
        success = evaluator.run()
        print(f"Success: {success}")
    except Exception as e:
        print(f"Failed with error: {e}")
        
    for record in logger.to_list():
        print(f"{record['phase']}: {record['status']} - {record.get('rendered')}")

if __name__ == "__main__":
    run_test()
