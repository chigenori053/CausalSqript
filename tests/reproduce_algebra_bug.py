import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry
from coherent.engine.reasoning.agent import ReasoningAgent
from coherent.engine.decision_theory import DecisionConfig
from coherent.engine.symbolic_engine import SymbolicEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reproduction")

def reproduce_issue():
    print("--- Reproducing Algebra Issue: 7a - 2a + 4 ---")
    
    # Initialize Engine
    symbolic = SymbolicEngine()
    comp_engine = ComputationEngine(symbolic)
    val_engine = ValidationEngine(comp_engine)
    hint_engine = HintEngine(comp_engine)
    registry = KnowledgeRegistry(root_path=Path("coherent/engine/knowledge"), engine=symbolic)
    
    decision_config = DecisionConfig(strategy="balanced")
    
    runtime = CoreRuntime(
        comp_engine, val_engine, hint_engine, 
        learning_logger=logger, 
        knowledge_registry=registry, 
        decision_config=decision_config
    )
    
    agent = ReasoningAgent(runtime)
    
    input_expr = "7a - 2a + 4"
    print(f"Input: {input_expr}")
    
    try:
        hypothesis = agent.think(input_expr)
        
        if hypothesis:
            print(f"Hypothesis Generated:")
            print(f"  Rule ID: {hypothesis.rule_id}")
            print(f"  Next Expr: {hypothesis.next_expr}")
            print(f"  Metadata: {hypothesis.metadata}")
        else:
            print("No hypothesis generated.")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_issue()
