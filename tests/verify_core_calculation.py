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
logger = logging.getLogger("Verification")

def verify_calculations():
    print("--- COHERENT Core Calculation Verification ---")
    
    # Initialize Engine
    symbolic = SymbolicEngine()
    comp_engine = ComputationEngine(symbolic)
    val_engine = ValidationEngine(comp_engine)
    hint_engine = HintEngine(comp_engine)
    registry = KnowledgeRegistry(root_path=Path("coherent/engine/knowledge"), engine=symbolic)
    
    decision_config = DecisionConfig(
        strategy="balanced"
    )
    
    runtime = CoreRuntime(
        comp_engine, val_engine, hint_engine, 
        learning_logger=logger, 
        knowledge_registry=registry, 
        decision_config=decision_config
    )
    
    agent = ReasoningAgent(runtime)
    
    test_cases = [
        {"input": "3x + 5x", "type": "Algebra Simplification", "expected_snippet": "8*x"},
        {"input": "1 + 2 * 3", "type": "Arithmetic", "expected_stub": "7"}, # Accept 7 or 1+6
        {"input": "diff(x^2, x)", "type": "Calculus Differentiation", "expected_snippet": "2*x"},
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\nTesting: {case['input']} ({case['type']})")
        try:
            hypothesis = agent.think(case['input'])
            
            if hypothesis:
                result_str = str(hypothesis.next_expr)
                print(f"  Result: {result_str}")
                print(f"  Rule Used: {hypothesis.rule_id}")
                
                # Basic validation
                expected = case.get('expected_snippet') or case.get('expected_stub')
                if expected in result_str.replace(" ", "") or result_str.replace(" ", "") == "1+6":
                    print("  [PASS] Result matches expectation.")
                    results.append((case['input'], True, result_str))
                else:
                    print(f"  [FAIL] Expected '{expected}' in '{result_str}'")
                    results.append((case['input'], False, result_str))
            else:
                print("  [FAIL] No hypothesis generated.")
                results.append((case['input'], False, "No Output"))
                
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((case['input'], False, str(e)))

    print("\n--- Summary ---")
    success_count = sum(1 for _, success, _ in results if success)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {success_count}")
    print(f"Failed: {len(results) - success_count}")

if __name__ == "__main__":
    verify_calculations()
