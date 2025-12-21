import sys
import os
from coherent.engine.language.semantic_parser import RuleBasedSemanticParser
from coherent.engine.reasoning.agent import ReasoningAgent
from coherent.core.state import State
from coherent.core.action import Action
from coherent.core.action_types import ActionType
from coherent.core.executor import ActionExecutor
from coherent.engine.core_runtime import CoreRuntime
from ui.app import get_system

# Mock System Init
system = get_system()
parser = system["semantic_parser"]
agent = system["agent"]
executor = system["executor"]
runtime = system['runtime']

def reproduce():
    text = "Factorize x^2 + 5x + 6"
    print(f"Input: {text}")
    
    # 1. Parse
    ir = parser.parse(text)
    print(f"Parsed Task: {ir.task}")
    print(f"Parsed Domain: {ir.math_domain}")
    
    if not ir.inputs:
        print("Error: No inputs extracted.")
        return
        
    extracted_expr = ir.inputs[0].value
    print(f"Extracted Expression: '{extracted_expr}'")
    
    # 2. State
    state = State(
        task_goal=ir.task,
        initial_inputs=ir.inputs,
        current_expression=extracted_expr
    )
    runtime.set(extracted_expr)
    
    # 3. Agent Act
    print("Agent Acting...")
    action = agent.act(state)
    print(f"Action Type: {action.type}")
    print(f"Action Name: {action.name}")
    print(f"Action Inputs: {action.inputs}")
    
    # 4. Agent Think (Direct check)
    print("Direct Agent Think Check:")
    hyp = agent.think(extracted_expr)
    if hyp:
        print(f"Hypothesis Next Expr: '{hyp.next_expr}'")
    else:
        print("No Hypothesis found.")

if __name__ == "__main__":
    reproduce()
