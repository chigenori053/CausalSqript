import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from coherent.engine.language.semantic_parser import RuleBasedSemanticParser
from coherent.core.state import State
from coherent.core.executor import ActionExecutor
from coherent.core.tracer import Tracer
from coherent.core.action_types import ActionType
from coherent.engine.language.semantic_types import TaskType

# Re-use the App's system initialization logic for consistency
# But simpler for CLI
from ui.app import get_system

def run_e2e(input_text: str):
    print(f"--- Starting E2E Verification for: '{input_text}' ---")
    
    # 1. Initialize System
    print("[1] Initializing System...")
    # Mock streamlit cache for get_system if needed, or just call logic
    # app.py's get_system is cached by streamlit, directly calling it might fail if st not internal 
    # but we can copy pertinent parts or try importing.
    # Actually app.py code imports 'st', so running it might need st context?
    # Let's just manually init what we need to avoid UI dependencies.
    
    from coherent.engine.symbolic_engine import SymbolicEngine
    from coherent.engine.computation_engine import ComputationEngine
    from coherent.engine.validation_engine import ValidationEngine
    from coherent.engine.hint_engine import HintEngine
    from coherent.engine.core_runtime import CoreRuntime
    from coherent.engine.knowledge_registry import KnowledgeRegistry
    from coherent.engine.reasoning.agent import ReasoningAgent
    from coherent.engine.tensor.engine import TensorLogicEngine
    from coherent.engine.tensor.converter import TensorConverter
    from coherent.engine.tensor.embeddings import EmbeddingRegistry
    from coherent.engine.fuzzy.judge import FuzzyJudge
    from coherent.engine.fuzzy.encoder import ExpressionEncoder
    from coherent.engine.fuzzy.metric import SimilarityMetric
    from coherent.engine.decision_theory import DecisionConfig

    # Init
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    
    knowledge_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "coherent", "engine", "knowledge")))
    knowledge_registry = KnowledgeRegistry(knowledge_path, sym_engine)
    
    decision_config = DecisionConfig(strategy="balanced")
    fuzzy_judge = FuzzyJudge(ExpressionEncoder(), SimilarityMetric(), decision_config=decision_config, symbolic_engine=sym_engine)
    val_engine = ValidationEngine(comp_engine, fuzzy_judge=fuzzy_judge, decision_engine=fuzzy_judge.decision_engine, knowledge_registry=knowledge_registry)
    hint_engine = HintEngine(comp_engine)
    
    runtime = CoreRuntime(comp_engine, val_engine, hint_engine, knowledge_registry=knowledge_registry)
    
    embedding_registry = EmbeddingRegistry()
    tensor_converter = TensorConverter(embedding_registry)
    tensor_engine = TensorLogicEngine(vocab_size=1000, embedding_dim=64)
    
    agent = ReasoningAgent(runtime, tensor_engine=tensor_engine, tensor_converter=tensor_converter)
    
    parser = RuleBasedSemanticParser()
    executor = ActionExecutor(runtime)
    tracer = Tracer()
    
    # 2. Parse Input
    print(f"[2] Parsing Input: '{input_text}'")
    ir = parser.parse(input_text)
    print(f"    Intent: {ir.task}, Domain: {ir.math_domain}")
    
    if not ir.inputs:
        print("ERROR: No extracted inputs.")
        return

    expr = ir.inputs[0].value
    print(f"    Extracted: {expr}")
    
    # 3. Initialize State & Episode
    ep_id = tracer.start_episode(input_text)
    print(f"[3] Started Episode: {ep_id}")
    
    state = State(
        task_goal=ir.task,
        initial_inputs=ir.inputs,
        current_expression=expr
    )
    
    # Set runtime focus
    runtime.set(expr)
    
    # 4. Reason Loop
    print("[4] Entering Reasoning Loop...")
    max_steps = 5
    
    for i in range(max_steps):
        print(f"\n--- Step {i+1} ---")
        print(f"Current State: {state.current_expression}")
        
        # Check if solved (simple heuristic for test: x = number)
        # Or let agent decide FINAL? (Not implemented in agent logic yet, stick to rule loop)
        
        # Look for action
        action = agent.act(state)
        print(f"Predicted Action: {action.type.name} - {action.name} (Conf: {action.confidence:.2f})")
        
        if action.type == ActionType.REJECT:
            print("Agent stopped (REJECT).")
            break
            
        # Execute
        result = executor.execute(action, state)
        print(f"Execution Result: {result.get('valid')} / {result.get('status')}")
        
        # Log
        tracer.log_step(state, action, result)
        
        if result.get("valid"):
            # Update state is handled by executor mostly, but let's confirm
            print(f"New Expression: {state.current_expression}")
            
            # Simple termination check
            # In real LM finalization, Agent should emit FINAL action.
            # Here we just stop if "x =" pattern or constant
            pass
        else:
            print("Action Failed Validity Check.")
            # In a real loop, might try again or ASK.
            break

    # 5. Finish
    tracer.end_episode("COMPLETED")
    print("\n[5] Episode Trace:")
    history = tracer.export_history()
    print(json.dumps(history[0]["steps"], indent=2, default=str))

if __name__ == "__main__":
    test_input = "Solve x^2 - 4 = 0"
    if len(sys.argv) > 1:
        test_input = sys.argv[1]
    run_e2e(test_input)
