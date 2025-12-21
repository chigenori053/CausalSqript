from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Tuple

from .generator import HypothesisGenerator
from .goal import GoalScanner
from .simulator import LookaheadSimulator
from .types import Hypothesis
# [NEW] Import Trainer
from coherent.optical.trainer import OpticalTrainer

if TYPE_CHECKING:
    from coherent.engine.core_runtime import CoreRuntime

# [NEW] Import Multimodal Integrator
from ..multimodal.integrator import MultimodalIntegrator

# [NEW] Import Memory Components
# [NEW] Import Memory Components
from coherent.memory.factory import get_vector_store
from coherent.memory.ast_generalizer import ASTGeneralizer
from coherent.memory.experience_manager import ExperienceManager
import json

# [NEW] Action Architecture Imports
from coherent.core.action import Action
from coherent.core.action_types import ActionType
from coherent.core.state import State

class ReasoningAgent:
    """
    Autonomous Reasoning Agent (System 2) for Coherent.
    Loops through Generate -> Simulate -> Evaluate to find the best next step.
    Supports Online Learning via OpticalTrainer and Multimodal Perception.
    Integrated with Long-Term Memory (Recall-First Architecture).
    """

    def __init__(self, runtime: CoreRuntime, tensor_engine=None, tensor_converter=None):
        self.runtime = runtime
        if not self.runtime.knowledge_registry:
            raise ValueError("ReasoningAgent requires a KnowledgeRegistry")
            
        self.registry = self.runtime.knowledge_registry
        self.symbolic_engine = self.runtime.computation_engine.symbolic_engine
        
        self.generator = HypothesisGenerator(
            self.registry, 
            self.symbolic_engine,
            optical_weights_path=None 
        )
        self.goal_scanner = GoalScanner(self.symbolic_engine)
        self.simulator = LookaheadSimulator(self.generator, self.registry, self.goal_scanner)

        # [NEW] Initialize Trainer linked to the Generator's Optical Layer
        self.trainer = OpticalTrainer(
            model=self.generator.optical_layer,
            vectorizer=self.generator.vectorizer
        )
        
        # [NEW] Multimodal Perception
        self.integrator = MultimodalIntegrator()
        
        # [NEW] Memory Core (Phase 3)
        self.vector_store = get_vector_store()
        self.generalizer = ASTGeneralizer()
        self.experience_manager = ExperienceManager(self.vector_store) 
 

    def remember_solution(self, initial_expr: str, solution_path: List[str]):
        """
        Learns from a successful solution path.
        Saves each step as an edge in the AST Network (Experience Memory).
        Also stores the full path pattern (Phase 2 legacy support).
        """
        # 1. Edge Learning (Phase 3)
        current_state = initial_expr
        
        # Note: solution_path contains Rule IDs. 
        # We need to simulate the path to get intermediate states if we want strict A->B edges.
        # However, `agent.think` loop doesn't return full trace easily here unless we passed it.
        # For this prototype, we will assume we can get states or we simplify.
        # SIMPLIFICATION: We only save the FIRST step edge for now (Source -> Rule -> Target).
        # To do this correctly, we need the result of applying Rule 1.
        
        # Let's try to apply the first rule to get the next state
        if solution_path:
            first_rule = solution_path[0]
            next_state = self.registry.apply_rule(current_state, first_rule)
            
            if next_state:
                # Generalize
                gen_source = self.generalizer.generalize(current_state)
                gen_target = self.generalizer.generalize(next_state)
                
                # Embed Source (using Multimodal Integrator's text encoder)
                # We need direct access or use integrator
                _, source_vec = self.integrator.process_input(gen_source, input_type="text")
                
                if source_vec:
                    self.experience_manager.save_edge(
                        source_state_gen=gen_source,
                        target_state_gen=gen_target,
                        rule_id=first_rule,
                        source_vector=source_vec
                    )

    def think(self, input_data: str, input_type: str = "text") -> Optional[Hypothesis]:
        """
        Executes one cycle of reasoning.
        Priority: 1. Memory Recall (Fast System 1) -> 2. Computation (Slow System 2).
        """
        # --- 1. Perception & Abstraction ---
        current_expr, semantic_vec = self.integrator.process_input(input_data, input_type)
        if not current_expr:
            return None
            
        gen_expr = self.generalizer.generalize(current_expr)
        
        # --- 2. Memory Recall (Recall-First) ---
        # Query the Experience Network for known transitions from this generalized state
        # We re-embed the GENERALIZED expression for structural retrieval
        # (Note: integrator generated vec for specific expr, we might want vec for generalized one)
        # For efficiency, let's reuse semantic_vec or re-encode if we want strict structural lookups.
        # Let's re-encode gen_expr to be safe about "structure".
        _, gen_vec = self.integrator.process_input(gen_expr, input_type="text")
        
        if gen_vec:
            memories = self.experience_manager.find_similar_edges(gen_vec, top_k=3)
            if memories:
                best_mem = memories[0]
                # If very similar (distance/score check implicit in vector store retrieval)
                # In real app, check score. For now, trust top 1 if exists.
                print(f"[Memory Recall] Found similar experience: {best_mem.rule_id} (-> {best_mem.next_expr})")
                
                # Create a Hypothesis from Memory
                rule_desc = f"Recalled from memory (Rule: {best_mem.rule_id})"
                hyp = Hypothesis(
                    id=f"mem_{best_mem.id}",
                    rule_id=best_mem.rule_id,
                    current_expr=current_expr,
                    next_expr=best_mem.next_expr, # Note: This might be generic form! 
                    score=0.99,
                    metadata={"source": "memory", "rule_description": rule_desc}
                )
                
                # Verify applicability (Reasoning Check)
                # Does this rule actually apply to current specific instance?
                computed_after = self.registry.apply_rule(current_expr, best_mem.rule_id)
                if computed_after:
                    hyp.next_expr = computed_after # Use computed specific result
                    self._add_explanation(hyp)
                    return hyp
                else:
                    print("[Memory Recall] Recalled rule failed application check. Falling back to compute.")

        # --- 3. Computation (Fallback) ---
        print("[Computation] Memory miss. Generating hypotheses...")
        candidates = self.generator.generate(current_expr)
        
        if not candidates:
            return None
            
        # Simulate & Evaluate
        scored_candidates = self.simulator.simulate(candidates, depth=1)
        
        if not scored_candidates:
            return None
            
        # Decision
        best_move = max(scored_candidates, key=lambda x: x.score)
        
        self._add_explanation(best_move)
        
        return best_move

    def retrain(self, training_data: List[Tuple[str, int]], epochs: int = 1) -> float:
        """
        Triggers a retraining session for the Optical Layer.
        
        Args:
            training_data: List of (expression, target_rule_idx) tuples.
            epochs: Number of epochs to train.
            
        Returns:
            avg_loss: The average loss over the last epoch.
        """
        print(f"Starting Optical Retraining with {len(training_data)} samples...")
        avg_loss = 0.0
        for epoch in range(epochs):
            loss = self.trainer.train_epoch(training_data)
            avg_loss = loss
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
            
        return avg_loss

    def _add_explanation(self, hypothesis: Hypothesis) -> None:
        """Generates a natural language explanation for the hypothesis."""
        rule_desc = hypothesis.metadata.get("rule_description", "Use a rule")
        hypothesis.explanation = f"{rule_desc} applied to transform the expression."

    def act(self, state: State) -> Action:
        """
        Predicts the next best Action given the current State.
        Standard interface for the Reasoning LM.
        """
        # 1. Extract context from State
        current_expr = state.current_expression
        
        # 2. Use existing 'think' logic to generate a Hypothesis
        # 'think' returns a Hypothesis(rule_id, next_expr, score, explanation)
        hypothesis = self.think(current_expr)
        
        # 3. Map Hypothesis -> Action
        if hypothesis:
            # If Hypothesis is from memory recall
            if hypothesis.metadata.get("source") == "memory":
                 # We could map this to RECALL action if we want to separate "Search" from "Apply"
                 # But sticking to P0 plan: Output the concrete APPLY_RULE action derived from memory.
                 # or if we strictly follow "Reasoning Agent loops", maybe RECALL is implicit.
                 pass

            return Action(
                type=ActionType.APPLY_RULE,
                name=hypothesis.rule_id,
                inputs={
                    "target": hypothesis.current_expr,
                    "next_state": hypothesis.next_expr,
                    "explanation": hypothesis.explanation
                },
                confidence=hypothesis.score,
                evidence={
                    "rule_id": hypothesis.rule_id,
                    "hypothesis_id": hypothesis.id,
                    "metadata": hypothesis.metadata
                }
            )
        else:
            # If no hypothesis found, what to do?
            # Maybe ASK for help or REJECT?
            # For now, if we are stuck, we might return a low confidence REJECT or STOP.
            return Action(
                type=ActionType.REJECT,
                name="no_solution_found",
                confidence=0.0,
                evidence={"reason": "Search exhausted with no candidates."}
            )
