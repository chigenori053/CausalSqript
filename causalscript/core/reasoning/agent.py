from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List

from .generator import HypothesisGenerator
from .goal import GoalScanner
from .simulator import LookaheadSimulator
from .types import Hypothesis

if TYPE_CHECKING:
    from ..core_runtime import CoreRuntime

class ReasoningAgent:
    """
    Autonomous Reasoning Agent (System 2) for CausalScript.
    Loops through Generate -> Simulate -> Evaluate to find the best next step.
    """

    def __init__(self, runtime: CoreRuntime):
        self.runtime = runtime
        if not self.runtime.knowledge_registry:
            raise ValueError("ReasoningAgent requires a KnowledgeRegistry")
            
        self.registry = self.runtime.knowledge_registry
        self.symbolic_engine = self.runtime.computation_engine.symbolic_engine
        
        self.generator = HypothesisGenerator(self.registry, self.symbolic_engine)
        self.goal_scanner = GoalScanner(self.symbolic_engine)
        self.simulator = LookaheadSimulator(self.generator, self.registry, self.goal_scanner)

    def think(self, current_expr: str) -> Optional[Hypothesis]:
        """
        Executes one cycle of reasoning to find the best next step.
        """
        # 1. Generate: Find candidate rules
        candidates = self.generator.generate(current_expr)
        
        if not candidates:
            return None
            
        # 2. Simulate & Evaluate: Lookahead and scoring
        # Use simple depth=1 for now as per design
        scored_candidates = self.simulator.simulate(candidates, depth=1)
        
        if not scored_candidates:
            return None
            
        # 3. Decision: Select best move
        best_move = max(scored_candidates, key=lambda x: x.score)
        
        # Add explanation (simple template for now)
        self._add_explanation(best_move)
        
        return best_move

    def _add_explanation(self, hypothesis: Hypothesis) -> None:
        """Generates a natural language explanation for the hypothesis."""
        rule_desc = hypothesis.metadata.get("rule_description", "Use a rule")
        hypothesis.explanation = f"{rule_desc} applied to transform the expression."
