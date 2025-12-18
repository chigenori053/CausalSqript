from typing import List, Optional, Tuple, Dict, Any
import uuid
import numpy as np
import torch # [NEW]

from ..knowledge_registry import KnowledgeRegistry
from ..symbolic_engine import SymbolicEngine
from .types import Hypothesis
from ..optical.vectorizer import FeatureExtractor
from ..optical.layer import OpticalInterferenceEngine

class HypothesisGenerator:
    """
    Generates candidate next steps (Hypotheses) using an Optical-Inspired Hybrid approach.
    1. Vectorize input expression.
    2. Optical Scattering (Score all rules) via PyTorch Layer.
    3. Select Top-k candidates.
    4. Symbolic Verification (Strict Check).
    """

    def __init__(self, registry: KnowledgeRegistry, engine: SymbolicEngine, 
                 optical_weights_path: Optional[str] = None):
        self.registry = registry
        self.engine = engine
        
        # Initialize Optical Components
        self.vectorizer = FeatureExtractor()
        
        # Mapping from output index to Rule ID
        self.rule_ids = [node.id for node in registry.nodes] 
        output_dim = len(self.rule_ids) if self.rule_ids else 100
        
        self.optical_layer = OpticalInterferenceEngine(
            weights_path=optical_weights_path, 
            input_dim=64, 
            memory_capacity=output_dim
        )
        # Set to eval mode for inference by default
        self.optical_layer.eval()

    def generate(self, expr: str) -> List[Hypothesis]:
        """
        Generates all valid hypotheses for the given expression using hybrid reasoning.
        """
        # Normalize equation syntax (a = b -> Eq(a, b))
        normalized_expr = self._normalize_input(expr)
        
        candidates = []
        
        try:
            # --- Optical Phase ---
            # 1. Vectorize
            # For MVP, we still rely on a simplified flow or fallback since full parsing might be heavy.
            # ideally: vector = self.vectorizer.vectorize(ast_node)
            
            # Using a placeholder zero vector if actual parsing is not integrated strictly here yet
            # as per previous implementation pattern.
            vector_np = np.zeros(64, dtype=np.float32) 
            
            # Convert to Tensor
            input_tensor = torch.from_numpy(vector_np)
            
            # 2. Optical Scoring (Inference)
            # Use no_grad for inference to save memory/compute
            with torch.no_grad():
                # [Batch=1, Dim]
                # returns intensity
                intensity = self.optical_layer(input_tensor)
                ambiguity = self.optical_layer.get_ambiguity(intensity)
                
            # Convert intensity to numpy for handling
            scores = intensity.squeeze().cpu().numpy()
            
            # 3. Candidate Generation (Hybrid)
            # We use the standard symbolic matching but enrich it with Optical Scores (Ambiguity).
            # (In a full trained system, we would prune based on 'scores')
            
            matched_rules = self.registry.match_rules(normalized_expr)
            
            for rule, next_expr in matched_rules:
                display_next = self._format_output(next_expr)
                h_id = str(uuid.uuid4())[:8]
                
                # Find score for this rule if possible
                rule_score = 0.0
                if rule.id in self.rule_ids:
                    idx = self.rule_ids.index(rule.id)
                    if idx < len(scores):
                        rule_score = float(scores[idx])
                
                hyp = Hypothesis(
                    id=h_id,
                    rule_id=rule.id,
                    current_expr=expr, 
                    next_expr=display_next,
                    metadata={
                        "rule_description": rule.description,
                        "rule_category": rule.category,
                        "rule_priority": rule.priority,
                        "ambiguity": ambiguity,     # [NEW] Optical Metric
                        "optical_score": rule_score # [NEW] Signal Intensity
                    }
                )
                candidates.append(hyp)

        except Exception as e:
            print(f"Optical Reasoning Error: {e}")
            # Fallback to pure symbolic matching
            matches = self.registry.match_rules(normalized_expr)
            for rule, next_expr in matches:
                display_next = self._format_output(next_expr)
                hyp = Hypothesis(
                    id=str(uuid.uuid4())[:8],
                    rule_id=rule.id,
                    current_expr=expr, 
                    next_expr=display_next,
                    metadata={
                        "rule_description": rule.description,
                        "rule_category": rule.category,
                    }
                )
                candidates.append(hyp)
            
        return candidates

    def _normalize_input(self, expr: str) -> str:
        """Converts user-friendly equation syntax to internal Eq() format if needed."""
        if "=" in expr and "Eq(" not in expr:
            parts = expr.split("=")
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                return f"Eq({lhs}, {rhs})"
        return expr

    def _format_output(self, expr: str) -> str:
        """Converts internal Eq() format back to user-friendly '=' syntax."""
        if expr.startswith("Eq(") and expr.endswith(")"):
            inner = expr[3:-1]
            if "," in inner:
                depth = 0
                split_idx = -1
                for i, char in enumerate(inner):
                    if char in "([{":
                        depth += 1
                    elif char in ")]}":
                        depth -= 1
                    elif char == "," and depth == 0:
                        split_idx = i
                        break
                
                if split_idx != -1:
                    lhs = inner[:split_idx].strip()
                    rhs = inner[split_idx+1:].strip()
                    return f"{lhs} = {rhs}"
        return expr
