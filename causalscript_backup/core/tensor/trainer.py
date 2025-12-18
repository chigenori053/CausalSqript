import torch
import torch.nn as nn
import torch.optim as optim
import json
from typing import List, Dict, Any
from pathlib import Path

from .engine import TensorLogicEngine
from .converter import TensorConverter

class TensorTrainer:
    """
    Trains the TensorLogicEngine using execution logs.
    """
    def __init__(self, engine: TensorLogicEngine, converter: TensorConverter, learning_rate: float = 0.01):
        self.engine = engine
        self.converter = converter
        self.optimizer = optim.Adam(self.engine.parameters(), lr=learning_rate)
        # Using CrossEntropyLoss? 
        # Output of engine for prediction is "scores" for each rule.
        # We need to map rule_id to an index for CrossEntropy.
        # Engine rule_weights are stored by rule_id string.
        # We need a stable mapping from rule_id -> class_index.
        self.rule_id_to_idx: Dict[str, int] = {}
        self.idx_to_rule_id: Dict[int, str] = {}
        
    def _prepare_rule_indices(self):
        """Builds index mapping for all registered rules."""
        # Ensure we know all rules currently in engine
        rule_ids = sorted(list(self.engine.rule_weights.keys()))
        self.rule_id_to_idx = {r: i for i, r in enumerate(rule_ids)}
        self.idx_to_rule_id = {i: r for i, r in enumerate(rule_ids)}

    def train_from_logs(self, log_paths: List[Path], epochs: int = 1):
        """
        Loads logs and runs training.
        """
        dataset = []
        
        # 1. Load Data
        for path in log_paths:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding='utf-8'))
            for entry in data:
                # We care about steps where a rule was applied
                if entry.get("rule_id") and entry.get("expression") and entry.get("status") == "ok": # Or "valid"
                    dataset.append({
                        "expr": entry["expression"],
                        "rule": entry["rule_id"]
                    })
        
        if not dataset:
            print("No training data found.")
            return

        # Ensure all rules in dataset are registered
        for data in dataset:
            self.engine.register_rule(data["rule"])
            
        self._prepare_rule_indices()
        
        # 2. Training Loop
        self.engine.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for item in dataset:
                self.optimizer.zero_grad()
                
                # Input
                expr_tensor = self.converter.encode(item["expr"])
                if expr_tensor.dim() == 0:
                     continue
                
                # Target
                target_rule = item["rule"]
                target_idx = self.rule_id_to_idx[target_rule]
                target_tensor = torch.tensor([target_idx], dtype=torch.long)
                
                # Forward
                # We need raw scores for all rules in fixed order
                # predict_rules returns top-k strings, we need raw logits for CrossEntropy
                logits = self._get_logits_for_all_rules(expr_tensor)
                
                # Loss
                # logits: [1, NumRules], target: [1]
                loss = nn.CrossEntropyLoss()(logits, target_tensor)
                
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataset)
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def _get_logits_for_all_rules(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Helper: compute scores for all rules in order of self.idx_to_rule_id.
        """
        # Get state vector
        # [Emb]
        state_vector = self.engine(state_tensor)
        
        # Compute scores
        # We want to enable gradient flow.
        # Score = weight + ... (same as predict_rules but ordered)
        scores = []
        for i in range(len(self.idx_to_rule_id)):
            rid = self.idx_to_rule_id[i]
            # Access parameter directly to keep graph
            weight = self.engine.rule_weights[rid]
            # In real model: dot product etc.
            # score = f(state, rule)
            score = weight # scalar
            scores.append(score)
            
        return torch.cat(scores).unsqueeze(0) # [1, NumRules]
