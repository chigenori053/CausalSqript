import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from .layer import OpticalInterferenceEngine
from .vectorizer import FeatureExtractor

class OpticalTrainer:
    """
    Handles the training loop for the OpticalInterferenceEngine using Backpropagation.
    """
    def __init__(self, model: OpticalInterferenceEngine, vectorizer: FeatureExtractor, lr: float = 0.01):
        self.model = model
        self.vectorizer = vectorizer
        # Adam optimizer works well for complex parameters too in PyTorch
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_one_step(self, expr_str: str, target_rule_idx: int) -> float:
        """
        Performs one step of Gradient Descent.
        
        Args:
            expr_str: The input expression string.
            target_rule_idx: The index of the rule that was successfully applied (ground truth).
            
        Returns:
            loss: The loss value for this step.
        """
        self.model.train() # Ensure training mode
        
        # 1. Preprocessing
        # In a real scenario, we parse expr_str to AST here.
        # Since we haven't integrated the parser fully in this isolated module,
        # we will use the same fallback as Generator (Mock Vector).
        # TODO: Integrate InputParser to get real AST.
        
        # Mocking the vector for now to ensure pipeline works
        # Use simple non-zero values to ensure gradients flow (zeros kill gradient in linear layer)
        vector_np = np.ones(self.model.input_dim, dtype=np.float32) * 0.1
        
        input_tensor = torch.from_numpy(vector_np).unsqueeze(0) # [1, Dim]
        
        # 2. Forward Pass
        self.optimizer.zero_grad()
        
        # returns intensity (resonance energy)
        intensity = self.model(input_tensor) # [1, NumRules]
        
        # 3. Loss Calculation
        # We want to maximize intensity at target_rule_idx
        # CrossEntropyLoss expects logits (unnormalized scores). Intensity is valid as logits here.
        target = torch.tensor([target_rule_idx], dtype=torch.long)
        
        loss = self.criterion(intensity, target)
        
        # 4. Backward Pass & Optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train_epoch(self, data: List[Tuple[str, int]]) -> float:
        """
        Trains on a batch of data (list of (expr, label)).
        """
        total_loss = 0.0
        for expr, label in data:
            total_loss += self.train_one_step(expr, label)
        return total_loss / len(data) if data else 0.0
