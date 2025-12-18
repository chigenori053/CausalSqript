import torch
import torch.nn as nn
from typing import Optional, List

class TensorRuleLayer(nn.Module):
    """
    Executes a logical rule using tensor contraction (einsum).
    Represents rules like: GP(x, z) <- P(x, y) ^ P(y, z)
    """
    def __init__(self, equation: str, activation: str = 'sigmoid'):
        """
        Args:
            equation: Einsum equation string (e.g., 'bxy,byz->bxz')
            activation: Activation function to apply ('sigmoid', 'relu', 'none')
        """
        super().__init__()
        self.equation = equation
        self.activation_name = activation
        
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply the rule to input tensors.
        
        Args:
            *inputs: Variable number of tensors matching the equation.
            
        Returns:
            Result tensor.
        """
        # Tensor contraction
        output = torch.einsum(self.equation, *inputs)
        
        # Apply activation
        if self.activation_name == 'sigmoid':
            return torch.sigmoid(output)
        elif self.activation_name == 'relu':
            return torch.relu(output)
        elif self.activation_name == 'tanh':
            return torch.tanh(output)
        
        return output

class LogicChainLayer(TensorRuleLayer):
    """
    Specialized layer for chain rules: A(x, y) ^ B(y, z) -> C(x, z)
    """
    def __init__(self):
        super().__init__('bxy,byz->bxz', activation='sigmoid')

class LogicAndLayer(TensorRuleLayer):
    """
    Specialized layer for intersection: A(x) ^ B(x) -> C(x)
    """
    def __init__(self):
        super().__init__('bx,bx->bx', activation='sigmoid')
