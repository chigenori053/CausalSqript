import torch
import re
from typing import List
from .embeddings import EmbeddingRegistry

class TensorConverter:
    """
    Converts symbolic expressions (strings) into Tensor representations.
    Uses EmbeddingRegistry to map tokens to IDs.
    """
    def __init__(self, registry: EmbeddingRegistry):
        self.registry = registry
        # Regex to tokenize: alphanumeric words, or individual operator symbols
        # Captures words like 'Parent', 'x', and symbols like '(', ')', ','
        self.token_pattern = re.compile(r'\w+|[^\w\s]')

    def tokenize(self, expr: str) -> List[str]:
        """
        Splits an expression string into tokens.
        Example: "Parent(x, y)" -> ["Parent", "(", "x", ",", "y", ")"]
        """
        return self.token_pattern.findall(expr)

    def encode(self, expr: str, register_new: bool = True) -> torch.Tensor:
        """
        Converts an expression string into a 1D tensor of IDs.
        
        Args:
            expr: The expression string (e.g., "Parent(x, y)").
            register_new: Whether to register new tokens in the registry.
            
        Returns:
            torch.Tensor: LongTensor of shape [sequence_length].
        """
        tokens = self.tokenize(expr)
        ids = [self.registry.get_id(t, register_if_missing=register_new) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, tensor: torch.Tensor) -> str:
        """
        Converts a tensor of IDs back into an expression string (reconstructed).
        Note: Original spacing is lost.
        """
        ids = tensor.tolist()
        tokens = []
        for idx in ids:
            term = self.registry.get_term(idx)
            if term is None:
                tokens.append("<?>")
            else:
                tokens.append(term)
        # Simple join, improving spacing would require more sophisticated logic
        return "".join(tokens)

    def batch_encode(self, exprs: List[str]) -> torch.Tensor:
        """
        Encodes a list of expressions into a padded batch tensor.
        
        Returns:
            torch.Tensor: Shape [Batch, MaxLen]
        """
        tensors = [self.encode(e) for e in exprs]
        # Pad sequence is not imported, let's just do manual padding for simplicity or use pad_sequence if available utils
        # Importing pad_sequence
        from torch.nn.utils.rnn import pad_sequence
        return pad_sequence(tensors, batch_first=True, padding_value=0)
