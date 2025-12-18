import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class TensorLogicEngine(nn.Module):
    """
    Neural network module for CausalScript reasoning.
    Manages embeddings for symbols and executes tensor-based logic rules.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Embedding layer: Maps term IDs to dense vectors
        # Using padding_idx=0 since EmbeddingRegistry reserves 0
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Trainable weights for rules (can be expanded later)
        self.rule_weights = nn.ParameterDict()
        
        self.to(device)
        
    def forward(self, state_tensor: torch.Tensor, rule_indices: Optional[torch.Tensor] = None):
        """
        Forward pass for the logic engine.
        
        Args:
            state_tensor: Tensor representing the current state (e.g., sequence of token IDs).
            rule_indices: Optional tensor of rule IDs to apply or evaluate.
            
        Returns:
            Output depends on the mode (prediction vs evaluation).
        """
        # Placeholder: Compute a "state vector" from input tokens using simple pooling
        # [Batch, Seq] -> [Batch, Seq, Emb] -> [Batch, Emb]
        embedded = self.embeddings(state_tensor)
        
        # Simple mean pooling for now to get a fixed-size state representation
        if embedded.dim() == 3:
            state_vector = embedded.mean(dim=1)
        else:
            state_vector = embedded.mean(dim=0, keepdim=True) # Single item
            
        return state_vector

    def register_rule(self, rule_id: str):
        """
        Registers a rule with a learnable weight parameter.
        """
        if rule_id not in self.rule_weights:
            # Initialize weight (scalar score)
            self.rule_weights[rule_id] = nn.Parameter(torch.randn(1))

    def predict_rules(self, state_tensor: torch.Tensor, top_k: int = 5) -> List[str]:
        """
        Predicts the most likely rules to apply given the current state.
        
        Args:
            state_tensor: [SeqLen] tensor of token IDs.
            top_k: Number of rules to return.
            
        Returns:
            List of rule_ids.
        """
        if not self.rule_weights:
            return []
            
        # Get state vector
        state_vector = self.forward(state_tensor)
        
        # Determine scores for each rule
        # For now, simplistic: score = rule.weight + dot(state, rules_embedding_placeholder)
        # Since we don't have rule embeddings yet, just use the rule weights (global bias)
        # This acts as a "prior"
        
        rule_ids = list(self.rule_weights.keys())
        scores = []
        
        for rid in rule_ids:
            # In a real model, this would be a proper network computation
            # score = f(state, rule)
            score = self.rule_weights[rid].item()
            scores.append(score)
            
        # Sort by score descending
        sorted_indices = torch.tensor(scores).argsort(descending=True)
        top_indices = sorted_indices[:top_k]
        
        return [rule_ids[i] for i in top_indices]

    def evaluate_state(self, state_tensors: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the 'goodness' (nearness to goal/simplicity) of a batch of states.
        
        Args:
            state_tensors: [Batch, SeqLen]
            
        Returns:
            torch.Tensor: [Batch] scores.
        """
        # [Batch, Emb]
        state_vectors = self.forward(state_tensors)
        
        # Dummy evaluation: norm of the vector (just to have some number)
        # In reality this would be a Value Network V(s)
        scores = torch.norm(state_vectors, dim=1)
        return scores


    def get_similarity(self, term_a_idx: int, term_b_idx: int) -> float:
        """
        Computes the cosine similarity between the embeddings of two terms.
        
        Args:
            term_a_idx: ID of the first term.
            term_b_idx: ID of the second term.
            
        Returns:
            Cosine similarity score (-1.0 to 1.0).
        """
        # Ensure indices are tensors on the correct device
        idx_a = torch.tensor([term_a_idx], device=self.device)
        idx_b = torch.tensor([term_b_idx], device=self.device)
        
        with torch.no_grad():
            vec_a = self.embeddings(idx_a)
            vec_b = self.embeddings(idx_b)
            score = F.cosine_similarity(vec_a, vec_b, dim=-1)
            
        return float(score.item())

    def update_embeddings(self, vocab_size: int):
        """
        Expands the embedding layer if the vocabulary grows.
        """
        current_vocab, dim = self.embeddings.weight.shape
        if vocab_size > current_vocab:
            new_embeddings = nn.Embedding(vocab_size, dim, padding_idx=0).to(self.device)
            # Copy existing weights
            with torch.no_grad():
                new_embeddings.weight[:current_vocab] = self.embeddings.weight
            self.embeddings = new_embeddings
