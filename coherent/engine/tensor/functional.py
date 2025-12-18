import torch

def prob_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Probabilistic AND (product t-norm).
    P(A ^ B) = P(A) * P(B) assuming independence.
    """
    return a * b

def prob_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Probabilistic OR (algebraic sum).
    P(A v B) = P(A) + P(B) - P(A ^ B)
    """
    return a + b - (a * b)

def prob_not(a: torch.Tensor) -> torch.Tensor:
    """
    Probabilistic NOT.
    P(~A) = 1 - P(A)
    """
    return 1.0 - a

def fuzzy_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fuzzy AND (Godel t-norm: min).
    """
    return torch.min(a, b)

def fuzzy_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fuzzy OR (max).
    """
    return torch.max(a, b)
