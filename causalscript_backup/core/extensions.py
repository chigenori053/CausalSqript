"""
Extensions interface for Modular Computation Engines.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class ComputationExtension(Protocol):
    """
    Protocol for computation extensions that can be lazy-loaded by CoreRuntime.
    Extensions are specialized engines (Calculus, LinearAlgebra, Stats, etc.)
    that are not required for every session.
    """
    
    def normalize_name(self) -> str:
        """Return the canonical name of this extension (e.g., 'calculus')."""
        ...
