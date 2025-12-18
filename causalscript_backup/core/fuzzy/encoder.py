"""Encoders for fuzzy reasoning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from .types import NormalizedExpr


@dataclass(frozen=True)
class MLVector:
    """MathLang vector representation."""

    data: tuple[float, ...]

    def dot(self, other: "MLVector") -> float:
        return sum(a * b for a, b in zip(self.data, other.data))

    def norm(self) -> float:
        return sum(value * value for value in self.data) ** 0.5

    @classmethod
    def zeros(cls, dimension: int) -> "MLVector":
        return cls(tuple(0.0 for _ in range(dimension)))


class ExpressionEncoder:
    """Deterministic encoder for expressions and text."""

    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension

    def encode_expr(self, expr: NormalizedExpr) -> MLVector:
        vector = [0.0] * self.dimension
        self._accumulate(vector, expr["raw"], "raw")
        self._accumulate(vector, expr["sympy"], "sympy")
        for index, token in enumerate(expr["tokens"]):
            self._accumulate(vector, f"{index}:{token}", "tok")
        return self._finalize(vector)

    def encode_text(self, text: str) -> MLVector:
        if not text:
            return MLVector.zeros(self.dimension)
        vector = [0.0] * self.dimension
        self._accumulate(vector, text, "text")
        return self._finalize(vector)

    def _accumulate(self, vector: list[float], payload: str, salt: str) -> None:
        hashed = hashlib.sha256(f"{salt}:{payload}".encode("utf-8")).digest()
        for i in range(self.dimension):
            byte = hashed[i % len(hashed)]
            value = (byte / 255.0) * 2.0 - 1.0
            vector[i] += value

    def normalize(self, text: str) -> NormalizedExpr:
        """Normalize raw text into a structured expression."""
        # Simple tokenization for now
        tokens = text.replace("(", " ( ").replace(")", " ) ").split()
        return {
            "raw": text,
            "sympy": text,  # Placeholder: ideally use specialized parser
            "tokens": tokens,
        }

    def _finalize(self, vector: list[float]) -> MLVector:
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return MLVector.zeros(self.dimension)
        normalized = tuple(value / norm for value in vector)
        return MLVector(normalized)
