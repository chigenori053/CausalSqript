"""Similarity metrics for fuzzy reasoning."""

from __future__ import annotations

from .encoder import MLVector


class SimilarityMetric:
    """Cosine similarity implementation."""

    def similarity(self, v1: MLVector, v2: MLVector) -> float:
        dot = v1.dot(v2)
        norm_product = v1.norm() * v2.norm()
        if norm_product == 0:
            return 0.0
        value = dot / norm_product
        return max(0.0, min(1.0, value))
