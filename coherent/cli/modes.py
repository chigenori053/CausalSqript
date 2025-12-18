"""Reusable evaluator runners for CLI entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from coherent.engine.evaluator import Evaluator
from coherent.engine.knowledge_registry import KnowledgeRegistry
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.polynomial_evaluator import PolynomialEvaluator
from coherent.engine.fuzzy.encoder import ExpressionEncoder
from coherent.engine.fuzzy.metric import SimilarityMetric
from coherent.engine.fuzzy.judge import FuzzyJudge


def run_symbolic_mode(program, learning_logger: LearningLogger) -> KnowledgeRegistry:
    symbolic_engine = SymbolicEngine()
    computation_engine = ComputationEngine(symbolic_engine)
    
    knowledge_registry = KnowledgeRegistry(
        root_path=Path(__file__).resolve().parents[1] / "engine" / "knowledge",
        engine=symbolic_engine,
    )
    
    fuzzy_judge = FuzzyJudge(
        encoder=ExpressionEncoder(),
        metric=SimilarityMetric(),
    )
    
    validation_engine = ValidationEngine(
        computation_engine, 
        fuzzy_judge=fuzzy_judge, 
        knowledge_registry=knowledge_registry
    )
    hint_engine = HintEngine(computation_engine)
    
    engine = CoreRuntime(
        computation_engine,
        validation_engine,
        hint_engine,
        knowledge_registry=knowledge_registry,
        learning_logger=learning_logger
    )

    evaluator = Evaluator(program, engine, learning_logger=learning_logger)
    evaluator.run()
    return knowledge_registry


def run_polynomial_mode(program, learning_logger: LearningLogger) -> None:
    symbolic_engine = SymbolicEngine()

    if symbolic_engine.has_sympy():

        def normalizer(expr: str) -> str:
            internal = symbolic_engine.to_internal(expr)
            return str(internal.expand())  # type: ignore[attr-defined]

    else:
        assignments = [
            {"x": 1, "y": 2, "z": 3},
            {"x": 2, "y": 1, "z": 0},
            {"a": 1, "b": 2, "c": 3},
        ]

        def normalizer(expr: str) -> str:
            values = []
            for assignment in assignments:
                try:
                    val = symbolic_engine.evaluate_numeric(expr, assignment)
                except Exception:
                    continue
                values.append(str(val))
            if not values:
                raise RuntimeError("Unable to normalize expression without SymPy.")
            return "|".join(values)

    fuzzy_judge = FuzzyJudge(
        encoder=ExpressionEncoder(),
        metric=SimilarityMetric(),
    )
    evaluator = PolynomialEvaluator(
        program,
        normalizer=normalizer,
        learning_logger=learning_logger,
        fuzzy_judge=fuzzy_judge,
    )
    evaluator.run()
