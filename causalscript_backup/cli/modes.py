"""Reusable evaluator runners for CLI entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from causalscript.core.evaluator import Evaluator
from causalscript.core.knowledge_registry import KnowledgeRegistry
from causalscript.core.learning_logger import LearningLogger
from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.computation_engine import ComputationEngine
from causalscript.core.core_runtime import CoreRuntime
from causalscript.core.validation_engine import ValidationEngine
from causalscript.core.hint_engine import HintEngine
from causalscript.core.polynomial_evaluator import PolynomialEvaluator
from causalscript.core.fuzzy.encoder import ExpressionEncoder
from causalscript.core.fuzzy.metric import SimilarityMetric
from causalscript.core.fuzzy.judge import FuzzyJudge


def run_symbolic_mode(program, learning_logger: LearningLogger) -> KnowledgeRegistry:
    symbolic_engine = SymbolicEngine()
    computation_engine = ComputationEngine(symbolic_engine)
    
    knowledge_registry = KnowledgeRegistry(
        root_path=Path(__file__).resolve().parents[1] / "core" / "knowledge",
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
