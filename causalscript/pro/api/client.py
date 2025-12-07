"""Simple wrappers that expose MathLang capabilities to pro notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from causalscript.core.causal import run_causal_analysis
from causalscript.core.parser import Parser
from causalscript.core.learning_logger import LearningLogger
from causalscript.core.evaluator import Evaluator, SymbolicEvaluationEngine
from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.knowledge_registry import KnowledgeRegistry


def _evaluate(source: str) -> Iterable[Dict[str, Any]]:
    program = Parser(source).parse()
    sym = SymbolicEngine()
    registry = KnowledgeRegistry(Path("core/knowledge"), sym)
    engine = SymbolicEvaluationEngine(sym, registry)
    logger = LearningLogger()
    Evaluator(program, engine, learning_logger=logger).run()
    return logger.to_list()


def run_causal_analysis_from_source(source: str) -> Dict[str, Any]:
    records = list(_evaluate(source))
    _, report = run_causal_analysis(records, include_graph=True)
    return report


def run_counterfactual(source: str, intervention: Dict[str, Any]) -> Dict[str, Any]:
    records = list(_evaluate(source))
    engine, _ = run_causal_analysis(records)
    return engine.counterfactual_result(intervention, records)


# Backwards-compatible names
run_causal_analysis = run_causal_analysis_from_source
