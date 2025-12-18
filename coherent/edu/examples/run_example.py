"""Utility script to run the bundled MathLang example program."""

from __future__ import annotations

from pathlib import Path

from coherent.engine.evaluator import Evaluator, SymbolicEvaluationEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.edu.dsl import EduParser


EXAMPLE_PATH = Path(__file__).with_name("pythagorean.mlang")


def main() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    program = EduParser(source).parse()
    logger = LearningLogger()
    symbolic = SymbolicEngine()
    knowledge = KnowledgeRegistry(Path("core/knowledge"), symbolic)
    engine = SymbolicEvaluationEngine(symbolic, knowledge)
    evaluator = Evaluator(program, engine=engine, learning_logger=logger)
    evaluator.run()
    for record in logger.to_list():
        rendered = record.get("rendered") or record.get("expression") or ""
        if rendered:
            print(rendered)


if __name__ == "__main__":
    main()
