"""Helpers to execute MathLang DSL snippets from notebook environments."""

from __future__ import annotations

from textwrap import dedent
from typing import Callable, Dict, List

from coherent.cli import modes as cli_modes
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.log_formatter import format_records
from coherent.engine.parser import Parser


Runner = Callable[[object, LearningLogger], object]


_RUNNERS: Dict[str, Runner] = {
    "symbolic": cli_modes.run_symbolic_mode,
    "polynomial": cli_modes.run_polynomial_mode,
}


def execute_mathlang(
    source: str,
    *,
    mode: str = "symbolic",
    include_meta: bool = True,
) -> List[str]:
    """Compile and execute MathLang DSL text and return formatted log lines."""

    normalized_source = dedent(source)
    if not normalized_source.strip():
        return []

    try:
        runner = _RUNNERS[mode.lower()]
    except KeyError as exc:  # pragma: no cover - defensive guard.
        supported = ", ".join(sorted(_RUNNERS))
        raise ValueError(f"Unsupported MathLang mode '{mode}'. Supported: {supported}") from exc

    program = Parser(normalized_source).parse()
    logger = LearningLogger()
    runner(program, logger)
    return format_records(logger.to_list(), include_meta=include_meta)
