"""IPython cell magic that lets notebooks run MathLang DSL snippets directly."""

from __future__ import annotations

import argparse
import shlex
from typing import Any

from .notebook_runner import execute_mathlang


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="%%mathlang", add_help=False)
    parser.add_argument(
        "--mode",
        choices=sorted({"symbolic", "polynomial"}),
        default="symbolic",
        help="MathLang evaluator mode (default: symbolic).",
    )
    parser.add_argument(
        "--no-meta",
        dest="include_meta",
        action="store_false",
        help="Omit explanation lines from the formatted output.",
    )
    parser.set_defaults(include_meta=True)
    return parser


def _parse_magic_args(line: str) -> argparse.Namespace:
    parser = _build_arg_parser()
    tokens = shlex.split(line, comments=False, posix=True)
    return parser.parse_args(tokens)


def _mathlang_magic(line: str, cell: str) -> None:
    """Entry point registered as %%mathlang cell magic."""

    args = _parse_magic_args(line)
    lines = execute_mathlang(
        cell,
        mode=args.mode,
        include_meta=args.include_meta,
    )
    if not lines:
        return
    print("\n".join(lines))


def load_ipython_extension(ipython: Any) -> None:
    """Notebook hook: %load_ext tools.notebook_magic."""

    ipython.register_magic_function(_mathlang_magic, "cell", "mathlang")


def unload_ipython_extension(ipython: Any) -> None:
    """Notebook hook for %unload_ext."""

    manager = getattr(ipython, "magics_manager", None)
    if manager is None:
        return
    registry = getattr(manager, "registry", None)
    if isinstance(registry, dict):
        registry.pop("mathlang", None)
