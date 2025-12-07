"""DSL helpers for MathLang Edu edition."""

from causalscript.core.parser import Parser as CoreParser


class EduParser(CoreParser):
    """Thin wrapper for future Edu-specific DSL extensions."""


__all__ = ["EduParser"]
