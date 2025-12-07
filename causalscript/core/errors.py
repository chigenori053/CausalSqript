"""Error classes for the CausalScript Core implementation."""

from __future__ import annotations


class CausalScriptError(Exception):
    """Base class for all CausalScript-specific errors."""


class SyntaxError(CausalScriptError):
    """Raised when the DSL contains invalid syntax."""


class MissingProblemError(CausalScriptError):
    """Raised when a problem declaration is missing before other statements."""


class InvalidStepError(CausalScriptError):
    """Raised when a transformation step is not equivalent to the previous expression."""


class InconsistentEndError(CausalScriptError):
    """Raised when the final expression does not match the expected end expression."""


class InvalidExprError(CausalScriptError):
    """Raised when symbolic or polynomial engines cannot interpret an expression."""


class EvaluationError(CausalScriptError):
    """Raised during expression evaluation."""


class ExtraContentError(CausalScriptError):
    """Raised when additional statements appear after an end declaration."""
