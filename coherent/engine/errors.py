"""Error classes for the Coherent Core implementation."""

from __future__ import annotations


class CoherentError(Exception):
    """Base class for all Coherent-specific errors."""


class SyntaxError(CoherentError):
    """Raised when the DSL contains invalid syntax."""


class MissingProblemError(CoherentError):
    """Raised when a problem declaration is missing before other statements."""


class InvalidStepError(CoherentError):
    """Raised when a transformation step is not equivalent to the previous expression."""


class InconsistentEndError(CoherentError):
    """Raised when the final expression does not match the expected end expression."""


class InvalidExprError(CoherentError):
    """Raised when symbolic or polynomial engines cannot interpret an expression."""


class EvaluationError(CoherentError):
    """Raised during expression evaluation."""


class ExtraContentError(CoherentError):
    """Raised when additional statements appear after an end declaration."""
