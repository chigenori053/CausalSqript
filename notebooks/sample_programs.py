"""Sample MathLang DSL programs for notebook or UI testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SampleProgram:
    """Container for reusable MathLang snippets."""

    id: str
    title: str
    description: str
    source: str


SAMPLE_PROGRAMS: List[SampleProgram] = [
    SampleProgram(
        id="binomial_square",
        title="Square of a binomial",
        description="Expands (x - y)^2 using distributive multiplication.",
        source="""problem: (x - y)^2
step: (x - y) (x - y)
step: x^2 - 2xy + y^2
end: done
""",
    ),
    SampleProgram(
        id="trinomial_square",
        title="Square of a trinomial",
        description="Expands (a + b + c)^2 step by step.",
        source="""problem: (a + b + c)^2
step: (a + b + c) (a + b + c)
step: a^2 + b^2 + c^2 + 2ab + 2ac + 2bc
end: done
""",
    ),
    SampleProgram(
        id="difference_of_squares",
        title="Difference of squares verification",
        description="Shows factoring of x^2 - 9 into (x - 3)(x + 3).",
        source="""problem: x^2 - 9
step: (x - 3) (x + 3)
step: x^2 + 3x - 3x - 9
step: x^2 - 9
end: done
""",
    ),
    SampleProgram(
        id="polynomial_simplification",
        title="Simplify polynomial addition",
        description="Combines like terms from two polynomials.",
        source="""problem: (2x^2 + 3x - 4) + (x^2 - 5x + 6)
step: 2x^2 + 3x - 4 + x^2 - 5x + 6
step: 3x^2 - 2x + 2
end: done
""",
    ),
]


def get_sample_program(program_id: str) -> SampleProgram:
    """Return a sample program by ID, raising KeyError if missing."""

    lookup: Dict[str, SampleProgram] = {program.id: program for program in SAMPLE_PROGRAMS}
    return lookup[program_id]


def list_sample_sources() -> Dict[str, str]:
    """Convenience helper returning {id: source} for quick selection."""

    return {program.id: program.source for program in SAMPLE_PROGRAMS}
