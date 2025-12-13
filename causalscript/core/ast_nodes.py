"""AST node definitions for the MathLang DSL."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Sequence, Union

# Core Expression Types
@dataclass(frozen=True)
class Int:
    value: int

@dataclass(frozen=True)
class Rat:
    p: int
    q: int

@dataclass(frozen=True)
class Sym:
    name: str

@dataclass(frozen=True)
class Add:
    terms: List['Expr']

@dataclass(frozen=True)
class Mul:
    factors: List['Expr']

@dataclass(frozen=True)
class Pow:
    base: 'Expr'
    exp: 'Expr'

@dataclass(frozen=True)
class Neg:
    expr: 'Expr'

@dataclass(frozen=True)
class Call:
    name: str
    args: List['Expr']

@dataclass(frozen=True)
class Div:
    left: 'Expr'
    right: 'Expr'

@dataclass(frozen=True)
class RationalNode:
    numerator: 'Expr'
    denominator: 'Expr'

Expr = Union[Int, Rat, Sym, Add, Mul, Pow, Neg, Call, Div, RationalNode]

@dataclass(frozen=True)
class Assignment:
    target: str
    expression: Expr

@dataclass(frozen=True)
class Show:
    identifier: str

@dataclass(frozen=True)
class PrepareNode:
    assignments: List['Assignment']

@dataclass(frozen=True)
class Step:
    before: Expr
    after: Expr
    label: str | None = None

@dataclass(frozen=True)
class Problem:
    name: str
    prepare: PrepareNode | None
    steps: List[Step]

# A program can contain multiple problems or other top-level statements.
Statement = Union[Problem, Assignment, Show]

@dataclass(frozen=True)
class Program:
    statements: List[Statement]


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def iter_child_nodes(node: Expr) -> Iterator[Expr]:
    """Yield direct child nodes for recursive traversals."""
    if isinstance(node, Add):
        yield from node.terms
    elif isinstance(node, Mul):
        yield from node.factors
    elif isinstance(node, Pow):
        yield node.base
        yield node.exp
    elif isinstance(node, Neg):
        yield node.expr
    elif isinstance(node, Call):
        yield from node.args
    elif isinstance(node, RationalNode):
        yield node.numerator
        yield node.denominator


def child_nodes(node: Expr) -> Sequence[Expr]:
    """Return the tuple of children for easier inspection/testing."""
    return tuple(iter_child_nodes(node))


# ---------------------------------------------------------------------------
# DSL Node definitions (MathLang Core v2 specification)
# ---------------------------------------------------------------------------

@dataclass
class Node:
    line: int | None = None


@dataclass
class ProgramNode(Node):
    body: List[Node] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.body is None:
            self.body = []


@dataclass
class ProblemNode(Node):
    expr: str = ""
    name: str | None = None
    mode: str | None = None  # arithmetic, algebra, calculus, etc.
    ast: object | None = None


@dataclass
class StepNode(Node):
    step_id: str | None = None
    expr: str = ""
    raw_expr: str | None = None
    ast: object | None = None
    before_expr: str | None = None
    note: str | None = None


@dataclass
class SubProblemNode(Node):
    expr: str = ""
    raw_expr: str = ""
    target_variable: str | None = None



@dataclass
class EndNode(Node):
    expr: str | None = None
    is_done: bool = False
    ast: object | None = None


@dataclass
class ExplainNode(Node):
    text: str = ""


@dataclass
class MetaNode(Node):
    data: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConfigNode(Node):
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeNode(Node):
    mode: str = "strict"


@dataclass
class PrepareNode(Node):
    kind: str = "list"
    expr: str | None = None
    directive: str | None = None
    statements: List[str] = field(default_factory=list)


@dataclass
class CounterfactualNode(Node):
    assume: Dict[str, str] = field(default_factory=dict)
    expect: str | None = None


@dataclass
class ScenarioNode(Node):
    name: str = ""
    assignments: Dict[str, str] = field(default_factory=dict)
