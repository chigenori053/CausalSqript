# CausalScript Causal Inference Engine – Implementation Spec (v1)

> Target: Python 3.12, existing CausalScript Core (Parser / Evaluator / SymbolicEngine / KnowledgeRegistry / LearningLogger)  
> Purpose: Add **causal inference capabilities** on top of the existing **step-by-step reasoning log**.

---

## 1. Scope

### 1.1 Goals

- Provide a **Causal Inference Engine** that can:
  - Build a **causal graph** over CausalScript events (problem / step / end / explain / error).
  - Answer simple **causal queries**:
    - “Which operation caused this error?”
    - “If we changed this step, would the result change?”
  - Suggest **interventions** / hints:
    - “Try fixing step 2 (addition) because later inconsistencies depend on it.”
- Integrate with:
  - `Evaluator` / `PolynomialEvaluator`
  - `LearningLogger`
  - `KnowledgeRegistry` (rule-level causal relations)

### 1.2 Non-Goals (v1)

- No automatic discovery from large datasets (no heavy ML training in v1).
- No continuous-time / probabilistic causal models (discrete steps only).
- No domain-specific causal models outside arithmetic / algebra in v1.

---

## 2. Directory & Module Layout

Add the following to the existing `mathlang/` repository (relative paths):

```text
mathlang/
  core/
    causal/
      __init__.py
      causal_types.py          # Data classes for nodes, edges, queries
      causal_graph.py          # Core SCM representation
      causal_engine.py         # High-level API, integration layer
      causal_analyzers.py      # Pre-baked analysis patterns (error backtracking, hinting)
  tests/
    test_causal_engine.py
    test_causal_integrations.py
```

---

## 3. Core Concepts & Data Model

### 3.1 Causal Node Types

Represent each “event” in CausalScript as a node in a causal graph.

```python
# core/causal/causal_types.py

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

class CausalNodeType(str, Enum):
    PROBLEM = "problem"
    STEP = "step"
    END = "end"
    EXPLAIN = "explain"
    ERROR = "error"
    RULE_APPLICATION = "rule_application"

@dataclass
class CausalNode:
    node_id: str                 # unique ID (e.g., "step-1", "rule-ARITH-ADD-001")
    node_type: CausalNodeType
    payload: Dict[str, Any]      # expression, rendered, rule_id, etc.
```

### 3.2 Causal Edge Types

Edges represent **direct causal influence** (not just correlation):

```python
class CausalEdgeType(str, Enum):
    STEP_TRANSITION = "step_transition"   # step i -> step i+1
    RULE_USAGE = "rule_usage"             # rule node -> step node
    ERROR_CAUSE = "error_cause"           # step/rule -> error
    EXPLAIN_LINK = "explain_link"         # explain node -> target step
```

```python
@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    edge_type: CausalEdgeType
    metadata: Dict[str, Any] | None = None
```

### 3.3 Causal Graph Interface

```python
# core/causal/causal_graph.py

from typing import Dict, List

class CausalGraph:
    """
    Lightweight, in-memory causal graph.
    Nodes: CausalNode
    Edges: CausalEdge
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.out_edges: Dict[str, List[CausalEdge]] = {}
        self.in_edges: Dict[str, List[CausalEdge]] = {}

    def add_node(self, node: CausalNode) -> None: ...
    def add_edge(self, edge: CausalEdge) -> None: ...

    def get_parents(self, node_id: str) -> list[CausalNode]: ...
    def get_children(self, node_id: str) -> list[CausalNode]: ...

    def ancestors(self, node_id: str) -> list[CausalNode]:
        """DFS/BFS to collect all ancestor nodes."
        ...

    def descendants(self, node_id: str) -> list[CausalNode]:
        """DFS/BFS to collect all descendant nodes."
        ...
```

---

## 4. Causal Engine – Public API

### 4.1 High-Level Class

```python
# core/causal/causal_engine.py

from pathlib import Path
from typing import Any, Dict, List, Optional

from .causal_graph import CausalGraph
from .causal_types import CausalNode, CausalNodeType, CausalEdge, CausalEdgeType

class CausalEngine:
    """
    Main entry point for causal reasoning in CausalScript.
    It consumes LearningLogger-like event streams and builds a causal graph.
    """

    def __init__(self):
        self.graph = CausalGraph()

    # ---- Building phase -------------------------------------------------

    def ingest_log_record(self, record: Dict[str, Any]) -> None:
        """
        Ingest a single LearningLogger record.

        Expected keys (minimal):
          - phase: "problem" | "step" | "end" | "error" | "explain" | ...
          - expression: str (optional for error/explain)
          - rendered: str
          - rule_id: str | None
          - step_index: int | None
          - status: str (e.g., "ok", "mistake", "fatal")
        This method:
          1. Creates/updates relevant CausalNode(s).
          2. Adds edges (step transitions, rule usage, error causes).
        """
        ...

    def ingest_log(self, records: List[Dict[str, Any]]) -> None:
        """Ingest multiple records in order."
        for r in records:
            self.ingest_log_record(r)

    # ---- Query phase ----------------------------------------------------

    def why_error(self, error_node_id: str) -> List[CausalNode]:
        """
        Backtrack from an ERROR node to identify direct/indirect causes.
        Strategy:
          - Trace ERROR_CAUSE edges upstream.
          - Return a ranked list of nodes (e.g., steps, rules).
        """
        ...

    def suggest_fix_candidates(self, error_node_id: str, limit: int = 3) -> List[CausalNode]:
        """
        Return candidate steps/rules that are good intervention points.
        Basic heuristic v1:
          - Use ancestors().
          - Prefer nodes of type STEP with invalid status, or rules frequently used.
        """
        ...

    def counterfactual_result(
        self,
        interventions: Dict[str, Any] | List[Dict[str, Any]],
        base_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Simulate counterfactual scenarios with one or more interventions:
          - Each intervention can replace / insert / delete a step (phase="step")
            or override an end expression (phase="end").
        Returns:
          - { "changed": bool,
              "diff_steps": [...],
              "diff_end": [...],
              "new_end_expr": str | None,
              "rerun_records": [...],
              "rerun_step_outcomes": [...],
              "rerun_first_error": dict | None,
              "rerun_last_phase": str | None }
        Implementation:
          - Clone `base_records`, apply interventions in order.
          - Re-run Evaluator (SymbolicEvaluationEngine) to obtain fresh logs.
          - Compare outcomes and return the structured report.
        """

    # ---- Export / Debug -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export graph as serializable dict for debugging or visualization."
        ...
```

---

## 5. Integration Points with Existing Core

### 5.1 LearningLogger → CausalEngine

Prerequisite: `LearningLogger` returns JSON-like records:

```python
record = {
    "phase": "step",
    "step_index": 1,
    "expression": "(3 + 5) * 4",
    "rendered": "(3 + 5) * 4",
    "status": "ok",
    "rule_id": "ARITH-ADD-001",  # optional
}
```

Integration example (Jupyter / CLI layer):

```python
from core.evaluator import Evaluator
from core.logging import LearningLogger  # assumed
from core.causal.causal_engine import CausalEngine

logger = LearningLogger()
causal_engine = CausalEngine()

evaluator = Evaluator(program_ast, engine, learning_logger=logger)
evaluator.run()

records = logger.to_dict()
causal_engine.ingest_log(records)
```

### 5.2 KnowledgeRegistry → RULE nodes

When `Evaluator` resolves a rule via `KnowledgeRegistry.match(before, after)` and obtains `rule_id`, the `LearningLogger` should include it. Then `CausalEngine` can:

- Create a `CausalNodeType.RULE_APPLICATION` node with `node_id = f"rule-{rule_id}"`.
- Connect it to step nodes via `CausalEdgeType.RULE_USAGE`.

This yields a causal chain like:

```text
[ProblemNode] --STEP_TRANSITION--> [Step1Node] --RULE_USAGE--> [Rule_ARITH-ADD-001]
```

### 5.3 Error Integration

Evaluator v2 does not raise on computation mistakes. Instead it logs:

- `phase = "step"`, `status = "mistake"`, `meta.reason = "invalid_step"` for wrong steps.
- `phase = "end"`, `status = "mistake"`, `meta.reason = "final_result_mismatch"` for wrong endings.
- `status = "fatal"` only for unrecoverable failures (syntax/order issues).

`CausalEngine.ingest_log_record` should:

- Treat any non-`ok` step record as an error source and synthesize a `CausalNodeType.ERROR` node linked via `ERROR_CAUSE`.
- Connect fatal records (if any) into the graph using their recorded phase.

---

## 6. Causal Analysis Patterns (causal_analyzers.py)

Provide reusable analysis helpers.

```python
# core/causal/causal_analyzers.py

from typing import List, Dict, Any
from .causal_engine import CausalEngine
from .causal_types import CausalNode, CausalNodeType

def explain_error(engine: CausalEngine, error_node_id: str) -> Dict[str, Any]:
    """
    High-level wrapper that returns a human-readable explanation:
      - which steps are likely responsible
      - which rules were misused
    Output example:
      {
        "error_id": "error-1",
        "cause_steps": ["step-2"],
        "cause_rules": ["ARITH-ADD-002"],
        "message": "Step 2 (addition) conflicts with earlier transformation."
      }
    """
    causes = engine.why_error(error_node_id)
    ...
```

---

## 7. Minimal Example

### 7.1 Program

```mathlang
problem: (3 + 5) * 4
step: 8 * 4
step: 32 * 4       # <-- incorrect step
end: 32
```

### 7.2 Simplified LearningLogger Records (conceptual)

```python
records = [
  {"phase": "problem", "step_index": 0, "expression": "(3 + 5) * 4",
   "rendered": "(3 + 5) * 4", "status": "ok"},
  {"phase": "step", "step_index": 1, "expression": "8 * 4",
   "rendered": "8 * 4", "status": "ok", "rule_id": "ARITH-ADD-001"},
  {"phase": "step", "step_index": 2, "expression": "32 * 4",
   "rendered": "32 * 4", "status": "mistake",
   "meta": {"reason": "invalid_step"}},
]
```

### 7.3 Causal Reasoning

```python
causal_engine = CausalEngine()
causal_engine.ingest_log(records)

error_node_id = "error-2"  # internal mapping defined in ingest_log_record
causes = causal_engine.why_error(error_node_id)
fix_candidates = causal_engine.suggest_fix_candidates(error_node_id)

# Expected:
#   - 'causes' includes STEP node "step-2"
#   - 'fix_candidates' suggests "step-2" as primary intervention point
```

---

## 8. Testing Plan

### 8.1 Unit Tests (`tests/test_causal_engine.py`)

1. **Node/Edge Creation**
   - Add nodes and edges manually, assert `ancestors`/`descendants` behavior.

2. **Log Ingestion**
   - Given a small `records` list, ensure:
     - Nodes are created for each phase.
     - STEP transitions are chained correctly.
     - RULE nodes are created when `rule_id` is present.

3. **Error Backtracking**
   - With one invalid step:
     - `why_error` returns the invalid step as primary cause.

4. **Fix Suggestion**
   - When there are multiple step nodes:
     - `suggest_fix_candidates` returns the most recent invalid step first.

5. **Counterfactual Simulation (v1 stub)**
   - Given a simple `intervention`, verify:
     - The method returns `dict` containing `changed` and `new_end_expr` keys.

### 8.2 Integration Tests (`tests/test_causal_integrations.py`)

- Build a tiny CausalScript script, run through:
  - `Parser` → `Evaluator` (+ `LearningLogger`) → `CausalEngine`.
- Assert:
  - Errors in evaluator result in ERROR nodes in causal graph.
  - Rule IDs appear as rule nodes and are connected to step nodes.

---

## 9. Implementation Phases

### Phase 1 – Minimal Graph & Error Backtracking

- Implement:
  - `CausalNode`, `CausalEdge`, `CausalGraph`.
  - `CausalEngine.ingest_log_record`, `ingest_log`.
  - `why_error`, `suggest_fix_candidates` (simple ancestor-based heuristic).
- Add unit tests and one integration test.

### Phase 2 – Rule-Level Causality

- Connect `KnowledgeRegistry` rule IDs to `RULE_APPLICATION` nodes.
- Extend analyzers to show:
  - “Which rule misuse might have caused this error?”

### Phase 3 – Counterfactual Prototype

- Implement `counterfactual_result` as:
  - Re-run evaluator with modified step expression.
  - Compare final `end` expression with original.

---

## 10. Coding Guidelines

- Language: Python 3.12, type hints required.
- No heavy external dependencies; standard library only for causal modules.
- Keep **pure logic** in `core/causal/*`; no I/O, no CLI.
- All public methods must have docstrings explaining:
  - Inputs / outputs
  - Assumptions (e.g., record schema)
