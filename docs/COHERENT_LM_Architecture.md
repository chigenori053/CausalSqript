# COHERENT: Reasoning Language Model Architecture
## "Language Model" as a Next Action Predictor

This document defines the architectural transformation of COHERENT into a Reasoning Language Model (LM). Unlike traditional LMs that predict the **next token**, COHERENT predicts the **next action**.

### Core Concept
- **Input**: Natural Language (Problem, Code Request, Student Answer)
- **Internal Representation**: Holographic Embeddings + AST/Semantic Graph + Context Memory
- **Output**: **Action** (Discrete reasoning step)
- **Mechanism**: Predict -> Simulate/Validate -> Decide

## Architecture Layers

### Layer A: Language I/O Adapter
Handles the interface between natural language and internal structured representations.
- **TextEncoder** (SemanticParser): Converts NL -> `SemanticIR` -> `State`.
- **TextRenderer**: Converts `Action` / `Result` -> Natural Language response (e.g., Japanese explanations).

### Layer B: COHERENT Core (The "Reasoning OS")
The brain that executes the OODA loop (Observe-Orient-Decide-Act).
1.  **OpticalStore (Memory)**: Recalls relevant past experiences (similar problems, efficient paths).
2.  **HypothesisGenerator**: Generates candidate `Action`s (Apply Rule, Use Tool, Request Info).
3.  **LookaheadSimulator**: Simulates actions to predict future states (Value Function).
4.  **ValidationEngine**: Verifies correctness (Math/Logic/Type checks).
5.  **DecisionEngine**: Controls the gate (Accept, Reject, Review, Ask for Clarification).

---

## Data Structures

### 1. Action Schema
The universal output unit of the system.
```json
{
  "type": "APPLY_RULE | CALL_TOOL | RECALL | ASK | FINAL | REJECT",
  "name": "distribute_property",
  "inputs": { "target": "x(y+z)", "args": [] },
  "confidence": 0.95,
  "ambiguity": 0.05,
  "evidence": {
    "rule_id": "math.algebra.distribute",
    "memory_hit": true
  }
}
```

### 2. State Schema
The snapshot of the problem solving context.
- **Goal**: `TaskType` (SOLVE, VERIFY, etc.)
- **Current Expression/AST**: The current mathematical or code state.
- **History**: List of previous Actions and Results.
- **Constraints**: Precision, timeout, forbidden methods.
- **Memory Context**: Active holographic embeddings.

---

## Roadmap

### P0: Action Schema & Execution Foundation (Current Focus)
- Define `Action` and `State` classes.
- Implement `ActionExecutor` to apply actions and update state.
- Implement `Tracer` to log the episode (RL/Learning dataset).
- **Goal**: System outputs standardized JSON Actions.

### P1: Education Domain (Math)
- End-to-End flow: Word Problem -> Equation -> Steps -> Verification.
- Modules: `ProblemUnderstanding`, `StepValidator` (Enhanced), `HintPolicy`.
- **Goal**: Stable step-by-step math solver with "Ask for Clarification" capability.

### P2: Coding Domain
- Flow: Request -> Plan -> Code Generation -> Test/Verify -> Fix.
- Modules: `CodePlanner`, `CodeToolchain` (Linter/Test), `RepoMemory`.
- **Goal**: Coding Agent running on the same "Reasoning OS".

---

## Learning Strategy
The "Language Model" learns from the logs of this system:
- **Policy**: Predict `$Action` given `$State`.
- **Value**: Predict `$SuccessProbability` given `$State`.
