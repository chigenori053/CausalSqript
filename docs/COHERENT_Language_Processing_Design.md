
# COHERENT Language Processing Capability Design Specification

## 1. Purpose
This document defines the architecture and design principles for implementing controllable, testable, and learnable language processing capabilities within the COHERENT system.

The goal is to transform natural language inputs into executable semantic structures without relying on black-box reasoning models.

## 2. Non-Goals
- General conversational AI
- Free-form text generation for reasoning
- LLM-only problem solving

## 3. Design Principles
- All language input must be converted into structured semantics.
- Reasoning authority remains in CoreRuntime and ReasoningAgent.
- Language understanding must be learnable and reusable.

## 4. Architecture Overview
Natural Language → Semantic Parser → Semantic IR → CoreRuntime → Optical Memory

## 5. Language Capability Layers
### L1 Surface Processing
Normalization, symbol completion, multilingual handling.

### L2 Semantic Parsing
Extract intent, domain, and mathematical objects.

### L3 Intent Classification
Enumerated intent outputs with confidence scores.

### L4 Semantic Intermediate Representation (SIR)
Defines task, domain, goals, inputs, constraints, ambiguity, and metadata.

## 6. Semantic IR Schema
```json
{
  "task": "solve | verify | hint | explain",
  "math_domain": "arithmetic | algebra | calculus | linear_algebra",
  "goal": { "type": "final_value | transformation | proof" },
  "inputs": [{ "type": "expression", "value": "(x - y)^2" }],
  "constraints": { "symbolic_only": true },
  "explanation_level": 0,
  "language_meta": { "original_language": "ja", "ambiguity": 0.23 }
}
```

## 7. Semantic Parser Design
Hybrid rule-based and LLM-based parser.
LLM output must be strict JSON with no reasoning.

## 8. CoreRuntime Integration
Intent-based routing to solving, verification, hint, or explanation engines.

## 9. Optical Memory Integration
Language semantics stored as reusable experience units enabling recall-first reasoning.

## 10. Ambiguity Handling
Ambiguity scores inform DecisionEngine to accept, review, or request clarification.

## 11. Testing Strategy
- Paraphrase equivalence
- Ambiguous instruction handling
- Student-like malformed input

## 12. Roadmap
Phase 1: IR definition and intent routing.
Phase 2: Optical memory integration.
Phase 3: Ambiguity-aware educational UX.

## 13. Conclusion
This design enables COHERENT to become an executable semantic intelligence system rather than a conversational model.
