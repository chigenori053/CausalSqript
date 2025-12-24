# COHERENT Edu Hint Generation Engine – Agent Design Spec

(abridged header for agent-readable markdown)

## Purpose
This document defines the updated Hint Generation subsystem for COHERENT Edu, optimized for agent-based reasoning, deterministic control, and educational co-creation.

## Core Principle
Hints are generated based on *structural drift* between user input and expected expressions, combined with resonance observation and decision-theoretic action selection.

## Components
- GapAnalyzer
- HintPolicy (DecisionEngine-based)
- HintRenderer
- Telemetry (ThoughtEvent)

## Actions
Suppress | Hint | Partial | Answer

## Drift Types
REPRESENTATION_EQUIV
LIKE_TERMS_UNMERGED
DISTRIBUTIVE_MISAPPLIED
SIGN_ERROR
COEFFICIENT_ARITH_ERROR
TERM_DROPPED_OR_ADDED
UNRELATED_TRANSFORM

## Decision Logic
- Resonance is an observation, not a threshold.
- Action is selected by expected utility maximization.
- Answer is forbidden in High-understanding state.

## Interfaces
See structured schemas for HintRequest / HintResponse.

## Logging
All hints and user interactions are stored as ThoughtEvent.

## Acceptance Criteria
- Drift-based hint switching
- No Answer in High state
- Agent-explainable decision trace
