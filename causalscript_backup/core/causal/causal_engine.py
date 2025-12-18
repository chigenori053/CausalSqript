"""High-level causal engine built on top of LearningLogger records."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..ast_nodes import EndNode, ExplainNode, ProblemNode, ProgramNode, StepNode
from ..evaluator import Evaluator, SymbolicEvaluationEngine
from ..knowledge_registry import KnowledgeRegistry
from ..learning_logger import LearningLogger
from ..symbolic_engine import SymbolicEngine
from .causal_graph import CausalGraph
from .causal_types import (
    CausalEdge,
    CausalEdgeType,
    CausalNode,
    CausalNodeType,
)
from ..decision_theory import DecisionEngine, DecisionConfig, DecisionAction


class CausalEngine:
    """Consumes LearningLogger records and exposes causal reasoning helpers."""

    def __init__(self, decision_config: DecisionConfig | None = None) -> None:
        self.graph = CausalGraph()
        self.decision_engine = DecisionEngine(decision_config or DecisionConfig(strategy="balanced"))
        self.reset()

    # ------------------------------------------------------------------ #
    # Building phase                                                     #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset the engine state and clear the graph."""
        self.graph = CausalGraph()
        self._records: List[Dict[str, Any]] = []
        self._node_counters: Dict[CausalNodeType, int] = {}
        self._node_order: Dict[str, int] = {}
        self._order_counter = 0
        self._last_flow_node_id: Optional[str] = None
        self._last_step_node_id: Optional[str] = None
        self._last_rule_node_id: Optional[str] = None
        self._phase_counts: Dict[str, int] = {}
        self._step_nodes: List[str] = []
        self._error_nodes: List[str] = []
        self._node_status: Dict[str, str] = {}

    def ingest_log(self, records: List[Dict[str, Any]], *, reset_graph: bool = True) -> None:
        """Ingest multiple LearningLogger records in order."""
        if reset_graph:
            self.reset()
        for record in records:
            self.ingest_log_record(record)

    def ingest_log_record(self, record: Dict[str, Any]) -> str:
        """Ingest a single record and update the graph."""
        stored_record = dict(record)
        self._records.append(stored_record)
        phase = (stored_record.get("phase") or "").lower()
        node_type = self._node_type_from_phase(phase)
        node_id = self._next_node_id(node_type)
        payload = {"record": stored_record}
        node = CausalNode(node_id=node_id, node_type=node_type, payload=payload)
        self._register_node(node)
        status = stored_record.get("status", "")
        self._node_status[node_id] = status or ""
        self._phase_counts[phase] = self._phase_counts.get(phase, 0) + 1

        if node_type == CausalNodeType.PROBLEM:
            self._last_flow_node_id = node_id
        elif node_type == CausalNodeType.STEP:
            self._handle_step(node_id, stored_record)
        elif node_type == CausalNodeType.END:
            self._connect_flow(self._last_flow_node_id, node_id)
            self._last_flow_node_id = node_id
        elif node_type == CausalNodeType.ERROR:
            self._handle_error_node(node_id, stored_record)
        elif node_type == CausalNodeType.EXPLAIN:
            self._handle_explain_node(node_id, stored_record)

        return node_id

    # ------------------------------------------------------------------ #
    # Query phase                                                        #
    # ------------------------------------------------------------------ #

    def why_error(self, error_node_id: str) -> List[CausalNode]:
        """Trace potential causes for a given ERROR node."""
        results = self._why_error_with_depth(error_node_id)
        return [node for node, _ in results]

    def _why_error_with_depth(self, error_node_id: str) -> List[tuple[CausalNode, int]]:
        """Trace potential causes and return (node, depth)."""
        if error_node_id not in self.graph.nodes:
            return []
        ranked: List[tuple[int, int, CausalNode]] = []
        visited = {error_node_id}
        queue: deque[tuple[str, int]] = deque([(error_node_id, 0)])
        while queue:
            current_id, depth = queue.popleft()
            for edge in self.graph.in_edges.get(current_id, []):
                parent_id = edge.source_id
                if parent_id in visited:
                    continue
                visited.add(parent_id)
                queue.append((parent_id, depth + 1))
                parent_node = self.graph.nodes[parent_id]
                if parent_node.node_type in {CausalNodeType.STEP, CausalNodeType.RULE_APPLICATION}:
                    ranked.append((depth + 1, self._node_order.get(parent_id, -1), parent_node))
        ranked.sort(key=lambda item: (item[0], -item[1]))
        return [(node, depth) for depth, _, node in ranked]

    def _calculate_cause_probability(self, node: CausalNode, distance: int) -> float:
        """
        Calculate heuristic probability that a node is the cause.
        Distance is the graph distance from the error node.
        """
        # Distance starts at 1.
        # 1 -> 0.8
        # 2 -> 0.4
        if distance <= 0:
            return 0.0
            
        base_prob = 0.8 / distance
        
        # Boost probability for Rule Applications (often root causes)
        if node.node_type == CausalNodeType.RULE_APPLICATION:
            base_prob *= 1.2
            
        return min(1.0, base_prob)

    def suggest_fix_candidates(self, error_node_id: str, limit: int = 3) -> List[CausalNode]:
        """Suggest intervention points for a given error using Decision Theory."""
        # 1. Get raw candidates with depth
        raw_candidates = self._why_error_with_depth(error_node_id)
        
        if not raw_candidates:
            return []
            
        # Lazy initialization of heuristics components
        if not hasattr(self, "_misuse_detector"):
             from ..heuristics import MisusePatternDetector
             from ..symbolic_engine import SymbolicEngine
             # We need a symbolic engine instance. We can build a fresh one or store one.
             # Ideally we share one, but CausalEngine doesn't persist one currently.
             # _build_evaluation_engine creates one, let's allow reuse.
             sym = SymbolicEngine()
             self._misuse_detector = MisusePatternDetector(sym)

        scored_candidates: List[tuple[float, CausalNode, Dict[str, Any]]] = []
        
        for node, depth in raw_candidates:
            # 2. Estimate Probability using actual depth
            prob = self._calculate_cause_probability(node, depth)
            
            # 2b. Heuristic Intensification: Check for Misuse Patterns
            misuse_info = {}
            if node.node_type == CausalNodeType.STEP:
                # Find previous step/problem to get 'before' expression
                before_expr = self._get_before_expr(node.node_id)
                current_expr = node.payload.get("record", {}).get("expression")
                
                if before_expr and current_expr:
                    misuse_name = self._misuse_detector.detect_misuse(before_expr, current_expr)
                    if misuse_name:
                        # Significant probability boost for detected misuse
                        prob = max(prob, 0.95)
                        misuse_info = {"misuse_detected": misuse_name}

            # 3. Decision
            action, utility, _ = self.decision_engine.decide(prob)
            
            # 4. Filter & Score
            if action != DecisionAction.REJECT:
                # Store extra info if needed (e.g. for explanation)
                # But return type is List[CausalNode], so we can't easily return metadata 
                # unless we attach it to the node or trust the caller doesn't need it yet.
                # Use a temporary attribute or just rely on the boosting.
                if misuse_info:
                    # In a real system, we might want to attach this to a result object.
                    # For now, we just let it influence the ranking.
                    pass
                scored_candidates.append((utility, node, misuse_info))
                
        # 5. Sort by Utility (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Return nodes (limit applied)
        return [node for _, node, _ in scored_candidates[:limit]]

    def _get_before_expr(self, node_id: str) -> Optional[str]:
        """Retrieve the expression of the node immediately preceding this one in the flow."""
        in_edges = self.graph.in_edges.get(node_id, [])
        for edge in in_edges:
            if edge.edge_type == CausalEdgeType.STEP_TRANSITION:
                parent = self.graph.nodes.get(edge.source_id)
                if parent and parent.payload.get("record"):
                    return parent.payload["record"].get("expression")
        return None

    def counterfactual_result(
        self,
        interventions: Dict[str, Any] | List[Dict[str, Any]],
        base_records: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """
        Prototype counterfactual reasoning.

        Supports multiple interventions (steps/end expressions) and returns a detailed
        rerun report so callers can inspect how the simulated execution behaved.
        """
        records = deepcopy(base_records if base_records is not None else self._records)
        if not records:
            return {
                "changed": False,
                "new_end_expr": None,
                "diff_steps": [],
                "diff_end": [],
                "interventions": [],
                "rerun_success": False,
                "rerun_error": "No records available.",
                "rerun_records": [],
                "rerun_first_error": None,
                "rerun_last_phase": None,
                "rerun_step_outcomes": [],
            }
        normalized = self._normalize_interventions(interventions)
        diff_steps: List[Dict[str, Any]] = []
        diff_end: List[Dict[str, Any]] = []
        changed = False
        for intervention in normalized:
            outcome = self._apply_intervention(records, intervention)
            if not outcome["changed"]:
                continue
            changed = True
            if outcome["type"] == "step":
                diff_steps.append(outcome["detail"])
            elif outcome["type"] == "end":
                diff_end.append(outcome["detail"])
        rerun_result = self._rerun_records(records)
        return {
            "changed": changed,
            "new_end_expr": rerun_result["end_expr"],
            "diff_steps": diff_steps,
            "diff_end": diff_end,
            "interventions": normalized,
            "rerun_success": rerun_result["success"],
            "rerun_error": rerun_result["error"],
            "rerun_records": rerun_result["records"],
            "rerun_first_error": rerun_result["first_error"],
            "rerun_last_phase": rerun_result["last_phase"],
            "rerun_step_outcomes": rerun_result["step_outcomes"],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable view of the current graph and records."""
        return {
            "graph": self.graph.to_dict(),
            "records": deepcopy(self._records),
            "errors": list(self._error_nodes),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _node_type_from_phase(self, phase: str) -> CausalNodeType:
        mapping = {
            "problem": CausalNodeType.PROBLEM,
            "step": CausalNodeType.STEP,
            "end": CausalNodeType.END,
            "explain": CausalNodeType.EXPLAIN,
            "error": CausalNodeType.ERROR,
        }
        return mapping.get(phase, CausalNodeType.EXPLAIN)

    def _next_node_id(self, node_type: CausalNodeType) -> str:
        counter = self._node_counters.get(node_type, 0) + 1
        self._node_counters[node_type] = counter
        if node_type == CausalNodeType.RULE_APPLICATION:
            return f"rule-{counter}"
        return f"{node_type.value}-{counter}"

    def _register_node(self, node: CausalNode) -> None:
        is_new = node.node_id not in self.graph.nodes
        self.graph.add_node(node)
        if is_new:
            self._node_order[node.node_id] = self._order_counter
            self._order_counter += 1

    def _connect_flow(self, source_id: Optional[str], target_id: str) -> None:
        if source_id is None:
            return
        edge = CausalEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=CausalEdgeType.STEP_TRANSITION,
        )
        self.graph.add_edge(edge)

    def _handle_step(self, node_id: str, record: Dict[str, Any]) -> None:
        self._connect_flow(self._last_flow_node_id, node_id)
        self._last_flow_node_id = node_id
        self._last_step_node_id = node_id
        self._step_nodes.append(node_id)
        metadata = record.get("meta") or {}
        rule_id = record.get("rule_id")
        if rule_id:
            rule_node_id = f"rule-{rule_id}"
            rule_node = CausalNode(
                node_id=rule_node_id,
                node_type=CausalNodeType.RULE_APPLICATION,
                payload={"rule_id": rule_id, "rule_meta": metadata.get("rule")},
            )
            self._register_node(rule_node)
            self.graph.add_edge(
                CausalEdge(
                    source_id=rule_node_id,
                    target_id=node_id,
                    edge_type=CausalEdgeType.RULE_USAGE,
                )
            )
            self._last_rule_node_id = rule_node_id
        else:
            self._last_rule_node_id = None

        status = (record.get("status") or "").lower()
        if status and status not in {"ok", "info"}:
            synthesized_error = {
                "phase": "error",
                "status": status,
                "rendered": record.get("rendered"),
                "expression": record.get("expression"),
            }
            error_node_id = self._next_node_id(CausalNodeType.ERROR)
            error_node = CausalNode(
                node_id=error_node_id,
                node_type=CausalNodeType.ERROR,
                payload={"record": synthesized_error, "source_step_id": node_id},
            )
            self._register_node(error_node)
            self._handle_error_node(error_node_id, synthesized_error)

    def _handle_error_node(self, node_id: str, record: Dict[str, Any]) -> None:
        status = record.get("status")
        metadata = {"status": status} if status else None
        if self._last_step_node_id:
            self.graph.add_edge(
                CausalEdge(
                    source_id=self._last_step_node_id,
                    target_id=node_id,
                    edge_type=CausalEdgeType.ERROR_CAUSE,
                    metadata=metadata,
                )
            )
        if self._last_rule_node_id:
            self.graph.add_edge(
                CausalEdge(
                    source_id=self._last_rule_node_id,
                    target_id=node_id,
                    edge_type=CausalEdgeType.ERROR_CAUSE,
                    metadata=metadata,
                )
            )
        if self._last_flow_node_id and self._last_flow_node_id not in {
            self._last_step_node_id,
            self._last_rule_node_id,
        }:
            self.graph.add_edge(
                CausalEdge(
                    source_id=self._last_flow_node_id,
                    target_id=node_id,
                    edge_type=CausalEdgeType.ERROR_CAUSE,
                    metadata=metadata,
                )
            )
        self._error_nodes.append(node_id)

    def _handle_explain_node(self, node_id: str, record: Dict[str, Any]) -> None:
        meta = record.get("meta") or {}
        target_id = meta.get("target_node_id") or meta.get("target_step_id")
        if target_id and target_id in self.graph.nodes:
            self.graph.add_edge(
                CausalEdge(
                    source_id=node_id,
                    target_id=target_id,
                    edge_type=CausalEdgeType.EXPLAIN_LINK,
                    metadata={"reason": meta.get("reason")},
                )
            )

    def _step_priority(self, node: CausalNode) -> tuple[int, int]:
        status = (self._node_status.get(node.node_id) or "").lower()
        priority = 0 if status and status not in {"ok", "info"} else 1
        order = self._node_order.get(node.node_id, -1)
        return (priority, -order)

    def _find_last_end_expression(self, records: List[Dict[str, Any]]) -> Optional[str]:
        for rec in reversed(records):
            if (rec.get("phase") or "").lower() == "end":
                return rec.get("expression")
        return None

    def _rerun_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        program = self._program_from_records(records)
        if program is None:
            return {
                "success": False,
                "end_expr": None,
                "error": "Cannot reconstruct program from records.",
                "records": [],
                "first_error": None,
                "last_phase": None,
                "step_outcomes": [],
            }
        engine = self._build_evaluation_engine()
        logger = LearningLogger()
        evaluator = Evaluator(program, engine, learning_logger=logger)
        try:
            evaluator.run()
            success = True
            error_message = None
        except Exception as exc:  # pragma: no cover - defensive fallback
            success = False
            error_message = str(exc)
        new_records = logger.to_list()
        end_expr = self._find_last_end_expression(new_records)
        first_error = next(
            (
                rec
                for rec in new_records
                if (rec.get("phase") or "").lower() in {"step", "end"}
                and (rec.get("status") or "").lower() not in {"ok"}
            ),
            None,
        )
        last_phase = new_records[-1]["phase"] if new_records else None
        return {
            "success": success,
            "end_expr": end_expr,
            "error": error_message,
            "records": new_records,
            "first_error": first_error,
            "last_phase": last_phase,
            "step_outcomes": self._collect_step_outcomes(new_records),
        }

    def _build_evaluation_engine(self) -> SymbolicEvaluationEngine:
        symbolic = SymbolicEngine()
        knowledge: KnowledgeRegistry | None = None
        knowledge_path = self._default_knowledge_path()
        if knowledge_path.exists():
            knowledge = KnowledgeRegistry(knowledge_path, symbolic)
        return SymbolicEvaluationEngine(symbolic, knowledge)

    def _default_knowledge_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "knowledge"

    def _program_from_records(self, records: List[Dict[str, Any]]) -> Optional[ProgramNode]:
        program = ProgramNode()
        has_problem = False
        for record in records:
            phase = (record.get("phase") or "").lower()
            if phase == "problem":
                expr = record.get("expression")
                if not expr:
                    continue
                program.body.append(ProblemNode(expr=expr))
                has_problem = True
            elif phase == "step" and has_problem:
                expr = record.get("expression")
                if expr is None:
                    continue
                meta = record.get("meta") or {}
                step_id = record.get("step_id") or meta.get("step_id")
                program.body.append(StepNode(step_id=step_id, expr=expr))
            elif phase == "end" and has_problem:
                expr = record.get("expression")
                is_done = expr is None
                program.body.append(EndNode(expr=expr, is_done=is_done))
            elif phase == "explain" and has_problem:
                text = record.get("rendered") or record.get("expression")
                if text:
                    program.body.append(ExplainNode(text=text))
        if not has_problem:
            return None
        if not any(isinstance(node, EndNode) for node in program.body):
            program.body.append(EndNode(expr=None, is_done=True))
        return program

    def _normalize_interventions(
        self,
        interventions: Dict[str, Any] | List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not interventions:
            return []
        if isinstance(interventions, dict):
            data = [interventions]
        else:
            data = list(interventions)
        normalized: List[Dict[str, Any]] = []
        for entry in data:
            phase = (entry.get("phase") or entry.get("target") or "step").lower()
            normalized.append(
                {
                    "phase": phase,
                    "index": entry.get("index") or entry.get("step_index"),
                    "step_id": entry.get("step_id"),
                    "expression": entry.get("expression"),
                    "rendered": entry.get("rendered"),
                    "action": (entry.get("action") or "replace").lower(),
                    "status": entry.get("status"),
                    "rule_id": entry.get("rule_id"),
                    "meta": entry.get("meta"),
                }
            )
        return normalized

    def _apply_intervention(
        self,
        records: List[Dict[str, Any]],
        intervention: Dict[str, Any],
    ) -> Dict[str, Any]:
        phase = intervention.get("phase", "step")
        if phase == "end":
            return self._apply_end_intervention(records, intervention)
        return self._apply_step_intervention(records, intervention)

    def _apply_step_intervention(
        self,
        records: List[Dict[str, Any]],
        intervention: Dict[str, Any],
    ) -> Dict[str, Any]:
        target_index = intervention.get("index")
        action = intervention.get("action") or "replace"
        expression = intervention.get("expression")
        if not isinstance(target_index, int) or target_index <= 0:
            return {"changed": False, "type": "step", "detail": {}}
        counter = 0
        for idx, rec in enumerate(records):
            if (rec.get("phase") or "").lower() != "step":
                continue
            counter += 1
            if counter != target_index:
                continue

            if action == "delete":
                removed = records.pop(idx)
                return {
                    "changed": True,
                    "type": "step",
                    "detail": {
                        "phase": "step",
                        "index": target_index,
                        "action": "delete",
                        "old_expression": removed.get("expression"),
                    },
                }

            if action in {"insert_before", "insert_after"}:
                if expression is None:
                    return {"changed": False, "type": "step", "detail": {}}
                new_record = self._build_intervention_step_record(expression, intervention)
                insert_pos = idx if action == "insert_before" else idx + 1
                records.insert(insert_pos, new_record)
                return {
                    "changed": True,
                    "type": "step",
                    "detail": {
                        "phase": "step",
                        "index": target_index,
                        "action": action,
                        "new_expression": expression,
                    },
                }

            if expression is None:
                return {"changed": False, "type": "step", "detail": {}}
            updated = dict(rec)
            old_expr = updated.get("expression")
            if old_expr == expression:
                return {"changed": False, "type": "step", "detail": {}}
            updated["expression"] = expression
            if intervention.get("rendered"):
                updated["rendered"] = intervention["rendered"]
            if intervention.get("status"):
                updated["status"] = intervention["status"]
            if intervention.get("rule_id"):
                updated["rule_id"] = intervention["rule_id"]
            if intervention.get("meta"):
                updated["meta"] = intervention["meta"]
            records[idx] = updated
            return {
                "changed": True,
                "type": "step",
                "detail": {
                    "phase": "step",
                    "index": target_index,
                    "action": action,
                    "old_expression": old_expr,
                    "new_expression": expression,
                },
            }
        return {"changed": False, "type": "step", "detail": {}}

    def _apply_end_intervention(
        self,
        records: List[Dict[str, Any]],
        intervention: Dict[str, Any],
    ) -> Dict[str, Any]:
        expression = intervention.get("expression")
        if expression is None:
            return {"changed": False, "type": "end", "detail": {}}
        target_index = intervention.get("index") or -1
        end_positions = [idx for idx, rec in enumerate(records) if (rec.get("phase") or "").lower() == "end"]
        if not end_positions:
            return {"changed": False, "type": "end", "detail": {}}
        if target_index == -1:
            position = end_positions[-1]
            selected = len(end_positions)
        else:
            if not isinstance(target_index, int) or target_index <= 0 or target_index > len(end_positions):
                return {"changed": False, "type": "end", "detail": {}}
            position = end_positions[target_index - 1]
            selected = target_index
        updated = dict(records[position])
        old_expr = updated.get("expression")
        if old_expr == expression:
            return {"changed": False, "type": "end", "detail": {}}
        updated["expression"] = expression
        updated["rendered"] = intervention.get("rendered", updated.get("rendered"))
        if intervention.get("status"):
            updated["status"] = intervention["status"]
        if intervention.get("meta"):
            updated["meta"] = intervention["meta"]
        records[position] = updated
        return {
            "changed": True,
            "type": "end",
            "detail": {
                "phase": "end",
                "index": selected,
                "action": intervention.get("action", "replace"),
                "old_expression": old_expr,
                "new_expression": expression,
            },
        }

    def _build_intervention_step_record(
        self,
        expression: str,
        intervention: Dict[str, Any],
    ) -> Dict[str, Any]:
        rendered = intervention.get("rendered") or f"Intervention step: {expression}"
        meta = dict(intervention.get("meta") or {})
        meta.setdefault("intervention", True)
        return {
            "phase": "step",
            "expression": expression,
            "rendered": rendered,
            "status": intervention.get("status") or "intervention",
            "rule_id": intervention.get("rule_id"),
            "meta": meta,
        }

    def _collect_step_outcomes(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outcomes: List[Dict[str, Any]] = []
        counter = 0
        for rec in records:
            if (rec.get("phase") or "").lower() != "step":
                continue
            counter += 1
            outcomes.append(
                {
                    "index": counter,
                    "expression": rec.get("expression"),
                    "status": rec.get("status"),
                    "rendered": rec.get("rendered"),
                    "rule_id": rec.get("rule_id"),
                }
            )
        return outcomes


__all__ = ["CausalEngine"]
