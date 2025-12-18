"""Output helpers shared between CLI entry points."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from coherent.logic import CausalEngine, run_causal_analysis
from coherent.engine.knowledge_registry import KnowledgeRegistry


def print_records(records: List[dict], stream: Optional[object] = None) -> None:
    if stream is None:
        stream = sys.stdout
    for record in records:
        rendered = record.get("rendered") or record.get("expression") or ""
        if rendered:
            print(rendered, file=stream)


def print_causal_summary(
    records: List[dict],
    *,
    rule_info: Dict[str, Dict[str, str]] | None = None,
    stream: Optional[object] = None,
) -> None:
    if not records:
        return
    engine, report = run_causal_analysis(records, rule_info=rule_info)
    error_ids: List[str] = report.get("errors", [])
    if not error_ids:
        return
    summaries = report.get("explanations", [])
    if stream is None:
        stream = sys.stdout
    print("\n== Causal Analysis ==", file=stream)
    rule_details: Dict[str, Dict[str, str]] = report.get("rule_details", {})
    for idx, error_id in enumerate(error_ids):
        summary = summaries[idx] if idx < len(summaries) else {"message": "", "cause_steps": [], "cause_rules": []}
        message = summary.get("message") or f"Error {error_id}"
        print(f"{error_id}: {message}", file=stream)
        steps = summary.get("cause_steps") or []
        rules = summary.get("cause_rules") or []
        if steps:
            print(f"  steps: {', '.join(steps)}", file=stream)
        if rules:
            print(f"  rules: {', '.join(rules)}", file=stream)
            for rule_id in rules:
                detail = rule_details.get(rule_id)
                if detail and detail.get("description"):
                    print(f"    {rule_id}: {detail['description']}", file=stream)
        fixes = engine.suggest_fix_candidates(error_id)
        if fixes:
            fix_ids = ", ".join(node.node_id for node in fixes)
            print(f"  fix candidates: {fix_ids}", file=stream)


def print_counterfactual_summary(records: List[dict], intervention_str: str) -> None:
    intervention = _parse_counterfactual_json(intervention_str)
    if intervention is None:
        print("Invalid counterfactual JSON; skipping simulation.", file=sys.stderr)
        return
    engine = CausalEngine()
    engine.ingest_log(records)
    result = engine.counterfactual_result(intervention, records)
    print("\n== Counterfactual Simulation ==")
    if not result["changed"]:
        print("No changes applied.")
        return
    for step_diff in result.get("diff_steps", []):
        action = step_diff.get("action", "replace")
        print(
            f"Step {step_diff.get('index')} {action}: {step_diff.get('old_expression')} -> {step_diff.get('new_expression')}"
        )
    for end_diff in result.get("diff_end", []):
        print(
            f"End {end_diff.get('index')} {end_diff.get('action', 'replace')}: {end_diff.get('old_expression')} -> {end_diff.get('new_expression')}"
        )
    print(f"Rerun success: {result['rerun_success']} (last phase: {result['rerun_last_phase']})")
    if result.get("rerun_first_error"):
        print(f"First error after intervention: {result['rerun_first_error']}")
    print(f"New end expression: {result.get('new_end_expr')}")


def _parse_counterfactual_json(value: str) -> Dict[str, Any] | List[Dict[str, Any]] | None:
    try:
        data = json.loads(value)
        if isinstance(data, (dict, list)):
            return data
        return None
    except Exception:
        return None


def extract_run_settings(records: List[dict]) -> tuple[str | None, dict[str, Any]]:
    mode: str | None = None
    config: dict[str, Any] = {}
    for record in records:
        phase = record.get("phase")
        meta = record.get("meta") or {}
        if phase == "mode" and isinstance(meta.get("mode"), str):
            mode = meta["mode"]
        elif phase == "config":
            options = meta.get("options")
            if isinstance(options, dict):
                config.update(options)
    return mode, config


def should_run_causal_analysis(records: List[dict]) -> bool:
    mode, config = extract_run_settings(records)
    if "causal" in config:
        return bool(config["causal"])
    if mode in {"causal", "cf"}:
        return True
    return True


def rule_metadata_from_registry(
    registry: KnowledgeRegistry | None,
) -> Dict[str, Dict[str, str]]:
    source = registry
    if source is None:
        try:
            from coherent.engine.symbolic_engine import SymbolicEngine

            symbolic = SymbolicEngine()
            source = KnowledgeRegistry(
                base_path=Path(__file__).resolve().parents[1] / "engine" / "knowledge",
                engine=symbolic,
            )
        except Exception:  # pragma: no cover
            return {}
    return {node.id: node.to_metadata() for node in source.nodes}
