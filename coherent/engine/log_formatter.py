"""Helpers to convert learning logger records into human-friendly messages."""

from __future__ import annotations

from typing import List, Sequence
from .renderers import ContentRenderer, RenderContext


def format_record_label(record: dict) -> str:
    phase = (record.get("phase") or "").strip()
    if phase == "step":
        idx = record.get("step_index")
        if isinstance(idx, int):
            return f"step{idx}"
    if phase:
        return phase
    return "record"


def _expression_for(record: dict) -> str:
    expr = record.get("expression")
    if expr:
        return str(expr).strip()
    rendered = record.get("rendered")
    if not rendered:
        return ""
    rendered_text = str(rendered).strip()
    phase = (record.get("phase") or "").strip().lower()
    if phase:
        prefix = f"{phase}:"
        lowered = rendered_text.lower()
        if lowered.startswith(prefix):
            remainder = rendered_text[len(prefix) :].lstrip()
            if remainder:
                return remainder
    return rendered_text


def format_record_message(record: dict, *, include_meta: bool = True) -> List[str]:
    """Format a single record into one or more textual lines."""

    label = format_record_label(record)
    
    # Get category from metadata
    meta = record.get("meta") or {}
    category = meta.get("category", "algebra")
    
    # Render
    ctx = RenderContext(
        expression=_expression_for(record),
        category=category,
        metadata=meta
    )
    rendered_expr = ContentRenderer.render_step(ctx)
    
    base = f"{label}: {rendered_expr}".rstrip() if rendered_expr else f"{label}:"
    lines: List[str] = [base]
    if not include_meta:
        return lines
    meta = record.get("meta") or {}
    reason = meta.get("reason")
    if reason:
        lines.append(f"message: {reason}")
    explanation = meta.get("explanation")
    if explanation:
        lines.append(f"explain: {explanation}")
    expected = meta.get("expected")
    if expected:
        lines.append(f"expected: {expected}")
    return lines


def format_records(records: Sequence[dict], *, include_meta: bool = True) -> List[str]:
    """Format all learning records into textual lines."""

    lines: List[str] = []
    for record in records:
        lines.extend(format_record_message(record, include_meta=include_meta))
    return lines


def print_formatted_records(records: Sequence[dict], *, include_meta: bool = True) -> None:
    """Convenience printer that sends formatted lines to stdout."""

    for line in format_records(records, include_meta=include_meta):
        print(line)
