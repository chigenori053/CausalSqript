"""Minimal YAML loader/dumper used when PyYAML is unavailable."""

from __future__ import annotations

import ast as _ast
import json
from dataclasses import dataclass
from typing import Any, List, Tuple


def safe_load(stream: Any) -> Any:
    """Parse YAML/JSON text into Python data structures."""
    text = _ensure_text(stream)
    stripped = text.lstrip()
    if not stripped:
        return None

    # First try JSON for speed and to support files that already use it.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        parser = _SimpleYAMLParser(text)
        return parser.parse()


def dump(data: Any, stream: Any | None = None, **kwargs: Any) -> str | None:
    """Serialize Python data as a readable JSON string."""
    sort_keys = kwargs.get("sort_keys", False)
    allow_unicode = kwargs.get("allow_unicode", False)
    text = json.dumps(data, indent=2, ensure_ascii=not allow_unicode, sort_keys=sort_keys)
    if stream is None:
        return text
    stream.write(text)
    return None


def _ensure_text(stream: Any) -> str:
    if hasattr(stream, "read"):
        return stream.read()
    return str(stream)


def _strip_comment(line: str) -> str:
    result = []
    in_single = in_double = False
    for char in line:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        result.append(char)
    return "".join(result)


def _split_inline_items(text: str) -> List[str]:
    items: List[str] = []
    current: List[str] = []
    in_single = in_double = False
    bracket_depth = 0
    for char in text:
        if char == "'" and not in_double:
            in_single = not in_single
            current.append(char)
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            current.append(char)
            continue
        if char in "([{" and not in_single and not in_double:
            bracket_depth += 1
        elif char in ")]}" and not in_single and not in_double:
            bracket_depth -= 1
        if char == "," and not in_single and not in_double and bracket_depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        items.append("".join(current).strip())
    return items


def _parse_key(raw_key: str) -> str:
    raw_key = raw_key.strip()
    if raw_key.startswith(("'", '"')) and raw_key.endswith(("'", '"')):
        try:
            return _ast.literal_eval(raw_key)
        except Exception:
            return raw_key.strip("'\"")
    return raw_key


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0] in "\"'":
        try:
            return _ast.literal_eval(value)
        except Exception:
            return value.strip("\"'")

    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in _split_inline_items(inner)]

    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        mapping = {}
        for item in _split_inline_items(inner):
            key, val = _split_key_value(item)
            mapping[_parse_key(key)] = _parse_scalar(val)
        return mapping

    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _split_key_value(content: str) -> Tuple[str, str]:
    in_single = in_double = False
    for idx, char in enumerate(content):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == ":" and not in_single and not in_double:
            return content[:idx], content[idx + 1 :]
    raise ValueError(f"Invalid mapping entry: {content}")


@dataclass
class _Line:
    indent: int
    content: str


class _SimpleYAMLParser:
    """Very small YAML subset parser sufficient for project config files."""

    def __init__(self, text: str) -> None:
        self.lines: List[_Line] = []
        for raw in text.splitlines():
            candidate = _strip_comment(raw).rstrip()
            if not candidate.strip():
                continue
            indent = len(candidate) - len(candidate.lstrip(" "))
            self.lines.append(_Line(indent, candidate.strip()))
        self.pos = 0

    def parse(self) -> Any:
        if not self.lines:
            return None
        return self._parse_block(self.lines[0].indent)

    def _parse_block(self, expected_indent: int) -> Any:
        if self.pos >= len(self.lines):
            return None
        line = self.lines[self.pos]
        if line.indent < expected_indent:
            return None
        if line.content.startswith("- "):
            return self._parse_list(expected_indent)
        return self._parse_mapping(expected_indent)

    def _parse_list(self, indent: int) -> List[Any]:
        items: List[Any] = []
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if line.indent < indent or not line.content.startswith("- "):
                break
            entry = line.content[2:].strip()
            self.pos += 1
            if not entry:
                items.append(self._parse_block(indent + 2))
                continue
            if ":" in entry:
                key, value = _split_key_value(entry)
                mapping: dict[str, Any] = {}
                if value.strip():
                    mapping[_parse_key(key)] = _parse_scalar(value)
                else:
                    mapping[_parse_key(key)] = self._parse_block(indent + 2)
                self._consume_mapping(mapping, indent + 2)
                items.append(mapping)
            else:
                items.append(_parse_scalar(entry))
        return items

    def _parse_mapping(self, indent: int) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if line.indent < indent:
                break
            if line.content.startswith("- ") and line.indent == indent:
                break
            self.pos += 1
            key, value = _split_key_value(line.content)
            parsed_key = _parse_key(key)
            value = value.strip()
            if value:
                mapping[parsed_key] = _parse_scalar(value)
            else:
                mapping[parsed_key] = self._parse_block(indent + 2)
        return mapping

    def _consume_mapping(self, mapping: dict[str, Any], indent: int) -> None:
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if line.indent < indent:
                break
            if line.content.startswith("- ") and line.indent == indent:
                break
            self.pos += 1
            key, value = _split_key_value(line.content)
            parsed_key = _parse_key(key)
            value = value.strip()
            if value:
                mapping[parsed_key] = _parse_scalar(value)
            else:
                mapping[parsed_key] = self._parse_block(indent + 2)


__all__ = ["safe_load", "dump"]
