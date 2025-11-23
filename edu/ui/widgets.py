"""Placeholder widgets for Edu notebooks/CLI demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ExampleOption:
    label: str
    file: str


class ExampleSelector:
    """Simple in-memory representation of a widget selection."""

    def __init__(self, options: List[ExampleOption]) -> None:
        self.options = options
        self.selected: ExampleOption | None = options[0] if options else None

    def select(self, label: str) -> ExampleOption | None:
        for option in self.options:
            if option.label == label:
                self.selected = option
                return option
        return None
