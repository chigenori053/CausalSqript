"""Exercise specification for MathLang validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Union

import yaml


@dataclass
class ExerciseSpec:
    """
    Specification for a mathematical exercise.
    
    Defines the target expression, validation mode, and optional constraints
    for validating student answers.
    
    Attributes:
        id: Unique identifier for the exercise
        target_expression: The correct answer expression
        validation_mode: Type of validation to perform
            - "symbolic_equiv": Check symbolic equivalence (default)
            - "exact_form": Check exact string match after normalization
            - "canonical_form": Check if expression matches canonical form
        canonical_form: Optional canonical form for format checking
        intermediate_steps: Optional list of valid intermediate steps
        hint_rules: Optional dictionary mapping error patterns to hints
        metadata: Optional additional metadata
    """
    
    id: str
    target_expression: str
    validation_mode: str = "symbolic_equiv"
    canonical_form: str | None = None
    intermediate_steps: list[str] | None = None
    hint_rules: dict[str, str] | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate the exercise specification."""
        valid_modes = {"symbolic_equiv", "exact_form", "canonical_form"}
        if self.validation_mode not in valid_modes:
            raise ValueError(
                f"Invalid validation_mode: {self.validation_mode}. "
                f"Must be one of {valid_modes}"
            )
        
        if self.validation_mode == "canonical_form" and self.canonical_form is None:
            raise ValueError(
                "canonical_form must be provided when validation_mode is 'canonical_form'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the specification to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExerciseSpec:
        """Create a specification from a dictionary."""
        # Filter out unknown keys to be safe
        known_keys = {
            "id", "target_expression", "validation_mode", "canonical_form",
            "intermediate_steps", "hint_rules", "metadata"
        }
        filtered_data = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered_data)


def load_exercise_spec(path: Union[str, Path]) -> ExerciseSpec:
    """
    Load an exercise specification from a JSON or YAML file.
    
    Args:
        path: Path to the file
        
    Returns:
        Loaded ExerciseSpec
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported or invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Exercise spec file not found: {path}")
        
    suffix = path.suffix.lower()
    
    with open(path, "r", encoding="utf-8") as f:
        if suffix == ".json":
            data = json.load(f)
        elif suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .json or .yaml")
            
    return ExerciseSpec.from_dict(data)


def save_exercise_spec(spec: ExerciseSpec, path: Union[str, Path]) -> None:
    """
    Save an exercise specification to a JSON or YAML file.
    
    Args:
        spec: ExerciseSpec to save
        path: Path to save to
    """
    path = Path(path)
    data = spec.to_dict()
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = path.suffix.lower()
    
    with open(path, "w", encoding="utf-8") as f:
        if suffix == ".json":
            json.dump(data, f, indent=2, ensure_ascii=False)
        elif suffix in (".yaml", ".yml"):
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .json or .yaml")
