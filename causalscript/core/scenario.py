from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Scenario:
    """Represents a specific context (variable assignments) for computation."""
    name: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScenarioGroup:
    """Manages a collection of scenarios."""
    scenarios: Dict[str, Scenario] = field(default_factory=dict)

    def add_scenario(self, name: str, context: Dict[str, Any]) -> None:
        self.scenarios[name] = Scenario(name, context)

    def get_scenario(self, name: str) -> Optional[Scenario]:
        return self.scenarios.get(name)

    def get_all_scenarios(self) -> List[Scenario]:
        return list(self.scenarios.values())
