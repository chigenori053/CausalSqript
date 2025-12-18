from typing import Any, Dict, Optional

class ContextManager:
    """
    Manages the execution context (variables, constants) for the Orchestrator.
    Handles mode switching and context persistence or isolation.
    """
    
    def __init__(self):
        self._global_context: Dict[str, Any] = {}
        self._current_mode: str = "default" # "Arithmetic", "Algebra", etc.

    @property
    def current_context(self) -> Dict[str, Any]:
        """Return the current active context."""
        return self._global_context

    def switch_context(self, mode: str):
        """
        Switch the execution mode.
        Currently, context is shared globally, but this hook allows for future isolation.
        """
        self._current_mode = mode
        # In a more advanced implementation, we might filter variables here:
        # e.g. if switching to Arithmetic, we might hide non-numeric variables.
        # For now, we trust the Parser/Engine to handle or reject context data.

    def set_variable(self, name: str, value: Any):
        self._global_context[name] = value

    def get_variable(self, name: str) -> Optional[Any]:
        return self._global_context.get(name)

    def clear(self):
        self._global_context.clear()
