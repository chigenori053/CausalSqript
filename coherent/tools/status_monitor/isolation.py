"""
Isolation Enforcement Utilities.
Proxies and decorators to enforce Status Policies.
"""
from typing import Any
from .manager import StatusManager

class ReadOnlyMemoryProxy:
    """
    Wraps a MemoryStore to prevent writes when writes are disabled.
    """
    def __init__(self, real_store: Any):
        self._store = real_store
        self._manager = StatusManager()

    def add(self, *args, **kwargs):
        if not self._manager.check_policy("memory_write"):
            # Log attempt and reject silently or raise?
            # Spec says "Forbidden".
            self._manager.report_error("MemoryProxy", "Write attempted in Read-Only Mode", safe_to_continue=True)
            raise PermissionError("Memory Write is disabled by System Policy.")
        return self._store.add(*args, **kwargs)

    def __getattr__(self, name: str):
        # Delegate other methods (read access)
        return getattr(self._store, name)

def guard_learning(func):
    """Decorator to skip learning if disabled."""
    def wrapper(*args, **kwargs):
        manager = StatusManager()
        if not manager.check_policy("learning"):
            # Silently skip or log?
            # Learning is usually a side effect, so silent skip is often safer for flow.
            # But let's log info.
            # print("Skipping learning due to policy.")
            return None
        return func(*args, **kwargs)
    return wrapper
