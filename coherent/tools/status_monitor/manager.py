"""
Status Manager for COHERENT.
Manages system state, enforces policies, and emits events.
"""
from datetime import datetime, timezone
from typing import Optional, List, Callable
import uuid
import logging

from .models import SystemStatus, SystemMode, SystemState, SystemStage, Policy, SystemEvent, EventSeverity

logger = logging.getLogger(__name__)

# Default Policies
POLICY_NORMAL = Policy(
    learning_enabled=True,
    memory_write_enabled=True,
    memory_read_only=False,
    sandbox_transform_enabled=False,
    compute_enabled=True,
    recall_enabled=True
)

POLICY_DEGRADED = Policy(
    learning_enabled=False,
    memory_write_enabled=False,
    memory_read_only=True,
    # Provisional: Sandbox allowed for diagnostics
    sandbox_transform_enabled=True, 
    compute_enabled=True,
    recall_enabled=True
)

POLICY_ISOLATION = Policy(
    learning_enabled=False,
    memory_write_enabled=False,
    memory_read_only=True,
    sandbox_transform_enabled=True,
    compute_enabled=False, # Restricted computation
    recall_enabled=True
)

POLICY_MAINTENANCE = Policy(
    learning_enabled=False,
    memory_write_enabled=False, # Careful manual only
    memory_read_only=True,
    sandbox_transform_enabled=True,
    compute_enabled=False,
    recall_enabled=False
)

class StatusManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StatusManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.system_id = f"sys-{uuid.uuid4().hex[:8]}"
        self._current_mode = SystemMode(
            state=SystemState.NORMAL,
            stage=SystemStage.NORMAL,
            since=datetime.now(timezone.utc),
            policy=POLICY_NORMAL,
            reason="System Initialization"
        )
        self._observers: List[Callable[[SystemEvent], None]] = []

    def get_status(self) -> SystemStatus:
        return SystemStatus(
            system_id=self.system_id,
            mode=self._current_mode
        )

    def add_observer(self, callback: Callable[[SystemEvent], None]):
        self._observers.append(callback)

    def _emit_event(self, type_: str, severity: EventSeverity, component: str, payload: dict):
        event = SystemEvent(
            event_id=uuid.uuid4().hex,
            type=type_,
            severity=severity,
            component=component,
            payload=payload
        )
        logger.info(f"EVENT [{severity}]: {type_} - {payload}")
        for obs in self._observers:
            obs(event)

    def transition_to(self, state: SystemState, reason: str = None):
        """Transition to a new state."""
        if self._current_mode.state == state:
            return

        old_state = self._current_mode.state
        new_stage = SystemStage.NORMAL # Default reset stage logic? Or depends on state.
        new_policy = POLICY_NORMAL

        if state == SystemState.NORMAL:
            new_policy = POLICY_NORMAL
            new_stage = SystemStage.NORMAL
        elif state == SystemState.DEGRADED:
            new_policy = POLICY_DEGRADED
            new_stage = SystemStage.ISOLATION_DIAG # Prepare for diagnosis
        elif state == SystemState.ISOLATION:
            new_policy = POLICY_ISOLATION
            new_stage = SystemStage.ISOLATION_DIAG
        elif state == SystemState.MAINTENANCE:
            new_policy = POLICY_MAINTENANCE
            new_stage = SystemStage.ISOLATION_DIAG # Manual diag

        self._current_mode = SystemMode(
            state=state,
            stage=new_stage,
            since=datetime.now(timezone.utc),
            policy=new_policy,
            reason=reason
        )

        self._emit_event(
            "STATE_ENTER", EventSeverity.WARN if state != SystemState.NORMAL else EventSeverity.INFO,
            "StatusManager",
            {"from": old_state, "to": state, "reason": reason}
        )

        if not new_policy.learning_enabled and old_state == SystemState.NORMAL:
             self._emit_event("LEARNING_HALTED", EventSeverity.WARN, "StatusManager", {})
        
        if new_policy.memory_read_only:
             self._emit_event("MEMORY_READONLY_ENABLED", EventSeverity.WARN, "StatusManager", {})


    def report_error(self, component: str, error_type: str, safe_to_continue: bool = False):
        """
        Report an error from a component. 
        If not safe_to_continue -> DEGRADED immediately.
        """
        payload = {"error": error_type, "safe": safe_to_continue}
        
        if not safe_to_continue:
            self.transition_to(SystemState.DEGRADED, reason=f"Critical Error in {component}: {error_type}")
            self._emit_event("CRITICAL_ERROR", EventSeverity.ERROR, component, payload)
        else:
            self._emit_event("ERROR", EventSeverity.WARN, component, payload)

    def check_policy(self, action: str) -> bool:
        """Check if action is allowed by current policy."""
        policy = self._current_mode.policy
        if action == "learning":
            return policy.learning_enabled
        if action == "memory_write":
            return policy.memory_write_enabled
        if action == "transform_sandbox":
            return policy.sandbox_transform_enabled
        return False
