"""
Pydantic models for the COHERENT Status Observation System.
Defines the schema for SystemStatus, Mode, Policy, and Events.
"""
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class SystemState(str, Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    ISOLATION = "ISOLATION"
    MAINTENANCE = "MAINTENANCE"

class SystemStage(str, Enum):
    NORMAL = "NORMAL"
    ISOLATION_DIAG = "ISOLATION_DIAG"
    WARMUP = "WARMUP"
    RECOVERY_READY = "RECOVERY_READY"

class Policy(BaseModel):
    learning_enabled: bool
    memory_write_enabled: bool
    memory_read_only: bool
    sandbox_transform_enabled: bool
    compute_enabled: bool
    recall_enabled: bool

class SystemMode(BaseModel):
    state: SystemState
    stage: SystemStage
    since: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: Optional[str] = None
    policy: Policy

class SystemStatus(BaseModel):
    schema_version: str = "coherent.status.v1"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    system_id: str
    mode: SystemMode

class EventSeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

class SystemEvent(BaseModel):
    schema_version: str = "coherent.event.v1"
    event_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    type: str # e.g. STATE_ENTER, LEARNING_HALTED
    severity: EventSeverity
    component: str
    payload: Dict[str, Any] = Field(default_factory=dict)
