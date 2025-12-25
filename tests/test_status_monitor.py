
import pytest
import json
from datetime import datetime
from coherent.tools.status_monitor import (
    StatusManager, SystemState, SystemStage, SystemStatus
)
from coherent.tools.status_monitor.sandbox import SandboxTransform
from coherent.tools.status_monitor.isolation import ReadOnlyMemoryProxy

@pytest.fixture
def manager():
    # Helper to clean up singleton? 
    # For now, just getting instance and forcing a reset if possible, 
    # or just transitioning to NORMAL at start.
    m = StatusManager()
    m.transition_to(SystemState.NORMAL, "Test Setup")
    return m

def test_status_schema_compliance(manager):
    """Verify JSON output complies with the spec."""
    status = manager.get_status()
    json_str = status.model_dump_json()
    data = json.loads(json_str)
    
    assert data["schema_version"] == "coherent.status.v1"
    assert data["mode"]["state"] == "NORMAL"
    assert "policy" in data["mode"]
    assert data["mode"]["policy"]["learning_enabled"] is True

def test_state_transitions(manager):
    """Test transitions to DEGRADED and ISOLATION."""
    # 1. Normal -> Degraded
    manager.transition_to(SystemState.DEGRADED, "Error detected")
    status = manager.get_status()
    assert status.mode.state == SystemState.DEGRADED
    assert status.mode.policy.learning_enabled is False
    assert status.mode.policy.memory_read_only is True

    # 2. Degraded -> Isolation
    manager.transition_to(SystemState.ISOLATION, "Confirmed anomaly")
    status = manager.get_status()
    assert status.mode.state == SystemState.ISOLATION
    assert status.mode.policy.compute_enabled is False

def test_sandbox_transform(manager):
    """Test SandboxTransform behavior in Isolation."""
    manager.transition_to(SystemState.ISOLATION, "Sandbox Test")
    
    sandbox = SandboxTransform()
    # Should work (mock)
    emb, stats = sandbox.encode("image_data")
    assert emb.dimension == 1024
    
    # Check if event was emitted? (would need mocking observer)
    events = []
    manager.add_observer(lambda e: events.append(e))
    
    sandbox.encode("more_data")
    assert len(events) > 0
    assert events[-1].type == "SANDBOX_TRANSFORM_RUN"

def test_memory_proxy_blocking(manager):
    """Test ReadOnlyMemoryProxy blocks writes in Degraded mode."""
    class MockStore:
        def add(self): return "added"
        def query(self): return "result"
    
    real_store = MockStore()
    proxy = ReadOnlyMemoryProxy(real_store)
    
    # Normal Mode -> Write OK
    manager.transition_to(SystemState.NORMAL)
    assert proxy.add() == "added"
    
    # Degraded Mode -> Write Blocked
    manager.transition_to(SystemState.DEGRADED)
    with pytest.raises(PermissionError):
        proxy.add()
        
    # Read still OK
    assert proxy.query() == "result"
