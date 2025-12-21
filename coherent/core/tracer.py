import json
import uuid
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from coherent.core.action import Action
from coherent.core.state import State
from coherent.core.action_types import ActionType

@dataclass
class StepRecord:
    """
    A single step in a reasoning episode.
    """
    step_id: int
    state_snapshot: Dict[str, Any] # Snapshot of state BEFORE action
    action: Dict[str, Any]         # The action taken
    result: Dict[str, Any]         # The result of the execution
    timestamp: float

@dataclass
class Episode:
    """
    A full problem-solving session.
    """
    episode_id: str
    problem_text: str  # Initial input
    steps: List[StepRecord] = field(default_factory=list)
    final_outcome: str = "UNKNOWN" # SUCCESS, FAILURE, ABORTED
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class Tracer:
    """
    Records the trajectory of the agent (State -> Action -> Result).
    Serves as the memory formation process and data generator for offline learning.
    """
    def __init__(self):
        self._current_episode: Optional[Episode] = None
        self._episodes: List[Episode] = []

    def start_episode(self, problem_text: str) -> str:
        """Starts a new tracing episode."""
        episode_id = str(uuid.uuid4())
        self._current_episode = Episode(
            episode_id=episode_id,
            problem_text=problem_text
        )
        return episode_id

    def log_step(self, state: State, action: Action, result: Any):
        """
        Logs a discrete step.
        IMPORTANT: Records state snapshot BEFORE it was mutated by this step's result 
        (if state is mutable, caller must ensure snapshoting or we handle it here).
        
        Ideally, log_step is called right before execution? 
        Or passed the 'before' state.
        
        The architecture doc says: State -> Action -> Result -> NextState
        So we should log: S_t, A_t, R_t. S_{t+1} is derived or next step's S_t.
        """
        if not self._current_episode:
            return

        # Create a lightweight snapshot of critical state info
        # We don't want deep copies of everything if large
        state_snapshot = {
            "expression": state.current_expression,
            "status": state.status,
            "goal": state.task_goal.value
        }

        step = StepRecord(
            step_id=len(self._current_episode.steps),
            state_snapshot=state_snapshot,
            action=action.to_dict(),
            result=result,
            timestamp=time.time()
        )
        self._current_episode.steps.append(step)

    def end_episode(self, outcome: str = "SUCCESS"):
        """Finalizes the current episode."""
        if self._current_episode:
            self._current_episode.final_outcome = outcome
            self._current_episode.end_time = time.time()
            self._episodes.append(self._current_episode)
            self._current_episode = None

    def get_latest_episode(self) -> Optional[Dict[str, Any]]:
        if self._episodes:
            return self._episodes[-1].to_dict()
        return None
        
    def export_history(self) -> List[Dict[str, Any]]:
        return [ep.to_dict() for ep in self._episodes]
