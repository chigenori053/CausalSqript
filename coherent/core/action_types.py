from enum import Enum

class ActionType(Enum):
    """
    Defines the discrete action space for the Reasoning LM.
    """
    APPLY_RULE = "APPLY_RULE"   # Symbolic rule application
    CALL_TOOL = "CALL_TOOL"     # External tool usage (Python interpreter, Plotter, etc.)
    RECALL = "RECALL"           # Memory retrieval (Optical Store)
    ASK = "ASK"                 # Ask user for clarification/input
    FINAL = "FINAL"             # Return final answer
    REJECT = "REJECT"           # Reject current path/state
    REVIEW = "REVIEW"           # Self-correction/Verification request
