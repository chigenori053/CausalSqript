"""
Decision Theory module for CausalScript.

This module implements the decision-theoretic framework for making judgments
under uncertainty (FuzzyJudge) and selecting optimal hints (HintEngine).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, Literal


class DecisionAction(str, Enum):
    """Possible actions for the judge."""
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


class DecisionState(int, Enum):
    """True state of the student's answer."""
    MATCH = 1      # The answer is essentially correct
    MISMATCH = -1  # The answer is incorrect


@dataclass
class UtilityMatrix:
    """
    Defines the utility (gain/loss) for each Action-State pair.
    
    Structure:
    {
        Action: {
            State.MATCH: utility_if_match,
            State.MISMATCH: utility_if_mismatch
        }
    }
    """
    accept: Dict[int, float]
    review: Dict[int, float]
    reject: Dict[int, float]

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[int, float]]) -> UtilityMatrix:
        return cls(
            accept=data["accept"],
            review=data["review"],
            reject=data["reject"]
        )


# Pre-defined strategies
STRATEGIES: Dict[str, UtilityMatrix] = {
    "balanced": UtilityMatrix(
        accept={DecisionState.MATCH: 100, DecisionState.MISMATCH: -50},
        review={DecisionState.MATCH: 50,  DecisionState.MISMATCH: -10},
        reject={DecisionState.MATCH: -20, DecisionState.MISMATCH: 50}
    ),
    "strict": UtilityMatrix(
        accept={DecisionState.MATCH: 100, DecisionState.MISMATCH: -100},
        review={DecisionState.MATCH: 20,  DecisionState.MISMATCH: 0},
        reject={DecisionState.MATCH: -10, DecisionState.MISMATCH: 100}
    ),
    "encouraging": UtilityMatrix(
        accept={DecisionState.MATCH: 100, DecisionState.MISMATCH: -10},
        review={DecisionState.MATCH: 80,  DecisionState.MISMATCH: -5},
        reject={DecisionState.MATCH: -100, DecisionState.MISMATCH: 20}
    ),
}


@dataclass
class DecisionConfig:
    """Configuration for the Decision Engine."""
    strategy: str = "balanced"
    algorithm: Literal["expected_utility", "minimax_regret"] = "expected_utility"
    ambiguity_threshold: float = 0.8
    custom_matrix: Optional[Dict] = None

    @property
    def matrix(self) -> UtilityMatrix:
        if self.custom_matrix:
            return UtilityMatrix.from_dict(self.custom_matrix)
        return STRATEGIES.get(self.strategy, STRATEGIES["balanced"])


class DecisionEngine:
    """
    Engine to make decisions based on probability and utility.
    """
    def __init__(self, config: DecisionConfig):
        self.config = config

    def decide(self, probability_match: float, ambiguity: float = 0.0) -> Tuple[DecisionAction, float, Dict[str, float]]:
        """
        Decide the best action given the probability of a match and ambiguity.

        Args:
            probability_match: The probability that the state is MATCH (0.0 to 1.0).
            ambiguity: The uncertainty/ambiguity of the signal (0.0 to 1.0).

        Returns:
            Tuple of (Selected Action, Expected Utility of that action, Debug info)
        """
        # High ambiguity forces a conservative calibration
        if ambiguity > self.config.ambiguity_threshold:
            # If slightly confident but very ambiguous, force REVIEW
            if probability_match > 0.4:
                return DecisionAction.REVIEW, 0.0, {"reason": "High Ambiguity", "ambiguity": ambiguity}
            else:
                return DecisionAction.REJECT, 0.0, {"reason": "High Ambiguity & Low Prob", "ambiguity": ambiguity}

        if self.config.algorithm == "minimax_regret":
            return self._decide_minimax_regret(probability_match)
        else:
            return self._decide_expected_utility(probability_match)

    def _decide_expected_utility(self, p_match: float) -> Tuple[DecisionAction, float, Dict[str, float]]:
        matrix = self.config.matrix
        p_mismatch = 1.0 - p_match

        # Calculate EU for each action
        # EU(a) = P(Match) * U(a, Match) + P(Mismatch) * U(a, Mismatch)
        
        eu_accept = p_match * matrix.accept[DecisionState.MATCH] + \
                    p_mismatch * matrix.accept[DecisionState.MISMATCH]
        
        eu_review = p_match * matrix.review[DecisionState.MATCH] + \
                    p_mismatch * matrix.review[DecisionState.MISMATCH]
        
        eu_reject = p_match * matrix.reject[DecisionState.MATCH] + \
                    p_mismatch * matrix.reject[DecisionState.MISMATCH]

        utilities = {
            DecisionAction.ACCEPT: eu_accept,
            DecisionAction.REVIEW: eu_review,
            DecisionAction.REJECT: eu_reject
        }

        best_action = max(utilities, key=utilities.get)
        return best_action, utilities[best_action], {k.value: v for k, v in utilities.items()}

    def _decide_minimax_regret(self, p_match: float) -> Tuple[DecisionAction, float, Dict[str, float]]:
        # This is a simplified implementation of Regret.
        # Regret(a, s) = Max_a'(U(a', s)) - U(a, s)
        # Expected Regret(a) = P(Match) * Regret(a, Match) + P(Mismatch) * Regret(a, Mismatch)
        # We want to minimize Expected Regret.
        
        matrix = self.config.matrix
        p_mismatch = 1.0 - p_match

        # 1. Find max utility for each state
        max_u_match = max(
            matrix.accept[DecisionState.MATCH],
            matrix.review[DecisionState.MATCH],
            matrix.reject[DecisionState.MATCH]
        )
        max_u_mismatch = max(
            matrix.accept[DecisionState.MISMATCH],
            matrix.review[DecisionState.MISMATCH],
            matrix.reject[DecisionState.MISMATCH]
        )

        # 2. Calculate regret for each action
        def calculate_expected_regret(action_utils: Dict[int, float]) -> float:
            regret_match = max_u_match - action_utils[DecisionState.MATCH]
            regret_mismatch = max_u_mismatch - action_utils[DecisionState.MISMATCH]
            return p_match * regret_match + p_mismatch * regret_mismatch

        er_accept = calculate_expected_regret(matrix.accept)
        er_review = calculate_expected_regret(matrix.review)
        er_reject = calculate_expected_regret(matrix.reject)

        regrets = {
            DecisionAction.ACCEPT: er_accept,
            DecisionAction.REVIEW: er_review,
            DecisionAction.REJECT: er_reject
        }

        # Select action with MINIMUM expected regret
        best_action = min(regrets, key=regrets.get)
        
        # Return negative regret as "utility" so higher is better in interface, 
        # or just return the regret value. Let's return -regret to keep consistent "higher is better"
        return best_action, -regrets[best_action], {k.value: v for k, v in regrets.items()}
