
import pytest
from coherent.engine.decision_theory import DecisionConfig, DecisionEngine, DecisionAction, DecisionState, UtilityMatrix

def test_utility_matrix_structure():
    matrix = UtilityMatrix(
        accept={DecisionState.MATCH: 10, DecisionState.MISMATCH: -10},
        review={DecisionState.MATCH: 5, DecisionState.MISMATCH: -5},
        reject={DecisionState.MATCH: -5, DecisionState.MISMATCH: 10}
    )
    assert matrix.accept[DecisionState.MATCH] == 10
    assert matrix.reject[DecisionState.MISMATCH] == 10

def test_decision_engine_balanced_strategy():
    config = DecisionConfig(strategy="balanced")
    engine = DecisionEngine(config)
    
    # High probability of match -> Should Accept
    action, utility, _ = engine.decide(0.95)
    assert action == DecisionAction.ACCEPT
    
    # Low probability of match -> Should Reject
    action, utility, _ = engine.decide(0.05)
    assert action == DecisionAction.REJECT
    
    # Middle probability -> Should Review (depending on exact values, but likely)
    # Balanced: Accept(100, -50), Review(50, -10), Reject(-20, 50)
    # P=0.5:
    # EU(Accept) = 0.5*100 + 0.5*-50 = 25
    # EU(Review) = 0.5*50 + 0.5*-10 = 20
    # EU(Reject) = 0.5*-20 + 0.5*50 = 15
    # So 0.5 might still be Accept or Review depending on exact threshold
    
    # Let's try P=0.4
    # EU(Accept) = 0.4*100 + 0.6*-50 = 40 - 30 = 10
    # EU(Review) = 0.4*50 + 0.6*-10 = 20 - 6 = 14
    # EU(Reject) = 0.4*-20 + 0.6*50 = -8 + 30 = 22
    # So P=0.4 -> Reject
    
    action, _, _ = engine.decide(0.4)
    assert action == DecisionAction.REJECT # Actually 0.4 gives Reject in balanced?
    # Wait, let's recheck calculation above.
    # EU(Reject) = 22, EU(Review) = 14. Yes, Reject.
    
    # Let's try to find a Review case.
    # We need Review > Accept AND Review > Reject
    # 50p - 10(1-p) > 100p - 50(1-p)
    # 60p - 10 > 150p - 50
    # 40 > 90p => p < 4/9 (approx 0.44)
    
    # 50p - 10(1-p) > -20p + 50(1-p)
    # 60p - 10 > -70p + 50
    # 130p > 60 => p > 6/13 (approx 0.46)
    
    # So Review is optimal between 0.46 and 0.44? Wait, 0.46 > 0.44.
    # This means Review is NEVER optimal in "balanced" strategy with these numbers!
    # Let's check the numbers in code:
    # accept={MATCH: 100, MISMATCH: -50}
    # review={MATCH: 50,  MISMATCH: -10}
    # reject={MATCH: -20, MISMATCH: 50}
    
    # Intersection Accept/Review:
    # 100p - 50(1-p) = 50p - 10(1-p)
    # 150p - 50 = 60p - 10
    # 90p = 40 => p = 4/9 = 0.444
    
    # Intersection Review/Reject:
    # 50p - 10(1-p) = -20p + 50(1-p)
    # 60p - 10 = -70p + 50
    # 130p = 60 => p = 6/13 = 0.461
    
    # Since 0.444 < 0.461, the "Review" region is non-existent or inverted?
    # If p > 0.444, Accept > Review.
    # If p < 0.461, Reject > Review.
    # So Review is dominated.
    
    # This is an interesting finding for the "balanced" strategy! 
    # It effectively acts as a binary classifier with threshold around 0.45.
    pass

def test_decision_engine_strict_strategy():
    config = DecisionConfig(strategy="strict")
    engine = DecisionEngine(config)
    
    # Strict: Accept(100, -100), Review(20, 0), Reject(-10, 100)
    # High penalty for False Positive (-100).
    
    # P=0.8
    # EU(Accept) = 0.8*100 + 0.2*-100 = 80 - 20 = 60
    # EU(Review) = 0.8*20 + 0.2*0 = 16
    # EU(Reject) = 0.8*-10 + 0.2*100 = -8 + 20 = 12
    # Accept wins.
    
    # P=0.6
    # EU(Accept) = 0.6*100 + 0.4*-100 = 60 - 40 = 20
    # EU(Review) = 0.6*20 = 12
    # EU(Reject) = 0.6*-10 + 0.4*100 = -6 + 40 = 34
    # Reject wins! Even at 0.6 probability!
    
    action, _, _ = engine.decide(0.6)
    assert action == DecisionAction.REJECT

def test_decision_engine_encouraging_strategy():
    config = DecisionConfig(strategy="encouraging")
    engine = DecisionEngine(config)
    
    # Encouraging: Accept(100, -10), Review(80, -5), Reject(-100, 20)
    
    # P=0.4
    # EU(Accept) = 0.4*100 + 0.6*-10 = 40 - 6 = 34
    # EU(Review) = 0.4*80 + 0.6*-5 = 32 - 3 = 29
    # EU(Reject) = 0.4*-100 + 0.6*20 = -40 + 12 = -28
    # Accept wins! Even at 0.4!
    
    action, _, _ = engine.decide(0.4)
    assert action == DecisionAction.ACCEPT

def test_minimax_regret():
    config = DecisionConfig(strategy="balanced", algorithm="minimax_regret")
    engine = DecisionEngine(config)
    
    # Just ensure it runs and returns a valid action
    action, utility, _ = engine.decide(0.5)
    assert isinstance(action, DecisionAction)
