
import pytest
pytest.importorskip("sympy")
from coherent.engine.heuristics import MisusePatternDetector
from coherent.engine.symbolic_engine import SymbolicEngine

@pytest.fixture
def detector():
    return MisusePatternDetector(SymbolicEngine())

def test_freshmans_dream(detector):
    # (a+b)^n -> a^n + b^n
    assert detector.detect_misuse("(x + y)**2", "x**2 + y**2") == "Freshman's Dream"
    # Python syntax for power is ** or ^ in Coherent parser, but detector sees Python string?
    # CausalEngine usually ingests parsed steps or string steps. 
    # symbolic_engine.match_structure uses sympy parser which handles ** and ^ if configured.
    # Let's assume standard python/cscript inputs.
    
    assert detector.detect_misuse("(x + y)^2", "x^2 + y^2") == "Freshman's Dream"
    
    # n=3
    assert detector.detect_misuse("(a + b)^3", "a^3 + b^3") == "Freshman's Dream"
    
    # Not a misuse if correct expansion? No, simple pattern match won't match correct expansion.
    # But we check for MISUSE pattern.
    
    # Condition check: n=1 is not a misuse 
    # (x+y)^1 -> x^1 + y^1 (x+y) is actually correct.
    assert detector.detect_misuse("(x + y)^1", "x + y") is None

def test_linear_trig(detector):
    # sin(a+b) -> sin(a) + sin(b)
    assert detector.detect_misuse("sin(x + y)", "sin(x) + sin(y)") == "Linear Sine"
    assert detector.detect_misuse("sin(2*x + 3)", "sin(2*x) + sin(3)") == "Linear Sine"
    
    # cos(a+b) -> cos(a) + cos(b)
    assert detector.detect_misuse("cos(x + y)", "cos(x) + cos(y)") == "Linear Cosine"

def test_sqrt_distribution(detector):
    res = detector.detect_misuse("sqrt(x + 4)", "sqrt(x) + 2")
    assert res in ["Sqrt Distribution", "Freshman's Dream"]
    
    res2 = detector.detect_misuse("sqrt(x + y)", "sqrt(x) + sqrt(y)")
    assert res2 in ["Sqrt Distribution", "Freshman's Dream"]

def test_no_false_positive_on_correct_steps(detector):
    # Correct expansion
    # (x+y)^2 -> x^2 + 2xy + y^2
    assert detector.detect_misuse("(x + y)^2", "x^2 + 2*x*y + y^2") is None
    
    # Correct trig identity
    # sin(2x) -> 2sin(x)cos(x)
    assert detector.detect_misuse("sin(2*x)", "2*sin(x)*cos(x)") is None

