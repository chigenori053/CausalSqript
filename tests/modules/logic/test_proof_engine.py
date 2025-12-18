import pytest
from coherent.engine.proof_engine import ProofEngine, Fact, TransitiveRule, SymmetricRule, SSSRule

def test_transitivity():
    engine = ProofEngine()
    engine.register_rule(TransitiveRule("Equal"))
    
    # Given: A = B, B = C
    engine.add_fact(Fact("Equal", ("A", "B")))
    engine.add_fact(Fact("Equal", ("B", "C")))
    
    # Goal: A = C
    goal = Fact("Equal", ("A", "C"))
    proof = engine.prove(goal)
    
    assert proof is not None
    assert len(proof) == 3 # Given, Given, Derived
    assert proof[-1].fact == goal
    assert proof[-1].rule_name == "Transitive(Equal)"

def test_symmetry():
    engine = ProofEngine()
    engine.register_rule(SymmetricRule("Equal"))
    
    engine.add_fact(Fact("Equal", ("A", "B")))
    
    goal = Fact("Equal", ("B", "A"))
    proof = engine.prove(goal)
    
    assert proof is not None
    assert proof[-1].fact == goal

def test_sss_congruence():
    engine = ProofEngine()
    engine.register_rule(SSSRule())
    
    # Triangle ABC and DEF
    # AB = DE, BC = EF, AC = DF
    engine.add_fact(Fact("SideEqual", ("AB", "DE")))
    engine.add_fact(Fact("SideEqual", ("BC", "EF")))
    engine.add_fact(Fact("SideEqual", ("AC", "DF")))
    
    # Goal: Congruent(ABC, DEF)
    # Note: Vertices are sorted in the rule implementation, so ABC and DEF
    goal = Fact("Congruent", ("ABC", "DEF"))
    proof = engine.prove(goal)
    
    assert proof is not None
    assert proof[-1].fact == goal
    assert proof[-1].rule_name == "SSS"
    assert len(proof[-1].precedents) == 3

def test_no_proof():
    engine = ProofEngine()
    engine.register_rule(TransitiveRule("Equal"))
    
    engine.add_fact(Fact("Equal", ("A", "B")))
    
    goal = Fact("Equal", ("A", "C"))
    proof = engine.prove(goal)
    
    assert proof is None
