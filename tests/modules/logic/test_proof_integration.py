import pytest
from coherent.engine.geometry_engine import GeometryEngine
from coherent.engine.proof_engine import Fact

def test_geometry_proof_integration():
    try:
        geo_engine = GeometryEngine()
    except ImportError:
        pytest.skip("SymPy not available")

    # Create two congruent triangles
    p1 = geo_engine.point(0, 0)
    p2 = geo_engine.point(3, 0)
    p3 = geo_engine.point(0, 4)
    t1 = geo_engine.triangle(p1, p2, p3)

    q1 = geo_engine.point(10, 10)
    q2 = geo_engine.point(13, 10)
    q3 = geo_engine.point(10, 14)
    t2 = geo_engine.triangle(q1, q2, q3)

    # Check congruence fact generation
    fact = geo_engine.get_congruence_fact(t1, "ABC", t2, "DEF")
    
    assert fact is not None
    assert isinstance(fact, Fact)
    assert fact.predicate == "Congruent"
    assert fact.args == ("ABC", "DEF")

    # Create a non-congruent triangle
    r1 = geo_engine.point(0, 0)
    r2 = geo_engine.point(5, 0)
    r3 = geo_engine.point(0, 5)
    t3 = geo_engine.triangle(r1, r2, r3)

    fact_fail = geo_engine.get_congruence_fact(t1, "ABC", t3, "GHI")
    assert fact_fail is None
