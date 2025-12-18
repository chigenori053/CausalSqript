"""Tests for the Geometry Engine."""

import pytest
try:
    import sympy
except ImportError:
    sympy = None

from coherent.engine.geometry_engine import GeometryEngine

@pytest.fixture
def engine():
    if sympy is None:
        pytest.skip("SymPy not installed")
    return GeometryEngine()

def test_primitives_creation(engine):
    p1 = engine.point(0, 0)
    p2 = engine.point(3, 4)
    line = engine.line(p1, p2)
    segment = engine.segment(p1, p2)
    ray = engine.ray(p1, p2)
    circle = engine.circle(p1, 5)
    poly = engine.polygon(p1, p2, engine.point(3, 0))
    
    assert p1.x == 0 and p1.y == 0
    assert line.p1 == p1 and line.p2 == p2
    assert circle.radius == 5

def test_calculations(engine):
    p1 = engine.point(0, 0)
    p2 = engine.point(3, 4)
    
    # Distance
    assert engine.distance(p1, p2) == 5
    
    # Midpoint
    mid = engine.midpoint(p1, p2)
    assert float(mid.x) == 1.5 and float(mid.y) == 2
    
    # Slope
    l = engine.line(p1, p2)
    assert engine.slope(l) == sympy.Rational(4, 3)
    
    # Area & Perimeter
    tri = engine.triangle(engine.point(0, 0), engine.point(3, 0), engine.point(0, 4))
    assert engine.area(tri) == 6
    assert engine.perimeter(tri) == 12
    
    # Circle
    c = engine.circle(p1, 5)
    assert engine.area(c) == 25 * sympy.pi
    assert engine.circumference(c) == 10 * sympy.pi
    assert float(engine.arc_length(c, 180)) == float(5 * sympy.pi)

def test_theorems(engine):
    # Pythagorean
    assert engine.check_pythagorean(3, 4, 5)
    assert not engine.check_pythagorean(3, 4, 6)
    
    # Congruence
    t1 = engine.triangle(engine.point(0, 0), engine.point(3, 0), engine.point(0, 4))
    t2 = engine.triangle(engine.point(1, 1), engine.point(4, 1), engine.point(1, 5))
    assert engine.check_congruence(t1, t2)
    
    # Similarity
    t3 = engine.triangle(engine.point(0, 0), engine.point(6, 0), engine.point(0, 8))
    assert engine.check_similarity(t1, t3)

def test_geometric_relations(engine):
    p1 = engine.point(0, 0)
    p2 = engine.point(1, 0)
    p3 = engine.point(0, 1)
    
    # Right angle
    assert engine.is_right_angle(p2, p1, p3)
    
    # Parallel
    l1 = engine.line(engine.point(0, 0), engine.point(1, 0))
    l2 = engine.line(engine.point(0, 1), engine.point(1, 1))
    assert engine.is_parallel(l1, l2)
    
    # Perpendicular
    l3 = engine.line(engine.point(0, 0), engine.point(0, 1))
    assert engine.is_perpendicular(l1, l3)

def test_solids(engine):
    # Cylinder
    cyl = engine.cylinder(radius=3, height=10)
    assert engine.volume(cyl) == 90 * sympy.pi
    assert engine.surface_area(cyl) == 60 * sympy.pi + 18 * sympy.pi # 78 pi

    # Sphere
    sph = engine.sphere(radius=3)
    assert engine.volume(sph) == 36 * sympy.pi
    assert engine.surface_area(sph) == 36 * sympy.pi

    # Cone
    cone = engine.cone(radius=3, height=4)
    assert engine.volume(cone) == 12 * sympy.pi
    # Slant height = sqrt(3^2 + 4^2) = 5
    # Surface area = pi*r*s + pi*r^2 = 15pi + 9pi = 24pi
    assert engine.surface_area(cone) == 24 * sympy.pi

    # Prism (Rectangular prism: base 3x4, height 10)
    base_area = 12
    base_perimeter = 14
    prism = engine.prism(base_area, 10, base_perimeter)
    assert engine.volume(prism) == 120
    assert engine.surface_area(prism) == 140 + 24 # 164

    # Pyramid (Square pyramid: base 6x6, height 4)
    # Slant height of face: sqrt(3^2 + 4^2) = 5
    base_area = 36
    base_perimeter = 24
    slant_height = 5
    pyramid = engine.pyramid(base_area, 4, base_perimeter, slant_height)
    assert engine.volume(pyramid) == 48 # (1/3)*36*4
    assert engine.surface_area(pyramid) == 60 + 36 # (1/2)*24*5 + 36 = 96
