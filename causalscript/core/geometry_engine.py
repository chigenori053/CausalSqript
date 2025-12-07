"""Geometry Engine for CausalScript Core (2D)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import math

from .errors import EvaluationError

try:
    import sympy
    from sympy import Point2D, Line2D, Segment2D, Ray2D, Polygon, Circle, RegularPolygon, Triangle
    from sympy.geometry import intersection
except ImportError:
    sympy = None
    Point2D = Any
    Line2D = Any
    Segment2D = Any
    Ray2D = Any
    Polygon = Any
    Circle = Any
    RegularPolygon = Any
    Triangle = Any
    Triangle = Any

# ------------------------------------------------------------------ #
# 3D Solids Classes                                                  #
# ------------------------------------------------------------------ #

class Solid3D:
    """Base class for 3D solids."""
    pass

class Prism(Solid3D):
    def __init__(self, base_area: Any, height: Any, base_perimeter: Any = None):
        self.base_area = base_area
        self.height = height
        self.base_perimeter = base_perimeter

class Pyramid(Solid3D):
    def __init__(self, base_area: Any, height: Any, base_perimeter: Any = None, slant_height: Any = None):
        self.base_area = base_area
        self.height = height
        self.base_perimeter = base_perimeter
        self.slant_height = slant_height

class Cylinder(Solid3D):
    def __init__(self, radius: Any, height: Any):
        self.radius = radius
        self.height = height

class Cone(Solid3D):
    def __init__(self, radius: Any, height: Any):
        self.radius = radius
        self.height = height

class Sphere(Solid3D):
    def __init__(self, radius: Any):
        self.radius = radius

class GeometryEngine:
    """
    Provides 2D geometry operations and calculations.
    
    Wraps sympy.geometry to provide a simplified API for:
    - Primitives: Points, Lines, Segments, Rays, Polygons, Circles
    - 3D Solids: Prism, Pyramid, Cylinder, Cone, Sphere
    - Calculations: Area, Perimeter, Distance, Midpoint, Slope, Volume, Surface Area
    - Theorems: Pythagorean, Congruence, Similarity
    """

    def __init__(self):
        if sympy is None:
            raise ImportError("SymPy is required for GeometryEngine.")

    # ------------------------------------------------------------------ #
    # Primitives Factories                                               #
    # ------------------------------------------------------------------ #

    def point(self, x: float, y: float) -> Point2D:
        """Create a 2D Point."""
        return Point2D(x, y)

    def line(self, p1: Point2D, p2: Point2D) -> Line2D:
        """Create a Line passing through two points."""
        return Line2D(p1, p2)

    def segment(self, p1: Point2D, p2: Point2D) -> Segment2D:
        """Create a Line Segment between two points."""
        return Segment2D(p1, p2)

    def ray(self, p1: Point2D, p2: Point2D) -> Ray2D:
        """Create a Ray starting at p1 and passing through p2."""
        return Ray2D(p1, p2)

    def circle(self, center: Point2D, radius: float) -> Circle:
        """Create a Circle with a center and radius."""
        return Circle(center, radius)

    def polygon(self, *points: Point2D) -> Polygon:
        """Create a Polygon from a list of points."""
        return Polygon(*points)
    
    def triangle(self, p1: Point2D, p2: Point2D, p3: Point2D) -> Triangle:
        """Create a Triangle from three points."""
        return Triangle(p1, p2, p3)

    # ------------------------------------------------------------------ #
    # 3D Solids Factories                                                #
    # ------------------------------------------------------------------ #

    def prism(self, base_area: Any, height: Any, base_perimeter: Any = None) -> Prism:
        """Create a Prism given base area and height."""
        return Prism(base_area, height, base_perimeter)

    def pyramid(self, base_area: Any, height: Any, base_perimeter: Any = None, slant_height: Any = None) -> Pyramid:
        """Create a Pyramid given base area and height."""
        return Pyramid(base_area, height, base_perimeter, slant_height)

    def cylinder(self, radius: Any, height: Any) -> Cylinder:
        """Create a Cylinder given radius and height."""
        return Cylinder(radius, height)

    def cone(self, radius: Any, height: Any) -> Cone:
        """Create a Cone given radius and height."""
        return Cone(radius, height)

    def sphere(self, radius: Any) -> Sphere:
        """Create a Sphere given radius."""
        return Sphere(radius)

    # ------------------------------------------------------------------ #
    # Calculations                                                       #
    # ------------------------------------------------------------------ #

    def distance(self, p1: Point2D, p2: Point2D) -> Any:
        """Calculate the Euclidean distance between two points."""
        return p1.distance(p2)

    def midpoint(self, p1: Point2D, p2: Point2D) -> Point2D:
        """Calculate the midpoint between two points."""
        return p1.midpoint(p2)

    def slope(self, entity: Union[Line2D, Segment2D, Ray2D]) -> Any:
        """Calculate the slope of a line, segment, or ray."""
        return entity.slope

    def length(self, entity: Union[Segment2D]) -> Any:
        """Calculate the length of a segment."""
        return entity.length

    def area(self, shape: Union[Polygon, Circle, Triangle]) -> Any:
        """Calculate the area of a shape."""
        return shape.area

    def perimeter(self, shape: Union[Polygon, Triangle]) -> Any:
        """Calculate the perimeter of a polygon."""
        return shape.perimeter

    def circumference(self, circle: Circle) -> Any:
        """Calculate the circumference of a circle."""
        return circle.circumference

    def arc_length(self, circle: Circle, angle_degrees: float) -> Any:
        """Calculate the arc length for a given central angle in degrees."""
        # Arc length = 2 * pi * r * (angle / 360)
        # SymPy Circle doesn't have a direct arc_length method for degrees
        return 2 * sympy.pi * circle.radius * (angle_degrees / 360)

    def line_equation(self, line: Line2D) -> str:
        """Return the equation of the line (ax + by + c = 0)."""
        return str(line.equation())

    def volume(self, solid: Solid3D) -> Any:
        """Calculate the volume of a 3D solid."""
        if isinstance(solid, Prism):
            return solid.base_area * solid.height
        elif isinstance(solid, Pyramid):
            return sympy.Rational(1, 3) * solid.base_area * solid.height
        elif isinstance(solid, Cylinder):
            return sympy.pi * solid.radius**2 * solid.height
        elif isinstance(solid, Cone):
            return sympy.Rational(1, 3) * sympy.pi * solid.radius**2 * solid.height
        elif isinstance(solid, Sphere):
            return sympy.Rational(4, 3) * sympy.pi * solid.radius**3
        raise ValueError(f"Unknown solid type: {type(solid)}")

    def surface_area(self, solid: Solid3D) -> Any:
        """Calculate the surface area of a 3D solid."""
        if isinstance(solid, Prism):
            if solid.base_perimeter is None:
                raise ValueError("Base perimeter is required for Prism surface area.")
            # Lateral area + 2 * Base area
            return solid.base_perimeter * solid.height + 2 * solid.base_area
        elif isinstance(solid, Pyramid):
            if solid.base_perimeter is None or solid.slant_height is None:
                raise ValueError("Base perimeter and slant height are required for Pyramid surface area.")
            # Lateral area + Base area
            return sympy.Rational(1, 2) * solid.base_perimeter * solid.slant_height + solid.base_area
        elif isinstance(solid, Cylinder):
            # 2*pi*r*h + 2*pi*r^2
            return 2 * sympy.pi * solid.radius * solid.height + 2 * sympy.pi * solid.radius**2
        elif isinstance(solid, Cone):
            # pi*r*s + pi*r^2 where s is slant height
            slant_height = sympy.sqrt(solid.radius**2 + solid.height**2)
            return sympy.pi * solid.radius * slant_height + sympy.pi * solid.radius**2
        elif isinstance(solid, Sphere):
            return 4 * sympy.pi * solid.radius**2
        raise ValueError(f"Unknown solid type: {type(solid)}")

    # ------------------------------------------------------------------ #
    # Theorem Verifiers                                                  #
    # ------------------------------------------------------------------ #

    def check_pythagorean(self, a: float, b: float, c: float) -> bool:
        """
        Check if three side lengths form a right triangle (a^2 + b^2 = c^2).
        Assumes c is the hypotenuse.
        """
        return sympy.simplify(a**2 + b**2 - c**2) == 0

    def check_congruence(self, t1: Triangle, t2: Triangle) -> bool:
        """Check if two triangles are congruent."""
        # SymPy Triangle doesn't have is_congruent, but is_similar checks for shape.
        # Congruence means similar AND equal area (or side lengths).
        return t1.is_similar(t2) and sympy.simplify(t1.area - t2.area) == 0

    def check_similarity(self, t1: Triangle, t2: Triangle) -> bool:
        """Check if two triangles are similar."""
        return t1.is_similar(t2)

    def is_right_angle(self, p1: Point2D, vertex: Point2D, p2: Point2D) -> bool:
        """Check if the angle formed by p1-vertex-p2 is 90 degrees."""
        l1 = Line2D(vertex, p1)
        l2 = Line2D(vertex, p2)
        return sympy.simplify(l1.angle_between(l2) - sympy.pi / 2) == 0

    def is_parallel(self, l1: Union[Line2D, Segment2D], l2: Union[Line2D, Segment2D]) -> bool:
        """Check if two lines or segments are parallel."""
        return Line2D(l1.p1, l1.p2).is_parallel(Line2D(l2.p1, l2.p2))

    def is_perpendicular(self, l1: Union[Line2D, Segment2D], l2: Union[Line2D, Segment2D]) -> bool:
        """Check if two lines or segments are perpendicular."""
        return Line2D(l1.p1, l1.p2).is_perpendicular(Line2D(l2.p1, l2.p2))

    # ------------------------------------------------------------------ #
    # Data Export                                                        #
    # ------------------------------------------------------------------ #

    def get_shape_data(self, shape: Any) -> Dict[str, Any]:
        """
        Export shape data as a dictionary for UI rendering.
        
        Args:
            shape: The geometric shape to export.
            
        Returns:
            Dictionary containing shape type and properties.
        """
        if sympy is None:
            return {"type": "unknown", "error": "SymPy not available"}

        if isinstance(shape, Point2D):
            return {"type": "point", "x": float(shape.x), "y": float(shape.y)}
        
        elif isinstance(shape, Line2D):
            # Line is defined by two points
            return {
                "type": "line", 
                "p1": [float(shape.p1.x), float(shape.p1.y)], 
                "p2": [float(shape.p2.x), float(shape.p2.y)],
                "equation": str(shape.equation())
            }
            
        elif isinstance(shape, Segment2D):
            return {
                "type": "segment", 
                "p1": [float(shape.p1.x), float(shape.p1.y)], 
                "p2": [float(shape.p2.x), float(shape.p2.y)],
                "length": float(shape.length)
            }
            
        elif isinstance(shape, Ray2D):
            return {
                "type": "ray", 
                "source": [float(shape.source.x), float(shape.source.y)], 
                "point": [float(shape.p2.x), float(shape.p2.y)]
            }
            
        elif isinstance(shape, Circle):
            return {
                "type": "circle", 
                "center": [float(shape.center.x), float(shape.center.y)], 
                "radius": float(shape.radius)
            }
            
        elif isinstance(shape, Triangle):
            return {
                "type": "triangle", 
                "vertices": [[float(v.x), float(v.y)] for v in shape.vertices],
                "area": float(shape.area)
            }
            
        elif isinstance(shape, Polygon):
            return {
                "type": "polygon", 
                "vertices": [[float(v.x), float(v.y)] for v in shape.vertices],
                "area": float(shape.area)
            }
            
        return {"type": "unknown", "str": str(shape)}

    # ------------------------------------------------------------------ #
    # Proof Integration                                                  #
    # ------------------------------------------------------------------ #

    def get_congruence_fact(self, t1: Triangle, t1_name: str, t2: Triangle, t2_name: str) -> Optional[Any]:
        """
        Generate a Congruent fact if two triangles are congruent.
        Returns None if not congruent or if Fact class cannot be imported.
        """
        if self.check_congruence(t1, t2):
            try:
                from .proof_engine import Fact
                return Fact("Congruent", (t1_name, t2_name))
            except ImportError:
                return None
        return None
