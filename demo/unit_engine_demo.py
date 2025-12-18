#!/usr/bin/env python3
"""
UnitEngine Demo

Demonstrates the capabilities of the UnitEngine:
- Unit conversion
- Simplification
- Dimensional consistency checking
- SI unit conversion
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.unit_engine import UnitEngine


def main():
    print("=" * 60)
    print("  UnitEngine Demo")
    print("=" * 60)

    # Initialize engine
    symbolic = SymbolicEngine()
    unit_engine = UnitEngine(symbolic)

    # 1. Unit Conversion
    print("\n1. Unit Conversion")
    conversions = [
        ("10 * meter", "centimeter"),
        ("1 * hour", "second"),
        ("100 * kilometer / hour", "meter / second"),
        ("1 * mile", "kilometer")
    ]

    for expr, target in conversions:
        try:
            result = unit_engine.convert(expr, target)
            print(f"  {expr} -> {target} : {result}")
        except Exception as e:
            print(f"  Error converting {expr} to {target}: {e}")

    # 2. Simplification
    print("\n2. Simplification")
    expressions = [
        "1000 * meter / kilometer",
        "1 * minute / second",
        "1 * kilogram * meter / second**2"  # Newton
    ]

    for expr in expressions:
        result = unit_engine.simplify(expr)
        print(f"  {expr} -> {result}")

    # 3. Dimensional Consistency
    print("\n3. Dimensional Consistency")
    consistency_checks = [
        "1 * meter + 20 * centimeter",
        "1 * meter + 1 * second",
        "10 * kilometer / hour - 5 * meter / second"
    ]

    for expr in consistency_checks:
        is_consistent = unit_engine.check_consistency(expr)
        status = "Consistent" if is_consistent else "Inconsistent"
        print(f"  {expr} : {status}")

    # 4. SI Unit Conversion
    print("\n4. SI Unit Conversion")
    si_conversions = [
        "1 * kilometer",
        "1 * hour",
        "100 * gram"
    ]

    for expr in si_conversions:
        result = unit_engine.get_si_units(expr)
        print(f"  {expr} -> {result}")

    print("\n" + "=" * 60)
    print("  Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
