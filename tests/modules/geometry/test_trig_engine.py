import math
import pytest

from coherent.engine.trig_engine import TrigHelper


def test_trig_unit_circle():
    helper = TrigHelper()
    point = helper.unit_circle_point(90, unit="degrees")
    assert point["sin"] == 1.0
    assert point["cos"] == 0.0


def test_trig_evaluate():
    helper = TrigHelper()
    value = helper.evaluate("sin", math.pi / 6)
    assert value == pytest.approx(0.5, rel=1e-6)


def test_trig_identity():
    helper = TrigHelper()
    result = helper.apply_identity("sin_double", math.pi / 6)
    # sin(2 * pi/6) = sin(pi/3) = sqrt(3)/2
    assert result == pytest.approx(math.sqrt(3) / 2, rel=1e-6)


def test_trig_sum_identity():
    helper = TrigHelper()
    result = helper.apply_identity("sin_sum", math.pi / 6, other_angle=math.pi / 3)
    assert result == pytest.approx(1.0, rel=1e-6)
