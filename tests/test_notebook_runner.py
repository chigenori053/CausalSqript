from textwrap import dedent

import pytest

from coherent.tools.notebook_runner import execute_mathlang


def test_execute_mathlang_accepts_notebook_friendly_code() -> None:
    source = dedent(
        """
        problem: (x - y)^2
        step1: (x - y)(x - y)
        step: x^2 -2xy +y^2
        end: done
        """
    )
    lines = execute_mathlang(source)
    assert "problem: (x - y)**2" in lines
    assert "step2: x^2 -2xy +y^2" in lines or "step2: x**2 - 2*x*y + y**2" in lines
    assert "end: done" in lines


def test_execute_mathlang_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        execute_mathlang(
            "problem: 1 + 1\nstep: 2\nend: 2",
            mode="invalid-mode",
        )
