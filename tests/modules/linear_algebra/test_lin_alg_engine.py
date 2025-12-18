import pytest

from coherent.engine.linear_algebra_engine import LinearAlgebraEngine


@pytest.fixture
def la_engine():
    return LinearAlgebraEngine()


def test_vector_operations(la_engine):
    assert la_engine.vector_add([1, 2], [3, 4]) == [4, 6]
    assert la_engine.vector_subtract([5, 4], [1, 2]) == [4, 2]
    assert la_engine.dot([1, 2, 3], [4, 5, 6]) == pytest.approx(32)
    assert la_engine.cross([1, 0, 0], [0, 1, 0]) == [0, 0, 1]


def test_matrix_operations(la_engine):
    m1 = [[1, 2], [3, 4]]
    m2 = [[5, 6], [7, 8]]
    added = la_engine.matrix_add(m1, m2)
    assert added == [[6, 8], [10, 12]]
    multiplied = la_engine.matrix_multiply(m1, m2)
    assert multiplied == [[19, 22], [43, 50]]
    transpose = la_engine.matrix_transpose(m1)
    assert transpose == [[1, 3], [2, 4]]
    det = la_engine.determinant(m1)
    assert det == pytest.approx(-2)


def test_linear_system_and_eigen(la_engine):
    solution = la_engine.solve_linear_system([[2, 1], [1, -1]], [4, 1])
    assert solution[0] == pytest.approx(5 / 3, rel=1e-6)
    eigenvals = la_engine.eigenvalues([[2, 0], [0, 3]])
    assert set(round(val, 6) for val in eigenvals) == {2.0, 3.0}
    eigenvectors = la_engine.eigenvectors([[2, 0], [0, 3]])
    assert any(abs(vec[0]) > 0 and abs(vec[1]) < 1e-6 for vec in eigenvectors)
