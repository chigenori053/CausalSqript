"""Linear algebra helpers for vectors and matrices."""

from __future__ import annotations

from typing import List, Sequence, Tuple

try:  # pragma: no cover
    import sympy as _sympy
except Exception:  # pragma: no cover
    _sympy = None


class LinearAlgebraEngine:
    """Provides vector and matrix operations."""

    def vector_add(self, v1: Sequence[float], v2: Sequence[float]) -> List[float]:
        self._assert_same_length(v1, v2)
        return [a + b for a, b in zip(v1, v2)]

    def vector_subtract(self, v1: Sequence[float], v2: Sequence[float]) -> List[float]:
        self._assert_same_length(v1, v2)
        return [a - b for a, b in zip(v1, v2)]

    def dot(self, v1: Sequence[float], v2: Sequence[float]) -> float:
        self._assert_same_length(v1, v2)
        return float(sum(a * b for a, b in zip(v1, v2)))

    def cross(self, v1: Sequence[float], v2: Sequence[float]) -> List[float]:
        if len(v1) != 3 or len(v2) != 3:
            raise ValueError("Cross product is defined for 3D vectors only.")
        a1, a2, a3 = v1
        b1, b2, b3 = v2
        return [
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1,
        ]

    def scalar_multiply(self, scalar: float, vector: Sequence[float]) -> List[float]:
        return [scalar * value for value in vector]

    def matrix_add(self, m1: Sequence[Sequence[float]], m2: Sequence[Sequence[float]]) -> List[List[float]]:
        self._assert_same_matrix_shape(m1, m2)
        return [
            [a + b for a, b in zip(row1, row2)]
            for row1, row2 in zip(m1, m2)
        ]

    def matrix_multiply(self, m1: Sequence[Sequence[float]], m2: Sequence[Sequence[float]]) -> List[List[float]]:
        if not m1 or not m2:
            raise ValueError("Matrices must not be empty.")
        rows, cols = len(m1), len(m2[0])
        inner = len(m1[0])
        if any(len(row) != inner for row in m1):
            raise ValueError("Matrix m1 has inconsistent row sizes.")
        if any(len(row) != cols for row in m2):
            raise ValueError("Matrix m2 has inconsistent row sizes.")
        if len(m2) != inner:
            raise ValueError("Matrix dimensions do not align for multiplication.")
        result = []
        for i in range(rows):
            result_row = []
            for j in range(cols):
                value = sum(m1[i][k] * m2[k][j] for k in range(inner))
                result_row.append(value)
            result.append(result_row)
        return result

    def matrix_transpose(self, matrix: Sequence[Sequence[float]]) -> List[List[float]]:
        if not matrix:
            return []
        width = len(matrix[0])
        if any(len(row) != width for row in matrix):
            raise ValueError("Matrix rows must have equal length.")
        return [[row[i] for row in matrix] for i in range(width)]

    def determinant(self, matrix: Sequence[Sequence[float]]) -> float:
        size = len(matrix)
        if size == 0:
            raise ValueError("Matrix must not be empty.")
        if any(len(row) != size for row in matrix):
            raise ValueError("Matrix must be square.")
        if size == 1:
            return float(matrix[0][0])
        if size == 2:
            return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
        if size == 3:
            a, b, c = matrix
            return float(
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
        raise ValueError("Determinant helper supports up to 3x3 matrices.")

    def solve_linear_system(
        self,
        coefficients: Sequence[Sequence[float]],
        constants: Sequence[float],
    ) -> List[float]:
        if not coefficients:
            raise ValueError("Coefficient matrix must not be empty.")
        if len(coefficients) != len(constants):
            raise ValueError("System must have same number of equations and constants.")
        if _sympy is not None:
            matrix = _sympy.Matrix(coefficients)
            vector = _sympy.Matrix(constants)
            try:
                solution = matrix.LUsolve(vector)
                return [float(val) for val in solution]
            except Exception as exc:  # pragma: no cover
                raise ValueError(str(exc)) from exc
        return self._gaussian_elimination(coefficients, constants)

    def eigenvalues(self, matrix: Sequence[Sequence[float]]) -> List[float]:
        if _sympy is not None:
            sym_matrix = _sympy.Matrix(matrix)
            vals = sym_matrix.eigenvals()
            return [float(val) for val in vals.keys()]
        size = len(matrix)
        if any(len(row) != size for row in matrix):
            raise ValueError("Matrix must be square.")
        if size == 2:
            a, b = matrix
            trace = a[0] + b[1]
            det = a[0] * b[1] - a[1] * b[0]
            discriminant = trace ** 2 - 4 * det
            if discriminant < 0:
                return []
            root = discriminant ** 0.5
            return [(trace + root) / 2, (trace - root) / 2]
        raise ValueError("Eigenvalue fallback supports up to 2x2 matrices.")

    def eigenvectors(self, matrix: Sequence[Sequence[float]]) -> List[List[float]]:
        if _sympy is not None:
            sym_matrix = _sympy.Matrix(matrix)
            vects = sym_matrix.eigenvects()
            eigenvectors: List[List[float]] = []
            for val, _, vectors in vects:
                for vec in vectors:
                    eigenvectors.append([float(x) for x in vec])
            return eigenvectors
        # simple fallback for 2x2
        values = self.eigenvalues(matrix)
        vectors: List[List[float]] = []
        for value in values:
            vec = self._eigenvector_2x2(matrix, value)
            if vec:
                vectors.append(vec)
        return vectors

    def _gaussian_elimination(
        self,
        coeffs: Sequence[Sequence[float]],
        constants: Sequence[float],
    ) -> List[float]:
        matrix = [list(row) + [const] for row, const in zip(coeffs, constants)]
        n = len(constants)

        for i in range(n):
            pivot = i + max(range(n - i), key=lambda k: abs(matrix[i + k][i]))
            matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
            pivot_val = matrix[i][i]
            if abs(pivot_val) < 1e-12:
                raise ValueError("Matrix is singular.")
            matrix[i] = [val / pivot_val for val in matrix[i]]
            for j in range(n):
                if j == i:
                    continue
                factor = matrix[j][i]
                matrix[j] = [
                    matrix[j][k] - factor * matrix[i][k] for k in range(n + 1)
                ]

        return [row[-1] for row in matrix]

    def _eigenvector_2x2(
        self,
        matrix: Sequence[Sequence[float]],
        eigenvalue: float,
    ) -> List[float] | None:
        a, b = matrix
        m11 = a[0] - eigenvalue
        m12 = a[1]
        m21 = b[0]
        m22 = b[1] - eigenvalue

        if abs(m11) > 1e-9 or abs(m12) > 1e-9:
            if abs(m12) > 1e-9:
                x = 1.0
                y = -m11 / m12
            elif abs(m11) > 1e-9:
                y = 1.0
                x = -m12 / m11
            else:
                x = 1.0
                y = 0.0
        elif abs(m21) > 1e-9 or abs(m22) > 1e-9:
            if abs(m22) > 1e-9:
                x = 1.0
                y = -m21 / m22
            elif abs(m21) > 1e-9:
                y = 1.0
                x = -m22 / m21
            else:
                x = 1.0
                y = 0.0
        else:
            return None
        return [float(x), float(y)]

    def _assert_same_length(self, v1: Sequence[float], v2: Sequence[float]) -> None:
        if len(v1) != len(v2):
            raise ValueError("Vectors must have the same length.")

    def _assert_same_matrix_shape(
        self,
        m1: Sequence[Sequence[float]],
        m2: Sequence[Sequence[float]],
    ) -> None:
        if len(m1) != len(m2):
            raise ValueError("Matrices must have the same shape.")
        for row1, row2 in zip(m1, m2):
            if len(row1) != len(row2):
                raise ValueError("Matrices must have the same shape.")
