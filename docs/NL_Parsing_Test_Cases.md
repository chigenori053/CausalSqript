# Natural Language Parsing Test Cases

Use these sentences to verify the Natural Language Processing capabilities in the `Agent Solver` tab of the UI.

## 1. Algebra (代数)

### English
- **Standard Equation**: `Solve x^2 - 4 = 0`
- **Factorization**: `Factorize x^2 + 5x + 6`
- **Inequality**: `Solve x + 3 > 5`
- **System of Equations**: `Solve x + y = 5 and x - y = 1`

### Japanese (日本語)
- **方程式**: `方程式 x^2 - 4 = 0 を解いて`
- **因数分解**: `x^2 + 5x + 6 を因数分解してください`
- **連立方程式**: `x + y = 5 と x - y = 1 を解く`

## 2. Calculus (微積分)

### English
- **Differentiation**: `Differentiate sin(x)`
- **Differentiation (Rule)**: `Find the derivative of x^2 * exp(x)`
- **Integration (Indefinite)**: `Integrate 1/x`
- **Integration (Definite)**: `Integrate x^2 from 0 to 1`

### Japanese (日本語)
- **微分**: `sin(x) を微分して`
- **導関数**: `x^2 の導関数を求めて`
- **積分**: `x^2 を 0 から 1 まで積分して`

## 3. General & Logic (一般・論理)

### English
- **Verification**: `Verify that (x+1)^2 = x^2 + 2x + 1`
- **Simplification**: `Simplify (x^2 + 2x + 1) / (x + 1)`
- **Explanation**: `Explain how to solve x^2 = 9`

### Japanese (日本語)
- **検証**: `(x+1)^2 = x^2 + 2x + 1 であることを確認して`
- **簡約化**: `(x^2 + 2x + 1) / (x + 1) を簡単にして`
- **解説**: `x^2 = 9 の解き方を教えて`

## 4. Edge Cases / Natural Phrasing (自然な言い回し)

- `Can you help me solve x^2 - 1 = 0?`
- `Please find the solution for sin(x) = 0`
- `x^2 - 4 = 0 の解を知りたい`
- `Is x=2 a solution to x^2=4?` (Intent: Verify)
