# Coherent Core — SymbolicEvaluationEngine 修正仕様書（Codex対応版）

Version: 1.0  
Status: Implementation Directive  
Audience: Code Model (Codex), Dev  
Scope: 修正対象 → SymbolicEvaluationEngine / SymbolicEngine / Evaluator v2  
Purpose: 記号式（symbolic expression）の評価と等価判定の誤動作を修正する

---

# 1. Problem Summary

## 現状の問題点

`SymbolicEvaluationEngine.check_step()` および `check_end()` は以下のような内部フローになっている：

```
evaluate(before_expr)
evaluate(after_expr)
valid = before_eval == after_eval
```

`evaluate(expr)` は以下を前提としている：

- context に変数の値が必要  
- context が空（{}）の場合、式に記号（x, y …）が含まれると Undefined symbols エラー  

このため、次のような典型的な記号式でも fatal となる：

```
problem: (x - y)^2
step1: (x - y)(x + y)
```

---

# 2. 修正方針

以下 3 点を Codex が実装可能な形で明示する。

### Rule 1: evaluate() を Step 判定に使用しない
- evaluate() は数値評価専用
- 記号式の等価判定には使用しない

### Rule 2: Step 等価判定は is_equiv() を使用する
```
valid = symbolic_engine.is_equiv(before, after)
```

### Rule 3: evaluate() の Undefined symbols は fatal にしない
- 記号式での evaluate は「not_evaluatable」フラグを返す
- fatal にしない

---

# 3. API 修正仕様

## 3.1 check_step() 修正版

### Before

```python
before_eval = self.evaluate(self._current_expr)
after_eval = self.evaluate(expr)
valid = before_eval == after_eval
```

### After (spec)

```
1. before = self._current_expr
2. after = expr
3. valid = symbolic_engine.is_equiv(before, after)
4. rule_id = knowledge_registry.match(before, after)
5. self._current_expr = after
6. return {"valid": valid, "rule_id": rule_id}
```

---

## 3.2 check_end() 修正版

```
before = self._current_expr
after = end_expr
valid = symbolic_engine.is_equiv(before, after)
status = ok / mistake
fatal = 構文エラー等のみ
```

---

# 4. SymbolicEngine.is_equiv() の仕様

## 入力
```
expr1: str
expr2: str
```

## 出力
```
True / False
```

## アルゴリズム

```
1. e1 = sympify(expr1)
2. e2 = sympify(expr2)
3. diff = simplify(e1 - e2)
4. if diff == 0:
       return True
5. else:
       fallback numeric sampling:
          assign small random numbers to free symbols
          compare evaluated e1 and e2 for multiple samples
6. 一致しなければ False
```

## fatal 条件
- sympify 失敗のみ（InvalidExprError）

---

# 5. evaluate() の新仕様

## 目的
- 数値評価専用関数
- 記号式の等価判定には使用しない

## 修正内容
- context が空で free_symbols がある場合：

```
return {"not_evaluatable": True}
```

- fatal を投げない

## fatal 条件
- context があるのに未定義変数が含まれる場合

---

# 6. Evaluator v2 側の修正点

### Before
- check_step() が EvaluationError を受けると fatal

### After
- EvaluationError のうち：
  - “not_evaluatable” → fatal ではない  
  - “構文エラー” → fatal  
  - “SymbolicEngine internal error” → fatal

### Step 判定
```
valid = engine.check_step(expr).valid
status = ok or mistake
```

---

# 7. Patch 指示（Codex向け）

```
[PATCH 1]
Replace evaluate() in check_step() and check_end()
with:
    valid = self.symbolic_engine.is_equiv(before, after)

[PATCH 2]
Modify evaluate() to return {"not_evaluatable": True}
when context is empty and free symbols exist.

[PATCH 3]
Evaluator._fatal() must ignore EvaluationError("not_evaluatable").

[PATCH 4]
Implement fallback numeric sampling in is_equiv().
```

---

# 8. Expected Behavior After Fix

### 正常な symbolic ステップ

```
problem: (x - y)^2
step1: (x - y)(x + y)
step2: x^2 - 2xy + y^2
end:   x^2 - 2xy + y^2
```

→ evaluate() は呼ばれない  
→ is_equiv() により適切に ok 判定  
→ fatal にならない  

### 壊れた式
```
step1: x^? + 3
```

→ sympify 失敗 → InvalidExprError → fatal

---

# End of Specification
