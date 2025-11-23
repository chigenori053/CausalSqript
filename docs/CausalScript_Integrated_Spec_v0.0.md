# CausalScript Integrated Architecture Specification  
**Version: v0.0 (Codex-Ready Edition)**  
**Status: Draft / Full Specification**

本仕様書は、CausalScript 言語および計算エンジンのコアアーキテクチャを  
Codex・LLM・AI エージェントが **正確に解析・実装できる形式**で記述した統合仕様書である。

---

# 0. Glossary（用語一覧）

| 用語 | 意味 |
|------|------|
| DSL | CausalScript の記述言語 |
| AST | 抽象構文木 |
| Evaluator | AST を逐次解釈し Engine に指示するモジュール |
| Engine | 数式状態とルール適用を管理 |
| SymbolicEngine | 厳密同値判定（T=0）を行う数学エンジン |
| Fuzzy Reasoning Engine | 曖昧推論（T>0スコア）を行うエンジン |
| Rule Engine | 四則・分数等の規則群 |
| LearningLogger | ステップの評価ログを保存するモジュール |

---

# 1. System Overview（システム概要）

CausalScript は以下の8つのコンポーネントから構成される。

1. **DSL Parser**
2. **AST Nodes**
3. **Evaluator**
4. **Engine**
5. **SymbolicEngine**
6. **Fuzzy Reasoning Engine**
7. **Rule Engine**
8. **LearningLogger**

全体データフロー：
DSL → Parser → AST → Evaluator → Engine
├ SymbolicEngine (strict)
└ FuzzyEngine (fuzzy)
↓
Rule Engine → Logger → Output


---

# 2. DSL Specification

## 2.1 基本文法

problem: <expression>

step: <expression>
explain: "<text>"

step: <expression>

end: <expression>



## 2.2 DSL キー仕様

| Key | Required | Description |
|-----|----------|-------------|
| `problem:` | Yes | 初期式 |
| `step:` | No | 中間ステップ |
| `explain:` | No | 直前ステップの説明 |
| `end:` | No | 最終宣言 |

## 2.3 数式仕様（初期バージョン）

- Operators: `+`, `-`, `*`, `/`
- Literals: integers
- parentheses: `(` `)`

---

# 3. AST Specification

```python
@dataclass
class ProblemNode:
    expr_src: str
    lineno: int

@dataclass
class StepNode:
    expr_src: str
    explain: str | None
    rule_id: str | None
    lineno: int

@dataclass
class EndNode:
    expr_src: str
    lineno: int

Parser は必ず以下を生成：
@dataclass
class ProgramAST:
    problem: ProblemNode
    steps: list[StepNode]
    end: EndNode | None

4. Parser Specification
4.1 Parser Interface
class Parser:
    def parse(self, text: str) -> ProgramAST:
        """DSL テキストを AST に変換する"""

4.2 パース要件

problem: は 1 回のみ登場

step: は 0 回以上

explain: は直前の step: に紐付く

空行・コメントは無視してよい

5. Engine Specification

Engine は「現在の式状態」「問題式」「最後に適用したルール ID」を保持する。
