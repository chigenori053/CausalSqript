# コアアーキテクチャ検証レポート

## 1. DSL入力検証
**ステータス: 実装済み**

-   **メカニズム**: `CoreRuntime` クラス（`core/core_runtime.py` 内）が `set()` および `check_step()` メソッドを通じてDSL入力を処理します。
-   **検証フロー**:
    1.  入力は `_normalize_expression` を通じて正規化されます。
    2.  `check_step` は前の状態（`before`）から新しい状態（`after`）への遷移を検証します。
    3.  多層的な検証戦略を採用しています：
        -   **記号的等価性**: `SymbolicEngine.is_equiv` を使用します。
        -   **スカラー等価性**: 式がスカラー倍の違いのみであるかを確認します（方程式の場合）。
        -   **シナリオ検証**: 複数の数値シナリオにわたって等価性を検証します（`check_equivalence_in_scenarios`）。
        -   **ファジィ判定**: 記号的なチェックが失敗した場合、`FuzzyJudge` にフォールバックします。

## 2. Symbolic Engineの最適化
**ステータス: 実装済み（アーキテクチャ上のニュアンスあり）**

-   **メカニズム**: アーキテクチャは、メインの `SymbolicEngine` と並行して、特化したエンジン（`CalculusEngine`, `LinearAlgebraEngine`, `StatsEngine`）を初期化します。
-   **最適化ロジック**:
    -   `CoreRuntime` は `ExpressionClassifier` を使用して現在の問題のドメイン（例：「calculus（微積分）」）を判定します。
    -   `generate_optimization_report()` メソッドはエンジンモードを明示的にレポートします：
        ```python
        "symbolic_engine_mode": "optimized" if "calculus" in self._current_domains else "standard"
        ```
    -   **考察**: *解法*（例：`calc_derivative`）には特化したエンジンが存在しますが、コアとなる*検証*ループ（`check_step`）は主に（SymPyをラップした）統一された `SymbolicEngine` を使用します。「最適化」とは、ステップごとに検証エンジンのインスタンス自体を入れ替えることではなく、システムがドメインを認識し、ドメイン固有のツールを利用可能にしていることを指します。

## 3. 計算ルールマップの最適化
**ステータス: 実装済み**

-   **メカニズム**: `KnowledgeRegistry`（`core/knowledge_registry.py` 内）が計算ルールを管理します。
-   **最適化ロジック**:
    -   `match()` メソッドは `context_domains` を受け取ります。
    -   現在のドメインに一致するルールマップの検索を優先します（例：ドメインが「calculus」の場合、微積分のマップを最初に検索します）。
    -   これにより、最も関連性の高いルールが最初にチェックされ、パフォーマンスと精度が最適化されます。

## 4. レポートレンダリングの最適化
**ステータス: 実装済み**

-   **メカニズム**: `LaTeXFormatter`（`core/latex_formatter.py` 内）がレンダリングを処理します。
-   **最適化ロジック**:
    -   `LaTeXFormatter` は `ExpressionClassifier` を使用して式のドメインを判定します。
    -   これらの `context_domains` を `SymbolicEngine.to_latex` に渡します。
    -   `SymbolicEngine.to_latex` はドメインに基づいて出力を適応させます（例：「algebra（代数）」コンテキストでは乗算記号を省略するなど）。
    -   `CoreRuntime.generate_optimization_report` はこの戦略を確認します：
        ```python
        "report_rendering_strategy": "latex_enhanced" if "calculus" in self._current_domains ... else "standard"
        ```

## 結論
要求された機能は現在のコアアーキテクチャに実装されています。システムは問題のドメインを正しく識別し、それを使用してルールマッチングやレポートレンダリングを最適化し、それに応じてSymbolic Engineのモードを報告しています。
