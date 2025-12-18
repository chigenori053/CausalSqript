<<<<<<< ours
# タスク管理シート

## 1. サマリー

本プロジェクト「Coherent」は、数学的な思考プロセスを可視化することを目的とした教育用ドメイン固有言語（DSL）です。コアアーキテクチャは、構文木（AST）と数理定理を適用するルールベースのエンジンを分離しており、教育的な透明性を重視した設計となっています。

## 2. 現在の進行状況

- **フェーズ1（完了）**: 基本的なパーサー（`core/parser.py`）とエバリュエーター（`core/evaluator.py`）は安定しており、リグレッションテストのみが必要です。
- **フェーズ2 & 3（進行中）**: 現在は多項式演算とロギング機能の開発に注力しています。
  - **多項式演算**: コアロジックはほぼ完成しており、`core/polynomial_evaluator.py` が実装済みです。主要な機能は `tests/test_polynomial.py` によってテストされています。
  - **ロギング**: `LearningLogger` が実装・統合され、評価ステップとルールメタデータを記録できることが確認されています。
- **最優先課題**: 上記機能のデモンストレーションを行うためのJupyterデモの作成が、現在の開発フェーズを完了させるための最終ステップとなります。

## 3. 優先度別タスクリスト

### 優先度：高

| ID | タスク名 | 内容 | 完了条件 |
| :-- | :--- | :--- | :--- |
| P3-01 | JupyterLabデモの作成 | `notebooks/Learning_Log_Demo.ipynb` を開発し、`PolynomialEvaluator` と `LearningLogger` の機能を示すデモを作成する。`docs/test/TestPlan.md` のシナリオを参考にする。 | デモノートブックが完成し、主要機能の動作が確認できる。 |

### 優先度：中

| ID | タスク名 | 内容 | 完了条件 |
| :-- | :--- | :--- | :--- |
| T-01 | テストカバレッジの強化 | `tests/` ディレクトリ内のテストを見直し、特に多項式エバリュエーターに関するより複雑なシナリオを追加する。 | 主要なロジックとエッジケースがテストでカバーされている。 |
| D-01 | ナレッジノードの改良とドキュメント化 | `core/knowledge/arithmetic/` 内のルールを見直し、`docs/knowledge_nodes_spec.md` との整合性を確保し、ドキュメントを整備する。 | 全てのナレッジノードに説明が付与され、仕様と一致している。 |

### 優先度：低 / 将来のタスク

| ID | タスク名 | 内容 | 完了条件 |
| :-- | :--- | :--- | :--- |
| P4-01 | フェーズ4の計画 | 教育リリースv0.9に向けた仕様策定を開始する。証明モジュールや自然言語統合などの将来的な拡張機能を検討する。 | 次期バージョンの開発ロードマップが策定されている。 |

## 4. 開発状況の強化に向けた提案

1.  **CI/CDの導入**: テストの自動化とコード品質チェック（Lint、フォーマット）を継続的に行うため、GitHub ActionsなどのCI/CDパイプラインを導入することを推奨します。これにより、手動でのテスト実行の手間を省き、品質を維持しやすくなります。
2.  **ドキュメントの拡充**: 現在のドキュメントは優れていますが、開発者が新たに参加しやすくなるよう、各モジュールのAPIリファレンスや、より詳細なチュートリアルを追加することが望ましいです。
3.  **ユーザーフィードバックの収集**: Jupyterデモが完成次第、ターゲットとなる教育関係者や学生からフィードバックを収集する仕組みを設けることで、プロジェクトの方向性をより実践的なものに調整できます。

以上
=======
# Coherent 開発タスク管理シート

## レジェンド
- **フェーズ**: README/仕様書で定義されたPhase 1〜4に対応。
- **優先度**: H (高) / M (中) / L (低)。
- **状態**: 未完了 / 進行中 / 完了（元の Backlog/Blocked/Review 等は備考に記載）。

## サマリー（2025-11-11 時点）
- **Core Engineリファクタリング**: `core/ast_nodes.py`, `core/parser.py`, `core/evaluator.py`, `core/optimizer.py`, `core/polynomial_evaluator.py` を `docs/Coherent_Core_Engine_Architecture_v1.md` に沿って更新済み。`tests/test_parser.py` / `tests/test_evaluator.py` / `tests/test_optimizer.py` もグリーンを維持している。
- **Knowledge Base整備**: `core/knowledge/__init__.py` と `core/knowledge/arithmetic/*.json` でKnowledgeRegistryと四則カテゴリの雛形を配置したが、`docs/knowledge_nodes_spec.md` に記載したカテゴリ分割・テスト連携は未完了のため P2-06 / P2-07 / P2-09 が残る。
- **Phase 3着手**: `core/logging.py` の `LearningLogger` を `core/evaluator.py` に接続し、`notebooks/Learning_Log_Demo.ipynb` で出力デモを確認。Notebook UI拡張とDemoパッケージ化（P3-01/02/03/04）を並行推進中。

## 進行評価
| フェーズ/領域 | 進捗率 | 状況・メモ |
|---------------|--------|-------------|
| Phase 1（DSL/CLI/Tests） | 100% | 仕様で定義されたP1タスクは全てDone。 |
| Phase 2（SymbolicAI & Knowledge） | 90% | コアエンジンを新アーキテクチャ仕様にリファクタリング完了。知識ノード再編（P2-06/P2-07）がBacklog。 |
| Phase 3（UI/Logs/Demo） | 25% | LearningLoggerとLearning LogデモNotebookを追加。追加タスク継続中。 |
| Phase 4（リリース準備） | 0% | タスク定義のみ。デモ完成後に着手。 |

## タスク一覧（ステータス別）

### 未完了
| ID | フェーズ | カテゴリ | タスク概要 | 完了条件 | 担当 | 優先度 | 状態 | 期限 | 備考/依存 |
|----|----------|----------|------------|----------|------|--------|-------|------|------------|
| P2-06 | Phase 2 | Knowledge Base | Core DSL向け KnowledgeRegistry設計 | `core/knowledge/`配下をカテゴリ分割しロードAPI/仕様を同期 | TBD | H | 未完了（Backlog） | 2025-12-05 | `docs/knowledge_nodes_spec.md`と`core/knowledge/__init__.py`の下地はあるが advancedカテゴリやロード順序設計、仕様追記が未完。 |
| P2-07 | Phase 2 | Knowledge Base | 四則演算カテゴリ別ノード整備 | `arithmetic/{addition,subtraction,multiplication,division}.json`に実用ルールを追加しテストを更新 | TBD | H | 未完了（Backlog） | 2025-12-10 | `core/knowledge/arithmetic/*.json`には各1ルールのみで、`Evaluator`/`PolynomialEvaluator`照合テスト（`tests/test_polynomial_scenario.py`等）が未連携。 |
| P2-09 | Phase 2 | テスト | Core DSL等価性検証テストの修正 | リファクタで無効化した等価性テストをCI緑に戻す | TBD | M | 未完了（Backlog） | 2025-11-16 | `tests/test_evaluator.py`の`_test_core_dsl_*`は先頭アンダースコアで収集されず、KnowledgeRegistry＋SymbolicEngineの統合確認がスキップされたまま。 |
| P3-03 | Phase 3 | Demo | Demoパッケージ化 | Notebook/CLI/シンボリック例をまとめた`demo/`配布物と実行スクリプトを整備 | TBD | H | 未完了（Backlog） | 2025-12-27 | `demo/`ディレクトリ未作成。P3-01/02とOPS-01の成果が前提。 |
| P3-04 | Phase 3 | UI | インタラクティブなNotebook UI | ipywidgetsを使いCoherentコードを実行できるUIを実装 | TBD | M | 未完了（Backlog） | 2025-12-15 | ベースは`notebooks/Learning_Log_Demo.ipynb`のみで、ウィジェット連携や`main.py`呼び出しUIが未実装。 |
| P4-01 | Phase 4 | リリース | Beta v0.9リリース準備 | リリースノート＋タグ付け＋README更新 | TBD | M | 未完了（Backlog） | 2026-01-15 | `docs/Coherent_SPECIFICATION.md`/`README.md`更新とデモ完成が依存。 |
| OPS-01 | Cross | ドキュメント | README/仕様書の二重管理整備 | 仕様更新時の差分手順テンプレ作成 | TBD | M | 未完了（Backlog） | 2024-11-15 | `Agent_ja.md`に運用メモはあるがテンプレ未作成。 |
| OPS-02 | Cross | 技術的負債 | Notebookフォーマット警告解消 | nbformat.normalize()等でMissingIDFieldWarningを解消 | TBD | L | 未完了（Backlog） | 2025-11-25 | `notebooks/Learning_Log_Demo.ipynb`の警告ログをCIで拾う仕組みが未整備。 |

### 進行中
| ID | フェーズ | カテゴリ | タスク概要 | 完了条件 | 担当 | 優先度 | 状態 | 期限 | 備考/依存 |
|----|----------|----------|------------|----------|------|--------|-------|------|------------|
| P3-01 | Phase 3 | UI | JupyterLabデモノートブック | `edu/examples/`配下にデモNotebookと手順書を配置 | TBD | H | 進行中 | 2025-12-20 | `notebooks/Learning_Log_Demo.ipynb`初版と`edu/examples/` CLIサンプルあり。UI磨き込みと手順書更新を継続。 |
| P3-02 | Phase 3 | 学習ログ | 学習ログ導出API | 評価過程をJSONログに保存する`Evaluator`拡張 | TBD | L | 進行中 | 2025-12-22 | `core/logging.py`の`LearningLogger`と`core/evaluator.py`連携済。ファイル永続化とCLIフラグ未提供。 |

### 完了
| ID | フェーズ | カテゴリ | タスク概要 | 完了条件 | 担当 | 優先度 | 状態 | 期限 | 備考/依存 |
|----|----------|----------|------------|----------|------|--------|-------|------|------------|
| P1-02 | Phase 1 | AST | ASTノード定義の整理 | `core/ast_nodes.py`に主要statement/expressionノードを網羅 | TBD | M | 完了 | 2025-11-05 | v1仕様に合わせて`ast.Program`/`ast.Step`等を再定義済。 |
| P1-03 | Phase 1 | Evaluator | ステップ実行とログ出力 | `core/evaluator.py`が実行順ログを生成、簡単なサンプルで再現性保証 | TBD | H | 完了 | 2025-11-10 | `Evaluator.run()`で一貫した`EvaluationResult`が得られ、`tests/test_evaluator.py`で検証。 |
| P1-04 | Phase 1 | テスト | Parser/Evaluator向けpytest整備 | `tests/test_parser.py`と`tests/test_evaluator.py`が主要ケースを網羅しCI緑 | TBD | H | 完了 | 2025-11-12 | `uv run pytest`でParser/Evaluator計41ケース成功（2025-11-09時点）。 |
| P1-05 | Phase 1 | CLI/UX | CLIとExamples整備 | `main.py`でファイル実行でき、`edu/examples/`配下にチュートリアルが存在 | TBD | M | 完了 | 2025-11-15 | CLI `python main.py edu/examples/polynomial_arithmetic.mlang` が動作、READMEにも手順記載。 |
| P1-06 | Phase 1 | テスト | エラーパスと境界値テスト | division-by-zero/未定義変数/多重unary等の失敗系がpytestで再現 | TBD | M | 完了 | 2025-11-18 | `tests/test_evaluator.py`でゼロ除算/未定義変数/多重unaryを網羅。 |
| P2-01 | Phase 2 | SymbolicAI | SymPy連携PoC | `core/symbolic_engine.py`で`simplify`/`explain`最小実装、2例以上のテスト付き | TBD | H | 完了 | 2025-12-03 | `tests/test_symbolic_engine.py`と`tests/test_cli.py`でPoC挙動を検証。 |
| P2-02 | Phase 2 | AST最適化 | AST最適化パス追加 | 冗長演算を統合する最適化が1ケース以上で効果検証 | TBD | M | 完了 | 2025-12-08 | `core/optimizer.py`新設＋`tests/test_optimizer.py`で定数畳み込み/代入展開を検証。 |
| P2-03 | Phase 2 | SymbolicAI | Evaluator統合＆trace拡張 | Evaluator/CLIからSymbolicEngineを呼び出し`show`出力で簡約結果を提示 | TBD | H | 完了 | 2025-12-05 | `Evaluator(symbolic_engine_factory)`導入、CLI `--symbolic-trace`で統合済。 |
| P2-04 | Phase 2 | インフラ | SymPy/Jupyter環境整備 | SymPyをCI/開発環境でインストールし`python main.py --symbolic`が実機で成功 | TBD | H | 完了 | 2025-11-25 | `uv`環境でSymPy導入済、CIも`uv run pytest`で成功（2025-11-09）。 |
| P2-05 | Phase 2 | SymbolicAI | 多項式四則演習スクリプト | CLI/`--polynomial`で演算結果を表示 | TBD | H | 完了 | 2025-12-02 | `edu/examples/polynomial_arithmetic.mlang`＋`tests/test_polynomial.py`/`tests/test_polynomial_scenario.py`で確認。 |
| P2-08 | Phase 2 | リファクタリング | コアエンジンを新アーキテクチャ仕様に準拠 | ast_nodes, parser, evaluator, optimizer等を更新しテストを修正 | TBD | H | 完了 | 2025-11-09 | `core/`各モジュールを`docs/Coherent_Core_Engine_Architecture_v1.md`準拠へ移行済。 |

## デモ版完成までのプロセス（案）
| ステップ | 期間目安 | 主担当タスク | 成果物/Exit Criteria | リスク・備考 |
|---------|-----------|---------------|----------------------|---------------|
| 1. Hello World実行保証 | 〜11/18 | P1-05, P2-04 | `python main.py -c "show 'Hello World'"`などで確実に表示でき、README/Specに手順記載 | **完了** |
| 2. 多項式四則メカニズム | 11/19〜12/02 | P2-01, P2-03, P2-05 | SymPy導入済みで、Evaluator/CLIが多項式の加減乗除・追跡説明を表示（四則演習スクリプト含む） | **完了** |
| 2.5 知識ノード再編 | 12/01〜12/10 | P2-06, P2-07 | Core DSL v1に合わせた KnowledgeRegistry と四則カテゴリ別ノードを実装・テスト | 新規タスク。Phase 2へ差し戻し |
| 3. 演習Notebook整備 | 12/03〜12/15 | P3-01, P3-02 | 「Hello World紹介＋多項式四則演習」のJupyter Notebookと学習ログ出力 | Notebookメンテ負荷、UI要件の追加発生 |
| 4. Demoパッケージ化 | 12/16〜12/27 | P3-03, OPS-01, P4-01 | `demo/`配布物、CLI/Notebook手順書、Release checklist更新、デモ発表用ビルド | ドキュメント差異や依存ライブラリバージョンずれ |

## デモ到達度チェック（Hello World & 多項式演習）
| 項目 | ステータス | 根拠/課題 |
|------|------------|------------|
| Hello World 表示 | Done | `python main.py -c "show 'Hello World'"`で動作確認済。 |
| 多項式 四則演習 | Done | P2-05にて演習サンプル・テストを実装完了。`--polynomial`フラグで実行可能。 |
| 演習 Notebook | In Progress | `notebooks/Learning_Log_Demo.ipynb` の自動実行テストに成功。インタラクティブ化や多項式演習への拡張は継続中。 |

## 追加メモ
- タスク追加時はIDをフェーズ別に連番で拡張（例: P2-03）。
- 依存関係が変わった場合は備考欄を更新し、必要に応じてAgent.mdに同期メモを記載。
- ステータス更新はPRレビュー時・スプリント開始時など定期イベントで行う。
>>>>>>> theirs
