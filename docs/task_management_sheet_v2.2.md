# Coherent Task Management Sheet v2.3
## フォーカス: Core Extended (Computation / Validation / Hinting) 実装

---

## Core Extended タスク

### High Priority
| ID | タスク | 内容 | 期限 | 状態 |
|----|---------|------|------|------|
| EXT-01 | ComputationEngine 実装 | `SymbolicEngine` および `SymPy` をラップし、`numeric_eval` や `simplify` を提供する `ComputationEngine` を実装する | Day22 | 完了 |
| EXT-02 | ValidationEngine 実装 | `ExerciseSpec` に基づき、数式の等価性判定や形式チェックを行う `ValidationEngine` を実装する | Day22 | 完了 |
| EXT-03 | HintEngine 実装 | 誤答パターンやシンボリックな差分に基づいてヒントを生成する `HintEngine` を実装する | Day23 | 完了 |
| EXT-04 | CoreRuntime & Evaluator 統合 | 3つのエンジンを統合する `CoreRuntime` を実装し、既存の `Evaluator` から呼び出せるようにする | Day23 | 完了 |

### Medium Priority
| ID | タスク | 内容 | 期限 | 状態 |
|----|---------|------|------|------|
| EXT-05 | ExerciseSpec I/O | `ExerciseSpec` の JSON/YAML 入出力スキーマを定義・実装し、問題定義を外部ファイルから読み込めるようにする | Day24 | 完了 |
| EXT-06 | CLI/Notebook デモ | 新しい Core Extended 機能（計算・検証・ヒント）を利用した CLI コマンドおよび Notebook デモを作成する | Day24 | 完了 |
| EXT-07 | UnitEngine 実装 | Core アーキテクチャに基づき、単位変換・次元解析を行う `UnitEngine` を実装する | Day24 | 完了 |

---

## 残存タスク (v2.2より継続)

### Medium Priority
| ID | タスク | 内容 | 期限 | 状態 |
|----|---------|------|------|------|
| UICLI-05 | Notebook テンプレ整備 | `edu/notebooks/edu_intro.ipynb` など教育向け 2 本、`pro/notebooks/pro_intro_causal.ipynb` など研究向け 2 本の計 4 本を作成し、Core API と UI ヘルパーの使い方を解説する | Day18 | 完了 |

### Low Priority
| ID | タスク | 内容 | 期限 | 状態 |
|----|---------|------|------|------|
| DSL-10 | DSL LSP/補完計画 v2 | v2.5 で増加した DSL セクションを対象に、補完/LSP 対応 PoC の要求事項とエディタ別の優先順位を整理した計画書を作成する | Day20 | 未着手 |
| UICLI-07 | UI テーマ設定反映 | `edu/config/edu_ui_settings.yaml` と `pro/config/pro_settings.yaml` のテーマ情報を UI レイヤーで読み込み、Notebook/CLI から切り替え可能にする | Day20 | 未着手 |
| UICLI-08 | シナリオ作成ツール | `edu/lessons/` や `pro/examples/` を自動生成できる簡易スクリプト／テンプレ CLI を `tools/` に追加する | Day21 | 未着手 |

---

## 新規対応 (v2.6 draft 反映)

### High Priority
| ID | タスク | 内容 | 期限 | 状態 |
|----|---------|------|------|------|
| RENDER-01 | LaTeX レンダリング回帰テスト | `core/latex_formatter.py` で指数・負数乗算の崩れ防止をテスト追加（2**3 が 23 にならないことを確認） | ASAP | 完了 |
| LOG-01 | 生入力ログ保持 | Evaluator で `raw_expr` を優先して LearningLogger に記録し、SymPy 正規化後ではなくユーザー入力を表示できるようにする | ASAP | 完了 |

### Medium Priority
| ID | タスク | 内容 | 期限 | 状態 |
|----|---------|------|------|------|
| RULE-05 | 減算構造マッチ改良 | `core/knowledge_registry.py` で `a - b` 構造が崩れないようマッチ前に表記を調整し、ARITH-CALC-SUB を検出しやすくする | ASAP | 完了 |
| EXEC-03 | Executor ライフサイクル改善 | `core/computation_engine.py` で ProcessPoolExecutor をインスタンス共有にし、シナリオ並列評価のオーバーヘッドを削減 | ASAP | 完了 |

---

## 開発状況サマリー
- **完了済み**: DSL v2.5 対応、Evaluator v2、LearningLogger v2、CausalEngine v1、CLI 基本実装 (v2.2 までの主要タスク)
- **現在**: `Coherent_Core_Extended_Spec.md` に基づく Core 機能の拡張フェーズ。計算・検証・ヒント生成機能の追加を行い、教育利用におけるフィードバック機能を強化する。
- **次ステップ**: `ComputationEngine` から順次実装を開始し、`CoreRuntime` で統合する。
