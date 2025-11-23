# CausalScript Reasoning Test Plan

このプランは、CausalScript が目標としている 3 つの柱
1. 計算ステップの正解/不正解認識
2. ルール判別および曖昧度推定（Fuzzy 判定）
3. 因果推論（因果グラフ／エラー解析）

を網羅的に検証するためのテストシナリオを整理したものです。各シナリオは既存テストまたは新規テストに対応し、実装済みであれば `pytest` によって自動化されています。

## 1. ステップ正しさ判定

| シナリオ | テスト | ポイント |
| --- | --- | --- |
| 問題→ステップ→終了の正常シーケンス | `tests/test_evaluator.py::test_evaluator_runs_problem_step_end` | ログ順序と完了判定を確認 |
| 無効ステップ検出 | `tests/test_evaluator.py::test_evaluator_records_mistake_for_invalid_step` | `status="mistake"` と `reason="invalid_step"` が記録される |
| 終了式不整合の検出 | `tests/test_evaluator.py::test_evaluator_records_mistake_for_end_mismatch` | 最終レコードが `status="mistake"` かつ `reason="final_result_mismatch"` になる |
| 計算ミスでもログ継続 | `tests/test_evaluator.py::test_evaluator_records_mistake_for_invalid_step` | 例外を投げずにログへ記録し、学習ログが最後まで残る |

## 2. ルール判別と曖昧度推定

| シナリオ | テスト | ポイント |
| --- | --- | --- |
| ルール一致時の `rule_id` 付与 | `tests/test_full_pipeline.py::test_full_reasoning_pipeline` | スタブ KnowledgeRegistry による rule_id 生成を確認 |
| 無効ステップ時の Fuzzy 判定ログ | `tests/test_evaluator.py::test_evaluator_logs_fuzzy_when_invalid_step` | FuzzyJudge の呼び出しと `phase=fuzzy` ログを確認 |
| 曖昧度スコア記録 | `tests/test_full_pipeline.py::test_full_reasoning_pipeline` | FuzzyJudge から返ったラベルがログに残ることを検証 |
| Fuzzy モジュール単体の計算 | `tests/test_fuzzy_engine.py` | 類似度計算とスコア組み立てのユニットテスト |
| 実データセットでの曖昧判定 | `tests/test_fuzzy_real_data.py` | `docs/data/fuzzy_samples.json` を用いて閾値設定と実シナリオ整合性を確認 |

## 3. 因果推論

| シナリオ | テスト | ポイント |
| --- | --- | --- |
| 因果グラフ基本操作 | `tests/test_causal_engine.py::{test_causal_graph_traversal_helpers,test_engine_ingest_creates_nodes_and_edges}` | ノード追加、親子関係、ルールノード接続を検証 |
| エラー原因推定と修正候補 | `tests/test_causal_engine.py::{test_why_error_returns_most_recent_invalid_step_first,test_suggest_fix_candidates_prioritizes_invalid_steps}` | `why_error`／`suggest_fix_candidates` のヒューリスティクス |
| カウンターファクト実行 | `tests/test_causal_engine.py::test_counterfactual_result_reports_changes` と `tests/test_full_pipeline.py::test_full_reasoning_pipeline` | 介入適用と Evaluator 再実行による結果比較 |
| Parser→Evaluator→Logger→CausalEngine の統合 | `tests/test_causal_integrations.py::test_causal_engine_builds_graph_from_learning_logger` | 学習ログから因果グラフを構築し、ルールノード・エラー原因を追跡 |

## 4. 総合シナリオ

`tests/test_full_pipeline.py::test_full_reasoning_pipeline` では以下を一度に検証します。

1. **ステップ判定**: 1 つ目のステップは正しく、2 つ目が `status="mistake"` でログ記録される。
2. **ルール判定 / Fuzzy**: 正しいステップに `rule_id` が付き、無効ステップで Fuzzy 判定が実行される。
3. **因果推論**: ログを CausalEngine に取り込み、`why_error`・`suggest_fix_candidates`・`counterfactual_result` が期待通りに動作する。

このテストプランをベースに、各シナリオに対応するテストの整備状況を定期的に確認することで、CausalScript の推論パイプライン全体が退行していないかを継続的に監視できます。
