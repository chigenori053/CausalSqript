# Development Report

## 概要
- CLI/Notebook から因果推論・カウンターファクトを体験できるデモフローを完成。
- `edu/examples/counterfactual_demo.mlang` & `--counterfactual` オプションで代表シナリオを実行可能。
- `edu.demo.edu_demo_runner` / `pro.demo_runner` を追加し、Edition別 CLI デモを簡単に実行可能。
- `.github/workflows/ci.yml` で pytest + CLI self-test + Notebook スモークテストを自動化。
- グラフ可視化ヘルパー (`graph_to_text/dot`) を公開し、`docs/demo/counterfactual_walkthrough.md` に手順を整理。
- FuzzyJudge/KnowledgeRegistry 向け実データ (`docs/data/fuzzy_samples.json`) を追加し、テスト (`tests/test_fuzzy_real_data.py`) で検証。
- README / Spec / Test Plan / Task Sheet を最新仕様に同期。

## 完了タスク
1. CORE-01: ルール/曖昧判定の本番データ化。
2. CORE-02: カウンターファクトシナリオ整備（DSL + CLI オプション + Eduデモランナー）。
3. CORE-03: 因果グラフ可視化を CLI/Notebook デモへ統合。
4. CORE-04: デモパッケージ（docs/demo/... + Edu/Pro ランナー）。
5. CORE-05: README / Spec / Test Plan 更新。
6. OPS-01: 文書同期ツール整備。
7. OPS-02: CI ワークフロー（pytest/CLI/Notebook）の自動化。

## 現在の状況
- pytest 51件すべて緑（CLI/Notebook 用の smoke test も CI へ導入済み）。
- CLI は実行結果 + `== Causal Analysis ==` + `== Counterfactual Simulation ==` を表示。
- Notebook では `run_causal_analysis` + `graph_to_text/dot` でグラフを可視化可能。
- 残タスク: OPS-02 (CI で pytest/lint/デモ self-test を常時監視)。

## 次のアクション
1. Pro 向け Notebook / API デモ (`pro/notebooks/pro_intro_causal.ipynb` など) を整備。
2. デモ資料（README/Docs）を外部公開形式へ整備し、Feedback 収集サイクルに備える。
