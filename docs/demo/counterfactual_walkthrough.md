# Counterfactual & Causal Analysis Walkthrough

`edu/examples/counterfactual_demo.mlang` は DSL v2.5 の代表的なサンプルで、以下のブロックをすべて含みます。

```text
meta / config / mode / prepare / problem / step (before/after/note) / end / counterfactual
```

1. 基本実行  
   `python main.py --file edu/examples/counterfactual_demo.mlang`  
   - `prepare` で宣言した `base_sum` と `scale_factor` がログに出力され、2つ目の step が `mistake` として因果解析に記録される。

2. CLI で介入を適用  
   `python main.py --file edu/examples/counterfactual_demo.mlang --counterfactual '{"phase": "step", "index": 2, "expression": "8 * factor"}'`  
   - `== Counterfactual Simulation ==` セクションで置換後の step/ end、`new_end_expr` を確認できる。

3. Demo Runner 経由で再生  
   `python -m edu.demo.edu_demo_runner counterfactual --with-counterfactual`  
   - `edu/config/edu_demo_config.yaml` のシナリオ定義からファイルと `counterfactual` ブロックを自動で解決する。

4. Notebook 連携  
```python
from core.causal.integration import run_causal_analysis
from core.causal.graph_utils import graph_to_text
records = logger.to_list()
engine, report = run_causal_analysis(records, include_graph=True)
print(graph_to_text(report["graph"]))
```

5. 補足：`docs/data/fuzzy_samples.json` を読み込み、`tests/test_fuzzy_real_data.py` と同じ手順で FuzzyJudge の実データ検証を再現可能。
