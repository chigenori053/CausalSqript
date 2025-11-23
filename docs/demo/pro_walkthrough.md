# Pro Edition Walkthrough

1. Polynomial counterfactualサンプル  
   `python -m pro.cli --file pro/examples/polynomial_analysis.mlang`
   - `prepare` で `observed_x` を宣言し、`counterfactual` ブロックで `observed_x: 4` を仮定できる。

2. Demo Runner（将来の pro 専用 CLI）: `python -m pro.demo_runner counterfactual`  
   - 現状はサンプル置き換え中のため、`pro/examples/polynomial_analysis.mlang` を直接 CLI で参照するのが最短。

3. Notebook（今後追加予定）: `pro/notebooks/pro_intro_causal.ipynb`  
   - `core.causal.integration.run_causal_analysis` と `pro/examples/polynomial_analysis.mlang` を組み合わせ、研究向け因果ログを可視化する。
