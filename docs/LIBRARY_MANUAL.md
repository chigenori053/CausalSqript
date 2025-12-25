# COHERENT Library User Manual

Version: 1.0.0
Date: 2025-12-25

## 概要 (Overview)

COHERENTは、光干渉原理を応用した推論検証・学習システムです。
本ライブラリは、Python環境において「計算過程の検証」「ホログラフィックメモリへの学習・想起」「システム状態の監視」機能を提供します。
フロントエンドアプリケーション（Web APIサーバー等）のバックエンドコアとして組み込むことを想定しています。

## インストール (Installation)

```bash
# プロジェクトルートにて
pip install .
```

推奨Pythonバージョン: 3.12+

## クイックスタート (Quick Start)

### 1. コアランタイムの初期化

```python
from coherent import CoreRuntime
from coherent.core.computation_engine import ComputationEngine
from coherent.core.validation_engine import ValidationEngine
from coherent.core.hint_engine import HintEngine
from coherent.core.symbolic_engine import SymbolicEngine

# エンジンの準備
symbolic = SymbolicEngine()
comp_engine = ComputationEngine(symbolic)
val_engine = ValidationEngine(comp_engine) # コンピュテーションエンジンを注入
hint_engine = HintEngine(comp_engine)

# ランタイムの生成
runtime = CoreRuntime(
    computation_engine=comp_engine,
    validation_engine=val_engine,
    hint_engine=hint_engine
)
```

### 2. 計算ステップの検証 (Step Verification)

```python
# 問題の設定
problem_expr = "x**2 + 2*x + 1"
runtime.set(problem_expr)

# ユーザー入力ステップの検証
step_expr = "(x + 1)**2"
result = runtime.check_step(step_expr)

print(f"Valid: {result['valid']}")
print(f"Status: {result['details'].get('status')}")
```

## ステータス監視機能 (Status Observation System)

COHERENTは、システムの健全性と学習の安全性を担保するため、厳格なステータス管理機能を備えています。

### ステータス・モード一覧

| State | 学習 (Learning) | メモリ書込 | Sandbox変換 | 説明 |
| :--- | :---: | :---: | :---: | :--- |
| **NORMAL** | ON | ON | OFF | 通常稼働中 |
| **DEGRADED** | OFF | OFF | ON | 異常検知、読み取り専用 |
| **ISOLATION** | OFF | OFF | ON | 完全隔離、診断モード |

### ステータスの取得

```python
from coherent import StatusManager

manager = StatusManager()
status = manager.get_status()

print(f"System ID: {status.system_id}")
print(f"State: {status.mode.state}")
print(f"Learning Enabled: {status.mode.policy.learning_enabled}")
```

### 異常時の挙動

システム内部で整合性エラー（NaN検出、次元不一致など）が発生すると、自動的に `DEGRADED` モードへ遷移し、以降の学習・メモリ書き込みが物理的にブロックされます。

## フロントエンド実装者への注意点

1.  **ステータスAPIの公開**: `StatusManager.get_status()` の結果をそのまま `/api/status` 等で返却することを強く推奨します。
2.  **イベントログ**: `StatusManager` は状態遷移時にイベントを発行します。監査ログとして記録してください。
3.  **非同期処理**: 計算・検証処理の一部（光学推論など）は並列実行されますが、Python API自体は現在同期的です。APIサーバーで使用する場合は `fastapi` 等でラップしてください。
