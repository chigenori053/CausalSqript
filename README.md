# COHERENT System 2.0

> **Intelligence in Phase.**
> Where Waves Become Logic.

COHERENTは、**光学干渉メモリ（Optical Holographic Memory）** と **アクション予測型推論（Action-Based Reasoning）** を融合させた次世代の論理推論システム（Reasoning LM）です。単なる「次のトークン」の予測ではなく、「次のアクション（思考のステップ）」を予測し、実行し、検証する自律的なループを実現します。

COHERENT is a next-generation **Reasoning LM** system that fuses **Optical Holographic Memory** with **Action-Based Reasoning**. Instead of merely predicting the "next token", it predicts, executes, and validates the "next action (reasoning step)", enabling an autonomous self-improving loop.

---

## Key Features / 主な特徴

### 1. Optical Holographic Memory (光学干渉メモリ)
脳の海馬のような長期的かつ連想的な記憶を実現します。
- **Resonance Recall**: 入力情報（波）と記憶（波）の干渉強度（Resonance）によって、最も関連性の高い過去の経験を瞬時に想起します。
- **Encoding**: テキスト、画像、音声を複素数テンソル（Holographic Tensor）にエンコードし、同一空間で扱います。

Acts like a hippocampus for long-term associative memory.
- **Resonance Recall**: Instantly recalls relevant past experiences based on the interference intensity (resonance) between input waves and stored waves.
- **Encoding**: Encodes text, images, and audio into complex-valued Holographic Tensors within a unified vector space.

### 2. Reasoning Agent (推論エージェント)
System 2（熟慮的思考）を担当するエージェントです。
- **Action-Based**: 思考を「ツールの使用」「ルールの適用」「検索」などの離散的な**Action**として出力します。
- **Self-Correction**: 実行結果（Execution Result）を観察し、自身の仮説を修正しながらゴールを目指します。

The agent responsible for System 2 (deliberate thinking).
- **Action-Based**: Outputs thoughts as discrete **Actions** (e.g., using a tool, applying a rule, searching memory).
- **Self-Correction**: Observes execution results and iteratively corrects its hypotheses to reach the goal.

### 3. Traceability & Learning (追跡可能性と学習)
- **Tracer**: すべての思考ステップ（State → Action → Result）をエピソードとして記録します。
- **Feedback Loop**: 成功したエピソードは光学メモリにフィードバックされ、「直感（System 1）」として定着します。

- **Tracer**: Records every reasoning step (State → Action → Result) as an episode.
- **Feedback Loop**: Successful episodes are fed back into the optical memory, solidifying them as "intuition" (System 1).

---

## Architecture / アーキテクチャ

| Component | Responsibility (役割) |
|-----------|-----------------------|
| **Layer A: Interface** | **Semantic Parser**: 自然言語を解析し、構造化されたタスク（Semantic IR）に変換します。<br>(Parses natural language into structured Semantic IR.) |
| **Layer B: Core** | **Action Executor**: アクションを実行し、システムの状態（State）を更新します。<br>**Tracer**: 実行ログを記録します。<br>(Executes actions, updates state, and logs episodes.) |
| **Layer C: Physics** | **Optical Engine**: 複素数演算による記憶の想起と干渉シミュレーション。<br>(Simulates memory recall and interference using complex arithmetic.) |

---

## Installation / インストール

**Requirements**: Python 3.12+

```bash
# Clone repository
git clone https://github.com/your-org/COHERENT.git
cd COHERENT

# Setup environment using uv
uv init
uv python install 3.12
uv sync
```

---

## Usage / 使用方法

### 1. Streamlit UI (Recommended)
リアルタイムで推論プロセスを可視化できるインタラクティブなUIです。

Interactive UI to visualize the reasoning process in real-time.

```bash
uv run streamlit run ui/app.py
```
- **Agent Solver**: 自然言語で問題を解かせることができます。（例: "Solve x^2 - 4 = 0", "Factorize x^2 + 5x + 6"）
- スクリーンショット通りに思考のステップ（Action）が表示されます。

- **Agent Solver**: Solve problems using natural language.
- Visualize reasoning steps (Actions) as seen in screenshots.

### 2. CLI / Script
スクリプトから直接エージェントを呼び出すことも可能です。

You can also invoke the agent directly from scripts.

```python
from ui.app import get_system
from coherent.core.state import State
from coherent.core.tracer import Tracer

# Initialize System
system = get_system()
agent = system["agent"]
executor = system["executor"]
tracer = Tracer()

# Start Episode
expression = "x^2 - 4 = 0"
state = State.from_string(expression)
tracer.start_episode(expression)

# Reasoning Loop
action = agent.act(state)
result = executor.execute(action, state)
tracer.log_step(state, action, result)

print(f"Action: {action.name}, Valid: {result['valid']}")
```

---

## Contribution / 貢献

COHERENTは現在、**Phase 2 (Coding Agent Integration)** に向けて開発進行中です。

COHERENT is currently under development towards **Phase 2 (Coding Agent Integration)**.

- **Tests**: `uv run pytest`
- **Docs**: `docs/` ディレクトリ配下の仕様書を参照してください。(See `docs/` for specifications.)
