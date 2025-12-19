# COHERENT システムアーキテクチャ概要 (COHERENT System Architecture Overview)

## 1. はじめに (Introduction)

COHERENTは、次世代の数学学習支援および自動推論システムです。「Recall-First（想起優先）」アーキテクチャを採用し、人間の認知プロセス（System 1の直感的想起とSystem 2の論理的推論のハイブリッド）を模倣することで、効率的かつ高度な問題解決能力を実現しています。

本ドキュメントでは、COHERENTシステムの全体像、各レイヤーの構成、および主要なアルゴリズムについて詳述します。

## 2. システムアーキテクチャ図 (System Architecture Diagram)

```mermaid
graph TD
    User[ユーザー / Frontend] --> Interface[Interface Layer (API/CLI)]
    Interface --> Orchestrator[Orchestration Layer (CoreRuntime)]
    
    subgraph "Orchestration Layer"
        Orchestrator --> Validation[ValidationEngine]
        Orchestrator --> Hint[HintEngine]
        Orchestrator --> Decision[Fuzzy Judge / DecisionEngine]
    end

    subgraph "Reasoning Layer (System 2)"
        ReasoningAgent[ReasoningAgent] --> Integrator[Multimodal Integrator]
        ReasoningAgent --> Generator[Hypothesis Generator]
        ReasoningAgent --> Simulator[Lookahead Simulator]
        Simulator --> GoalScanner[Goal Scanner]
    end
    
    subgraph "Memory Layer (System 1)"
        ExpMemory[Experience Memory (Evolutionary Graph)]
        OpticalStore[Optical Holographic Store]
        TensorEngine[Tensor Logic Engine]
    end

    subgraph "Computational Engine Layer"
        SymEngine[SymbolicEngine (SymPy)]
        CalcEngine[Calculus Module]
        LAEngine[Linear Algebra Module]
        StatsEngine[Statistics Module]
    end

    Orchestrator --> ReasoningAgent
    ReasoningAgent --> OpticalStore
    ReasoningAgent --> SymEngine
    ReasoningAgent --> TensorEngine
    Validation --> SymEngine
    Validation --> Decision
    OpticalStore -- Resonance/Recall --> ReasoningAgent
```

## 3. レイヤー構成と詳細 (Layer Descriptions)

### 3.1 Orchestration Layer (CoreRuntime)
システムの中心となる制御層です。ユーザーからの入力（数式、ステップ）を受け取り、検証、ヒント生成、および学習ログの記録を統括します。

*   **CoreRuntime (`core_runtime.py`)**: 全体のワークフローを管理します。`ComputationEngine`や`ValidationEngine`を統合し、拡張機能（Calculus, Linear Algebra）を必要に応じてロードします。
*   **ValidationEngine (`validation_engine.py`)**: ユーザーの入力したステップの正当性を検証します。厳密な数式等価性チェックに加え、曖昧さを含む判断（Fuzzy Judge）を行います。
*   **DecisionEngine (`decision_theory.py`)**: ベイズ決定理論に基づき、不確実性がある状況下での最適なアクション（Accept, Review, Reject）を決定します。

### 3.2 Reasoning Layer (System 2)
自律的に問題を解決するための推論層です。

*   **ReasoningAgent (`reasoning/agent.py`)**: システムの「脳」にあたるエージェントです。
    *   **Recall-First戦略**: まず過去の経験（Memory）からの想起を試み、解決策が見つからない場合のみ計算（Computation）による探索を行います。
*   **HypothesisGenerator (`generator.py`)**: 次のステップの候補（仮説）を生成します。
*   **LookaheadSimulator (`simulator.py`)**: 生成された仮説を数ステップ先までシミュレーションし、ゴールに近づくかどうかを評価します。

### 3.3 Memory Layer (System 1 - Holographic Memory)
高速な想起とパターンマッチングを担当する記憶層です。物理的な光学現象をシミュレーションした「光学ホログラフィックメモリ」を採用しています。

*   **OpticalFrequencyStore (`memory/optical_store.py`)**: ベクトルデータを複素振幅（ホログラム）として保存します。ユークリッド距離ではなく、位相共鳴（Resonance）によって類似度を判定します。
*   **OpticalInterferenceEngine (`optical/layer.py`)**: 光の干渉（Interference）と回折をPyTorch上でシミュレーションする計算コアです。

### 3.4 Engine Layer
数式処理や特定領域の計算を行うバックエンド層です。

*   **SymbolicEngine (`symbolic_engine.py`)**: SymPyをラップした数式処理エンジン。代数的な等価性判定や簡約化を行います。
*   **CalculusEngine (`calculus_engine.py`)**: 微積分（微分、積分）専用の処理モジュール。
*   **TensorEngine (`tensor/engine.py`)**: ニューロシンボリックな推論をサポートするためのテンソル演算エンジン。

---

## 4. 主要アルゴリズムと数式 (Core Algorithms & Mathematics)

### 4.1 光学ホログラフィックメモリ (Optical Holographic Memory)

ベクトルデータを光の波（複素数）としてエンコードし、干渉パターンとして保存・検索します。

#### A. 信号エンコーディング (Signal Encoding)
実数値ベクトル $\mathbf{v} \in \mathbb{R}^d$ を、複素単位球面上の点（振幅・位相）にマッピングします。現在は振幅変調（Amplitude Modulation）を使用しています。

$$
\mathbf{z} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} \quad (\text{Converting to Complex Float})
$$

※将来的には位相エンコーディング $\mathbf{z} = e^{i \mathbf{v}}$ への拡張が予定されています。

#### B. 共鳴と検索 (Resonance & Retrieval)
クエリ信号 $\mathbf{q}$ とメモリ内の各スロット $\mathbf{m}_i$ との間の共鳴強度（Resonance Intensity） $I$ を計算します。これは複素ドット積（エルミート内積）の絶対値に相当します。

$$
I_i = |\mathbf{q} \cdot \mathbf{m}_i^*| = \left| \sum_{k=1}^{d} q_k \cdot \overline{m}_{ik} \right|
$$

ここで、$I_i$ が高いほど、クエリと記憶が強く共鳴（類似）していることを示します。

#### C. 曖昧性 (Ambiguity)
システム全体の不確実性をエントロピー的な指標で評価します。

### 4.2 決定理論的検証 (Decision Theoretic Validation)

数式の等価性が微妙な場合（例：数値解法による近似等）、期待効用最大化（Expected Utility Maximization）に基づいて判定を行います。

#### 効用関数 (Utility Function)
アクション $a \in \{Accept, Review, Reject\}$ と 真の状態 $s \in \{Match, Mismatch\}$ 
に対し、効用 $U(a, s)$ を定義します（例：正解をRejectするコストは大きい）。

#### 期待効用 (Expected Utility)
確率 $P(Match)$ が与えられたときのアクション $a$ の期待効用：

$$
EU(a) = P(Match) \cdot U(a, Match) + P(Mismatch) \cdot U(a, Mismatch)
$$

システムは $a^* = \arg\max_a EU(a)$ となるアクションを選択します。

### 4.3 推論エージェント (Reasoning Agent Logic)

#### 想起優先ループ (Recall-First Loop)
1.  **Perception**: 入力 $x$ を一般化抽象構文木（AST Generalization） $G(x)$ に変換。
2.  **Recall**: 光学メモリ $M$ から $G(x)$ に共鳴する経験 $e$ を検索。
    $$ e^* = \text{query}(M, G(x)) $$
    もし共鳴強度 $I(e^*) > \theta$ ならば、記憶されたルール $R_e$ を適用して解とする。
3.  **Compute**: 想起に失敗した場合、探索アルゴリズム（BFS/MCTS等）を用いて計算的に解を導出する。
