"""
Coherent Streamlit UI - Integrated System Tester
"""

import streamlit as st
import sys
import os
import json
import time
import io
from contextlib import redirect_stdout
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core Imports
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine, HintPersona
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.latex_formatter import LaTeXFormatter
from coherent.engine.parser import Parser
from coherent.engine.evaluator import Evaluator
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.fuzzy.judge import FuzzyJudge
from coherent.engine.fuzzy.encoder import ExpressionEncoder
from coherent.engine.fuzzy.metric import SimilarityMetric
from coherent.engine.unit_engine import get_common_units
from coherent.engine.decision_theory import DecisionConfig
from coherent.engine.knowledge_registry import KnowledgeRegistry

# Reasoning & Memory Imports
from coherent.engine.tensor.engine import TensorLogicEngine
from coherent.engine.tensor.converter import TensorConverter
from coherent.engine.tensor.embeddings import EmbeddingRegistry
from coherent.engine.reasoning.agent import ReasoningAgent

# Language Processing
from coherent.engine.language.semantic_parser import RuleBasedSemanticParser
from coherent.engine.language.semantic_types import TaskType

import torch
import numpy as np
import matplotlib.pyplot as plt
import graphviz

# Page Config
st.set_page_config(
    page_title="Coherent System 2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = []

# --- System Initialization ---

@st.cache_resource
def get_system():
    """Initializes the full Coherent System (Runtime + Agent + Memory)."""
    
    # 1. Base Engines
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    
    # 2. Knowledge Registry
    knowledge_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "coherent", "engine", "knowledge")))
    knowledge_path.mkdir(parents=True, exist_ok=True)
    knowledge_registry = KnowledgeRegistry(knowledge_path, sym_engine)
    
    # 3. Validation & Fuzzy Judge (Default Config)
    decision_config = DecisionConfig(strategy="balanced")
    encoder = ExpressionEncoder()
    metric = SimilarityMetric()
    fuzzy_judge = FuzzyJudge(encoder, metric, decision_config=decision_config, symbolic_engine=sym_engine)
    decision_engine = fuzzy_judge.decision_engine
    
    val_engine = ValidationEngine(
        comp_engine, 
        fuzzy_judge=fuzzy_judge,
        decision_engine=decision_engine,
        knowledge_registry=knowledge_registry
    )
    
    # 4. Hint Engine
    hint_engine = HintEngine(comp_engine)
    
    # 5. Core Runtime
    logger = LearningLogger() 
    runtime = CoreRuntime(
        comp_engine, 
        val_engine, 
        hint_engine, 
        learning_logger=logger,
        knowledge_registry=knowledge_registry,
        decision_config=decision_config
    )
    
    # Inject units
    for name, unit in get_common_units().items():
        comp_engine.bind(name, unit)

    # 6. Tensor/Neuro-Symbolic Components
    embedding_registry = EmbeddingRegistry()
    tensor_converter = TensorConverter(embedding_registry)
    tensor_engine = TensorLogicEngine(vocab_size=1000, embedding_dim=64)
    
    # 7. Reasoning Agent (Recall-First)
    agent = ReasoningAgent(
        runtime,
        tensor_engine=tensor_engine,
        tensor_converter=tensor_converter
    )
    
    # 8. Utilities
    from coherent.engine.classifier import ExpressionClassifier
    classifier = ExpressionClassifier(sym_engine)
    formatter = LaTeXFormatter(sym_engine, classifier)
    
    # 9. Language Parser
    semantic_parser = RuleBasedSemanticParser()

    return {
        "runtime": runtime,
        "agent": agent,
        "comp_engine": comp_engine,
        "knowledge_registry": knowledge_registry,
        "formatter": formatter,
        "tensor_engine": tensor_engine,
        "val_engine": val_engine,
        "hint_engine": hint_engine,
        "sym_engine": sym_engine,
        "semantic_parser": semantic_parser
    }

# Load System
system = get_system()
runtime = system["runtime"]
agent = system["agent"]
formatter = system["formatter"]
parser = system["semantic_parser"]

# --- Helper Functions ---

def render_optical_memory(agent):
    """Visualizes the Optical Memory state as a heatmap."""
    try:
        optical_mem = agent.trainer.model.optical_memory 
        
        if optical_mem is None:
            return None
            
        energy = torch.abs(optical_mem).detach().cpu().numpy()
        
        display_rows = min(50, energy.shape[0])
        display_data = energy[:display_rows, :]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        cax = ax.imshow(display_data, aspect='auto', cmap='inferno', interpolation='nearest')
        ax.set_title(f"Optical Memory State (Top {display_rows} Slots)")
        ax.set_xlabel("Vector Dimension")
        ax.set_ylabel("Memory Slot")
        fig.colorbar(cax, orientation='vertical')
        
        return fig
    except Exception as e:
        st.error(f"Visualization Error: {e}")
        return None

# --- UI Layout ---

st.title("🌌 Coherent System 2.0")
st.markdown("Recall-First Reasoning & Optical Holographic Memory Interface")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    strategy_name = st.selectbox(
        "Decision Strategy", 
        ["balanced", "strict", "encouraging"],
        index=0
    )
    
    runtime.hint_persona = st.selectbox("Hint Persona", ["balanced", "sparta", "support"])
    runtime.validation_engine.fuzzy_judge.decision_engine.config.strategy = strategy_name
    
    st.divider()
    st.markdown("**System Status**")
    st.markdown(f"🧠 **Knowledge Rules**: {len(system['knowledge_registry'].nodes)}")
    mem_cap = agent.trainer.model.memory_capacity
    st.markdown(f"💡 **Optical Capacity**: {mem_cap}")

# Tabs
tab_solver, tab_tester, tab_train = st.tabs(["🧩 Agent Solver", "✅ Script Tester", "🧠 Optical Training"])

# --- TAB 1: Agent Solver (Step-by-Step) ---
with tab_solver:
    st.header("Step-by-Step Reasoning")
    st.markdown("Solve problems using the **Reasoning Agent** (System 2) backed by **Optical Memory** (System 1).")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        user_input = st.text_input("Enter Problem (Natural Language)", value="Solve (x - 2y)^2")
        solve_btn = st.button("Thinking Step", type="primary")
        
        reset_btn = st.button("Reset Problem")
        if reset_btn:
             st.session_state.agent_memory = []
             st.rerun()

    with col2:
        # NLP Understanding Display
        current_display = ""
        try:
            if user_input:
                ir = parser.parse(user_input)
                st.caption(f"🤖 **Detected Intent**: `{ir.task.name}` | **Domain**: `{ir.math_domain.name}`")
                
                extracted_math = ir.inputs[0].value if ir.inputs else ""
                
                if not st.session_state.agent_memory:
                     current_display = extracted_math if extracted_math else "(Waiting for valid input...)"
                else:
                     current_display = st.session_state.agent_memory[-1]['state']
        except Exception as e:
            st.error(f"Semantic Parsing Error: {e}")
            current_display = "Error"
        
        st.info(f"Current State: `{current_display}`")

    if solve_btn:
        start_state = None
        
        if not st.session_state.agent_memory:
             ir = parser.parse(user_input)
             if ir.task != TaskType.SOLVE:
                 st.warning(f"Currently only SOLVE task is supported. Detected: {ir.task}")
             elif not ir.inputs:
                 st.warning("Could not extract a mathematical expression.")
             else:
                 start_state = ir.inputs[0].value
        else:
             start_state = st.session_state.agent_memory[-1]['state']
             
        if start_state:
            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    hypothesis = agent.think(start_state)
                except Exception as e:
                    st.error(f"Agent Error: {e}")
                    hypothesis = None
            
            thought_log = f.getvalue()
            
            if hypothesis:
                step_record = {
                    "step": len(st.session_state.agent_memory) + 1,
                    "input": start_state,
                    "state": hypothesis.next_expr,
                    "rule": hypothesis.rule_id,
                    "score": hypothesis.score,
                    "explanation": hypothesis.explanation,
                    "log": thought_log
                }
                st.session_state.agent_memory.append(step_record)
            else:
                st.warning("Agent could not find a confident next step.")
                st.text(thought_log)

    if st.session_state.agent_memory:
        st.divider()
        st.subheader("Solution Path")
        
        for i, step in enumerate(st.session_state.agent_memory):
            with st.container():
                st.markdown(f"#### Step {i+1}")
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.latex(formatter.format_expression(step['state']))
                    st.caption(step['explanation'])
                with c2:
                    st.markdown(f"**Rule**: `{step['rule']}`")
                    st.markdown(f"**Confidence**: `{step['score']:.2f}`")
                    with st.expander("Machine Thoughts"):
                        st.code(step['log'], language="text")
                st.divider()

    with st.expander("👁️ Optical Memory State", expanded=True):
        fig = render_optical_memory(agent)
        if fig:
            st.pyplot(fig)


# --- TAB 2: Script Tester (Legacy/Batch) ---
with tab_tester:
    st.header("Validation Script Tester")
    
    default_script = """problem: (x - 2y)^2
step: (x - 2y)(x - 2y)
step: x(x - 2y) - 2y(x - 2y)
step: x^2 - 2xy - 2yx + 4y^2
step: x^2 - 4xy + 4y^2
end: x^2 - 4xy + 4y^2"""
    
    script_input = st.text_area("Validation Script", value=default_script, height=300)
    run_script = st.button("Validate Script", type="primary")
    
    if run_script:
        logger = system['runtime'].learning_logger
        logger.records = [] 
        
        try:
            parser_v = Parser(script_input)
            program = parser_v.parse()
            evaluator = Evaluator(program, system['runtime'], learning_logger=logger)
            success = evaluator.run()
            
            logs = logger.to_list()
            
            for record in logs:
                if record['phase'] == 'step':
                    status_icon = "✅" if record.get('status') == 'ok' else "❌"
                    st.markdown(f"{status_icon} **{record.get('expression')}**")
                    if record.get('status') != 'ok':
                        st.error(f"Status: {record.get('status')}")
                        if 'hint' in record.get('meta', {}):
                            st.info(f"Hint: {record['meta']['hint'].get('message')}")
                    
        except Exception as e:
            st.error(f"Error: {e}")


# --- TAB 3: Optical Training ---
with tab_train:
    st.header("Optical Layer Training")
    st.markdown("Train the agent's intuition (Optical Memory) from successful past experiences.")

    if st.button("Extract & Train from Logs"):
        logger = system['runtime'].learning_logger
        logs = logger.to_list()
        
        training_samples = []
        rule_ids = agent.generator.rule_ids
        
        for record in logs:
            if record.get('status') == 'ok' and record.get('phase') == 'step':
                expr = record.get('expression')
                meta = record.get('meta', {})
                rule_node = meta.get('rule', None) 
                
                if expr and rule_node:
                    rid = rule_node.get('id')
                    if rid and rid in rule_ids:
                        target_idx = rule_ids.index(rid)
                        training_samples.append((expr, target_idx))
        
        if training_samples:
            st.info(f"Found {len(training_samples)} samples.")
            with st.spinner("Training Optical Layer..."):
                loss = agent.retrain(training_samples, epochs=10)
            st.success(f"Training Complete. Loss: {loss:.4f}")
        else:
            st.warning("No valid training samples found in current session logs. Run a valid Validation Script first.")
