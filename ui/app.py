"""
CausalScript Streamlit UI - Test Interface
"""

import streamlit as st
import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.symbolic_engine import SymbolicEngine
from core.computation_engine import ComputationEngine
from core.validation_engine import ValidationEngine
from core.hint_engine import HintEngine
from core.core_runtime import CoreRuntime
from core.latex_formatter import LaTeXFormatter
from core.parser import Parser
from core.evaluator import Evaluator
from core.learning_logger import LearningLogger
from core.errors import CausalScriptError
from core.fuzzy.judge import FuzzyJudge
from core.fuzzy.encoder import ExpressionEncoder
from core.fuzzy.metric import SimilarityMetric
from core.unit_engine import get_common_units
from core.decision_theory import DecisionConfig
from core.hint_engine import HintPersona
from core.knowledge_registry import KnowledgeRegistry

# Page Config
st.set_page_config(
    page_title="CausalScript Logic Tester",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "scenario_logs" not in st.session_state:
    st.session_state.scenario_logs = []

# Initialize Engines (Cached)
@st.cache_resource
def get_engines():
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    
    # Initialize Fuzzy Judge
    encoder = ExpressionEncoder()
    metric = SimilarityMetric()
    fuzzy_judge = FuzzyJudge(encoder, metric)
    
    val_engine = ValidationEngine(comp_engine, fuzzy_judge=fuzzy_judge)
    hint_engine = HintEngine(comp_engine)
    from core.classifier import ExpressionClassifier
    classifier = ExpressionClassifier(sym_engine)
    formatter = LaTeXFormatter(sym_engine, classifier)
    
    # Initialize Knowledge Registry
    # Point to the knowledge root directory
    knowledge_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "knowledge")))
    # Create directory if it doesn't exist to avoid errors
    knowledge_path.mkdir(parents=True, exist_ok=True)
    knowledge_registry = KnowledgeRegistry(knowledge_path, sym_engine)
    
    # Inject common units into context
    for name, unit in get_common_units().items():
        comp_engine.bind(name, unit)
        
    return comp_engine, val_engine, hint_engine, formatter, knowledge_registry

comp_engine, val_engine, hint_engine, formatter, knowledge_registry = get_engines()

# --- Helper Functions ---

def render_test_report(step_index, step_data):
    """Renders a detailed test report for a single step."""
    status = step_data['status']
    is_ok = status == "ok"
    
    # Color coding
    if is_ok:
        border_color = "#28a745" # Green
        bg_color = "rgba(40, 167, 69, 0.1)"
    else:
        border_color = "#dc3545" # Red
        bg_color = "rgba(220, 53, 69, 0.1)"
        
    with st.container():
        st.markdown(f"""
        <div style="border-left: 5px solid {border_color}; padding-left: 15px; margin-bottom: 20px; background-color: {bg_color}; padding: 10px; border-radius: 5px;">
            <h4 style="margin: 0;">Step {step_index}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. Formula Display
        f_cols = st.columns(2)
        with f_cols[0]:
            st.caption("Input Formula")
            st.latex(step_data['latex'])
        with f_cols[1]:
            st.caption("Evaluated Formula")
            if 'evaluated' in step_data['analysis']:
                st.latex(step_data['analysis']['evaluated'])
            else:
                st.markdown("*Not available*")
        
        # 2. Analysis Grid
        cols = st.columns(3)
        
        # Col 1: Causal / Logic Analysis
        with cols[0]:
            st.markdown("**🧠 Causal Inference**")
            analysis = step_data.get('analysis', {})
            
            # Rule Info
            if 'rule' in analysis:
                rule = analysis['rule']
                st.success(f"**Rule Match**: `{rule.get('id', 'N/A')}`")
                st.caption(rule.get('description', ''))
            else:
                st.markdown("Rule Match: *None*")
                
            # Fuzzy Info
            if 'fuzzy_score' in analysis:
                score = analysis['fuzzy_score']
                label = analysis.get('fuzzy_label', 'N/A')
                st.markdown(f"**Fuzzy Score**: `{score:.3f}`")
                st.progress(score)
                st.caption(f"Label: {label}")
                
        # Col 2: Decision Theory
        with cols[1]:
            st.markdown("**⚖️ Decision Engine**")
            
            if 'decision_action' in analysis:
                action = analysis['decision_action']
                utility = analysis.get('decision_utility', 0.0)
                
                st.info(f"**Action**: `{action}`")
                st.markdown(f"**Utility**: `{utility:.2f}`")
                
                if 'decision_utils' in analysis:
                    with st.expander("Utility Breakdown"):
                        st.json(analysis['decision_utils'])
            else:
                st.markdown("*No decision data available*")

        # Col 3: Hint Generation
        with cols[2]:
            st.markdown("**💡 Hint System**")
            
            hint_data = step_data.get('hint_data')
            if hint_data:
                st.warning(f"**Message**: {hint_data.get('message')}")
                st.markdown(f"**Type**: `{hint_data.get('type')}`")
                
                details = hint_data.get('details', {})
                if details:
                    st.caption(f"Selection Utility: {details.get('selection_utility', 'N/A')}")
                    st.caption(f"Persona: {details.get('persona', 'N/A')}")
            elif not is_ok:
                st.error("No hint generated.")
            else:
                st.markdown("*Step Valid - No Hint Needed*")

# --- Layout ---

st.title("🧪 CausalScript Logic Tester")
st.markdown("""
This interface is designed to test:
1. **Formula Recognition**: Problem -> Step -> End flow.
2. **Step Evaluation**: Validity of each transformation.
3. **Causal Inference**: Rule matching, Decision Theory application, and Adaptive Hinting.
""")

st.divider()

# Configuration Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Decision Theory")
    strategy = st.selectbox(
        "Strategy", 
        ["balanced", "strict", "encouraging"],
        help="Defines the utility matrix for the Decision Engine."
    )
    
    st.subheader("Hint System")
    persona = st.selectbox(
        "Persona", 
        ["balanced", "sparta", "support"],
        help="Defines the personality and utility function for Hint Selection."
    )
    
    st.divider()
    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.logs = []
        st.rerun()

import graphviz

# ... (existing imports)

# --- Helper Functions ---

def generate_scenarios(variables):
    """Generates simple test scenarios based on variables."""
    scenarios = {}
    # Default test values
    test_values = [1, 2, 0, -1]
    
    for i, val in enumerate(test_values):
        scenario_name = f"Scenario {i+1}"
        context = {}
        for var in variables:
            context[str(var)] = val # Simple strategy: all vars get same value
            # We could vary them, but this is a start
        scenarios[scenario_name] = context
    return scenarios

def render_test_report(step_index, step_data):
    # ... (existing render_test_report code)
    pass # I will not replace this function, just keeping the structure for context if needed, 
         # but actually I need to replace the MAIN LOGIC below.

# ... (existing code)

# Main Input Area
col_input, col_results = st.columns([1, 1])

with col_input:
    st.subheader("📝 Input Script")
    
    with st.form("test_form"):
        st.caption("Enter the full script below using `problem:`, `step:`, and `end:` keywords.")
        
        default_script = """problem: (x - 2y)^2
step: (x - 2y)(x - 2y)
step: x(x - 2y) - 2y(x - 2y)
step: x^2 - 2xy - 2yx + 4y^2
step: x^2 - 4xy + 4y^2
end: x^2 - 4xy + 4y^2"""
        
        user_input = st.text_area("Script", height=400, value=default_script, help="Type your CausalScript here.")
        
        submitted = st.form_submit_button("Run Test", type="primary")
        
        if submitted:
            # Reset logs
            st.session_state.logs = []
            
            # Initialize Runtime
            logger = LearningLogger()
            runtime = CoreRuntime(
                comp_engine, 
                val_engine, 
                hint_engine, 
                learning_logger=logger,
                knowledge_registry=knowledge_registry,
                decision_config=DecisionConfig(strategy=strategy),
                hint_persona=persona
            )
            
            try:
                # Parse and Evaluate
                parser = Parser(user_input)
                program = parser.parse()
                evaluator = Evaluator(program, runtime, learning_logger=logger)
                success = evaluator.run()
                
                # Process Logs into History
                logs = logger.to_list()
                st.session_state.logs = logs # Store raw logs
                
                history_entry = {"steps": []}
                
                # Graphviz Visualization Data
                dot = graphviz.Digraph(comment='Calculation Tree')
                dot.attr(rankdir='TB')
                
                previous_step_id = None
                
                for i, record in enumerate(logs):
                    if record['phase'] in ['problem', 'step', 'end']:
                        current_expr = str(record['expression']) if record['expression'] else ""
                        meta = record.get('meta', {})
                        
                        # 1. Prepare Step Data for Report
                        step_data = {
                            "phase": record['phase'],
                            "latex": formatter.format_expression(current_expr) if current_expr else "",
                            "status": record.get('status', 'ok'),
                            "analysis": {},
                            "hint_data": None
                        }
                        
                        # ... (Extract Analysis logic same as before)
                        if 'rule' in meta:
                            step_data['analysis']['rule'] = meta['rule']
                        if 'fuzzy_score' in meta:
                            step_data['analysis']['fuzzy_score'] = meta['fuzzy_score']
                            step_data['analysis']['fuzzy_label'] = meta.get('fuzzy_label')
                        if 'evaluated' in meta:
                            step_data['analysis']['evaluated'] = formatter.format_expression(str(meta['evaluated']))
                        if 'decision_action' in meta:
                            step_data['analysis']['decision_action'] = meta['decision_action']
                            step_data['analysis']['decision_utility'] = meta.get('decision_utility')
                            step_data['analysis']['decision_utils'] = meta.get('decision_utils')
                        if 'hint' in meta:
                            step_data['hint_data'] = meta['hint']
                            
                        history_entry["steps"].append(step_data)
                        
                        # 2. Graphviz Node Creation
                        step_id = f"step_{i}"
                        label = f"{record['phase'].upper()}\n{current_expr}"
                        if record.get('status') != 'ok':
                            dot.node(step_id, label, shape='box', style='filled', fillcolor='#ffcccc')
                        else:
                            dot.node(step_id, label, shape='box', style='filled', fillcolor='#e6f3ff')
                            
                        if previous_step_id:
                            # Edge from previous step
                            edge_label = meta.get('rule', {}).get('id', '')
                            dot.edge(previous_step_id, step_id, label=edge_label)
                        
                        previous_step_id = step_id
                        
                        # 3. Parallel Verification (Scenario Evaluation)
                        try:
                            # Extract variables
                            # We use internal sympy object to get free symbols
                            internal_expr = comp_engine.symbolic_engine.to_internal(current_expr)
                            if hasattr(internal_expr, 'free_symbols') and internal_expr.free_symbols:
                                variables = internal_expr.free_symbols
                                scenarios = generate_scenarios(variables)
                                
                                # Run parallel evaluation
                                results = comp_engine.evaluate_in_scenarios(current_expr, scenarios)
                                
                                # Add scenario nodes
                                for s_name, s_result in results.items():
                                    s_id = f"{step_id}_{s_name}"
                                    # Format result
                                    res_str = str(s_result)
                                    if isinstance(s_result, float):
                                        res_str = f"{s_result:.2f}"
                                        
                                    dot.node(s_id, f"{s_name}\n{res_str}", shape='ellipse', style='dashed', fontsize='10')
                                    dot.edge(step_id, s_id, style='dotted', arrowhead='none')
                                    
                        except Exception as e:
                            # If symbol extraction or evaluation fails (e.g. syntax error), skip scenarios
                            pass

                st.session_state.history = [history_entry]
                st.session_state.graph = dot # Store graph
                
            except Exception as e:
                st.error(f"System Error: {e}")

with col_results:
    st.subheader("📊 Test Report")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Steps", "Visualization", "Raw Logs"])
    
    with tab1:
        if st.session_state.history:
            entry = st.session_state.history[-1]
            steps = entry.get("steps", [])
            if not steps:
                st.info("No steps recorded.")
            else:
                for i, step in enumerate(steps):
                    render_test_report(i + 1, step)
        else:
            st.info("Run a test to see results.")
            
    with tab2:
        if "graph" in st.session_state:
            st.graphviz_chart(st.session_state.graph)
        else:
            st.info("Run a test to generate visualization.")
            
    with tab3:
        if st.session_state.logs:
            st.json(st.session_state.logs)
        else:
            st.info("No logs available.")
