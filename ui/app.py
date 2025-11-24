"""
CausalScript Streamlit UI.
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

# Page Config
st.set_page_config(
    page_title="CausalScript UI",
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
if "step_count" not in st.session_state:
    st.session_state.step_count = 1

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
    formatter = LaTeXFormatter(sym_engine)
    
    # Inject common units into context
    for name, unit in get_common_units().items():
        comp_engine.bind(name, unit)
        
    return comp_engine, val_engine, hint_engine, formatter

comp_engine, val_engine, hint_engine, formatter = get_engines()

# --- Helper Functions ---

def render_card(title, latex_content, status, hint=None, corrected_form=None):
    """Renders a result card."""
    border_color = "#4CAF50" if status == "ok" else "#FF5252"
    bg_color = "rgba(76, 175, 80, 0.1)" if status == "ok" else "rgba(255, 82, 82, 0.1)"
    icon = "✅" if status == "ok" else "❌"
    
    st.markdown(
        f"""
        <div style="
            border: 2px solid {border_color};
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: {bg_color};
        ">
            <div style="font-weight: bold; margin-bottom: 5px; color: {border_color};">
                {icon} {title}
            </div>
        """,
        unsafe_allow_html=True
    )
    
    if latex_content:
        st.latex(latex_content)
        
    if corrected_form:
        st.markdown(
            f"""
            <div style="
                margin-top: 10px;
                padding: 10px;
                background-color: rgba(33, 150, 243, 0.1);
                border-radius: 5px;
                border-left: 4px solid #2196F3;
                color: #0D47A1;
            ">
                <strong>ℹ️ Corrected Form:</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.latex(corrected_form)
        
    if status != "ok" and hint:
        st.markdown(
            f"""
            <div style="
                margin-top: 10px;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.5);
                border-radius: 5px;
                border-left: 4px solid #FFC107;
                color: #856404;
            ">
                <strong>💡 Hint:</strong> {hint}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --- Layout ---

main_col, side_col = st.columns([3, 1.5])

with main_col:
    st.title("CausalScript Interface")
    
    # --- Output Section (Top) ---
    st.subheader("Evaluation Results")
    
    if st.session_state.history:
        last_entry = st.session_state.history[-1]
        
        if last_entry.get("type") == "script":
            for step in last_entry.get("steps", []):
                title = step['phase'].capitalize()
                if step['phase'] == 'step':
                    pass
                
                render_card(
                    title=title,
                    latex_content=step['latex'],
                    status=step['status'],
                    hint=step.get('hint'),
                    corrected_form=step.get('corrected_form_latex')
                )
                
        elif last_entry.get("type") == "error":
            st.error(f"Error: {last_entry['error']}")
    else:
        st.info("Enter a problem and steps below to see results here.")

    st.divider()

    # --- Input Section (Bottom) ---
    st.subheader("Input")
    
    tab_struct, tab_raw = st.tabs(["Structured Input", "Raw Script"])
    
    user_input = None
    process_input = False

    with tab_struct:
        with st.form("structured_form"):
            problem_input = st.text_input("Problem", placeholder="e.g., (x - 2y)^2")
            
            steps_input = st.text_area("Steps (One per line)", height=150, placeholder="x^2 - 4xy + 4y^2\n...")
            
            end_input = st.text_input("End (Final Answer)", placeholder="e.g., x^2 - 4xy + 4y^2")
            
            submitted_struct = st.form_submit_button("Evaluate")
            
            if submitted_struct:
                # Construct script
                script_lines = []
                if problem_input:
                    if problem_input.strip().lower().startswith("problem:"):
                        script_lines.append(problem_input)
                    else:
                        script_lines.append(f"problem: {problem_input}")
                
                if steps_input:
                    for line in steps_input.split('\n'):
                        if line.strip():
                            if line.strip().lower().startswith("step:"):
                                script_lines.append(line.strip())
                            else:
                                script_lines.append(f"step: {line.strip()}")
                            
                if end_input:
                    if end_input.strip().lower().startswith("end:"):
                        script_lines.append(end_input)
                    else:
                        script_lines.append(f"end: {end_input}")
                
                user_input = "\n".join(script_lines)
                process_input = True

    with tab_raw:
        with st.form("raw_form"):
            raw_input = st.text_area("Script", height=200, placeholder="problem: ...\nstep: ...\nend: ...")
            submitted_raw = st.form_submit_button("Run Script")
            if submitted_raw:
                user_input = raw_input
                process_input = True

    # Processing Logic
    if process_input and user_input:
        # Reset logs
        st.session_state.logs = []
        st.session_state.scenario_logs = []
        
        # We need to get a fresh logger but reuse engines
        logger = LearningLogger()
        # Re-instantiate runtime with the cached engines
        runtime = CoreRuntime(comp_engine, val_engine, hint_engine, learning_logger=logger)
        
        try:
            # 1. Parse and Evaluate (Main Flow)
            parser = Parser(user_input)
            program = parser.parse()
            evaluator = Evaluator(program, runtime, learning_logger=logger)
            success = evaluator.run()
            
            # Collect logs for "Automated Answering"
            logs = logger.to_list()
            for record in logs:
                st.session_state.logs.append(f"[{record['phase'].upper()}] {record.get('rendered', '')} ({record.get('status', '')})")

            # Build History Entry
            history_entry = {
                "input": user_input,
                "type": "script",
                "steps": []
            }

            last_valid_expr = None
            for record in logs:
                if record['phase'] in ['problem', 'step', 'end']:
                    current_expr = str(record['expression']) if record['expression'] else ""
                    
                    step_data = {
                        "latex": formatter.format_expression(current_expr) if current_expr else "",
                        "status": record.get('status', 'ok'),
                        "phase": record['phase']
                    }
                    
                    # Check for corrected form in details (from FuzzyJudge)
                    if 'meta' in record and record['meta'] and 'corrected_form' in record['meta']:
                         cf = record['meta']['corrected_form']
                         step_data['corrected_form_latex'] = formatter.format_expression(str(cf))
                    
                    if record['phase'] == 'problem':
                        last_valid_expr = current_expr
                    elif record['phase'] in ['step', 'end']:
                        if record.get('status') == 'ok':
                            last_valid_expr = current_expr
                        elif last_valid_expr:
                            # Generate hint (if not already in meta)
                            if 'meta' in record and record['meta'] and 'hint' in record['meta']:
                                step_data['hint'] = record['meta']['hint']['message']
                            else:
                                try:
                                    hint_res = hint_engine.generate_hint(current_expr, last_valid_expr)
                                    step_data['hint'] = hint_res.message
                                except Exception:
                                    pass

                    history_entry["steps"].append(step_data)

            st.session_state.history.append(history_entry)
            
            # 2. Parallel Computation (Scenario Logs)
            problem_expr = None
            for record in logs:
                if record['phase'] == 'problem':
                    problem_expr = str(record['expression'])
                    break
            
            if problem_expr:
                try:
                    # Default scenarios for demo
                    scenarios = {
                        "Scenario A (x=1, y=1)": {"x": 1, "y": 1},
                        "Scenario B (x=2, y=3)": {"x": 2, "y": 3},
                        "Scenario C (x=0, y=5)": {"x": 0, "y": 5}
                    }
                    results = comp_engine.evaluate_in_scenarios(problem_expr, scenarios)
                    st.session_state.scenario_logs.append(f"Evaluated '{problem_expr}' across {len(scenarios)} scenarios:")
                    st.session_state.scenario_logs.append(json.dumps(results, indent=2))
                except Exception as e:
                    st.session_state.scenario_logs.append(f"Scenario Eval Error: {e}")

            st.rerun()
            
        except Exception as e:
            st.session_state.logs.append(f"Error: {str(e)}")
            st.session_state.history.append({
                "input": user_input,
                "type": "error",
                "error": str(e)
            })
            st.rerun()

# --- Side Panel ---

with side_col:
    st.header("Logs & Analysis")
    
    # Tabbed Side Panel
    tab_auto, tab_scenario = st.tabs(["Automated Answering", "Scenario Logs"])
    
    with tab_auto:
        st.caption("Step-by-step calculation logs")
        if st.button("Clear Auto Logs", key="clear_auto"):
            st.session_state.logs = []
            st.rerun()
            
        log_container = st.container(height=600)
        with log_container:
            for log in st.session_state.logs:
                st.text(log)

    with tab_scenario:
        st.caption("Parallel computation results")
        if st.button("Clear Scenario Logs", key="clear_scenario"):
            st.session_state.scenario_logs = []
            st.rerun()
            
        scen_container = st.container(height=600)
        with scen_container:
            for log in st.session_state.scenario_logs:
                if log.startswith("{"):
                    st.json(json.loads(log))
                else:
                    st.text(log)
