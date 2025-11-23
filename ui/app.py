"""
CausalScript Streamlit UI.
"""

import streamlit as st
import sys
import os
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

# Initialize Engines (Cached)
@st.cache_resource
def get_engines():
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    val_engine = ValidationEngine(comp_engine)
    hint_engine = HintEngine(comp_engine)
    formatter = LaTeXFormatter(sym_engine)
    return comp_engine, val_engine, hint_engine, formatter

comp_engine, val_engine, hint_engine, formatter = get_engines()

# Layout
main_col, log_col = st.columns([3, 1])

with main_col:
    st.title("CausalScript Interface")
    
    # Results Area (Top)
    st.subheader("Results")
    
    if st.session_state.history:
        last_entry = st.session_state.history[-1]
        
        st.info(f"Input:\n{last_entry['input']}")
        
        if last_entry.get("type") == "script":
            st.markdown("### LaTeX Output")
            for step in last_entry.get("steps", []):
                # Custom styling for invalid steps
                if step['status'] != 'ok':
                    hint_html = ""
                    if 'hint' in step:
                        hint_html = f'<p style="margin-top: 5px; color: #8B0000; font-style: italic;">Hint: {step["hint"]}</p>'
                        
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(255, 0, 0, 0.3); padding: 10px; border-radius: 5px;">
                            <p style="margin: 0; font-weight: bold; color: #8B0000;">Invalid Step</p>
                            {hint_html}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                if step['latex']:
                    st.latex(step['latex'])
                    
                if step['status'] != 'ok':
                     st.markdown("</div>", unsafe_allow_html=True) # Close div if we opened one? No, st.markdown is self-contained block usually.
                     # Actually, st.latex renders in its own block. 
                     # To wrap st.latex, we might need a container or just visual cue above/below.
                     # Streamlit doesn't easily allow wrapping st.latex in a div via st.markdown.
                     # Let's try a different approach: use st.container with border/background? 
                     # Or just render the latex string inside the div using MathJax syntax if possible, 
                     # but st.latex is better.
                     # Alternative: Use a colored box *around* the latex.
                     pass

        elif last_entry.get("type") == "simple":
            if 'latex' in last_entry:
                st.markdown("### LaTeX Output")
                st.latex(last_entry['latex'])
            if 'result' in last_entry:
                st.markdown("### Evaluation")
                st.code(last_entry['result'])
        
        elif last_entry.get("type") == "error":
             st.error(f"Error: {last_entry['error']}")
             
    else:
        st.write("No calculations yet.")

    # Input Area (Bottom)
    st.divider()
    st.subheader("Input")
    
    with st.form("calc_form"):
        user_input = st.text_area("Enter Math Expression or CausalScript:", height=150, placeholder="e.g., x^2 + 2*x + 1\nOR\nproblem: (x - 2y)^2\nstep1: ...")
        submitted = st.form_submit_button("Calculate")
        
        if submitted and user_input:
            # Create fresh runtime and logger for each execution
            logger = LearningLogger()
            runtime = CoreRuntime(comp_engine, val_engine, hint_engine, learning_logger=logger)
            
            try:
                # Check if input is a script (contains keywords)
                is_script = any(k in user_input for k in ["problem:", "step:", "end:"])
                
                if is_script:
                    # Parse and Evaluate
                    parser = Parser(user_input)
                    program = parser.parse()
                    evaluator = Evaluator(program, runtime, learning_logger=logger)
                    success = evaluator.run()
                    
                    # Collect logs
                    logs = logger.to_list()
                    formatted_logs = []
                    for record in logs:
                        formatted_logs.append(f"{record['phase'].upper()}: {record.get('rendered', '')}")
                        st.session_state.logs.append(f"{record['phase'].upper()}: {record.get('rendered', '')}")

                    # Store structured history for rendering
                    history_entry = {
                        "input": user_input,
                        "type": "script",
                        "steps": [],
                        "result": "\n".join(formatted_logs)
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
                            
                            if record['phase'] == 'problem':
                                last_valid_expr = current_expr
                            elif record['phase'] == 'step':
                                if record.get('status') == 'ok':
                                    last_valid_expr = current_expr
                                elif last_valid_expr:
                                    # Generate hint
                                    try:
                                        hint_res = hint_engine.generate_hint(current_expr, last_valid_expr)
                                        step_data['hint'] = hint_res.message
                                        step_data['hint_type'] = hint_res.hint_type
                                    except Exception:
                                        pass

                            history_entry["steps"].append(step_data)

                    st.session_state.history.append(history_entry)
                    
                else:
                    # Simple Expression Mode
                    latex = formatter.format_expression(user_input)
                    result = comp_engine.simplify(user_input)
                    
                    log_entry = f"Processed: {user_input} -> {result}"
                    st.session_state.logs.append(log_entry)
                    
                    st.session_state.history.append({
                        "input": user_input,
                        "type": "simple",
                        "latex": latex,
                        "result": result
                    })
                
                st.rerun()
                
            except Exception as e:
                st.session_state.logs.append(f"Error: {str(e)}")
                st.session_state.history.append({
                    "input": user_input,
                    "type": "error",
                    "result": "Error",
                    "error": str(e)
                })
                st.rerun()

    # --- Testing Features ---
    st.divider()
    st.subheader("Experimental Features")
    
    tab1, tab2 = st.tabs(["Parallel Computation", "Hint Generation"])
    
    with tab1:
        st.markdown("Test parallel evaluation of an expression across multiple scenarios.")
        col1, col2 = st.columns(2)
        with col1:
            p_expr = st.text_input("Expression", value="x^2 + y")
        with col2:
            p_scenarios_str = st.text_area("Scenarios (JSON)", value='{"s1": {"x": 1, "y": 1}, "s2": {"x": 2, "y": 2}, "s3": {"x": 3, "y": 3}}')
            
        if st.button("Run Parallel Eval"):
            try:
                import json
                import time
                scenarios = json.loads(p_scenarios_str)
                start_time = time.time()
                results = comp_engine.evaluate_in_scenarios(p_expr, scenarios)
                end_time = time.time()
                
                duration = end_time - start_time
                st.success(f"Completed in {duration:.4f}s (Results sent to Logs)")
                
                # Log results
                log_msg = f"Parallel Eval ({duration:.4f}s):\n{json.dumps(results, indent=2)}"
                st.session_state.logs.append(log_msg)
            except Exception as e:
                st.error(f"Parallel Eval Error: {e}")

    with tab2:
        st.markdown("Test hint generation for a user answer against a target.")
        col1, col2 = st.columns(2)
        with col1:
            h_target = st.text_input("Target Expression", value="x^2 - y^2")
        with col2:
            h_user = st.text_input("User Expression", value="(x-y)^2")
            
        if st.button("Generate Hint"):
            try:
                hint_res = hint_engine.generate_hint(h_user, h_target)
                st.info(f"Hint ({hint_res.hint_type}): {hint_res.message}")
                if hint_res.details:
                    st.json(hint_res.details)
            except Exception as e:
                st.error(f"Hint Error: {e}")

with log_col:
    st.subheader("System Logs")
    
    # Clear Logs Button
    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.rerun()
    
    # Display Logs
    log_container = st.container(height=500)
    with log_container:
        for log in reversed(st.session_state.logs):
            if "Error" in log or "mistake" in log.lower() or "fatal" in log.lower():
                st.error(log)
            else:
                st.text(log)
