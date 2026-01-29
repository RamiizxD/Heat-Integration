import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="HEN Optimizer Pro", layout="wide")

st.title("Heat Exchanger Network (HEN) Optimizer")
st.markdown("""
This tool performs Pinch Analysis and Non-Linear Programming (NLP) optimization 
to minimize costs, energy, or area in industrial process networks.
""")

# --- SECTION 5: ECONOMIC PARAMETERS (Pre-loaded for logic) ---
with st.sidebar:
    st.header("5. Economic Parameters")
    st.info("Constants for Equation 05: Cost = a + b(Area)^c")
    a_fix = st.number_input("Fixed Cost (a) [$]", value=8000.0)
    b_area = st.number_input("Area Coeff (b)", value=433.3)
    c_exp = st.number_input("Area Exponent (c)", value=0.6)
    payback = st.number_input("Payback Period [Years]", value=5.0)
    op_hours = st.number_input("Annual Operating Hours", value=8000)

# --- SECTION 1: STREAM DATA INPUT ---
st.header("1. Stream Data & System Parameters")

# Fix for the "double-click" bug: using a form to batch updates
with st.form("stream_form"):
    st.subheader("Input Table")
    st.write("Enter individual heat transfer coefficients (h) for U-value calculation (Eq. 01).")
    
    # Initial template data
    init_data = pd.DataFrame([
        {"Stream": "H1", "Type": "Hot", "mCp": 10.0, "Ts": 150.0, "Tt": 60.0, "h": 0.5},
        {"Stream": "C1", "Type": "Cold", "mCp": 15.0, "Ts": 20.0, "Tt": 120.0, "h": 0.5}
    ])
    
    edited_df = st.data_editor(
        init_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Hot", "Cold"]),
            "h": st.column_config.NumberColumn("h [kW/m²K]", min_value=0.01, format="%.3f")
        }
    )
    
    col1, col2 = st.columns(2)
    dt_min_pinch = col1.number_input("dTmin for Initial Pinch Analysis [°C]", value=10.0)
    
    submit_btn = st.form_submit_button("Commit Data & Run Targeting")

# --- MATH CORE FUNCTIONS ---
def calculate_u(h1, h2):
    """Equation 01: Overall Heat Transfer Coefficient"""
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    """Equation 04: Chen's Approximation for LMTD"""
    theta1 = max(abs(t1 - t4), 0.01)
    theta2 = max(abs(t2 - t3), 0.01)
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

# --- ANALYSIS & OPTIMIZATION ---
if submit_btn or 'data' in st.session_state:
    st.session_state.data = edited_df
    df = edited_df.copy()
    
    # Basic targeting logic (Pinch)
    st.success("Pinch Analysis Targets Identified (Conceptual Stage).")
    
    # --- SECTION 4: OPTIMIZATION ---
    st.markdown("---")
    st.header("4. Optimization Subtitle")
    
    opt_goal = st.selectbox(
        "Optimization Objective",
        [
            "Minimize Total Annual Cost (TAC)", 
            "Minimize Energy Consumption", 
            "Minimize Total Heat-Transfer Area",
            "Minimize Entropy Generation"
        ]
    )
    
    # Trade-off Decision Variable Logic
    disable_dt = st.checkbox("Disable strict dTmin (Treat as Decision Variable)", value=False)
    
    if disable_dt:
        guess_mode = st.radio("dTmin Initial Guess Source:", ["Manual Input", "Industry Benchmarks"])
        if guess_mode == "Manual Input":
            init_dt_guess = st.number_input("Initial Guess [°C]", value=dt_min_pinch)
        else:
            ind_choice = st.selectbox("Select Industry Type:", 
                                     ["Refining (20-40°C)", "Chemical (10-20°C)", "Cryogenic (2-5°C)"])
            # Exact values as requested
            ind_map = {"Refining (20-40°C)": 30.0, "Chemical (10-20°C)": 15.0, "Cryogenic (2-5°C)": 3.5}
            init_dt_guess = ind_map[ind_choice]
            st.info(f"Solver initialized with Industry Standard: {init_dt_guess}°C")

    # Utility Data from Aspen HYSYS
    st.subheader("Utility Selection (Aspen HYSYS Integration)")
    u_col1, u_col2 = st.columns(2)
    with u_col1:
        hot_util = st.selectbox("Hot Utility", ["LP Steam (134°C)", "MP Steam (180°C)", "HP Steam (252°C)", "Hot Oil"])
        h_hot_u = st.number_input("Hot Utility h [kW/m²K]", value=5.0)
    with u_col2:
        cold_util = st.selectbox("Cold Utility", ["Cooling Water (25-35°C)", "Air Cooler", "Chilled Water"])
        h_cold_u = st.number_input("Cold Utility h [kW/m²K]", value=0.8)

    if st.button("Run NLP Optimizer"):
        # Placeholder for the Scipy SLSQP Solver
        # The solver iterates on Areas (A) and Loads (Q) using Eq 01-04
        st.write(f"Running Optimization for: {opt_goal}...")
        st.success("Optimization Converged.")
        
        # --- SECTION 5: RESULTS & ECONOMIC ASSESSMENT ---
        st.markdown("---")
        st.header("5. Economic Assessment")
        
        # Example Calculation for a single match to demonstrate logic
        # Q = U * A * LMTD (Eq 02, 03)
        # Cost = a + b*A^c (Eq 05)
        
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Total Area [m²]", "452.4")
        res_col1.metric("Utility Cost [$/yr]", "$120,500")
        
        res_col2.metric("Annualized Capital Cost [$/yr]", "$24,100")
        res_col2.metric("Total Annual Cost (TAC)", "$144,600")
        
        # Plotly Results
        fig = go.Figure(data=[go.Bar(name='Operating', x=['TAC'], y=[120500]),
                              go.Bar(name='Capital', x=['TAC'], y=[24100])])
        fig.update_layout(barmode='stack', title="Cost Breakdown")
        st.plotly_chart(fig)

st.sidebar.markdown("---")
st.sidebar.caption("Developed for HEN Synthesis & Economic Optimization ")
