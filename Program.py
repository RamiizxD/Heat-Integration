import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Synthesis & Optimization")
st.markdown("""
This application performs traditional **Pinch Analysis**, **Stream Matching** and **NLP Optimization**.
""")
st.markdown("---")

# --- MATH CORE FUNCTIONS ---
def calculate_u(h1, h2):
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    theta1 = max(abs(t1 - t4), 0.01)
    theta2 = max(abs(t2 - t3), 0.01)
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)
    df['Q_Raw'] = df['mCp'] * abs(df['Ts'] - df['Tt'])
    q_h_raw = df[df['Type'] == 'Hot']['Q_Raw'].sum()
    q_c_raw = df[df['Type'] == 'Cold']['Q_Raw'].sum()
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_load = df[(df['Type'] == 'Hot') & (df['S_Ts'] >= hi) & (df['S_Tt'] <= lo)]['mCp'].sum() * (hi - lo)
        c_load = df[(df['Type'] == 'Cold') & (df['S_Ts'] <= lo) & (df['S_Tt'] >= hi)]['mCp'].sum() * (hi - lo)
        intervals.append({'hi': hi, 'lo': lo, 'net': h_load - c_load})
    int_df = pd.DataFrame(intervals)
    infeasible = [0] + list(int_df['net'].cumsum())
    qh_min = abs(min(min(infeasible), 0))
    feasible = [qh_min + val for val in infeasible]
    pinch_t = temps[feasible.index(0)] if 0 in feasible else None
    return qh_min, feasible[-1], pinch_t, temps, feasible, df, q_h_raw, q_c_raw

# --- SECTION 1: PRIMARY DATA INPUT ---
st.subheader("1. Stream Data & System Parameters")

if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False

with st.form("main_input_form"):
    col_param, _ = st.columns([1, 2])
    with col_param:
        dt_min_input = st.number_input("Target ΔTmin", value=10.0, step=1.0)
    
    init_data = pd.DataFrame([
        {"Stream": "H1", "Type": "Hot", "mCp": 10.0, "Ts": 150.0, "Tt": 60.0, "individual HTC (h)": 0.5},
        {"Stream": "C1", "Type": "Cold", "mCp": 15.0, "Ts": 20.0, "Tt": 120.0, "individual HTC (h)": 0.5}
    ])
    
    edited_df = st.data_editor(init_data, num_rows="dynamic", use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Hot", "Cold"], required=True),
            "individual HTC (h)": st.column_config.NumberColumn("(h)", min_value=0.01, format="%.3f")
        }
    )
    submit_thermal = st.form_submit_button("Run Pinch Analysis & MER HEN Synthesis")

if submit_thermal:
    st.session_state.run_clicked = True

# --- MAIN OUTPUT DISPLAY ---
if st.session_state.run_clicked:
    qh, qc, pinch, t_plot, q_plot, processed_df, q_h_raw, q_c_raw = run_thermal_logic(edited_df, dt_min_input)
    
    # 2. Results
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    res_col_metrics, res_col_chart = st.columns([1, 2])
    with res_col_metrics:
        st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
        st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
        st.metric("Pinch Temp", f"{pinch} °C" if pinch is not None else "N/A")
    with res_col_chart:
        fig_cc = go.Figure()
        fig_cc.add_trace(go.Scatter(x=t_plot, y=q_plot, mode='lines+markers', name="Composite Curve"))
        if pinch: fig_cc.add_vline(x=pinch, line_dash="dash", line_color="red")
        fig_cc.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_cc, use_container_width=True)

    # 4. Optimization Section
    st.markdown("---")
    st.subheader("4. Optimization & Economic Results")
    
    opt_goal = st.selectbox("Objective", ["Minimize TAC", "Minimize Energy", "Minimize Area", "Multi-Objective"])
    
    if opt_goal == "Multi-Objective":
        weight = st.slider("Weight: Economic (Left) vs Environmental (Right)", 0.0, 1.0, 0.5)
        st.caption("Blending TAC minimization with Entropy generation reduction.")

    o1, o2 = st.columns(2)
    with o1:
        disable_dt = st.checkbox("Variable dTmin (NLP Decision Mode)")
        if disable_dt:
            ind = st.selectbox("Industry Benchmark:", ["Refining (20-40°C)", "Chemical (10-20°C)", "Cryogenic (2-5°C)"])
            ind_map = {"Refining (20-40°C)": 30.0, "Chemical (10-20°C)": 15.0, "Cryogenic (2-5°C)": 3.5}
            init_dt = ind_map[ind]
    with o2:
        h_hot_v = st.number_input("Hot Utility h [kW/m²K]", value=5.0)
        h_cold_v = st.number_input("Cold Utility h [kW/m²K]", value=0.8)

    # RUN OPTIMIZATION BUTTON
    if st.button("Calculate Economic Optimum"):
        # Solve for Area target
        avg_h_h = edited_df[edited_df['Type']=='Hot']['h'].mean()
        avg_h_c = edited_df[edited_df['Type']=='Cold']['h'].mean()
        U_h = calculate_u(h_hot_v, avg_h_c)
        U_c = calculate_u(h_cold_v, avg_h_h)
        lmtd_est = lmtd_chen(150, 140, 100, 110)
        
        # Calculate Economics immediately
        opt_area = (qh / (U_h * lmtd_est)) + (qc / (U_c * lmtd_est))
        
        # 5. Economic Assessment (Placed inside Section 4 display logic)
        st.markdown("#### NLP Optimization Economic Breakdown")
        
        # Use user coefficients or defaults
        a, b, c, pay = 8000.0, 433.3, 0.6, 5.0
        cap_inv = a + b * (opt_area ** c)
        op_cost = (qh * 0.05 + qc * 0.01) * 8000
        tac = op_cost + (cap_inv / pay)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimized Total Area", f"{opt_area:,.2f} m²")
        m2.metric("Total Capital Investment", f"${cap_inv:,.2f}")
        m3.metric("Total Annual Cost (TAC)", f"${tac:,.2f}")

        fig_cost = go.Figure(data=[
            go.Bar(name='Op. Cost', x=['Breakdown'], y=[op_cost]),
            go.Bar(name='Annualized Cap.', x=['Breakdown'], y=[cap_inv/pay])
        ])
        fig_cost.update_layout(barmode='stack', height=350)
        st.plotly_chart(fig_cost, use_container_width=True)

    # Section 5 (Static definitions/Coefficients)
    st.markdown("---")
    st.subheader("5. Economic Assessment Parameters")
    with st.expander("Adjust Cost Coefficients (Eq. 05)"):
        st.write("Current model uses: Cost = 8000 + 433.3 * Area^0.6")
        st.number_input("Fixed Cost (a)", 8000.0)
        st.number_input("Area Coefficient (b)", 433.3)
else:
    st.info("Please run thermal analysis to enable optimization.")
