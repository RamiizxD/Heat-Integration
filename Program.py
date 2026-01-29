import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Synthesis")
st.markdown("Testing Benchmark: **Linnhoff 4-Stream Classic**")
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
    # Ensure numeric types
    df['mCp'] = pd.to_numeric(df['mCp'])
    df['Ts'] = pd.to_numeric(df['Ts'])
    df['Tt'] = pd.to_numeric(df['Tt'])
    
    # Standard Shifting Logic (Global Shift)
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)
    
    df['Q_Raw'] = df['mCp'] * abs(df['Ts'] - df['Tt'])
    
    # Interval Analysis
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        mid = (hi + lo) / 2
        
        # Calculate net heat in this interval
        h_cp_sum = df[(df['Type'] == 'Hot') & (df['S_Ts'] >= hi) & (df['S_Tt'] <= lo)]['mCp'].sum()
        c_cp_sum = df[(df['Type'] == 'Cold') & (df['S_Ts'] <= lo) & (df['S_Tt'] >= hi)]['mCp'].sum()
        
        net_q = (h_cp_sum - c_cp_sum) * (hi - lo)
        intervals.append({'hi': hi, 'lo': lo, 'net': net_q})
    
    int_df = pd.DataFrame(intervals)
    cumulative = [0] + list(int_df['net'].cumsum())
    
    # Cascade Analysis
    qh_min = abs(min(min(cumulative), 0))
    feasible_cascade = [qh_min + val for val in cumulative]
    
    # Find Pinch (where heat flow is zero)
    pinch_t = None
    if 0 in feasible_cascade:
        pinch_t = temps[feasible_cascade.index(0)]
        
    qc_min = feasible_cascade[-1]
    
    return qh_min, qc_min, pinch_t, temps, feasible_cascade, df

# --- SECTION 1: PRIMARY DATA INPUT ---
st.subheader("1. Stream Data & System Parameters")

if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False

with st.form("main_input_form"):
    col_param, _ = st.columns([1, 2])
    with col_param:
        dt_min_input = st.number_input("Target ΔTmin", value=10.0, step=1.0)
    
    # BENCHMARK DATA PRE-LOADED
    init_data = pd.DataFrame([
        {"Stream": "H1", "Type": "Hot", "mCp": 30.0, "Ts": 170.0, "Tt": 60.0, "h": 0.5},
        {"Stream": "H2", "Type": "Hot", "mCp": 15.0, "Ts": 150.0, "Tt": 30.0, "h": 0.5},
        {"Stream": "C1", "Type": "Cold", "mCp": 20.0, "Ts": 20.0, "Tt": 135.0, "h": 0.5},
        {"Stream": "C2", "Type": "Cold", "mCp": 40.0, "Ts": 80.0, "Tt": 140.0, "h": 0.5}
    ])
    
    edited_df = st.data_editor(init_data, num_rows="dynamic", use_container_width=True)
    submit_thermal = st.form_submit_button("Run Analysis")

if submit_thermal:
    st.session_state.run_clicked = True

# --- MAIN OUTPUT DISPLAY ---
if st.session_state.run_clicked:
    qh, qc, pinch, t_plot, q_plot, processed_df = run_thermal_logic(edited_df, dt_min_input)
    
    st.markdown("---")
    st.subheader("2. Results & Grand Composite Curve")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Hot Utility (Qh)", f"{qh:,.1f} kW")
    m2.metric("Cold Utility (Qc)", f"{qc:,.1f} kW")
    m3.metric("Shifted Pinch Temp", f"{pinch}°C" if pinch is not None else "None")

    # Plotting the Grand Composite Curve
    fig_gcc = go.Figure()
    fig_gcc.add_trace(go.Scatter(x=q_plot, y=t_plot, mode='lines+markers', name="Heat Cascade"))
    fig_gcc.update_layout(title="Grand Composite Curve", xaxis_title="Net Heat Flow (kW)", yaxis_title="Shifted Temp (°C)")
    st.plotly_chart(fig_gcc, use_container_width=True)
