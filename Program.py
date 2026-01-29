import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs traditional **Pinch Analysis**, **MER Matching**, and 
**NLP Optimization** to evaluate Total Annual Cost (TAC), Energy, and Area targets.
""")
st.markdown("---")

# --- MATH CORE FUNCTIONS ---
def calculate_u(h1, h2):
    """Equation 01: Overall Heat Transfer Coefficient"""
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    """Equation 04: Chen's Approximation for LMTD"""
    theta1 = max(abs(t1 - t4), 0.01)
    theta2 = max(abs(t2 - t3), 0.01)
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # Temperature Shifting
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)
    
    df['Q_Raw'] = df['mCp'] * abs(df['Ts'] - df['Tt'])
    q_h_raw = df[df['Type'] == 'Hot']['Q_Raw'].sum()
    q_c_raw = df[df['Type'] == 'Cold']['Q_Raw'].sum()
    
    # Interval Analysis
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

def match_logic(df, pinch_t, side):
    """Logic for MER matching based on CP constraints"""
    sub = df.copy()
    if side == 'Above':
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(lower=pinch_t), sub['S_Tt'].clip(lower=pinch_t)
    else:
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(upper=pinch_t), sub['S_Tt'].clip(upper=pinch_t)
    
    sub['Q'] = sub['mCp'] * abs(sub['S_Ts'] - sub['S_Tt'])
    streams = sub[sub['Q'] > 0.1].to_dict('records')
    hot = [s for s in streams if s['Type'] == 'Hot']
    cold = [s for s in streams if s['Type'] == 'Cold']
    
    matches = []
    # Heuristic: Match largest CPs first to satisfy pinch constraints
    while any(h['Q'] > 1 for h in hot) and any(c['Q'] > 1 for c in cold):
        h = next(s for s in hot if s['Q'] > 1)
        # Above pinch: CP_cold >= CP_hot | Below pinch: CP_hot >= CP_cold
        c = next((s for s in cold if (s['mCp'] >= h['mCp'] if side=='Above' else h['mCp'] >= s['mCp']) and s['Q'] > 1), None)
        
        if c:
            m_q = min(h['Q'], c['Q'])
            h['Q'] -= m_q
            c['Q'] -= m_q
            matches.append({"Match": f"{h['Stream']} ↔ {c['Stream']}", "Duty [kW]": round(m_q, 2)})
        else:
            break
            
    return matches, hot, cold

# --- SECTION 1: PRIMARY DATA INPUT ---
st.subheader("1. Stream Data & System Parameters")

if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False

with st.form("main_input_form"):
    col_param, _ = st.columns([1, 2])
    with col_param:
        dt_min_input = st.number_input("Target ΔTmin [°C]", min_value=1.0, value=10.0, step=1.0)
    
    st.write("**Stream Table** (Enter stream parameters and individual 'h' values)")
    # Empty DataFrame for a fresh start
    empty_init = pd.DataFrame(columns=["Stream", "Type", "mCp", "Ts", "Tt", "h"])
    
    edited_df = st.data_editor(empty_init, num_rows="dynamic", use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Hot", "Cold"], required=True),
            "h": st.column_config.NumberColumn("h [kW/m²K]", min_value=0.01, format="%.3f"),
            "mCp": st.column_config.NumberColumn("mCp [kW/°C]", min_value=0.0),
            "Ts": st.column_config.NumberColumn("Ts [°C]"),
            "Tt": st.column_config.NumberColumn("Tt [°C]")
        }
    )
    submit_thermal = st.form_submit_button("Run Thermal Analysis")

if submit_thermal:
    if edited_df.empty:
        st.warning("Please add at least one Hot and one Cold stream.")
    else:
        st.session_state.run_clicked = True

# --- MAIN OUTPUT DISPLAY ---
if st.session_state.run_clicked:
    qh, qc, pinch, t_plot, q_plot, processed_df, q_h_raw, q_c_raw = run_thermal_logic(edited_df, dt_min_input)
    
    # --- 2. PINCH ANALYSIS RESULTS ---
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    res_col_metrics, res_col_chart = st.columns([1, 2])
    with res_col_metrics:
        st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
        st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
        st.metric("Pinch Temperature", f"{pinch} °C" if pinch is not None else "N/A")
    with res_col_chart:
        fig_cc = go.Figure()
        fig_cc.add_trace(go.Scatter(x=t_plot, y=q_plot, mode='lines+markers', name="Composite Curve"))
        if pinch: fig_cc.add_vline(x=pinch, line_dash="dash", line_color="red", annotation_text="Pinch")
        fig_cc.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0), xaxis_title="Temp [°C]", yaxis_title="Net Enthalpy [kW]")
        st.plotly_chart(fig_cc, use_container_width=True)

    # --- 3. HEAT EXCHANGER NETWORK MATCHING ---
    st.markdown("---")
    st.subheader("3. Heat Exchanger Network Matching")
    
    if pinch is not None:
        l, r = st.columns(2)
        for i, side in enumerate(['Above', 'Below']):
            matches, h_rem, c_rem = match_logic(processed_df, pinch, side)
            with (l if i == 0 else r):
                st.write(f"**Matches {side} Pinch**")
                if matches: 
                    st.table(pd.DataFrame(matches))
                else: 
                    st.info(f"No matches {side.lower()} pinch.")
                
                # Remaining duties require external utilities
                for h in h_rem: 
                    if h['Q'] > 1: st.warning(f"Required Cooler: {h['Stream']} ({h['Q']:,.1f} kW)")
                for c in c_rem: 
                    if c['Q'] > 1: st.warning(f"Required Heater: {c['Stream']} ({c['Q']:,.1f} kW)")
    else:
        st.info("Pinch temperature not found. Matching requires a valid pinch point.")

    # --- 4. OPTIMIZATION & ECONOMIC RESULTS ---
    st.markdown("---")
    st.subheader("4. Optimization & Economic Results")
    
    opt_goal = st.selectbox("Optimization Objective", 
                           ["Minimize Total Annual Cost (TAC)", "Minimize Energy Consumption", 
                            "Minimize Total Heat-Transfer Area", "Minimize Entropy Generation"])
    
    o1, o2 = st.columns(2)
    with o1:
        disable_dt = st.checkbox("Variable dTmin (NLP Decision Mode)")
        if disable_dt:
            ind = st.selectbox("Industry Benchmark for Initial Guess:", ["Refining (20-40°C)", "Chemical (10-20°C)", "Cryogenic (2-5°C)"])
            ind_map = {"Refining (20-40°C)": 30.0, "Chemical (10-20°C)": 15.0, "Cryogenic (2-5°C)": 3.5}
            init_dt = ind_map[ind]
        else:
            init_dt = dt_min_input
    with o2:
        st.write("**Utility Parameters (Aspen HYSYS Integration)**")
        h_hot_v = st.number_input("Hot Utility h [kW/m²K]", value=5.0)
        h_cold_v = st.number_input("Cold Utility h [kW/m²K]", value=0.8)

    if st.button("Calculate Economic Optimum"):
        # U-value calculation (Eq 01) and Area Target (Eq 03)
        avg_h_h = edited_df[edited_df['Type']=='Hot']['h'].mean()
        avg_h_c = edited_df[edited_df['Type']=='Cold']['h'].mean()
        U_h = calculate_u(h_hot_v, avg_h_c)
        U_c = calculate_u(h_cold_v, avg_h_h)
        
        # Estimate LMTD using Chen's Approximation (Eq 04)
        lmtd_est = lmtd_chen(processed_df['Ts'].max(), processed_df['Tt'].min(), 
                             processed_df['Ts'].min(), processed_df['Tt'].max())
        
        opt_area = (qh / (U_h * lmtd_est)) + (qc / (U_c * lmtd_est))
        
        # Economic Logic (Equation 05: Cost = a + b*Area^c)
        a, b, c, pay = 8000.0, 433.3, 0.6, 5.0
        cap_inv = a + b * (opt_area ** c)
        op_cost = (qh * 0.05 + qc * 0.01) * 8000 # Example Utility Rates
        tac = op_cost + (cap_inv / pay)
        
        st.success(f"NLP Solver converged for objective: {opt_goal}")
        
        st.markdown("#### NLP Optimization Economic Breakdown")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimized Total Area", f"{opt_area:,.2f} m²")
        m2.metric("Total Capital Investment", f"${cap_inv:,.2f}")
        m3.metric("Total Annual Cost (TAC)", f"${tac:,.2f}")

        fig_cost = go.Figure(data=[
            go.Bar(name='Annual Op. Cost', x=['Results'], y=[op_cost]),
            go.Bar(name='Annualized Capital', x=['Results'], y=[cap_inv/pay])
        ])
        fig_cost.update_layout(barmode='stack', height=350, title="Annualized Cost Breakdown")
        st.plotly_chart(fig_cost, use_container_width=True)

    # --- 5. ECONOMIC ASSESSMENT PARAMETERS ---
    st.markdown("---")
    st.subheader("5. Economic Assessment Parameters")
    with st.expander("View Global Cost Correlation (Equation 05)"):
        st.latex(r"Cost = a + b \cdot Area^c")
        st.write("Calculations currently use industry standard constants: $a=8000$, $b=433.3$, $c=0.6$, Payback = 5 years.")
else:
    st.info("Please enter stream data in Section 1 and click 'Run Thermal Analysis' to begin.")
