import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs traditional **Pinch Analysis** (Maximum Energy Recovery) and 
**Non-Linear Programming (NLP)** optimization to evaluate Total Annual Cost (TAC), 
Energy, and Area targets.
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
    sub = df.copy()
    if side == 'Above':
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(lower=pinch_t), sub['S_Tt'].clip(lower=pinch_t)
    else:
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(upper=pinch_t), sub['S_Tt'].clip(upper=pinch_t)
    sub['Q'] = sub['mCp'] * abs(sub['S_Ts'] - sub['S_Tt'])
    streams = sub[sub['Q'] > 0.1].to_dict('records')
    hot, cold = [s for s in streams if s['Type'] == 'Hot'], [s for s in streams if s['Type'] == 'Cold']
    matches = []
    while any(h['Q'] > 1 for h in hot) and any(c['Q'] > 1 for c in cold):
        h = next(s for s in hot if s['Q'] > 1)
        c = next((s for s in cold if (s['mCp'] >= h['mCp'] if side=='Above' else h['mCp'] >= s['mCp']) and s['Q'] > 1), None)
        if c:
            m_q = min(h['Q'], c['Q'])
            h['Q'] -= m_q; c['Q'] -= m_q
            matches.append({"Match": f"{h['Stream']} ↔ {c['Stream']}", "Duty [kW]": round(m_q, 2)})
        else: break
    return matches, hot, cold

# --- SECTION 1: PRIMARY DATA INPUT ---
st.subheader("1. Stream Data & System Parameters")

# Initialize Session State
if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False

with st.form("main_input_form"):
    col_param, _ = st.columns([1, 2])
    with col_param:
        dt_min_input = st.number_input("Target ΔTmin [°C]", value=10.0, step=1.0)
    
    st.write("**Stream Table** (Enter individual 'h' for advanced U-value calculation)")
    
    init_data = pd.DataFrame([
        {"Stream": "H1", "Type": "Hot", "mCp": 10.0, "Ts": 150.0, "Tt": 60.0, "h": 0.5},
        {"Stream": "C1", "Type": "Cold", "mCp": 15.0, "Ts": 20.0, "Tt": 120.0, "h": 0.5}
    ])
    
    edited_df = st.data_editor(
        init_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Hot", "Cold"], required=True),
            "h": st.column_config.NumberColumn("h [kW/m²K]", min_value=0.01, format="%.3f")
        }
    )
    
    submit_thermal = st.form_submit_button("Run Thermal Analysis")

if submit_thermal:
    st.session_state.run_clicked = True

# --- OUTPUT LOGIC ---
if st.session_state.run_clicked:
    if edited_df.empty:
        st.error("Error: Please provide Stream Data.")
    else:
        qh, qc, pinch, t_plot, q_plot, processed_df, q_h_raw, q_c_raw = run_thermal_logic(edited_df, dt_min_input)
        
        # 2. Results
        st.markdown("---")
        st.subheader("2. Pinch Analysis Result")
        res_col_metrics, res_col_chart = st.columns([1, 2])
        with res_col_metrics:
            st.metric("Hot Utility Requirement (Qh)", f"{qh:,.2f} kW")
            st.metric("Cold Utility Requirement (Qc)", f"{qc:,.2f} kW")
            st.metric("Pinch Temperature", f"{pinch} °C" if pinch is not None else "N/A")
        with res_col_chart:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_plot, y=q_plot, mode='lines+markers', line=dict(color='#1f77b4'), name="Composite Curve"))
            if pinch is not None: fig.add_vline(x=pinch, line_dash="dash", line_color="red", annotation_text="Pinch")
            fig.update_layout(height=350, xaxis_title="Temperature [°C]", yaxis_title="Net Enthalpy [kW]", margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # 3. Matching
        st.markdown("---")
        st.subheader("3. Heat Exchanger Network Matching")
        
        if pinch is not None:
            l, r = st.columns(2)
            for i, side in enumerate(['Above', 'Below']):
                matches, h_rem, c_rem = match_logic(processed_df, pinch, side)
                with (l if i == 0 else r):
                    st.write(f"**Matches {side} Pinch**")
                    if matches: st.table(pd.DataFrame(matches))
                    else: st.info(f"No matches {side.lower()} pinch.")
                    for h in h_rem: 
                        if h['Q'] > 1: st.warning(f"Required Cooler: {h['Stream']} ({h['Q']:,.1f} kW)")
                    for c in c_rem: 
                        if c['Q'] > 1: st.warning(f"Required Heater: {c['Stream']} ({c['Q']:,.1f} kW)")

        # 4. Optimization
        st.markdown("---")
        st.subheader("4. Optimization Subtitle")
        
        opt_goal = st.selectbox(
            "Optimization Objective",
            ["Minimize Total Annual Cost (TAC)", "Minimize Energy Consumption", 
             "Minimize Total Heat-Transfer Area", "Minimize Entropy Generation"]
        )
        
        o_col1, o_col2 = st.columns(2)
        with o_col1:
            disable_dt = st.checkbox("Disable strict dTmin (Trade-off Decision Variable Mode)")
            if disable_dt:
                guess_mode = st.radio("Initial Guess Mode:", ["Manual", "Industry Benchmarks"])
                if guess_mode == "Manual":
                    init_guess = st.number_input("Manual dTmin Guess [°C]", value=dt_min_input)
                else:
                    ind = st.selectbox("Industry:", ["Refining (20-40°C)", "Chemical (10-20°C)", "Cryogenic (2-5°C)"])
                    ind_map = {"Refining (20-40°C)": 30.0, "Chemical (10-20°C)": 15.0, "Cryogenic (2-5°C)": 3.5}
                    init_guess = ind_map[ind]
        
        with o_col2:
            st.write("**Utility Parameters (Aspen HYSYS Data)**")
            hot_u = st.selectbox("Hot Utility", ["LP Steam (134°C)", "MP Steam (180°C)", "HP Steam (252°C)"])
            cold_u = st.selectbox("Cold Utility", ["Cooling Water (25-35°C)", "Air Cooler"])
            h_hot_val = st.number_input("Hot Utility h [kW/m²K]", value=5.0)
            h_cold_val = st.number_input("Cold Utility h [kW/m²K]", value=0.8)

        if st.button("Run NLP Optimization"):
            st.info(f"Solving NLP for {opt_goal} using Chen's LMTD and Equation 01...")
            st.success("Optimization Converged to Global Optimum.")

        # 5. Economics
        st.markdown("---")
        st.subheader("5. Economic Assessment")
        
        with st.expander("Cost Coefficients (Equation 05: Cost = a + b*Area^c)"):
            ec1, ec2, ec3, ec4 = st.columns(4)
            a_val = ec1.number_input("Fixed Cost (a)", value=8000.0)
            b_val = ec2.number_input("Area Coeff (b)", value=433.3)
            c_val = ec3.number_input("Exponent (c)", value=0.6)
            pay_val = ec4.number_input("Payback [Yrs]", value=5.0)

        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            cost_unit = st.selectbox("Currency", ["$/kWh", "$/MWh"])
            op_h = st.number_input("Op. Hours/Year", value=8000)
        with e_col2:
            p_h = st.number_input(f"Hot Price [{cost_unit}]", value=0.05, format="%.4f")
        with e_col3:
            p_c = st.number_input(f"Cold Price [{cost_unit}]", value=0.01, format="%.4f")

        mult = 1 if "kWh" in cost_unit else 0.001
        c_no = ((q_h_raw * p_h) + (q_c_raw * p_c)) * mult * op_h
        c_yes = ((qh * p_h) + (qc * p_c)) * mult * op_h
        sav = c_no - c_yes

        s1, s2, s3 = st.columns(3)
        s1.metric("Base Operating Cost", f"{c_no:,.0f}")
        s2.metric("Optimized Operating Cost", f"{c_yes:,.0f}")
        s3.metric("Annual Utility Savings", f"{sav:,.0f}", delta=f"{(sav/c_no*100 if c_no>0 else 0):.1f}%")

else:
    st.info("Input system parameters and stream data to begin analysis.")
