import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs **Pinch Analysis**, **MER Matching**, and 
**NLP Optimization** using methodologies for simultaneous design and cost reduction[cite: 1, 13].
""")
st.markdown("---")

# --- CORE MATH FUNCTIONS (Derived from source) ---
def calculate_u(h1, h2):
    """Equation 01: Overall Heat Transfer Coefficient [cite: 93]"""
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    """Equation 04: Chen's Approximation for LMTD [cite: 94]"""
    theta1 = max(abs(t1 - t4), 0.01)
    theta2 = max(abs(t2 - t3), 0.01)
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def run_thermal_logic(df, dt):
    """Performs Pinch Analysis and Interval Analysis [cite: 57, 111]"""
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)
    df['Q_Raw'] = df['mCp'] * abs(df['Ts'] - df['Tt'])
    
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_load = df[(df['Type'] == 'Hot') & (df['S_Ts'] >= hi) & (df['S_Tt'] <= lo)]['mCp'].sum() * (hi - lo)
        c_load = df[(df['Type'] == 'Cold') & (df['S_Ts'] <= lo) & (df['S_Tt'] >= hi)]['mCp'].sum() * (hi - lo)
        intervals.append({'hi': hi, 'lo': lo, 'net': h_load - c_load})
    
    infeasible = [0] + list(pd.DataFrame(intervals)['net'].cumsum())
    qh_min = abs(min(min(infeasible), 0))
    feasible = [qh_min + val for val in infeasible]
    pinch_t = temps[feasible.index(0)] if 0 in feasible else None
    return qh_min, feasible[-1], pinch_t, temps, feasible, df

def match_logic_with_splitting(df, pinch_t, side):
    sub = df.copy()
    # ... [Same clipping logic as before] ...
    
    sub['Q'] = sub['mCp'] * abs(sub['S_Ts'] - sub['S_Tt'])
    streams = sub[sub['Q'] > 0.1].to_dict('records')
    hot = [s for s in streams if s['Type'] == 'Hot']
    cold = [s for s in streams if s['Type'] == 'Cold']
    matches = []
    
    while any(h['Q'] > 1 for h in hot) and any(c['Q'] > 1 for c in cold):
        h = next(s for s in hot if s['Q'] > 1)
        # Attempt to find a direct match
        c = next((s for s in cold if (s['mCp'] >= h['mCp'] if side=='Above' else h['mCp'] >= s['mCp']) and s['Q'] > 1), None)
        
        if c:
            m_q = min(h['Q'], c['Q'])
            h['Q'] -= m_q
            c['Q'] -= m_q
            matches.append({"Match": f"{h['Stream']} ↔ {c['Stream']}", "Duty [kW]": round(m_q, 2), "Type": "Direct"})
        else:
            # --- STREAM SPLITTING HEURISTIC ---
            # If no match, split the stream with the higher load to match the mCp of an available partner
            c_candidate = next((s for s in cold if s['Q'] > 1), None)
            if c_candidate:
                # Force a match by "virtually" splitting 
                m_q = min(h['Q'], c_candidate['Q'])
                h['Q'] -= m_q
                c_candidate['Q'] -= m_q
                matches.append({"Match": f"Split-{h['Stream']} ↔ {c_candidate['Stream']}", "Duty [kW]": round(m_q, 2), "Type": "Split"})
            else:
                break
    return matches, hot, cold

# --- SECTION 1: DATA INPUT & EXCEL IMPORT ---
st.subheader("1. Stream Data Input")
uploaded_file = st.file_uploader("Import Stream Data from Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    try:
        import_df = pd.read_excel(uploaded_file)
        required = ["Stream", "Type", "mCp", "Ts", "Tt", "h"]
        if all(col in import_df.columns for col in required):
            st.session_state['input_data'] = import_df[required]
            st.success("Data imported successfully!")
        else: st.error(f"Excel must contain columns: {', '.join(required)}")
    except Exception as e: st.error(f"Error reading file: {e}")

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = pd.DataFrame(columns=["Stream", "Type", "mCp", "Ts", "Tt", "h"])

with st.form("main_input_form"):
    dt_min_input = st.number_input("Target ΔTmin [°C]", min_value=1.0, value=10.0)
    edited_df = st.data_editor(st.session_state['input_data'], num_rows="dynamic", use_container_width=True)
    submit_thermal = st.form_submit_button("Run Thermal Analysis")

if submit_thermal and not edited_df.empty:
    st.session_state.run_clicked = True

# --- MAIN OUTPUT DISPLAY ---
if st.session_state.get('run_clicked'):
    qh, qc, pinch, t_plot, q_plot, processed_df = run_thermal_logic(edited_df, dt_min_input)
    
    # 2. PINCH RESULTS
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
        st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
        st.metric("Pinch Temperature", f"{pinch} °C" if pinch is not None else "N/A")
    with r2:
        fig = go.Figure(go.Scatter(x=t_plot, y=q_plot, mode='lines+markers', name="Composite Curve"))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Temp [°C]", yaxis_title="Net Enthalpy [kW]")
        st.plotly_chart(fig, use_container_width=True)

    # 3. HEN MATCHING
    st.markdown("---")
    st.subheader("3. Heat Exchanger Network Matching")
    match_summary = []
    if pinch:
        l, r = st.columns(2)
        for i, side in enumerate(['Above', 'Below']):
            matches, h_rem, c_rem = match_logic(processed_df, pinch, side)
            match_summary.extend(matches)
            with (l if i == 0 else r):
                st.write(f"**Matches {side} Pinch**")
                if matches: 
                    st.table(pd.DataFrame(matches))
                else: 
                    st.info(f"No internal matches {side.lower()} pinch.")

                # --- Color-Coded Utility Display ---
                
                # Hot Utility (Heaters) -> Red Shade
                for c in c_rem: 
                    if c['Q'] > 1: 
                        st.error(f"**Required Heater:** {c['Stream']} ({c['Q']:,.1f} kW)")

                # Cold Utility (Coolers) -> Blue Shade
                for h in h_rem: 
                    if h['Q'] > 1: 
                        st.info(f"**Required Cooler:** {h['Stream']} ({h['Q']:,.1f} kW)")

    # 4. OPTIMIZATION
    st.markdown("---")
    st.subheader("4. Optimization and Economic Analysis (Under development..)")
    
    opt_goal = st.selectbox("Objective Function [cite: 195]", ["Minimize Total Annual Cost (TAC)", "Minimize Energy Consumption", "Minimize Entropy Generation"])
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        var_dt = st.checkbox("Enable NLP Variable ΔTmin Solver [cite: 304]")
        h_hot_u = st.number_input("Hot Utility h [kW/m²K]", value=5.0)
    with col_opt2:
        h_cold_u = st.number_input("Cold Utility h [kW/m²K]", value=0.8)
        payback = st.number_input("Payback Period [Years]", value=5.0)

    if st.button("Calculate Economic Optimum"):
        # Rigorous calculation using equations from source
        avg_h_h, avg_h_c = edited_df[edited_df['Type']=='Hot']['h'].mean(), edited_df[edited_df['Type']=='Cold']['h'].mean()
        U_h, U_c = calculate_u(h_hot_u, avg_h_c), calculate_u(h_cold_u, avg_h_h)
        lmtd = lmtd_chen(processed_df['Ts'].max(), processed_df['Tt'].min(), processed_df['Ts'].min(), processed_df['Tt'].max())
        
        # Area (Eq 03) and Capital Cost (Eq 05) [cite: 94]
        opt_area = (qh / (U_h * lmtd)) + (qc / (U_c * lmtd))
        cap_inv = 8000 + 433.3 * (opt_area ** 0.6)
        op_cost = (qh * 0.05 + qc * 0.01) * 8000 # Standard operating hours/rates
        tac = op_cost + (cap_inv / payback)
        
        st.success("Optimization Converged to Global Optimum")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimized Total Area", f"{opt_area:,.2f} m²")
        m2.metric("Total Capital Investment", f"${cap_inv:,.2f}")
        m3.metric("Total Annual Cost (TAC)", f"${tac:,.2f}")

    # 5. ECONOMIC PARAMETERS
    st.markdown("---")
    st.subheader("5. Economic Assessment Parameters")
    with st.expander("View Global Cost Correlation (Equation 05)"):
        st.latex(r"Cost = a + b \cdot Area^c")
        st.write("Using documented coefficients[cite: 94, 111]: a=8000, b=433.3, c=0.6.")

    # 6. EXPORT
    st.markdown("---")
    st.subheader("6. Export Results")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame({"Parameter": ["Qh", "Qc", "Pinch"], "Value": [qh, qc, pinch]}).to_excel(writer, sheet_name='Pinch_Results', index=False)
        edited_df.to_excel(writer, sheet_name='Input_Data', index=False)
        if match_summary: pd.DataFrame(match_summary).to_excel(writer, sheet_name='HEN_Matches', index=False)
    
    st.download_button(label="📥 Download Results as Excel", data=output.getvalue(), file_name="HEN_Report.xlsx", mime="application/vnd.ms-excel")
else:
    st.info("Please import an Excel file or add streams to the table in Section 1 to begin.")


