import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs **Pinch Analysis**, **MER Matching with Stream Splitting**, and 
**Economic Optimization**.
""")
st.markdown("---")

# --- CORE MATH FUNCTIONS ---
def calculate_u(h1, h2):
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    theta1 = max(abs(t1 - t4), 0.01)
    theta2 = max(abs(t2 - t3), 0.01)
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # Temperature Shifting: Cold is shifted UP by dt
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)
    
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_mcp = df[(df['Type'] == 'Hot') & (df['S_Ts'] >= hi) & (df['S_Tt'] <= lo)]['mCp'].sum()
        c_mcp = df[(df['Type'] == 'Cold') & (df['S_Ts'] <= lo) & (df['S_Tt'] >= hi)]['mCp'].sum()
        intervals.append({'hi': hi, 'lo': lo, 'net': (h_mcp - c_mcp) * (hi - lo)})
    
    infeasible = [0] + list(pd.DataFrame(intervals)['net'].cumsum())
    qh_min = abs(min(min(infeasible), 0))
    feasible = [qh_min + val for val in infeasible]
    pinch_t = temps[feasible.index(0)] if 0 in feasible else None
    
    return qh_min, feasible[-1], pinch_t, temps, feasible, df

# --- ADDED: DGS-RWCE ALGORITHM & ECONOMIC INPUTS ---

# 1. HARD-CODED ALGORITHM PARAMETERS (DGS-RWCE Logic)
# These are internal values from the paper that govern the search behavior.
DGS_CONFIG = {
    "N_HD": 3,              # Number of hot main nodes
    "N_CD": 3,              # Number of cold main nodes
    "N_FH": 2,              # Number of hot stream branches
    "N_FC": 2,              # Number of cold stream branches
    "DELTA_L": 50.0,        # Max walk step for heat loads (Random Walk)
    "THETA": 1.0,           # Small perturbation increment for Q_DEP search
    "P_GEN": 0.01,          # Probability of generating a new heat exchanger
    "P_INCENTIVE": 0.005,   # Probability of 'Incentive Heat Load' (0.5%)
    "MAX_ITER": 100000,     # Number of iterations (Adjustable for speed)
    "ANNUAL_FACTOR": 0.2    # Typically used to annualize capital (e.g., 5-year payout)
}

# 2. USER-PROVIDED INPUTS (UI Integration)
# We will place these inside Section 4 of your Streamlit app.
def render_optimization_inputs():
    st.markdown("### 4. Optimization & Economics Parameters")
    
    with st.expander("Economic Coefficients (Plant Specific)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.number_input("Fixed Investment [a] ($)", value=8000.0)
            c_hu = st.number_input("Hot Utility Cost ($/kW·yr)", value=80.0)
        with col2:
            b = st.number_input("Area Coefficient [b] ($/m²)", value=800.0)
            c_cu = st.number_input("Cold Utility Cost ($/kW·yr)", value=20.0)
        with col3:
            c = st.number_input("Area Exponent [c]", value=0.8, step=0.01)
            
    return {
        "a": a, "b": b, "c": c, 
        "c_hu": c_hu, "c_cu": c_cu
    }

# 3. DATA CONVERSION UTILITY
# This converts your Streamlit 'edited_df' into a format the Optimizer can use.
def prepare_optimizer_data(df):
    hot_streams = df[df['Type'] == 'Hot'].to_dict('records')
    cold_streams = df[df['Type'] == 'Cold'].to_dict('records')
    return hot_streams, cold_streams

# --- END OF INPUTS SECTION ---

def match_logic_with_splitting(df, pinch_t, side):
    sub = df.copy()
    if side == 'Above':
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(lower=pinch_t), sub['S_Tt'].clip(lower=pinch_t)
    else:
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(upper=pinch_t), sub['S_Tt'].clip(upper=pinch_t)
    
    sub['Q_Total'] = sub['mCp'] * abs(sub['S_Ts'] - sub['S_Tt'])
    total_duties = sub.set_index('Stream')['Q_Total'].to_dict()
    sub['Q'] = sub['Q_Total']
    
    streams = sub[sub['Q'] > 0.1].to_dict('records')
    hot = [s for s in streams if s['Type'] == 'Hot']
    cold = [s for s in streams if s['Type'] == 'Cold']
    matches = []
    
    while any(h['Q'] > 1 for h in hot) and any(c['Q'] > 1 for c in cold):
        h = next(s for s in hot if s['Q'] > 1)
        c = next((s for s in cold if (s['mCp'] >= h['mCp'] if side=='Above' else h['mCp'] >= s['mCp']) and s['Q'] > 1), None)
        
        is_split = False
        if not c:
            c = next((s for s in cold if s['Q'] > 1), None)
            is_split = True
            
        if c:
            m_q = min(h['Q'], c['Q'])
            h_ratio = m_q / total_duties[h['Stream']] if total_duties[h['Stream']] > 0 else 0
            ratio_text = f"{round(h_ratio, 2)} " if h_ratio < 0.99 else ""
            match_str = f"{ratio_text}Stream {h['Stream']} ↔ {c['Stream']}"
            
            h['Q'] -= m_q
            c['Q'] -= m_q
            matches.append({
                "Match": match_str, 
                "Duty [kW]": round(m_q, 2), 
                "Type": "Split" if is_split or (0 < h_ratio < 0.99) else "Direct"
            })
        else:
            break
    return matches, hot, cold

def find_q_dep(h_stream, c_stream, econ_params, current_tac):
    """
    Finds the heat load Q where the new TAC equals the current TAC.
    Uses the 'Small Perturbation Method' from the paper.
    """
    # 1. Initialize variables
    q_ne = 1.0  # Start with a small 1kW load
    theta = DGS_CONFIG["THETA"]
    
    # Calculate U for this specific match
    u_match = calculate_u(h_stream['h'], c_stream['h'])
    
    # 2. Maximum possible heat load for this match (thermodynamic limit)
    # Q = mCp * (T_in - T_target)
    q_max_h = h_stream['mCp'] * (h_stream['Ts'] - h_stream['Tt'])
    q_max_c = c_stream['mCp'] * (c_stream['Tt'] - c_stream['Ts'])
    q_limit = min(q_max_h, q_max_c)

    # 3. Iteration loop
    while q_ne < q_limit:
        # Calculate resulting temperatures for LMTD
        tho = h_stream['Ts'] - (q_ne / h_stream['mCp'])
        tco = c_stream['Ts'] + (q_ne / c_stream['mCp'])
        
        # Check temperature feasibility (delta T > 0)
        if (h_stream['Ts'] - tco) <= 0 or (tho - c_stream['Ts']) <= 0:
            break
            
        # A. Calculate Capital Cost of this new unit
        lmtd = lmtd_chen(h_stream['Ts'], tho, c_stream['Ts'], tco)
        area = q_ne / (u_match * lmtd)
        investment = econ_params['a'] + econ_params['b'] * (area ** econ_params['c'])
        annualized_inv = investment * DGS_CONFIG['ANNUAL_FACTOR']
        
        # B. Calculate Revenue (Utility Savings)
        # Every kW recovered saves 1kW of Hot Utility AND 1kW of Cold Utility
        savings = q_ne * (econ_params['c_hu'] + econ_params['c_cu'])
        
        # C. Check Delta TAC
        # In a single-match case: Delta TAC = Investment - Savings
        delta_tac = annualized_inv - savings
        
        # If delta_tac is near zero or negative, we found a viable load!
        if delta_tac <= 0:
            return round(q_ne, 2)
            
        # Perturb: Increase Q and try again
        q_ne += np.random.uniform(0.5, 1.5) * theta
        
    return None # No equilibrium point found for this stream pair

# --- SECTION 1: DATA INPUT & EXCEL IMPORT ---
    st.subheader("1. Stream Data Input")
uploaded_file = st.file_uploader("Import Stream Data from Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    try:
        import_df = pd.read_excel(uploaded_file)
        st.session_state['input_data'] = import_df
        st.success("Data imported successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")

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
    
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
        st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
        st.metric("Pinch Temperature (Hot)", f"{pinch} °C" if pinch is not None else "N/A")
        st.metric("Pinch Temperature (Cold)", f"{pinch - dt_min_input} °C" if pinch is not None else "N/A")
    with r2:
        fig = go.Figure(go.Scatter(x=q_plot, y=t_plot, mode='lines+markers', name="GCC"))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Net Heat Flow [kW]", yaxis_title="Shifted Temp [°C]")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Heat Exchanger Network Matching (MER)")
    
    match_summary = []
    if pinch is not None:
        l, r = st.columns(2)
        for i, side in enumerate(['Above', 'Below']):
            matches, h_rem, c_rem = match_logic_with_splitting(processed_df, pinch, side)
            match_summary.extend(matches)
            with (l if i == 0 else r):
                st.write(f"**Matches {side} Pinch**")
                if matches:
                    st.dataframe(pd.DataFrame(matches), use_container_width=True)
                else:
                    st.info("No internal matches possible.")
                for c in c_rem: 
                    if c['Q'] > 1: st.error(f"Required Heater: {c['Stream']} ({c['Q']:,.1f} kW)")
                for h in h_rem: 
                    if h['Q'] > 1: st.info(f"Required Cooler: {h['Stream']} ({h['Q']:,.1f} kW)")
    
    st.markdown("---")
    st.markdown("---")
    st.subheader("4. Optimization and Economic Analysis")
    
    # 1. Call the new input function
    econ_params = render_optimization_inputs()
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        h_hot_u = st.number_input("Hot Utility h [kW/m²K]", value=5.0)
    with col_opt2:
        h_cold_u = st.number_input("Cold Utility h [kW/m²K]", value=0.8)

if st.button("Calculate Economic Optimum"):
        # --- VALIDATION CHECK ---
        # Check if 'h' column exists and if it contains any zeros or NaN values
        if 'h' not in edited_df.columns or edited_df['h'].isnull().any() or (edited_df['h'] <= 0).any():
            st.warning("Individual heat transfer coefficients are necessary for this part. Please fill them in the input table before trying again.")
        else:
            # 1. Baseline TAC Calculation
            avg_h_h = edited_df[edited_df['Type']=='Hot']['h'].mean()
            avg_h_c = edited_df[edited_df['Type']=='Cold']['h'].mean()
            
            U_h = calculate_u(h_hot_u, avg_h_c)
            U_c = calculate_u(h_cold_u, avg_h_h)
            
            lmtd_base = lmtd_chen(processed_df['Ts'].max(), processed_df['Tt'].min(), 
                                  processed_df['Ts'].min(), processed_df['Tt'].max())
            
            opt_area = (qh / (U_h * lmtd_base)) + (qc / (U_c * lmtd_base))
            cap_inv = econ_params['a'] + econ_params['b'] * (opt_area ** econ_params['c'])
            annual_opex = (qh * econ_params['c_hu']) + (qc * econ_params['c_cu'])
            baseline_tac = annual_opex + (cap_inv * DGS_CONFIG['ANNUAL_FACTOR'])
            
            # Display Results
            st.markdown("#### Baseline Economic Breakdown")
            m1, m2, m3 = st.columns(3)
            m1.metric("Est. Total Area", f"{opt_area:,.2f} m²")
            m2.metric("Total Capital (CAPEX)", f"${cap_inv:,.2f}")
            m3.metric("Baseline TAC", f"${baseline_tac:,.2f}/yr")

            st.markdown("---")

            # 2. DGS-RWCE Search Logic
            st.write("### Dynamic Generation Strategy: Finding Viable Matches")
            hot_streams, cold_streams = prepare_optimizer_data(edited_df)
            found_matches = []
            
            for hs in hot_streams:
                for cs in cold_streams:
                    q_dep = find_q_dep(hs, cs, econ_params, baseline_tac)
                    if q_dep:
                        found_matches.append({
                            "Hot Stream": hs['Stream'],
                            "Cold Stream": cs['Stream'],
                            "Recommended Load [kW]": q_dep,
                            "Type": "DGS Equilibrium" if q_dep < 0.7 * hs['mCp']*(hs['Ts']-hs['Tt']) else "Incentive Strategy"
                        })
            
            if found_matches:
                st.success(f"DGS-RWCE identified {len(found_matches)} cost-effective equipment generations!")
                st.dataframe(pd.DataFrame(found_matches), use_container_width=True)
            else:
                st.info("No cost-neutral matches found with current parameters.")
    st.subheader("5. Export Results")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Save Match results
        if match_summary:
            pd.DataFrame(match_summary).to_excel(writer, sheet_name='HEN_Matches', index=False)
        # Save input data for reference
        edited_df.to_excel(writer, sheet_name='Input_Data', index=False)
        # Save Pinch metrics
        pd.DataFrame({"Metric": ["Qh", "Qc", "Pinch Hot", "Pinch Cold"], "Value": [qh, qc, pinch, pinch-dt_min_input]}).to_excel(writer, sheet_name='Pinch_Summary', index=False)
    
    st.download_button(label="📥 Download HEN Report (Excel)", data=output.getvalue(), file_name="HEN_Full_Analysis.xlsx", mime="application/vnd.ms-excel")



