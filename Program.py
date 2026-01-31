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
    if h1 <= 0 or h2 <= 0:
        return 0
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

# --- DGS-RWCE ALGORITHM & ECONOMIC INPUTS ---
DGS_CONFIG = {
    "N_HD": 3,
    "N_CD": 3,
    "N_FH": 2,
    "N_FC": 2,
    "DELTA_L": 50.0,
    "THETA": 1.0,
    "P_GEN": 0.01,
    "P_INCENTIVE": 0.005,
    "MAX_ITER": 100000,
    "ANNUAL_FACTOR": 0.2
}

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
    return {"a": a, "b": b, "c": c, "c_hu": c_hu, "c_cu": c_cu}

def prepare_optimizer_data(df):
    hot_streams = df[df['Type'] == 'Hot'].to_dict('records')
    cold_streams = df[df['Type'] == 'Cold'].to_dict('records')
    return hot_streams, cold_streams

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
    q_ne = 1.0
    theta = DGS_CONFIG["THETA"]
    u_match = calculate_u(h_stream.get('h', 0), c_stream.get('h', 0))
    if u_match <= 0: return None

    q_max_h = h_stream['mCp'] * (h_stream['Ts'] - h_stream['Tt'])
    q_max_c = c_stream['mCp'] * (c_stream['Tt'] - c_stream['Ts'])
    q_limit = min(q_max_h, q_max_c)

    while q_ne < q_limit:
        tho = h_stream['Ts'] - (q_ne / h_stream['mCp'])
        tco = c_stream['Ts'] + (q_ne / c_stream['mCp'])
        if (h_stream['Ts'] - tco) <= 0 or (tho - c_stream['Ts']) <= 0: break
        lmtd = lmtd_chen(h_stream['Ts'], tho, c_stream['Ts'], tco)
        area = q_ne / (u_match * lmtd)
        annualized_inv = (econ_params['a'] + econ_params['b'] * (area ** econ_params['c'])) * DGS_CONFIG['ANNUAL_FACTOR']
        savings = q_ne * (econ_params['c_hu'] + econ_params['c_cu'])
        if (annualized_inv - savings) <= 0: return round(q_ne, 2)
        q_ne += np.random.uniform(0.5, 1.5) * theta
    return None

def run_random_walk(initial_matches, hot_streams, cold_streams, econ_params):
    import copy
    best_matches = copy.deepcopy(initial_matches)
    
    def calculate_network_tac(matches):
        total_inv = 0
        total_q_recovered = 0
        for m in matches:
            q = m['Recommended Load [kW]']
            h_s = next(s for s in hot_streams if s['Stream'] == m['Hot Stream'])
            c_s = next(s for s in cold_streams if s['Stream'] == m['Cold Stream'])
            u = calculate_u(h_s['h'], c_s['h'])
            tho = h_s['Ts'] - (q / h_s['mCp'])
            tco = c_s['Ts'] + (q / c_s['mCp'])
            if (h_s['Ts'] - tco) <= 0.1 or (tho - c_s['Ts']) <= 0.1: return float('inf')
            lmtd = lmtd_chen(h_s['Ts'], tho, c_s['Ts'], tco)
            area = q / (u * lmtd)
            inv = econ_params['a'] + econ_params['b'] * (area ** econ_params['c'])
            total_inv += inv
            total_q_recovered += q
        return (total_inv * DGS_CONFIG['ANNUAL_FACTOR']) - (total_q_recovered * (econ_params['c_hu'] + econ_params['c_cu']))

    current_best_score = calculate_network_tac(best_matches)
    for _ in range(500):
        if not best_matches: break
        idx = np.random.randint(0, len(best_matches))
        original_q = best_matches[idx]['Recommended Load [kW]']
        step = np.random.uniform(-1, 1) * DGS_CONFIG['DELTA_L']
        new_q = max(1.0, original_q + step)
        best_matches[idx]['Recommended Load [kW]'] = new_q
        new_score = calculate_network_tac(best_matches)
        if new_score < current_best_score: current_best_score = new_score
        else: best_matches[idx]['Recommended Load [kW]'] = original_q
    return best_matches, current_best_score

# --- SECTION 1: DATA INPUT ---
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
                if matches: st.dataframe(pd.DataFrame(matches), use_container_width=True)
                else: st.info("No internal matches possible.")
                for c in c_rem: 
                    if c['Q'] > 1: st.error(f"Required Heater: {c['Stream']} ({c['Q']:,.1f} kW)")
                for h in h_rem: 
                    if h['Q'] > 1: st.info(f"Required Cooler: {h['Stream']} ({h['Q']:,.1f} kW)")

    st.markdown("---")
    st.subheader("4. Optimization and Economic Analysis")
    econ_params = render_optimization_inputs()
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1: h_hot_u = st.number_input("Hot Utility h [kW/m²K]", value=5.0)
    with col_opt2: h_cold_u = st.number_input("Cold Utility h [kW/m²K]", value=0.8)

    # Initialize variables to avoid NameErrors in Export Section
    found_matches = []
    refined_matches = []
    savings = 0

    if st.button("Calculate Economic Optimum"):
        if 'h' not in edited_df.columns or edited_df['h'].isnull().any() or (edited_df['h'] <= 0).any():
            st.warning("Individual heat transfer coefficients are necessary for this part. Please fill them in the input table before trying again.")
        else:
            # Baseline TAC Calculation
            avg_h_h = edited_df[edited_df['Type']=='Hot']['h'].mean()
            avg_h_c = edited_df[edited_df['Type']=='Cold']['h'].mean()
            U_h, U_c = calculate_u(h_hot_u, avg_h_c), calculate_u(h_cold_u, avg_h_h)
            lmtd_base = lmtd_chen(processed_df['Ts'].max(), processed_df['Tt'].min(), processed_df['Ts'].min(), processed_df['Tt'].max())
            opt_area = (qh / (U_h * lmtd_base)) + (qc / (U_c * lmtd_base))
            cap_inv = econ_params['a'] + econ_params['b'] * (opt_area ** econ_params['c'])
            annual_opex = (qh * econ_params['c_hu']) + (qc * econ_params['c_cu'])
            baseline_tac = annual_opex + (cap_inv * DGS_CONFIG['ANNUAL_FACTOR'])
            
            # Find Viable Starting Matches
            hot_streams, cold_streams = prepare_optimizer_data(edited_df)
            for hs in hot_streams:
                for cs in cold_streams:
                    q_dep = find_q_dep(hs, cs, econ_params, baseline_tac)
                    if q_dep:
                        found_matches.append({
                            "Hot Stream": hs['Stream'], "Cold Stream": cs['Stream'],
                            "Recommended Load [kW]": q_dep,
                            "Type": "DGS Equilibrium" if q_dep < 0.7 * hs['mCp']*(hs['Ts']-hs['Tt']) else "Incentive Strategy"
                        })
            
            if found_matches:
                with st.status("Evolving Network via Random Walk...", expanded=True) as status:
                    refined_matches, savings = run_random_walk(found_matches, hot_streams, cold_streams, econ_params)
                    status.update(label="Evolution Complete!", state="complete", expanded=False)
                
                st.markdown("### Optimized Heat Recovery Network")
                st.dataframe(pd.DataFrame(refined_matches), use_container_width=True)
                st.metric("Potential Extra Savings from Optimization", f"${abs(savings):,.2f}/yr")
            else:
                st.info("No cost-neutral matches found with current parameters.")

    # --- SECTION 5: EXPORT RESULTS ---
    st.markdown("---")
    st.subheader("5. Export Results")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        final_matches = refined_matches if refined_matches else match_summary
        if final_matches:
            pd.DataFrame(final_matches).to_excel(writer, sheet_name='HEN_Matches', index=False)
        edited_df.to_excel(writer, sheet_name='Input_Data', index=False)
        pd.DataFrame({"Metric": ["Qh", "Qc", "Pinch Hot", "Pinch Cold"], 
                      "Value": [qh, qc, pinch, pinch-dt_min_input if pinch else None]}).to_excel(writer, sheet_name='Pinch_Summary', index=False)
    
    st.download_button(label="📥 Download HEN Report (Excel)", 
                       data=output.getvalue(), 
                       file_name="HEN_Full_Analysis.xlsx", 
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
