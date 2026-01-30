import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import copy

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
    theta1 = max(t1 - t4, 0.001) 
    theta2 = max(t2 - t3, 0.001)
    if abs(theta1 - theta2) < 0.01: return theta1
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
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
    "N_HD": 3, "N_CD": 3, "N_FH": 2, "N_FC": 2,
    "DELTA_L": 50.0, "THETA": 1.0, "P_GEN": 0.01,
    "P_INCENTIVE": 0.005, "MAX_ITER": 100000, "ANNUAL_FACTOR": 0.2
}

def render_optimization_inputs():
    st.markdown("### 4. Optimization & Economics Parameters")
    with st.expander("Economic Coefficients (Plant Specific)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.number_input("Fixed Investment [a] ($)", value=8000.0)
            c_hu = st.number_input("Hot Utility Cost ($/kW·yr)", value=80.0)
        with col2:
            b = st.number_input("Area Coefficient [b] ($/m²)", value=1200.0)
            c_cu = st.number_input("Cold Utility Cost ($/kW·yr)", value=20.0)
        with col3:
            c = st.number_input("Area Exponent [c]", value=0.6, step=0.01)
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

def find_q_dep(h_stream, c_stream, econ_params, dt_min):
    q_ne = 1.0
    theta = DGS_CONFIG["THETA"]
    u_match = calculate_u(h_stream.get('h', 0), c_stream.get('h', 0))
    if u_match <= 0: return None

    q_max_h = h_stream['mCp'] * abs(h_stream['Ts'] - h_stream['Tt'])
    q_max_c = c_stream['mCp'] * abs(c_stream['Tt'] - c_stream['Ts'])
    q_limit = min(q_max_h, q_max_c)

    while q_ne < q_limit:
        tho = h_stream['Ts'] - (q_ne / h_stream['mCp'])
        tco = c_stream['Ts'] + (q_ne / c_stream['mCp'])
        if (h_stream['Ts'] - tco) < dt_min or (tho - c_stream['Ts']) < dt_min: break
        
        lmtd = lmtd_chen(h_stream['Ts'], tho, c_stream['Ts'], tco)
        area = q_ne / (u_match * lmtd)
        annualized_inv = (econ_params['a'] + econ_params['b'] * (area ** econ_params['c'])) * DGS_CONFIG['ANNUAL_FACTOR']
        savings = q_ne * (econ_params['c_hu'] + econ_params['c_cu'])
        if (annualized_inv - savings) <= 0: return round(q_ne, 2)
        q_ne += np.random.uniform(0.5, 1.5) * theta
    return None

def run_random_walk(initial_matches, hot_streams, cold_streams, econ_params, dt_min):
    best_matches = copy.deepcopy(initial_matches)
    
    # --- STREAM HEAT LOAD TRACKING LOGIC ---
    def calculate_network_tac(matches):
        rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
        rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in cold_streams}
        total_inv = 0
        
        for m in matches:
            q = m['Recommended Load [kW]']
            if q <= 0: continue
            
            h_s = next(s for s in hot_streams if s['Stream'] == m['Hot Stream'])
            c_s = next(s for s in cold_streams if s['Stream'] == m['Cold Stream'])
            
            # Constraints: Cannot exchange more than stream availability
            if q > rem_h[m['Hot Stream']] + 0.1 or q > rem_c[m['Cold Stream']] + 0.1:
                return float('inf')

            tho = h_s['Ts'] - (q / h_s['mCp'])
            tco = c_s['Ts'] + (q / c_s['mCp'])
            
            if (h_s['Ts'] - tco) < dt_min or (tho - c_s['Ts']) < dt_min: 
                return float('inf')
            
            rem_h[m['Hot Stream']] -= q
            rem_c[m['Cold Stream']] -= q
            
            u = calculate_u(h_s['h'], c_s['h'])
            lmtd = lmtd_chen(h_s['Ts'], tho, c_s['Ts'], tco)
            area = q / (u * lmtd)
            total_inv += (econ_params['a'] + econ_params['b'] * (area ** econ_params['c']))

        actual_qh = sum(max(0, val) for val in rem_c.values())
        actual_qc = sum(max(0, val) for val in rem_h.values())
        opex = (actual_qh * econ_params['c_hu']) + (actual_qc * econ_params['c_cu'])
        return opex + (total_inv * DGS_CONFIG['ANNUAL_FACTOR'])

    current_best_score = calculate_network_tac(best_matches)
    for _ in range(2000):
        if not best_matches: break
        idx = np.random.randint(0, len(best_matches))
        original_q = best_matches[idx]['Recommended Load [kW]']
        step = np.random.uniform(-1, 1) * DGS_CONFIG['DELTA_L']
        best_matches[idx]['Recommended Load [kW]'] = max(0.0, original_q + step)
        
        new_score = calculate_network_tac(best_matches)
        if new_score < current_best_score:
            current_best_score = new_score
        else:
            best_matches[idx]['Recommended Load [kW]'] = original_q
            
    return [m for m in best_matches if m['Recommended Load [kW]'] > 0.1], current_best_score

# --- UI LOGIC ---
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

if st.session_state.get('run_clicked'):
    qh, qc, pinch, t_plot, q_plot, processed_df = run_thermal_logic(edited_df, dt_min_input)
    
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
        st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
        st.metric("Pinch Temperature", f"{pinch} °C" if pinch is not None else "N/A")
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

    econ_params = render_optimization_inputs() 
    
    # MER Baseline Calc
    total_mer_q = sum(m['Duty [kW]'] for m in match_summary)
    area_mer = total_mer_q / (0.5 * 20.0) # Estimated
    cap_mer = econ_params['a'] + econ_params['b'] * (area_mer ** econ_params['c'])
    ann_cap_mer = cap_mer * DGS_CONFIG['ANNUAL_FACTOR']
    opex_mer = (qh * econ_params['c_hu']) + (qc * econ_params['c_cu'])
    tac_mer = opex_mer + ann_cap_mer

    st.markdown("#### MER Economic Breakdown")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Capital Cost", f"${cap_mer:,.2f}", f"(${ann_cap_mer:,.2f}/yr)")
    m_col2.metric("Annual Operating Cost", f"${opex_mer:,.2f}/yr")
    m_col3.metric("Total Annual Cost (TAC)", f"${tac_mer:,.2f}/yr")

    st.markdown("---")
    st.subheader("4. Optimization and Economic Analysis")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1: h_hot_u = st.number_input("Hot Utility h [kW/m²K]", value=5.0)
    with col_opt2: h_cold_u = st.number_input("Cold Utility h [kW/m²K]", value=0.8)

    total_q_h_base = edited_df[edited_df['Type']=='Cold'].apply(lambda x: x['mCp']*abs(x['Ts']-x['Tt']), axis=1).sum()
    total_q_c_base = edited_df[edited_df['Type']=='Hot'].apply(lambda x: x['mCp']*abs(x['Ts']-x['Tt']), axis=1).sum()

    if st.button("Calculate Economic Optimum"):
        hot_streams, cold_streams = prepare_optimizer_data(edited_df)
        found_matches = []
        
        # Search for feasible pairings
        for hs in hot_streams:
            for cs in cold_streams:
                q_dep = find_q_dep(hs, cs, econ_params, dt_min_input)
                if q_dep:
                    found_matches.append({
                        "Hot Stream": hs['Stream'], "Cold Stream": cs['Stream'],
                        "Recommended Load [kW]": q_dep
                    })
        
        if found_matches:
            with st.status("Evolving Network via Random Walk...", expanded=True) as status:
                refined_matches, tac_opt = run_random_walk(found_matches, hot_streams, cold_streams, econ_params, dt_min_input)
                status.update(label="Evolution Complete!", state="complete", expanded=False)
            
            # --- FINAL CALCULATION & FORMATTING ---
            display_matches = []
            final_cap = 0
            rem_h_final = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
            
            for m in refined_matches:
                q = m['Recommended Load [kW]']
                hs = next(s for s in hot_streams if s['Stream'] == m['Hot Stream'])
                cs = next(s for s in cold_streams if s['Stream'] == m['Cold Stream'])
                
                # Split ratio calc
                ratio = q / (hs['mCp'] * abs(hs['Ts'] - hs['Tt']))
                ratio_text = f"{round(ratio, 2)} " if ratio < 0.99 else ""
                
                u = calculate_u(hs['h'], cs['h'])
                tho = hs['Ts'] - (q / hs['mCp'])
                tco = cs['Ts'] + (q / cs['mCp'])
                l_val = lmtd_chen(hs['Ts'], tho, cs['Ts'], tco)
                area = q / (u * l_val)
                final_cap += (econ_params['a'] + econ_params['b'] * (area ** econ_params['c']))
                
                display_matches.append({
                    "Match": f"{ratio_text}Stream {hs['Stream']} ↔ {cs['Stream']}",
                    "Duty [kW]": round(q, 2),
                    "Area [m²]": round(area, 2)
                })

            ann_cap_opt = final_cap * DGS_CONFIG['ANNUAL_FACTOR']
            actual_qh_opt = max(0, total_q_h_base - sum(m['Recommended Load [kW]'] for m in refined_matches)) # Approx for summary
            opex_opt = tac_opt - ann_cap_opt

            st.dataframe(pd.DataFrame(display_matches), use_container_width=True)

            o_col1, o_col2, o_col3 = st.columns(3)
            o_col1.metric("Capital Cost", f"${final_cap:,.2f}", f"(${ann_cap_opt:,.2f}/yr)")
            o_col2.metric("Annual Operating Cost", f"${opex_opt:,.2f}/yr")
            o_col3.metric("Total Annual Cost (TAC)", f"${tac_opt:,.2f}/yr")

            st.markdown("---")
            st.subheader("5. Comparison")
            opex_no_int = (total_q_h_base * econ_params['c_hu']) + (total_q_c_base * econ_params['c_cu'])
            comparison_df = pd.DataFrame({
                "Metric": ["Capital Cost ($)", "Operating Cost ($/yr)", "TAC ($/yr)"],
                "No Integration": ["0.00", f"{opex_no_int:,.2f}", f"{opex_no_int:,.2f}"],
                "MER Setup": [f"{cap_mer:,.2f}", f"{opex_mer:,.2f}", f"{tac_mer:,.2f}"],
                "Optimized": [f"{final_cap:,.2f}", f"{opex_opt:,.2f}", f"{tac_opt:,.2f}"]
            })
            st.table(comparison_df)
