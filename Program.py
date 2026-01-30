import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pygad

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs **Pinch Analysis**, **MER Matching with Stream Splitting**, and 
**Economic Optimization** using **Genetic Algorithm (PyGAD)**.
""")
st.markdown("---")

# --- CORE MATH FUNCTIONS ---
def calculate_u(h1, h2, h_unit_factor=1.0):
    if h1 <= 0 or h2 <= 0:
        return 0
    h1_converted = h1 * h_unit_factor
    h2_converted = h2 * h_unit_factor
    return 1 / ((1/h1_converted) + (1/h2_converted))

def lmtd_chen(t1, t2, t3, t4):
    theta1 = max(t1 - t4, 0.001) 
    theta2 = max(t2 - t3, 0.001)
    if abs(theta1 - theta2) < 0.01: 
        return theta1
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def validate_dataframe(df):
    required_cols = ["Stream", "Type", "mCp", "Ts", "Tt", "h"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    if df.empty:
        return False, "DataFrame is empty. Please add stream data."
    return True, "Valid"

def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # Shift temperatures
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

# --- GENETIC ALGORITHM HELPERS ---
GA_CONFIG = {
    "num_generations": 100,
    "num_parents_mating": 10,
    "sol_per_pop": 50,
    "mutation_percent_genes": 15,
    "min_split_ratio": 0.10
}

def calculate_hex_capital(area, econ_params):
    if econ_params.get("formula") == "benchmark":
        return econ_params["cost_coef"] * (area ** econ_params["cost_exp"])
    else:
        total_capital = econ_params["a"] + econ_params["b"] * (area ** econ_params["c"])
        return total_capital * econ_params["ann_factor"]

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
        h = next((s for s in hot if s['Q'] > 1), None)
        if not h: break
        c = next((s for s in cold if (s['mCp'] >= h['mCp'] if side=='Above' else h['mCp'] >= s['mCp']) and s['Q'] > 1), None)
        is_split = False
        if not c:
            c = next((s for s in cold if s['Q'] > 1), None)
            is_split = True
            
        if c:
            m_q = min(h['Q'], c['Q'])
            h_ratio = m_q / total_duties[h['Stream']] if total_duties[h['Stream']] > 0 else 0
            if h_ratio >= GA_CONFIG["min_split_ratio"] or h_ratio >= 0.99:
                ratio_text = f"{round(h_ratio, 2)} " if h_ratio < 0.99 else ""
                matches.append({
                    "Match": f"{ratio_text}Stream {h['Stream']} ↔ {c['Stream']}",
                    "Duty [kW]": round(m_q, 2), 
                    "Type": "Split" if is_split or (0 < h_ratio < 0.99) else "Direct",
                    "Side": side
                })
                h['Q'] -= m_q
                c['Q'] -= m_q
            else: break
        else: break
    return matches, hot, cold

def calculate_mer_capital_properly(match_summary, processed_df, econ_params, pinch_t, dt_min, qh_util, qc_util):
    cap_mer = 0
    h_factor = econ_params.get('h_factor', 1.0)
    pinch_real_hot = pinch_t
    pinch_real_cold = pinch_t - dt_min
    
    for m in match_summary:
        duty = m['Duty [kW]']
        if duty <= 0: continue
        side = m.get('Side', 'Above')
        
        try:
            match_str = m['Match']
            if ' ' in match_str and match_str.split()[0].replace('.', '').isdigit():
                match_str = ' '.join(match_str.split()[1:])
            
            parts = match_str.replace('Stream ', '').split(' ↔ ')
            h_id, c_id = parts[0].strip(), parts[1].strip()
            
            h_stream = processed_df[(processed_df['Stream'].astype(str) == h_id) & (processed_df['Type'] == 'Hot')].iloc[0]
            c_stream = processed_df[(processed_df['Stream'].astype(str) == c_id) & (processed_df['Type'] == 'Cold')].iloc[0]
            
            if side == 'Above':
                thi, tci = h_stream['Ts'], max(c_stream['Ts'], pinch_real_cold)
            else:
                thi, tci = min(h_stream['Ts'], pinch_real_hot), c_stream['Ts']
            
            tho = thi - (duty / h_stream['mCp'])
            tco = tci + (duty / c_stream['mCp'])
            
            u = calculate_u(h_stream['h'], c_stream['h'], h_factor)
            lmtd = lmtd_chen(thi, tho, tci, tco)
            if u > 0 and lmtd > 0:
                cap_mer += calculate_hex_capital(duty / (u * lmtd), econ_params)
        except: continue
        
    # Utility Heaters/Coolers
    h_hu, h_cu = econ_params.get('h_hu', 4.8), econ_params.get('h_cu', 1.6)
    if qh_util > 0.1:
        u_hu = calculate_u(h_hu, processed_df[processed_df['Type'] == 'Cold']['h'].mean(), h_factor)
        if u_hu > 0: cap_mer += calculate_hex_capital(qh_util / (u_hu * 50.0), econ_params)
    if qc_util > 0.1:
        u_cu = calculate_u(processed_df[processed_df['Type'] == 'Hot']['h'].mean(), h_cu, h_factor)
        if u_cu > 0: cap_mer += calculate_hex_capital(qc_util / (u_cu * 30.0), econ_params)
    
    return cap_mer

# --- GA IMPLEMENTATION ---
def fitness_function(ga_instance, solution, solution_idx):
    match_pairs = st.session_state['ga_match_pairs']
    hot_streams = st.session_state['ga_hot_streams']
    cold_streams = st.session_state['ga_cold_streams']
    econ_params = st.session_state['ga_econ_params']
    dt_min = st.session_state['ga_dt_min']
    
    rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
    rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in cold_streams}
    total_inv = 0
    h_factor = econ_params.get('h_factor', 1.0)
    
    for i, pair in enumerate(match_pairs):
        q = solution[i] * pair['max_duty']
        if q <= 0.1: continue
        hs, cs = pair['hot_stream'], pair['cold_stream']
        
        if q > rem_h[hs['Stream']] + 0.1 or q > rem_c[cs['Stream']] + 0.1: return -1e10
        
        tho, tco = hs['Ts'] - (q / hs['mCp']), cs['Ts'] + (q / cs['mCp'])
        if (hs['Ts'] - tco) < dt_min or (tho - cs['Ts']) < dt_min: return -1e10
        
        rem_h[hs['Stream']] -= q
        rem_c[cs['Stream']] -= q
        
        u = calculate_u(hs['h'], cs['h'], h_factor)
        lmtd = lmtd_chen(hs['Ts'], tho, cs['Ts'], tco)
        if u <= 0 or lmtd <= 0: return -1e10
        total_inv += calculate_hex_capital(q / (u * lmtd), econ_params)

    opex = (sum(max(0, v) for v in rem_c.values()) * econ_params['c_hu']) + \
           (sum(max(0, v) for v in rem_h.values()) * econ_params['c_cu'])
    return -(opex + total_inv)

# --- UI LOGIC ---
st.subheader("1. Stream Data Input")
if st.button("Load Example Data"):
    st.session_state['input_data'] = pd.DataFrame({
        "Stream": [1, 2, 3, 4], "Type": ["Hot", "Hot", "Cold", "Cold"],
        "mCp": [10.0, 15.0, 13.0, 12.0], "Ts": [180, 150, 60, 90],
        "Tt": [60, 30, 160, 140], "h": [0.5, 0.6, 0.4, 0.5]
    })

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = pd.DataFrame(columns=["Stream", "Type", "mCp", "Ts", "Tt", "h"])

with st.form("main_input_form"):
    dt_min_input = st.number_input("Target ΔTmin [°C]", min_value=1.0, value=10.0)
    edited_df = st.data_editor(st.session_state['input_data'], num_rows="dynamic", use_container_width=True)
    if st.form_submit_button("Run Thermal Analysis"):
        is_valid, msg = validate_dataframe(edited_df)
        if is_valid:
            st.session_state.run_clicked, st.session_state.edited_df, st.session_state.dt_min = True, edited_df, dt_min_input
        else: st.error(msg)

if st.session_state.get('run_clicked'):
    qh, qc, pinch, t_plot, q_plot, processed_df = run_thermal_logic(st.session_state.edited_df, st.session_state.dt_min)
    
    st.subheader("2. Pinch Analysis Result")
    col1, col2 = st.columns([1, 2])
    col1.metric("Hot Utility", f"{qh:,.2f} kW")
    col1.metric("Cold Utility", f"{qc:,.2f} kW")
    col1.metric("Pinch Temp", f"{pinch} °C" if pinch else "N/A")
    
    fig = go.Figure(go.Scatter(x=q_plot, y=t_plot, mode='lines+markers', name="GCC"))
    col2.plotly_chart(fig, use_container_width=True)

    # Simplified Economic Inputs
    econ_params = {"formula": "benchmark", "cost_coef": 1000.0, "cost_exp": 0.6, "c_hu": 80.0, "c_cu": 20.0, "h_factor": 1.0}
    
    # MER Calc
    matches_all = []
    if pinch:
        for side in ['Above', 'Below']:
            m, _, _ = match_logic_with_splitting(processed_df, pinch, side)
            matches_all.extend(m)
    
    cap_mer = calculate_mer_capital_properly(matches_all, processed_df, econ_params, pinch, st.session_state.dt_min, qh, qc)
    st.metric("MER Total Annual Cost", f"${(cap_mer + (qh*80) + (qc*20)):,.2f}/yr")
    if matches_all: st.table(pd.DataFrame(matches_all))
