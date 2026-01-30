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
**Economic Optimization** using **Genetic Algorithm (PyGAD)** based on literature-standard TAC formulas.
""")
st.markdown("---")

# --- CORE MATH FUNCTIONS ---
def calculate_u(h1, h2, h_unit_factor=1.0):
    """Calculate overall heat transfer coefficient with unit conversion."""
    if h1 <= 0 or h2 <= 0:
        return 0
    h1_converted = h1 * h_unit_factor
    h2_converted = h2 * h_unit_factor
    return 1 / ((1/h1_converted) + (1/h2_converted))

def lmtd_chen(t1, t2, t3, t4):
    """Chen (1987) approximation for LMTD to prevent numerical errors with tiny temp differences."""
    theta1 = max(t1 - t4, 0.001) 
    theta2 = max(t2 - t3, 0.001)
    if abs(theta1 - theta2) < 0.01: 
        return theta1
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def validate_dataframe(df):
    """Validate that the DataFrame has all required columns and valid numeric types."""
    required_cols = ["Stream", "Type", "mCp", "Ts", "Tt", "h"]
    if not all(col in df.columns for col in required_cols):
        return False, "Missing required columns."
    if df.empty:
        return False, "DataFrame is empty."
    return True, "Valid"

def run_thermal_logic(df, dt):
    """Perform pinch analysis to find minimum utility requirements."""
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # Shift temperatures based on Type
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)
    
    # Temperature intervals for GCC
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

# --- GENETIC ALGORITHM CONFIGURATION ---
GA_CONFIG = {
    "num_generations": 100,
    "num_parents_mating": 10,
    "sol_per_pop": 50,
    "mutation_percent_genes": 15,
    "min_split_ratio": 0.10 
}

def calculate_hex_capital(area, econ_params):
    """Calculate annual capital cost ($/yr) for a single heat exchanger."""
    if econ_params.get("formula") == "benchmark":
        return econ_params["cost_coef"] * (area ** econ_params["cost_exp"])
    else:
        total_capital = econ_params["a"] + econ_params["b"] * (area ** econ_params["c"])
        return total_capital * econ_params["ann_factor"]

def calculate_mer_capital_properly(match_summary, processed_df, econ_params, pinch_t, dt_min, qh_util, qc_util):
    """Calculate MER capital cost by splitting systems at the pinch."""
    cap_mer = 0
    h_factor = econ_params.get('h_factor', 1.0)
    p_hot = pinch_t
    p_cold = pinch_t - dt_min
    
    for m in match_summary:
        duty = m['Duty [kW]']
        side = m.get('Side', 'Above')
        try:
            match_parts = m['Match'].replace('Stream ', '').split(' ↔ ')
            h_id, c_id = match_parts[0].strip(), match_parts[1].strip()
            
            h_s = processed_df[(processed_df['Stream'].astype(str) == h_id) & (processed_df['Type'] == 'Hot')].iloc[0]
            c_s = processed_df[(processed_df['Stream'].astype(str) == c_id) & (processed_df['Type'] == 'Cold')].iloc[0]
            
            # Correct inlet clamping based on Pinch side
            thi = h_s['Ts'] if side == 'Above' else min(h_s['Ts'], p_hot)
            tci = max(c_s['Ts'], p_cold) if side == 'Above' else c_s['Ts']
            
            tho, tco = thi - (duty/h_s['mCp']), tci + (duty/c_s['mCp'])
            u = calculate_u(h_s['h'], c_s['h'], h_factor)
            lmtd = lmtd_chen(thi, tho, tci, tco)
            
            if u > 0 and lmtd > 0:
                cap_mer += calculate_hex_capital(duty / (u * lmtd), econ_params)
        except: continue
        
    # Utility units capital
    for q, h_u, l_u in [(qh_util, econ_params['h_hu'], 50.0), (qc_util, econ_params['h_cu'], 30.0)]:
        if q > 0.1:
            u_u = calculate_u(h_u, 1.6, h_factor)
            if u_u > 0: cap_mer += calculate_hex_capital(q / (u_u * l_u), econ_params)
            
    return cap_mer

# --- GENETIC ALGORITHM LOGIC ---
def fitness_function(ga_instance, solution, solution_idx):
    """Minimize Total Annual Cost (TAC)."""
    match_pairs = st.session_state['ga_match_pairs']
    h_streams = st.session_state['ga_hot_streams']
    c_streams = st.session_state['ga_cold_streams']
    econ = st.session_state['ga_econ_params']
    dt_min = st.session_state['ga_dt_min']
    
    rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in h_streams}
    rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in c_streams}
    total_inv = 0
    
    for i, pair in enumerate(match_pairs):
        q = solution[i] * pair['max_duty']
        if q < 0.1: continue
        
        hs, cs = pair['hot_stream'], pair['cold_stream']
        if q > rem_h[hs['Stream']] + 0.1 or q > rem_c[cs['Stream']] + 0.1: return -1e10
        
        tho, tco = hs['Ts'] - (q/hs['mCp']), cs['Ts'] + (q/cs['mCp'])
        if (hs['Ts'] - tco) < dt_min or (tho - cs['Ts']) < dt_min: return -1e10
        
        rem_h[hs['Stream']] -= q
        rem_c[cs['Stream']] -= q
        
        u = calculate_u(hs['h'], cs['h'], econ['h_factor'])
        lmtd = lmtd_chen(hs['Ts'], tho, cs['Ts'], tco)
        if u > 0 and lmtd > 0: total_inv += calculate_hex_capital(q / (u * lmtd), econ)

    opex = (sum(max(0, v) for v in rem_c.values()) * econ['c_hu']) + (sum(max(0, v) for v in rem_h.values()) * econ['c_cu'])
    return -(opex + total_inv)

# --- STREAMLIT UI ---
# (Standard input forms and result rendering logic)
# [Detailed Streamlit UI components omitted for brevity, following the structure in Python Program.txt]

if __name__ == "__main__":
    # Main logic execution and UI rendering
    pass
``` [cite: 52, 107, 108, 122]
