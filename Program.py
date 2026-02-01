import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("---")

# --- CORE MATH FUNCTIONS ---
def run_thermal_logic(df, dt):
    """
    Standard Problem Table Algorithm.
    Calculates Minimum Energy Targets (Qh, Qc) and the Pinch Temperature.
    """
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # 1. Shift Temperatures for Analysis (Hot - dt/2, Cold + dt/2)
    # This is purely for the algebraic 'Problem Table' calculation.
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'] - dt/2, df['Ts'] + dt/2)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'] - dt/2, df['Tt'] + dt/2)
    
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    
    # Calculate Heat Balance in each interval
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_mcp = df[(df['Type'] == 'Hot') & (df['S_Ts'] >= hi) & (df['S_Tt'] <= lo)]['mCp'].sum()
        c_mcp = df[(df['Type'] == 'Cold') & (df['S_Ts'] <= lo) & (df['S_Tt'] >= hi)]['mCp'].sum()
        intervals.append({'hi': hi, 'lo': lo, 'net': (h_mcp - c_mcp) * (hi - lo)})
    
    # Cascade analysis to find minimum utilities
    net_heat = pd.DataFrame(intervals)['net']
    cascade = [0] + list(net_heat.cumsum())
    min_val = min(cascade)
    qh_min = abs(min(min_val, 0)) # The amount we must add to make the cascade non-negative
    qc_min = cascade[-1] + qh_min
    
    feasible_cascade = [val + qh_min for val in cascade]
    
    # Identify Pinch Temperature (Shifted)
    try:
        pinch_idx = feasible_cascade.index(0)
        pinch_shifted = temps[pinch_idx]
    except ValueError:
        pinch_shifted = None
        
    # GCC Plot Data (Shifted Temp vs Net Heat Flow)
    # We construct points for the step graph
    gcc_t = []
    gcc_q = []
    for i, q in enumerate(feasible_cascade):
        gcc_t.append(temps[i])
        gcc_q.append(q)
    
    return qh_min, qc_min, pinch_shifted, gcc_t, gcc_q, df

def match_logic_with_splitting(df, pinch_s, side):
    """
    MER Matching Logic: Checks streams against the Pinch Boundary.
    """
    sub = df.copy()
    
    # Filter streams based on Shifted Pinch Temperature
    if side == 'Above':
        # Hot streams > Pinch, Cold streams > Pinch
        # Note: Logic uses Shifted Temps for boundary check
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(lower=pinch_s), sub['S_Tt'].clip(lower=pinch_s)
    else:
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(upper=pinch_s), sub['S_Tt'].clip(upper=pinch_s)
    
    # Calculate duty available in this zone
    sub['Q_Zone'] = sub['mCp'] * abs(sub['S_Ts'] - sub['S_Tt'])
    sub['Q'] = sub['Q_Zone'] # Working duty variable
    
    # Filter out streams with no duty in this zone
    streams = sub[sub['Q'] > 0.01].to_dict('records')
    hot = [s for s in streams if s['Type'] == 'Hot']
    cold = [s for s in streams if s['Type'] == 'Cold']
    matches = []
    
    # Greedy Matching Algorithm
    while any(h['Q'] > 1 for h in hot) and any(c['Q'] > 1 for c in cold):
        h = next(s for s in hot if s['Q'] > 1)
        
        # MER Criterion: Above Pinch (CP_hot <= CP_cold), Below Pinch (CP_hot >= CP_cold)
        if side == 'Above':
            candidates = [c for c in cold if c['mCp'] >= h['mCp'] and c['Q'] > 1]
        else:
            candidates = [c for c in cold if h['mCp'] >= c['mCp'] and c['Q'] > 1]
            
        c = candidates[0] if candidates else None
        
        is_split = False
        # If no match found satisfying CP rule, force a match (stream splitting required scenario)
        if not c:
            c = next((s for s in cold if s['Q'] > 1), None)
            is_split = True
            
        if c:
            m_q = min(h['Q'], c['Q'])
            match_str = f"Stream {h['Stream']} ↔ {c['Stream']}"
            
            h['Q'] -= m_q
            c['Q'] -= m_q
            matches.append({
                "Match": match_str, 
                "Duty [kW]": round(m_q, 2), 
                "Type": "Split/Non-Opt" if is_split else "Direct"
            })
        else:
            break
            
    return matches, hot, cold

def get_composite_curve_points(df, stream_type, start_enthalpy=0):
    """
    Generates (Temperature, Enthalpy) points for a Composite Curve.
    Uses ACTUAL temperatures.
    """
    subset = df[df['Type'] == stream_type].copy()
    if subset.empty:
        return [], []
    
    # Get all unique interval temperatures
    temps = sorted(pd.concat([subset['Ts'], subset['Tt']]).unique())
    if stream_type == 'Hot':
        temps = sorted(temps, reverse=True) # Hot streams go High -> Low
    else:
        temps = sorted(temps) # Cold streams go Low -> High

    H_points = [start_enthalpy]
    T_points = [temps[0]]
    
    current_H = start_enthalpy
    
    for i in range(len(temps)-1):
        t_start, t_end = temps[i], temps[i+1]
        
        # Identify streams active in this temp interval
        if stream_type == 'Hot':
            active = subset[(subset['Ts'] >= t_start) & (subset['Tt'] <= t_end)]
            # For Hot: dT is positive (High - Low)
            delta_t = t_start - t_end 
        else:
            active = subset[(subset['Ts'] <= t_start) & (subset['Tt'] >= t_end)] # Logic depends on sort
            # Let's simplify: simply check if interval is within stream bounds
            # Since we sorted temps, we just check overlap
            low, high = min(t_start, t_end), max(t_start, t_end)
            active = subset[(subset['Ts'] <= low) &
