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
            active = subset[(subset['Ts'] <= low) & (subset['Tt'] >= high) | 
                            (subset['Ts'] >= high) & (subset['Tt'] <= low)]
            delta_t = high - low

        mcp_sum = active['mCp'].sum()
        delta_h = mcp_sum * delta_t
        
        current_H += delta_h
        
        T_points.append(t_end)
        H_points.append(current_H)
        
    return T_points, H_points

# --- SECTION 1: DATA INPUT ---
st.subheader("1. Stream Data Input")
uploaded_file = st.file_uploader("Import Stream Data from Excel (.xlsx)", type=["xlsx"])

if 'input_data' not in st.session_state:
    # Default example data
    st.session_state['input_data'] = pd.DataFrame([
        [1, 'Hot', 2.0, 150, 60],
        [2, 'Hot', 1.0, 90, 60],
        [3, 'Cold', 3.0, 20, 125],
        [4, 'Cold', 0.5, 25, 100]
    ], columns=["Stream", "Type", "mCp", "Ts", "Tt"])

if uploaded_file:
    try:
        st.session_state['input_data'] = pd.read_excel(uploaded_file)
        st.success("Data imported!")
    except Exception as e: st.error(f"Error: {e}")

with st.form("main_input_form"):
    dt_min_input = st.number_input("Target ΔTmin [°C]", min_value=1.0, value=10.0, step=1.0)
    edited_df = st.data_editor(st.session_state['input_data'], num_rows="dynamic", use_container_width=True)
    submit_thermal = st.form_submit_button("Run Thermal Analysis")

if submit_thermal and not edited_df.empty:
    st.session_state.run_clicked = True

# --- MAIN OUTPUT DISPLAY ---
if st.session_state.get('run_clicked'):
    qh_min, qc_min, pinch_s, gcc_t, gcc_q, processed_df = run_thermal_logic(edited_df, dt_min_input)
    
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Hot Utility (Qh)", f"{qh_min:,.2f} kW")
    col2.metric("Cold Utility (Qc)", f"{qc_min:,.2f} kW")
    
    # Calculate Actual Pinch Temps from Shifted Pinch
    if pinch_s is not None:
        p_hot = pinch_s + dt_min_input/2
        p_cold = pinch_s - dt_min_input/2
        col3.metric("Pinch Temperature", f"{p_hot:.1f}°C / {p_cold:.1f}°C")
    else:
        col3.metric("Pinch Temperature", "None")

    # --- SECTION 3: GRAPHICAL REPRESENTATION ---
    st.markdown("---")
    st.subheader("3. Graphical Representation of Heat Loads")
    
    g1, g2 = st.columns(2)
    
    with g1:
        st.write("**Composite Curves**")
        st.caption(f"Cold Curve shifted vertically by Qh_min ({qh_min:.2f} kW) to achieve minimum approach of {dt_min_input}°C.")
        
        # 1. Generate Hot Curve Points (Actual T, Cumulative H)
        h_t, h_h = get_composite_curve_points(edited_df, 'Hot', start_enthalpy=0)
        
        # 2. Generate Cold Curve Points (Actual T, Cumulative H)
        # CRITICAL: Start Enthalpy = Qh_min. This "raises" the curve on the Y-axis.
        c_t, c_h = get_composite_curve_points(edited_df, 'Cold', start_enthalpy=qh_min)
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=h_t, y=h_h, name="Hot Composite", line=dict(color='red', width=3)))
        fig_comp.add_trace(go.Scatter(x=c_t, y=c_h, name="Cold Composite", line=dict(color='blue', width=3)))
        
        fig_comp.update_layout(
            xaxis_title="Actual Temperature [°C]", 
            yaxis_title="Enthalpy / Heat Load [kW]",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with g2:
        st.write("**Grand Composite Curve**")
        st.caption("Shifted Temperature vs. Net Heat Flow")
        # Ensure T is on X, Q is on Y
        fig_gcc = go.Figure(go.Scatter(
            x=gcc_t, y=gcc_q, 
            mode='lines+markers', 
            name="GCC", 
            fill='tozeroy', 
            line=dict(color='green')
        ))
        fig_gcc.update_layout(
            xaxis_title="Shifted Temperature [°C]", 
            yaxis_title="Net Heat Flow [kW]",
            height=500
        )
        st.plotly_chart(fig_gcc, use_container_width=True)

    # --- SECTION 4: MATCHING ---
    st.markdown("---")
    st.subheader("4. Heat Exchanger Network Matching (MER)")
    
    if pinch_s is not None:
        l, r = st.columns(2)
        match_summary = []
        for i, side in enumerate(['Above', 'Below']):
            matches, h_rem, c_rem = match_logic_with_splitting(processed_df, pinch_s, side)
            match_summary.extend(matches)
            
            with (l if i == 0 else r):
                st.markdown(f"**Matches {side} Pinch**")
                if matches:
                    st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
                else:
                    st.info("No matches found.")
                
                # Show Remaining Duties (Utilities)
                if c_rem:
                    for c in c_rem:
                        if c['Q'] > 1: st.error(f"Heater required: {c['Stream']} ({c['Q']:,.1f} kW)")
                if h_rem:
                    for h in h_rem:
                        if h['Q'] > 1: st.info(f"Cooler required: {h['Stream']} ({h['Q']:,.1f} kW)")
        
        # Export Button
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(match_summary).to_excel(writer, sheet_name='Matches', index=False)
        st.download_button("📥 Download Design", output.getvalue(), "HEN_Design.xlsx")
    else:
        st.warning("No Pinch Point detected. The process might be a threshold problem or fully satisfied.")
