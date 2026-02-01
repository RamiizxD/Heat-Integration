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
    Standard Pinch Analysis (Problem Table) to find Energy Targets.
    """
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # 1. Shift Temperatures (Hot - dt/2, Cold + dt/2)
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'] - dt/2, df['Ts'] + dt/2)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'] - dt/2, df['Tt'] + dt/2)
    
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    
    # 2. Problem Table Cascade
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_mcp = df[(df['Type'] == 'Hot') & (df['S_Ts'] >= hi) & (df['S_Tt'] <= lo)]['mCp'].sum()
        c_mcp = df[(df['Type'] == 'Cold') & (df['S_Ts'] <= lo) & (df['S_Tt'] >= hi)]['mCp'].sum()
        intervals.append({'hi': hi, 'lo': lo, 'net': (h_mcp - c_mcp) * (hi - lo)})
    
    net_heat = pd.DataFrame(intervals)['net']
    cascade = [0] + list(net_heat.cumsum())
    
    # 3. Calculate Minimum Utility
    qh_min = abs(min(min(cascade), 0)) # Max deficit
    qc_min = cascade[-1] + qh_min
    
    feasible_cascade = [val + qh_min for val in cascade]
    pinch_shifted = temps[feasible_cascade.index(0)] if 0 in feasible_cascade else None
    
    # GCC Data (Shifted Temp vs Net Heat)
    return qh_min, qc_min, pinch_shifted, temps, feasible_cascade, df

def get_curve_points(df, stream_type, start_enthalpy=0):
    """ Generates standard (Enthalpy, Temp) points for a Composite Curve """
    subset = df[df['Type'] == stream_type].copy()
    if subset.empty: return [], []
    
    # Sort Temperatures: Hot (High->Low), Cold (Low->High)
    temps = sorted(pd.concat([subset['Ts'], subset['Tt']]).unique())
    if stream_type == 'Hot': temps.reverse() # Descending for Hot
    
    H_points = [start_enthalpy]
    T_points = [temps[0]]
    current_H = start_enthalpy
    
    for i in range(len(temps)-1):
        t_start, t_end = temps[i], temps[i+1]
        
        # Find active streams in this interval
        if stream_type == 'Hot':
            # Hot goes High to Low
            active = subset[(subset['Ts'] >= t_start) & (subset['Tt'] <= t_end)]
            delta_t = t_start - t_end
        else:
            # Cold goes Low to High
            active = subset[(subset['Ts'] <= t_start) & (subset['Tt'] >= t_end)]
            delta_t = t_end - t_start
            
        mcp = active['mCp'].sum()
        delta_h = mcp * delta_t
        current_H += delta_h
        
        T_points.append(t_end)
        H_points.append(current_H)
        
    return np.array(H_points), np.array(T_points)

def match_logic_with_splitting(df, pinch_s, side):
    """ (Standard MER Matching Logic kept as requested) """
    sub = df.copy()
    if side == 'Above':
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(lower=pinch_s), sub['S_Tt'].clip(lower=pinch_s)
    else:
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(upper=pinch_s), sub['S_Tt'].clip(upper=pinch_s)
    
    sub['Q'] = sub['mCp'] * abs(sub['S_Ts'] - sub['S_Tt'])
    streams = sub[sub['Q'] > 0.01].to_dict('records')
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
            matches.append({"Match": f"Stream {h['Stream']} ↔ {c['Stream']}", "Duty [kW]": round(m_q, 2)})
            h['Q'] -= m_q
            c['Q'] -= m_q
        else: break
    return matches, hot, cold

# --- SECTION 1: DATA INPUT ---
st.subheader("1. Stream Data Input")
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = pd.DataFrame([
        [1, 'Hot', 2.0, 150, 60],
        [2, 'Hot', 1.0, 90, 60],
        [3, 'Cold', 3.0, 20, 125],
        [4, 'Cold', 0.5, 25, 100]
    ], columns=["Stream", "Type", "mCp", "Ts", "Tt"])

with st.form("main_input_form"):
    dt_min_input = st.number_input("Target ΔTmin [°C]", min_value=1.0, value=10.0)
    edited_df = st.data_editor(st.session_state['input_data'], num_rows="dynamic", use_container_width=True)
    submit_thermal = st.form_submit_button("Run Thermal Analysis")

if submit_thermal: st.session_state.run_clicked = True

# --- MAIN OUTPUT DISPLAY ---
if st.session_state.get('run_clicked'):
    qh, qc, pinch_s, gcc_t, gcc_q, processed_df = run_thermal_logic(edited_df, dt_min_input)
    
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
    c2.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
    if pinch_s: c3.metric("Pinch Temp (Shifted)", f"{pinch_s:.1f}°C")

    # --- SECTION 3: GRAPHICAL REPRESENTATION ---
    st.markdown("---")
    st.subheader("3. Graphical Representation")
    
    g1, g2 = st.columns(2)
    
    with g1:
        st.write("**Composite Curves (Adjusted)**")
        
        # 1. Get Standard Curves (Hot @ 0, Cold @ Qh)
        h_h, h_t = get_curve_points(edited_df, 'Hot', 0)
        c_h, c_t = get_curve_points(edited_df, 'Cold', qh)
        
        # 2. IMPLEMENT USER LOGIC: Find Max Vertical Distance
        if len(h_h) > 0 and len(c_h) > 0:
            # Create a common X-axis (Enthalpy) grid for comparison
            # We look at the overlapping range of Enthalpy
            min_h = max(h_h.min(), c_h.min())
            max_h = min(h_h.max(), c_h.max())
            
            # Generate grid points (include original points that fall in range)
            grid_h = sorted(list(set(
                [x for x in h_h if min_h <= x <= max_h] + 
                [x for x in c_h if min_h <= x <= max_h]
            )))
            
            if grid_h:
                # Interpolate Temperatures at these grid points
                t_hot_interp = np.interp(grid_h, h_h, h_t)
                t_cold_interp = np.interp(grid_h, c_h, c_t)
                
                # Check (y_hot - y_cold) -> (t_hot - t_cold)
                diffs = t_hot_interp - t_cold_interp
                
                # Find MAX difference
                max_diff = np.max(diffs)
                
                # Add Max Diff to Cold Curve Y-coordinates (Raise it up)
                c_t_raised = c_t + max_diff
                
                st.caption(f"Cold Curve raised vertically by Max ΔT ({max_diff:.2f}°C) to separate curves.")
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=h_h, y=h_t, name="Hot Composite", line=dict(color='green', width=3)))
                # Plotting ONLY the raised curve, not the original
                fig_comp.add_trace(go.Scatter(x=c_h, y=c_t_raised, name="Cold Composite (Raised)", line=dict(color='black', width=3)))
                
                fig_comp.update_layout(xaxis_title="Enthalpy [kW]", yaxis_title="Temperature [°C]", height=500)
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("Curves do not overlap on Enthalpy axis.")
        else:
            st.warning("Insufficient data to plot curves.")

    with g2:
        st.write("**Grand Composite Curve**")
        fig_gcc = go.Figure(go.Scatter(x=gcc_q, y=gcc_t, fill='tozerox', line=dict(color='green')))
        fig_gcc.update_layout(xaxis_title="Net Heat Flow [kW]", yaxis_title="Shifted Temp [°C]", height=500)
        st.plotly_chart(fig_gcc, use_container_width=True)

    # --- SECTION 4: MATCHING ---
    st.markdown("---")
    st.subheader("4. HEN Matching")
    if pinch_s:
        l, r = st.columns(2)
        match_summary = []
        for i, side in enumerate(['Above', 'Below']):
            matches, h_rem, c_rem = match_logic_with_splitting(processed_df, pinch_s, side)
            match_summary.extend(matches)
            with (l if i == 0 else r):
                st.write(f"**Matches {side} Pinch**")
                if matches: st.dataframe(pd.DataFrame(matches), hide_index=True)
                else: st.info("No matches found.")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(match_summary).to_excel(writer, sheet_name='Matches', index=False)
        st.download_button("📥 Download HEN Design", output.getvalue(), "HEN.xlsx")
