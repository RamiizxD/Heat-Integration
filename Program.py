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
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # Problem Table Shifting
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'] - dt/2, df['Ts'] + dt/2)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'] - dt/2, df['Tt'] + dt/2)
    
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_mcp = df[(df['Type'] == 'Hot') & (df['S_Ts'] >= hi) & (df['S_Tt'] <= lo)]['mCp'].sum()
        c_mcp = df[(df['Type'] == 'Cold') & (df['S_Ts'] <= lo) & (df['S_Tt'] >= hi)]['mCp'].sum()
        intervals.append({'hi': hi, 'lo': lo, 'net': (h_mcp - c_mcp) * (hi - lo)})
    
    net_heat = pd.DataFrame(intervals)['net']
    cascade = [0] + list(net_heat.cumsum())
    qh_min = abs(min(min(cascade), 0))
    qc_min = cascade[-1] + qh_min
    feasible_cascade = [val + qh_min for val in cascade]
    
    pinch_shifted = temps[feasible_cascade.index(0)] if 0 in feasible_cascade else None
    
    return qh_min, qc_min, pinch_shifted, temps, feasible_cascade, df

def get_curve_points(df, stream_type, start_enthalpy=0):
    subset = df[df['Type'] == stream_type].copy()
    if subset.empty: return np.array([]), np.array([])
    
    # Get unique temperatures
    temps = sorted(pd.concat([subset['Ts'], subset['Tt']]).unique())
    # For Hot streams, we generally calculate high to low, but for plotting X=Temp, we need sorted
    H_points = [start_enthalpy]
    T_points = [temps[0]]
    current_H = start_enthalpy
    
    for i in range(len(temps)-1):
        t_low, t_high = temps[i], temps[i+1]
        active = subset[((subset['Ts'] <= t_low) & (subset['Tt'] >= t_high)) | 
                        ((subset['Ts'] >= t_high) & (subset['Tt'] <= t_low))]
        delta_h = active['mCp'].sum() * (t_high - t_low)
        current_H += delta_h
        T_points.append(t_high)
        H_points.append(current_H)
        
    return np.array(T_points), np.array(H_points)

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
    if pinch_s: 
        st.write(f"**Pinch Point:** {pinch_s + dt_min_input/2}°C (Hot) / {pinch_s - dt_min_input/2}°C (Cold)")

    # --- SECTION 3: GRAPHICAL REPRESENTATION ---
    st.markdown("---")
    st.subheader("3. Graphical Representation (Temp X, Enthalpy Y)")
    
    g1, g2 = st.columns(2)
    
    with g1:
        st.write("**Composite Curves (Raised Cold Curve)**")
        
        # Get base points: X = Temp, Y = Enthalpy
        # Start hot at 0, start cold at Qh (standard initial gap)
        h_t, h_h = get_curve_points(edited_df, 'Hot', 0)
        c_t, c_h = get_curve_points(edited_df, 'Cold', qh)
        
        if len(h_t) > 0 and len(c_t) > 0:
            # Create common Temp range for comparison
            min_t = max(h_t.min(), c_t.min())
            max_t = min(h_t.max(), c_t.max())
            grid_t = np.linspace(min_t, max_t, 100)
            
            # Interpolate Enthalpy (Y) at these Temperatures (X)
            h_h_interp = np.interp(grid_t, h_t, h_h)
            c_h_interp = np.interp(grid_t, c_t, c_h)
            
            # Calculate max vertical difference (Enthalpy gap)
            diffs = h_h_interp - c_h_interp
            max_y_diff = np.max(diffs) if len(diffs) > 0 else 0
            
            # Raise the Cold Curve
            c_h_raised = c_h + max_y_diff
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=h_t, y=h_h, name="Hot Composite", line=dict(color='green', width=3)))
            fig_comp.add_trace(go.Scatter(x=c_t, y=c_h_raised, name="Cold Composite (Raised)", line=dict(color='black', width=3)))
            
            fig_comp.update_layout(xaxis_title="Temperature [°C]", yaxis_title="Enthalpy [kW]", height=500)
            st.plotly_chart(fig_comp, use_container_width=True)

    with g2:
        st.write("**Grand Composite Curve**")
        # X = Shifted Temp, Y = Net Heat
        fig_gcc = go.Figure(go.Scatter(x=gcc_t, y=gcc_q, fill='tozeroy', line=dict(color='green')))
        fig_gcc.update_layout(xaxis_title="Shifted Temperature [°C]", yaxis_title="Net Heat Flow [kW]", height=500)
        st.plotly_chart(fig_gcc, use_container_width=True)
