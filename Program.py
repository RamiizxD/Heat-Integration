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
    
    # Problem Table Shifting (Used to find the shift distance Qh_min)
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
    
    # Cascade Analysis to find Max Deficit (The "Distance")
    net_heat = pd.DataFrame(intervals)['net']
    cascade = [0] + list(net_heat.cumsum())
    
    # qh_min is the amount we need to shift the cold curve to restore feasibility
    qh_min = abs(min(min(cascade), 0))
    qc_min = cascade[-1] + qh_min
    
    feasible_cascade = [val + qh_min for val in cascade]
    
    pinch_shifted = temps[feasible_cascade.index(0)] if 0 in feasible_cascade else None
    
    # Return GCC points for plotting later
    return qh_min, qc_min, pinch_shifted, temps, feasible_cascade, df

def get_composite_curve_points(df, stream_type, start_enthalpy=0):
    subset = df[df['Type'] == stream_type].copy()
    if subset.empty: return [], []
    
    # Sort actual temperatures to build the curve segment by segment
    temps = sorted(pd.concat([subset['Ts'], subset['Tt']]).unique())
    H_points = [start_enthalpy]
    T_points = [temps[0]]
    current_H = start_enthalpy
    
    for i in range(len(temps)-1):
        t_low, t_high = temps[i], temps[i+1]
        
        # Identify streams active in this temperature range
        active = subset[((subset['Ts'] <= t_low) & (subset['Tt'] >= t_high)) | 
                        ((subset['Ts'] >= t_high) & (subset['Tt'] <= t_low))]
        
        if not active.empty:
            delta_h = active['mCp'].sum() * (t_high - t_low)
            current_H += delta_h
            
        T_points.append(t_high)
        H_points.append(current_H)
        
    return H_points, T_points

def match_logic_with_splitting(df, pinch_s, side):
    sub = df.copy()
    if side == 'Above':
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(lower=pinch_s), sub['S_Tt'].clip(lower=pinch_s)
    else:
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(upper=pinch_s), sub['S_Tt'].clip(upper=pinch_s)
    
    sub['Q_Total'] = sub['mCp'] * abs(sub['S_Ts'] - sub['S_Tt'])
    sub['Q'] = sub['Q_Total']
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
            match_str = f"Stream {h['Stream']} ↔ {c['Stream']}"
            h['Q'] -= m_q
            c['Q'] -= m_q
            matches.append({"Match": match_str, "Duty [kW]": round(m_q, 2), "Type": "Split" if is_split else "Direct"})
        else:
            break
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

if submit_thermal and not edited_df.empty:
    st.session_state.run_clicked = True

# --- MAIN OUTPUT DISPLAY ---
if st.session_state.get('run_clicked'):
    qh, qc, pinch_s, gcc_t, gcc_q, processed_df = run_thermal_logic(edited_df, dt_min_input)
    
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
    c2.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
    if pinch_s:
        c3.metric("Pinch (Hot/Cold)", f"{pinch_s + dt_min_input/2:.1f}°C / {pinch_s - dt_min_input/2:.1f}°C")

    # --- SECTION 3: GRAPHICAL REPRESENTATION ---
    st.markdown("---")
    st.subheader("3. Graphical Representation of Heat Loads")
    g1, g2 = st.columns(2)
    
    with g1:
        st.write("**Composite Curves**")
        st.caption(f"Cold Curve shifted by {qh:.2f} kW to maintain min gap.")
        
        # 1. Hot Curve: Starts at Enthalpy 0
        h_q, h_t = get_composite_curve_points(edited_df, 'Hot', start_enthalpy=0)
        
        # 2. Cold Curve: Starts at Enthalpy = Qh (The 'Raised' distance)
        c_q, c_t = get_composite_curve_points(edited_df, 'Cold', start_enthalpy=qh)
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=h_q, y=h_t, name="Hot Composite", line=dict(color='green', width=3)))
        fig_comp.add_trace(go.Scatter(x=c_q, y=c_t, name="Cold Composite", line=dict(color='black', width=3)))
        
        # Matches user image format: X=Enthalpy, Y=Temp
        fig_comp.update_layout(
            xaxis_title="Enthalpy [kW]", 
            yaxis_title="Temperature [°C]", 
            height=500
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with g2:
        st.write("**Grand Composite Curve**")
        fig_gcc = go.Figure(go.Scatter(x=gcc_q, y=gcc_t, mode='lines', fill='tozerox', line=dict(color='green')))
        fig_gcc.update_layout(xaxis_title="Net Heat Flow [kW]", yaxis_title="Shifted Temperature [°C]", height=500)
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
                st.write(f"**Matches {side} Pinch**")
                if matches: 
                    st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
                else: 
                    st.info("No internal matches possible.")
                
                for c in c_rem: 
                    if c['Q'] > 1: st.error(f"**Heater:** Stream {c['Stream']} ({c['Q']:,.1f} kW)")
                for h in h_rem: 
                    if h['Q'] > 1: st.info(f"**Cooler:** Stream {h['Stream']} ({h['Q']:,.1f} kW)")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(match_summary).to_excel(writer, sheet_name='Matches', index=False)
        st.download_button(label="📥 Download HEN Report", data=output.getvalue(), file_name="HEN_Design.xlsx")
    else:
        st.warning("No Pinch Point detected.")
