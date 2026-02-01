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
    
    # SHIFT LOGIC: Hot stays same, Cold shifts UP by dTmin
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
    
    # The pinch occurs at the shifted temperature scale
    pinch_shifted = temps[feasible.index(0)] if 0 in feasible else None
    
    return qh_min, feasible[-1], pinch_shifted, temps, feasible, df

# --- SECTION 1: DATA INPUT ---
st.subheader("1. Stream Data Input")
uploaded_file = st.file_uploader("Import Stream Data from Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    try:
        import_df = pd.read_excel(uploaded_file)
        st.session_state['input_data'] = import_df
        st.success("Data imported!")
    except Exception as e: st.error(f"Error: {e}")

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = pd.DataFrame(columns=["Stream", "Type", "mCp", "Ts", "Tt"])

with st.form("main_input_form"):
    dt_min_input = st.number_input("Target ΔTmin [°C]", min_value=1.0, value=10.0)
    edited_df = st.data_editor(st.session_state['input_data'], num_rows="dynamic", use_container_width=True)
    submit_thermal = st.form_submit_button("Run Thermal Analysis")

if submit_thermal and not edited_df.empty:
    st.session_state.run_clicked = True

# --- MAIN OUTPUT DISPLAY ---
if st.session_state.get('run_clicked'):
    qh, qc, pinch_s, t_plot, q_plot, processed_df = run_thermal_logic(edited_df, dt_min_input)
    
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    r1, r2 = st.columns([1, 2])
    
    with r1:
        st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
        st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
        # Shifted back to actual scales
        p_hot = pinch_s
        p_cold = pinch_s - dt_min_input if pinch_s is not None else None
        st.metric("Pinch Temp (Hot)", f"{p_hot:.1f} °C" if p_hot else "N/A")
        st.metric("Pinch Temp (Cold)", f"{p_cold:.1f} °C" if p_cold else "N/A")

    with r2:
        st.write("**Shifted Temperature Table**")
        st.dataframe(processed_df[['Stream', 'Type', 'Ts', 'Tt', 'S_Ts', 'S_Tt']], use_container_width=True)

    # --- SECTION 3: GRAPHICAL REPRESENTATION ---
    st.markdown("---")
    st.subheader("3. Graphical Representation of Heat Loads")
    g1, g2 = st.columns(2)
    
    with g1:
        st.write("**Composite Curves (Hot & Shifted Cold)**")
        # Hot Composite (Actual)
        hot_df = edited_df[edited_df['Type'] == 'Hot'].copy()
        h_temps = sorted(pd.concat([hot_df['Ts'], hot_df['Tt']]).unique())
        h_q = [0]
        for i in range(len(h_temps)-1):
            lo, hi = h_temps[i], h_temps[i+1]
            mcp_sum = hot_df[(hot_df['Ts'] >= hi) & (hot_df['Tt'] <= lo) | (hot_df['Ts'] <= lo) & (hot_df['Tt'] >= hi)]['mCp'].sum()
            h_q.append(h_q[-1] + mcp_sum * (hi - lo))
        
        # Cold Composite (Shifted UP by dTmin)
        cold_df = edited_df[edited_df['Type'] == 'Cold'].copy()
        c_temps_actual = sorted(pd.concat([cold_df['Ts'], cold_df['Tt']]).unique())
        c_temps_shifted = [t + dt_min_input for t in c_temps_actual]
        c_q = [qh] 
        for i in range(len(c_temps_actual)-1):
            lo, hi = c_temps_actual[i], c_temps_actual[i+1]
            mcp_sum = cold_df[(cold_df['Ts'] <= lo) & (cold_df['Tt'] >= hi) | (cold_df['Ts'] >= hi) & (cold_df['Tt'] <= lo)]['mCp'].sum()
            c_q.append(c_q[-1] + mcp_sum * (hi - lo))

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=h_temps, y=h_q, name="Hot Composite (Actual)", line=dict(color='red', width=3)))
        fig_comp.add_trace(go.Scatter(x=c_temps_shifted, y=c_q, name="Cold Composite (Shifted)", line=dict(color='blue', dash='dash')))
        fig_comp.update_layout(xaxis_title="Temperature [°C]", yaxis_title="Heat Load [kW]")
        st.plotly_chart(fig_comp, use_container_width=True)

    with g2:
        st.write("**Grand Composite Curve**")
        fig_gcc = go.Figure(go.Scatter(x=t_plot, y=q_plot, mode='lines+markers', name="GCC", fill='tozeroy', line=dict(color='green')))
        fig_gcc.update_layout(xaxis_title="Shifted Temperature [°C]", yaxis_title="Net Heat Flow [kW]")
        st.plotly_chart(fig_gcc, use_container_width=True)

    # --- SECTION 4: MATCHING ---
    st.markdown("---")
    st.subheader("4. Heat Exchanger Network Matching (MER)")
    # Logic follows pinch_s (the boundary where shifted curves touch)
    if pinch_s:
        # (Matching logic omitted for brevity but uses pinch_s as the cut point)
        st.info(f"Pinch point identified at {pinch_s}°C (Shifted scale). Matches should be designed separately Above and Below this point.")
