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
    
    pinch_shifted = temps[feasible.index(0)] if 0 in feasible else None
    
    return qh_min, feasible[-1], pinch_shifted, temps, feasible, df

def match_logic_with_splitting(df, pinch_s, side, dt):
    sub = df.copy()
    # Logic for matching based on the shifted pinch boundary
    if side == 'Above':
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(lower=pinch_s), sub['S_Tt'].clip(lower=pinch_s)
    else:
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(upper=pinch_s), sub['S_Tt'].clip(upper=pinch_s)
    
    sub['Q_Total'] = sub['mCp'] * abs(sub['S_Ts'] - sub['S_Tt'])
    total_duties = sub.set_index('Stream')['Q_Total'].to_dict()
    
    sub['Q'] = sub['Q_Total']
    streams = sub[sub['Q'] > 0.1].to_dict('records')
    hot = [s for s in streams if s['Type'] == 'Hot']
    cold = [s for s in streams if s['Type'] == 'Cold']
    matches = []
    
    while any(h['Q'] > 1 for h in hot) and any(c['Q'] > 1 for c in cold):
        h = next(s for s in hot if s['Q'] > 1)
        # MER rules: Above Pinch (mCp_hot <= mCp_cold), Below Pinch (mCp_hot >= mCp_cold)
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
            matches.append({"Match": match_str, "Duty [kW]": round(m_q, 2), "Type": "Split" if is_split else "Direct"})
        else:
            break
    return matches, hot, cold

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
        p_hot = pinch_s
        p_cold = pinch_s - dt_min_input if pinch_s is not None else None
        st.metric("Pinch Temp (Hot)", f"{p_hot:.1f} °C" if p_hot else "N/A")
        st.metric("Pinch Temp (Cold)", f"{p_cold:.1f} °C" if p_cold else "N/A")
    with r2:
        st.write("**Temperature Data (Actual vs Shifted)**")
        st.dataframe(processed_df[['Stream', 'Type', 'Ts', 'Tt', 'S_Ts', 'S_Tt']], use_container_width=True)

    # --- SECTION 3: GRAPHICAL REPRESENTATION ---
    st.markdown("---")
    st.subheader("3. Graphical Representation of Heat Loads")
    g1, g2 = st.columns(2)
    with g1:
        st.write("**Composite Curves (Hot vs Shifted Cold)**")
        # Build Actual Hot Curve
        hot_df = edited_df[edited_df['Type'] == 'Hot'].copy()
        h_t = sorted(pd.concat([hot_df['Ts'], hot_df['Tt']]).unique())
        h_q_vals = [0]
        for i in range(len(h_t)-1):
            mcp_sum = hot_df[((hot_df['Ts'] >= h_t[i+1]) & (hot_df['Tt'] <= h_t[i])) | ((hot_df['Ts'] <= h_t[i]) & (hot_df['Tt'] >= h_t[i+1]))]['mCp'].sum()
            h_q_vals.append(h_q_vals[-1] + mcp_sum * (h_t[i+1] - h_t[i]))
        
        # Build Shifted Cold Curve (Starting at Qh to align at Pinch)
        cold_df = edited_df[edited_df['Type'] == 'Cold'].copy()
        c_t_actual = sorted(pd.concat([cold_df['Ts'], cold_df['Tt']]).unique())
        c_t_shifted = [t + dt_min_input for t in c_t_actual]
        c_q_vals = [qh]
        for i in range(len(c_t_actual)-1):
            mcp_sum = cold_df[((cold_df['Ts'] <= c_t_actual[i]) & (cold_df['Tt'] >= c_t_actual[i+1])) | ((cold_df['Ts'] >= c_t_actual[i+1]) & (cold_df['Tt'] <= c_t_actual[i]))]['mCp'].sum()
            c_q_vals.append(c_q_vals[-1] + mcp_sum * (c_t_actual[i+1] - c_t_actual[i]))

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=h_t, y=h_q_vals, name="Hot (Actual)", line=dict(color='red', width=3)))
        fig_comp.add_trace(go.Scatter(x=c_t_shifted, y=c_q_vals, name="Cold (Shifted)", line=dict(color='blue', dash='dash')))
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
    if pinch_s:
        l, r = st.columns(2)
        match_summary = []
        for i, side in enumerate(['Above', 'Below']):
            matches, h_rem, c_rem = match_logic_with_splitting(processed_df, pinch_s, side, dt_min_input)
            match_summary.extend(matches)
            with (l if i == 0 else r):
                st.write(f"**Matches {side} Pinch**")
                if matches:
                    st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
                else: st.info("No internal matches possible.")
                for c in c_rem: 
                    if c['Q'] > 1: st.error(f"**Heater:** Stream {c['Stream']} ({c['Q']:,.1f} kW)")
                for h in h_rem: 
                    if h['Q'] > 1: st.info(f"**Cooler:** Stream {h['Stream']} ({h['Q']:,.1f} kW)")
    
    # Export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(match_summary).to_excel(writer, sheet_name='Matches', index=False)
    st.download_button(label="📥 Download HEN Report", data=output.getvalue(), file_name="HEN_Design.xlsx")
