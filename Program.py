import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs **Pinch Analysis**, **MER Matching with Stream Splitting**, and 
**Economic Optimization**.
""")
st.markdown("---")

# --- CORE MATH FUNCTIONS ---
def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # Shifting temperatures
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'] - dt/2, df['Ts'] + dt/2)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'] - dt/2, df['Tt'] + dt/2)
    
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

def match_logic_with_splitting(df, pinch_t, side):
    sub = df.copy()
    # Logic for matching based on shifted pinch
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
        # Simplified MER matching rule (mCp comparison)
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
    qh, qc, pinch, t_plot, q_plot, processed_df = run_thermal_logic(edited_df, dt_min_input)
    
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
    c2.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
    c3.metric("Pinch (Shifted)", f"{pinch} °C" if pinch is not None else "N/A")

    # --- SECTION 3: GRAPHICAL REPRESENTATION ---
    st.markdown("---")
    st.subheader("3. Graphical Representation of Heat Loads")
    
    g1, g2 = st.columns(2)
    
    with g1:
        st.write("**Composite Curves**")
        # Logic to build Hot/Cold Composite Curves
        # Hot Composite
        hot_df = edited_df[edited_df['Type'] == 'Hot'].copy()
        h_temps = sorted(pd.concat([hot_df['Ts'], hot_df['Tt']]).unique(), reverse=True)
        h_q = [0]
        for i in range(len(h_temps)-1):
            hi, lo = h_temps[i], h_temps[i+1]
            mcp_sum = hot_df[(hot_df['Ts'] >= hi) & (hot_df['Tt'] <= lo)]['mCp'].sum()
            h_q.append(h_q[-1] + mcp_sum * (hi - lo))
        
        # Cold Composite (Shifted by Qh to show overlap)
        cold_df = edited_df[edited_df['Type'] == 'Cold'].copy()
        c_temps = sorted(pd.concat([cold_df['Ts'], cold_df['Tt']]).unique())
        c_q = [qh]
        for i in range(len(c_temps)-1):
            lo, hi = c_temps[i], c_temps[i+1]
            mcp_sum = cold_df[(cold_df['Ts'] <= lo) & (cold_df['Tt'] >= hi)]['mCp'].sum()
            c_q.append(c_q[-1] + mcp_sum * (hi - lo))

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=h_q, y=h_temps, name="Hot Composite", line=dict(color='red')))
        fig_comp.add_trace(go.Scatter(x=c_q, y=c_temps, name="Cold Composite", line=dict(color='blue')))
        fig_comp.update_layout(xaxis_title="Heat Load [kW]", yaxis_title="Temperature [°C]", height=400)
        st.plotly_chart(fig_comp, use_container_width=True)

    with g2:
        st.write("**Grand Composite Curve**")
        fig_gcc = go.Figure(go.Scatter(x=q_plot, y=t_plot, mode='lines+markers', name="GCC", fill='tozerox', line=dict(color='green')))
        fig_gcc.update_layout(xaxis_title="Net Heat Flow [kW]", yaxis_title="Shifted Temperature [°C]", height=400)
        st.plotly_chart(fig_gcc, use_container_width=True)

    # --- SECTION 4: HEN MATCHING ---
    st.markdown("---")
    st.subheader("4. Heat Exchanger Network Matching (MER)")
    
    match_summary = []
    if pinch is not None:
        l, r = st.columns(2)
        for i, side in enumerate(['Above', 'Below']):
            matches, h_rem, c_rem = match_logic_with_splitting(processed_df, pinch, side)
            match_summary.extend(matches)
            with (l if i == 0 else r):
                st.write(f"**Matches {side} Pinch**")
                if matches: 
                    m_df = pd.DataFrame(matches)
                    st.dataframe(m_df, use_container_width=True, hide_index=True)
                else: 
                    st.info("No internal matches possible.")

                for c in c_rem: 
                    if c['Q'] > 1: st.error(f"**Required Heater:** Stream {c['Stream']} ({c['Q']:,.1f} kW)")
                for h in h_rem: 
                    if h['Q'] > 1: st.info(f"**Required Cooler:** Stream {h['Stream']} ({h['Q']:,.1f} kW)")
    else:
        st.warning("No Pinch Point detected.")

    st.markdown("---")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(match_summary).to_excel(writer, sheet_name='Matches', index=False)
    st.download_button(label="📥 Download HEN Report", data=output.getvalue(), file_name="HEN_Design.xlsx")
