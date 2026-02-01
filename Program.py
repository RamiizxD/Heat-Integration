import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")
st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("---")

# --- 1. PRIMARY DATA INPUT ---
st.subheader("1. Stream Data & System Parameters")
col_param, _ = st.columns([1, 2])
with col_param:
    dt_min = st.number_input("Minimum Temperature Difference (ΔTmin) [°C]", value=None, step=1.0, placeholder="Required")

empty_df = pd.DataFrame(columns=["Stream", "Type", "mCp", "Ts", "Tt"])
edited_df = st.data_editor(
    empty_df, 
    num_rows="dynamic", 
    use_container_width=True,
    column_config={
        "Type": st.column_config.SelectboxColumn("Stream Type", options=["Hot", "Cold"], required=True),
        "mCp": st.column_config.NumberColumn("mCp [kW/°C]", format="%.2f"),
        "Ts": st.column_config.NumberColumn("Supply Temp [°C]", format="%.1f"),
        "Tt": st.column_config.NumberColumn("Target Temp [°C]", format="%.1f"),
    }
)

run_analysis = st.button("Run Thermal Analysis")

# --- LOGIC ENGINE ---
def calculate_pinch(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)
    
    df['Q_Total'] = df['mCp'] * abs(df['Ts'] - df['Tt'])
    q_hot_raw = df[df['Type'] == 'Hot']['Q_Total'].sum()
    q_cold_raw = df[df['Type'] == 'Cold']['Q_Total'].sum()
    
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_load = df[(df['Type'] == 'Hot') & (np.maximum(df['S_Ts'], df['S_Tt']) >= hi) & (np.minimum(df['S_Ts'], df['S_Tt']) <= lo)]['mCp'].sum() * (hi - lo)
        c_load = df[(df['Type'] == 'Cold') & (np.maximum(df['S_Ts'], df['S_Tt']) >= hi) & (np.minimum(df['S_Ts'], df['S_Tt']) <= lo)]['mCp'].sum() * (hi - lo)
        intervals.append({'net': h_load - c_load})
    
    infeasible = [0] + list(pd.DataFrame(intervals)['net'].cumsum())
    qh_min = abs(min(min(infeasible), 0))
    feasible = [qh_min + val for val in infeasible]
    qc_min = feasible[-1]
    pinch_t = temps[feasible.index(0)] if 0 in feasible else None
    
    return qh_min, qc_min, pinch_t, temps, feasible, q_hot_raw, q_cold_raw

# --- REFINED MATCHING LOGIC ---
def get_matches(df, pinch_t, side):
    sub = df.copy()
    sub[['mCp', 'Ts', 'Tt']] = sub[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # Define shifted bounds for integration
    if side == 'Above':
        # Shifted cold: T+dt. If shifted > pinch_t, it's above pinch.
        sub['Eff_Ts'] = np.where(sub['Type'] == 'Hot', sub['Ts'], sub['Ts'] + dt_min)
        sub['Eff_Tt'] = np.where(sub['Type'] == 'Hot', sub['Tt'], sub['Tt'] + dt_min)
        sub['Q'] = sub['mCp'] * (np.maximum(sub['Eff_Ts'], sub['Eff_Tt']) - np.maximum(pinch_t, np.minimum(sub['Eff_Ts'], sub['Eff_Tt'])))
    else:
        sub['Eff_Ts'] = np.where(sub['Type'] == 'Hot', sub['Ts'], sub['Ts'] + dt_min)
        sub['Eff_Tt'] = np.where(sub['Type'] == 'Hot', sub['Tt'], sub['Tt'] + dt_min)
        sub['Q'] = sub['mCp'] * (np.minimum(pinch_t, np.maximum(sub['Eff_Ts'], sub['Eff_Tt'])) - np.minimum(sub['Eff_Ts'], sub['Eff_Tt']))
    
    sub = sub[sub['Q'] > 0.1].copy()
    hot = sub[sub['Type'] == 'Hot'].to_dict('records')
    cold = sub[sub['Type'] == 'Cold'].to_dict('records')
    
    matches = []
    # Exhaustive loop to find all possible pairings
    for h in hot:
        for c in cold:
            if h['Q'] > 0.1 and c['Q'] > 0.1:
                # mCp Rule check
                rule_pass = (c['mCp'] >= h['mCp']) if side == 'Above' else (h['mCp'] >= c['mCp'])
                if rule_pass:
                    m_q = min(h['Q'], c['Q'])
                    h['Q'] -= m_q
                    c['Q'] -= m_q
                    matches.append({"Match": f"{h['Stream']} ↔ {c['Stream']}", "Duty (kW)": round(m_q, 2)})
    
    return matches, hot, cold

# --- OUTPUT DISPLAY ---
if run_analysis:
    if dt_min is None or edited_df.empty:
        st.error("Error: Please provide ΔTmin and Stream Data.")
    else:
        qh, qc, pinch, t_plot, q_plot, q_h_raw, q_c_raw = calculate_pinch(edited_df, dt_min)
        
        # --- 2. PINCH ANALYSIS RESULT ---
        st.markdown("---")
        st.subheader("2. Pinch Analysis Result")
        res_col_metrics, res_col_chart = st.columns([1, 2])
        
        with res_col_metrics:
            st.metric("Hot Utility Requirement (Qh_min)", f"{qh:,.2f} kW")
            st.metric("Cold Utility Requirement (Qc_min)", f"{qc:,.2f} kW")
            st.metric("Pinch Temperature (Hot Scale)", f"{pinch} °C" if pinch is not None else "N/A")
            
        with res_col_chart:
            fig = go.Figure()
            # SWAPPED: Temperature on X, Enthalpy on Y
            fig.add_trace(go.Scatter(x=t_plot, y=q_plot, mode='lines+markers', line=dict(color='#1f77b4', width=3)))
            if pinch is not None:
                fig.add_vline(x=pinch, line_dash="dash", line_color="red", annotation_text="Pinch")
            fig.update_layout(height=400, xaxis_title="Temperature (Hot Scale) [°C]", yaxis_title="Net Enthalpy [kW]", template="none")
            st.plotly_chart(fig, use_container_width=True)

        # --- 3. HEAT MATCHING ---
        st.markdown("---")
        st.subheader("3. Heat Exchanger Network Matching")
        if pinch is not None:
            m_above, h_a, c_a = get_matches(edited_df, pinch, 'Above')
            m_below, h_b, c_b = get_matches(edited_df, pinch, 'Below')
            
            ma_col, mb_col = st.columns(2)
            with ma_col:
                st.write("**Matches Above Pinch**")
                if m_above: st.table(pd.DataFrame(m_above))
                else: st.info("No process-to-process matches found above pinch.")
                for h in h_a: 
                    if h['Q'] > 1: st.warning(f"Remaining Hot Duty ({h['Stream']}): {h['Q']:,.1f} kW (Requires Utility)")
            with mb_col:
                st.write("**Matches Below Pinch**")
                if m_below: st.table(pd.DataFrame(m_below))
                else: st.info("No process-to-process matches found below pinch.")
                for c in c_b: 
                    if c['Q'] > 1: st.warning(f"Remaining Cold Duty ({c['Stream']}): {c['Q']:,.1f} kW (Requires Utility)")
        else:
            st.warning("No Pinch point identified; HEN matching constraints not applicable.")

        # --- 4. ECONOMIC ASSESSMENT ---
        st.markdown("---")
        st.subheader("4. Economic Assessment")
        calc_savings = st.checkbox("Enable annual utility cost comparison?")
        
        if calc_savings:
            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                cost_unit = st.selectbox("Unit", ["$/kWh", "$/MWh", "€/kWh", "£/kWh"])
                op_hours = st.number_input("Operating Hours/Year", value=8760)
            with e_col2:
                price_hot = st.number_input(f"Hot Utility Price [{cost_unit}]", format="%.4f")
            with e_col3:
                price_cold = st.number_input(f"Cold Utility Price [{cost_unit}]", format="%.4f")

            multiplier = 1 if "kWh" in cost_unit else 0.001
            cost_no_int = ((q_h_raw * price_hot) + (q_c_raw * price_cold)) * multiplier * op_hours
            cost_int = ((qh * price_hot) + (qc * price_cold)) * multiplier * op_hours
            savings = cost_no_int - cost_int

            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Base Case Utility Cost", f"{cost_no_int:,.2f}")
            s_col2.metric("Integrated Utility Cost", f"{cost_int:,.2f}")
            s_col3.metric("Annual Savings", f"{savings:,.2f}", delta=f"{(savings/cost_no_int*100 if cost_no_int > 0 else 0):.1f}%")
else:
    st.info("Awaiting technical data input.")
