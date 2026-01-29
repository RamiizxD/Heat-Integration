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
        "mCp": st.column_config.NumberColumn("mCp", format="%.2f"),
        "Ts": st.column_config.NumberColumn("Supply Temp", format="%.1f"),
        "Tt": st.column_config.NumberColumn("Target Temp", format="%.1f"),
    }
)

# Initialize Session State
if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False

if st.button("Run Thermal Analysis"):
    st.session_state.run_clicked = True

# --- LOGIC ENGINE ---
def run_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)
    
    df['Q_Raw'] = df['mCp'] * abs(df['Ts'] - df['Tt'])
    q_h_raw = df[df['Type'] == 'Hot']['Q_Raw'].sum()
    q_c_raw = df[df['Type'] == 'Cold']['Q_Raw'].sum()
    
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_load = df[(df['Type'] == 'Hot') & (df['S_Ts'] >= hi) & (df['S_Tt'] <= lo)]['mCp'].sum() * (hi - lo)
        c_load = df[(df['Type'] == 'Cold') & (df['S_Ts'] <= lo) & (df['S_Tt'] >= hi)]['mCp'].sum() * (hi - lo)
        intervals.append({'hi': hi, 'lo': lo, 'net': h_load - c_load})
    
    int_df = pd.DataFrame(intervals)
    infeasible = [0] + list(int_df['net'].cumsum())
    qh_min = abs(min(min(infeasible), 0))
    feasible = [qh_min + val for val in infeasible]
    pinch_t = temps[feasible.index(0)] if 0 in feasible else None
    
    return qh_min, feasible[-1], pinch_t, temps, feasible, df, q_h_raw, q_c_raw

def match_logic(df, pinch_t, side):
    sub = df.copy()
    if side == 'Above':
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(lower=pinch_t), sub['S_Tt'].clip(lower=pinch_t)
    else:
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(upper=pinch_t), sub['S_Tt'].clip(upper=pinch_t)
    sub['Q'] = sub['mCp'] * abs(sub['S_Ts'] - sub['S_Tt'])
    streams = sub[sub['Q'] > 0.1].to_dict('records')
    hot, cold = [s for s in streams if s['Type'] == 'Hot'], [s for s in streams if s['Type'] == 'Cold']
    matches = []
    while any(h['Q'] > 1 for h in hot) and any(c['Q'] > 1 for c in cold):
        h = next(s for s in hot if s['Q'] > 1)
        c = next((s for s in cold if (s['mCp'] >= h['mCp'] if side=='Above' else h['mCp'] >= s['mCp']) and s['Q'] > 1), None)
        if c:
            m_q = min(h['Q'], c['Q'])
            h['Q'] -= m_q; c['Q'] -= m_q
            matches.append({"Match": f"{h['Stream']} ↔ {c['Stream']}", "Duty [kW]": round(m_q, 2)})
        else: break
    return matches, hot, cold

# --- OUTPUT DISPLAY ---
if st.session_state.run_clicked:
    if dt_min is None or edited_df.empty:
        st.error("Error: Please provide ΔTmin and Stream Data.")
        st.session_state.run_clicked = False
    else:
        qh, qc, pinch, t_plot, q_plot, processed_df, q_h_raw, q_c_raw = run_logic(edited_df, dt_min)
        
        # 2. Results
        st.markdown("---")
        st.subheader("2. Pinch Analysis Result")
        res_col_metrics, res_col_chart = st.columns([1, 2])
        with res_col_metrics:
            st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
            st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
            st.metric("Pinch Temp", f"{pinch} °C" if pinch is not None else "N/A")
        with res_col_chart:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_plot, y=q_plot, mode='lines+markers', line=dict(color='#1f77b4')))
            if pinch is not None: fig.add_vline(x=pinch, line_dash="dash", line_color="red")
            fig.update_layout(height=400, xaxis_title="Temperature [°C]", yaxis_title="Net Enthalpy [kW]")
            st.plotly_chart(fig, use_container_width=True)

        # 3. Matching
        st.markdown("---")
        st.subheader("3. Heat Exchanger Network Matching")
        if pinch is not None:
            l, r = st.columns(2)
            for i, side in enumerate(['Above', 'Below']):
                matches, h_rem, c_rem = match_logic(processed_df, pinch, side)
                with (l if i == 0 else r):
                    st.write(f"**Matches {side} Pinch**")
                    if matches: st.table(pd.DataFrame(matches))
                    else: st.info(f"No matches {side.lower()} pinch.")
                    for h in h_rem: 
                        if h['Q'] > 1: st.warning(f"Cooler: {h['Stream']} ({h['Q']:,.1f} kW)")
                    for c in c_rem: 
                        if c['Q'] > 1: st.warning(f"Heater: {c['Stream']} ({c['Q']:,.1f} kW)")

        # 4. Economics
        st.markdown("---")
        st.subheader("4. Economic Assessment")
        calc_savings = st.radio("Calculate utility savings?", ["No", "Yes"], index=0)
        
        if calc_savings == "Yes":
            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                cost_unit = st.selectbox("Currency Unit", ["$/kWh", "$/MWh", "€/kWh", "£/kWh"])
                op_hours = st.number_input("Annual Operating Hours", value=8760)
            with e_col2:
                price_hot = st.number_input(f"Hot Utility Price [{cost_unit}]", format="%.4f", key="p_hot")
            with e_col3:
                price_cold = st.number_input(f"Cold Utility Price [{cost_unit}]", format="%.4f", key="p_cold")

            multiplier = 1 if "kWh" in cost_unit else 0.001
            cost_no_int = ((q_h_raw * price_hot) + (q_c_raw * price_cold)) * multiplier * op_hours
            cost_int = ((qh * price_hot) + (qc * price_cold)) * multiplier * op_hours
            savings = cost_no_int - cost_int

            st.markdown("#### Annual Utility Comparison")
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Cost (No Integration)", f"{cost_no_int:,.2f}")
            s_col2.metric("Cost (With Integration)", f"{cost_int:,.2f}")
            s_col3.metric("Total Annual Savings", f"{savings:,.2f}", delta=f"{(savings/cost_no_int*100 if cost_no_int > 0 else 0):.1f}%")
else:

    st.info("Input system parameters and stream data to begin analysis.")
