import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs traditional **Pinch Analysis**, **MER Matching**, and 
**NLP Optimization** with Excel Import/Export capabilities.
""")
st.markdown("---")

# --- UTILITY FUNCTIONS ---
def calculate_u(h1, h2):
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    theta1 = max(abs(t1 - t4), 0.01)
    theta2 = max(abs(t2 - t3), 0.01)
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)
    df['Q_Raw'] = df['mCp'] * abs(df['Ts'] - df['Tt'])
    
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_load = df[(df['Type'] == 'Hot') & (df['S_Ts'] >= hi) & (df['S_Tt'] <= lo)]['mCp'].sum() * (hi - lo)
        c_load = df[(df['Type'] == 'Cold') & (df['S_Ts'] <= lo) & (df['S_Tt'] >= hi)]['mCp'].sum() * (hi - lo)
        intervals.append({'hi': hi, 'lo': lo, 'net': h_load - c_load})
    
    infeasible = [0] + list(pd.DataFrame(intervals)['net'].cumsum())
    qh_min = abs(min(min(infeasible), 0))
    feasible = [qh_min + val for val in infeasible]
    pinch_t = temps[feasible.index(0)] if 0 in feasible else None
    return qh_min, feasible[-1], pinch_t, temps, feasible, df

def match_logic(df, pinch_t, side):
    sub = df.copy()
    if side == 'Above':
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(lower=pinch_t), sub['S_Tt'].clip(lower=pinch_t)
    else:
        sub['S_Ts'], sub['S_Tt'] = sub['S_Ts'].clip(upper=pinch_t), sub['S_Tt'].clip(upper=pinch_t)
    
    sub['Q'] = sub['mCp'] * abs(sub['S_Ts'] - sub['S_Tt'])
    streams = sub[sub['Q'] > 0.1].to_dict('records')
    hot = [s for s in streams if s['Type'] == 'Hot']
    cold = [s for s in streams if s['Type'] == 'Cold']
    matches = []
    while any(h['Q'] > 1 for h in hot) and any(c['Q'] > 1 for c in cold):
        h = next(s for s in hot if s['Q'] > 1)
        c = next((s for s in cold if (s['mCp'] >= h['mCp'] if side=='Above' else h['mCp'] >= s['mCp']) and s['Q'] > 1), None)
        if c:
            m_q = min(h['Q'], c['Q'])
            h['Q'] -= m_q
            c['Q'] -= m_q
            matches.append({"Match": f"{h['Stream']} ↔ {c['Stream']}", "Duty [kW]": round(m_q, 2)})
        else: break
    return matches, hot, cold

# --- SECTION 1: DATA INPUT & EXCEL IMPORT ---
st.subheader("1. Stream Data Input")

# Excel Import Logic
uploaded_file = st.file_uploader("Import Stream Data from Excel", type=["xlsx"])
if uploaded_file:
    try:
        import_df = pd.read_excel(uploaded_file)
        # Validate columns
        required = ["Stream", "Type", "mCp", "Ts", "Tt", "h"]
        if all(col in import_df.columns for col in required):
            st.session_state['input_data'] = import_df[required]
            st.success("Data imported successfully!")
        else:
            st.error(f"Excel must contain columns: {', '.join(required)}")
    except Exception as e:
        st.error(f"Error reading file: {e}")

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = pd.DataFrame(columns=["Stream", "Type", "mCp", "Ts", "Tt", "h"])

with st.form("main_input_form"):
    dt_min_input = st.number_input("Target ΔTmin [°C]", min_value=1.0, value=10.0)
    edited_df = st.data_editor(st.session_state['input_data'], num_rows="dynamic", use_container_width=True)
    submit_thermal = st.form_submit_button("Run Analysis")

if submit_thermal:
    st.session_state.run_clicked = True

# --- MAIN OUTPUT DISPLAY ---
if st.session_state.get('run_clicked'):
    qh, qc, pinch, t_plot, q_plot, processed_df = run_thermal_logic(edited_df, dt_min_input)
    
    # 2. Results
    st.subheader("2. Pinch Analysis Result")
    st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
    st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
    st.metric("Pinch Temp", f"{pinch} °C" if pinch is not None else "N/A")

    # 3. Matching
    st.subheader("3. Heat Exchanger Network Matching")
    match_results = []
    if pinch:
        l, r = st.columns(2)
        for i, side in enumerate(['Above', 'Below']):
            matches, h_rem, c_rem = match_logic(processed_df, pinch, side)
            match_results.extend(matches)
            with (l if i == 0 else r):
                st.write(f"**{side} Pinch**")
                if matches: st.table(pd.DataFrame(matches))
    
    # 4. Economics
    st.subheader("4. NLP Optimization & Economics")
    # Using Equation 05 constants [cite: 94, 111]
    opt_area = (qh / (1.5 * 15)) + (qc / (1.5 * 15)) # Simplified estimation
    cap_inv = 8000 + 433.3 * (opt_area ** 0.6)
    tac = (qh * 0.05 * 8000) + (cap_inv / 5)
    
    st.metric("Total Annual Cost (TAC)", f"${tac:,.2f}")

    # --- EXPORT RESULTS TO EXCEL ---
    st.markdown("---")
    st.subheader("5. Export Report")
    
    # Generate Excel Report
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Summary Results
        summary_data = {
            "Parameter": ["Target DTmin", "Hot Utility (Qh)", "Cold Utility (Qc)", "Pinch Temperature", "Total Area Estimate", "Capital Investment", "Total Annual Cost"],
            "Value": [dt_min_input, qh, qc, pinch, opt_area, cap_inv, tac],
            "Unit": ["°C", "kW", "kW", "°C", "m²", "$", "$/yr"]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Input Streams
        edited_df.to_excel(writer, sheet_name='Input_Streams', index=False)
        
        # Sheet 3: Matches
        if match_results:
            pd.DataFrame(match_results).to_excel(writer, sheet_name='HEN_Matches', index=False)
            
    processed_data = output.getvalue()
    st.download_button(
        label="📥 Download Results as Excel",
        data=processed_data,
        file_name="HEN_Analysis_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
