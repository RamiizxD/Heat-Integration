import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import copy

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("---")

# --- CORE MATH FUNCTIONS ---
def calculate_u(h1, h2):
    if h1 <= 0 or h2 <= 0: return 0
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    theta1 = max(abs(t1 - t4), 0.1)
    theta2 = max(abs(t2 - t3), 0.1)
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
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
    pinch_t = temps[feasible.index(0)] if 0 in feasible else None
    return qh_min, feasible[-1], pinch_t, temps, feasible, df

# --- OPTIMIZATION HELPERS ---
def prepare_optimizer_data(df):
    hot_streams = df[df['Type'] == 'Hot'].to_dict('records')
    cold_streams = df[df['Type'] == 'Cold'].to_dict('records')
    return hot_streams, cold_streams

def find_q_dep(h_stream, c_stream, econ_params):
    q_ne = 1.0
    u_match = calculate_u(h_stream.get('h', 0), c_stream.get('h', 0))
    if u_match <= 0: return None
    q_limit = min(h_stream['mCp'] * (h_stream['Ts'] - h_stream['Tt']), c_stream['mCp'] * (c_stream['Tt'] - c_stream['Ts']))
    while q_ne < q_limit:
        tho, tco = h_stream['Ts'] - (q_ne / h_stream['mCp']), c_stream['Ts'] + (q_ne / c_stream['mCp'])
        if (h_stream['Ts'] - tco) <= 0 or (tho - c_stream['Ts']) <= 0: break
        lmtd = lmtd_chen(h_stream['Ts'], tho, c_stream['Ts'], tco)
        area = q_ne / (u_match * lmtd)
        inv = (econ_params['a'] + econ_params['b'] * (area ** econ_params['c'])) * 0.2
        if (inv - q_ne * (econ_params['c_hu'] + econ_params['c_cu'])) <= 0: return round(q_ne, 2)
        q_ne += 2.0
    return None

def run_random_walk(initial_matches, hot_streams, cold_streams, econ_params):
    best_matches = copy.deepcopy(initial_matches)
    def calc_score(matches):
        t_inv, t_q = 0, 0
        for m in matches:
            q = m['Recommended Load [kW]']
            hs = next(s for s in hot_streams if s['Stream'] == m['Hot Stream'])
            cs = next(s for s in cold_streams if s['Stream'] == m['Cold Stream'])
            u = calculate_u(hs['h'], cs['h'])
            tho, tco = hs['Ts'] - (q / hs['mCp']), cs['Ts'] + (q / cs['mCp'])
            if (hs['Ts'] - tco) <= 0.1 or (tho - cs['Ts']) <= 0.1: return float('inf')
            area = q / (u * lmtd_chen(hs['Ts'], tho, cs['Ts'], tco))
            t_inv += (econ_params['a'] + econ_params['b'] * (area ** econ_params['c']))
            t_q += q
        return (t_inv * 0.2) - (t_q * (econ_params['c_hu'] + econ_params['c_cu']))
    
    score = calc_score(best_matches)
    for _ in range(500):
        idx = np.random.randint(0, len(best_matches))
        old_q = best_matches[idx]['Recommended Load [kW]']
        best_matches[idx]['Recommended Load [kW]'] = max(1.0, old_q + np.random.uniform(-50, 50))
        new_score = calc_score(best_matches)
        if new_score < score: score = new_score
        else: best_matches[idx]['Recommended Load [kW]'] = old_q
    return best_matches, score

# --- SECTION 1: DATA INPUT ---
st.subheader("1. Stream Data Input")
uploaded_file = st.file_uploader("Import Stream Data from Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    try:
        st.session_state['input_data'] = pd.read_excel(uploaded_file)
        st.success("Excel data loaded!")
    except Exception as e:
        st.error(f"Error: {e}")

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = pd.DataFrame(columns=["Stream", "Type", "mCp", "Ts", "Tt", "h"])

with st.form("input_form"):
    dt_min = st.number_input("Target ΔTmin [°C]", value=10.0)
    edited_df = st.data_editor(st.session_state['input_data'], num_rows="dynamic", use_container_width=True)
    if st.form_submit_button("Run Analysis"): st.session_state.run_clicked = True

# --- SECTION 2: THERMAL RESULTS ---
if st.session_state.get('run_clicked') and not edited_df.empty:
    qh, qc, pinch, t_plot, q_plot, proc_df = run_thermal_logic(edited_df, dt_min)
    
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    r1, r2 = st.columns([1, 2])
    r1.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
    r1.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
    with r2:
        fig = go.Figure(go.Scatter(x=q_plot, y=t_plot, mode='lines+markers', name="GCC"))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Net Heat Flow [kW]", yaxis_title="Shifted Temp [°C]")
        st.plotly_chart(fig, use_container_width=True)

    # --- SECTION 4: ECONOMICS ---
    st.markdown("---")
    st.subheader("4. Optimization & Economics")
    col1, col2, col3 = st.columns(3)
    with col1: a = st.number_input("Fixed Investment ($)", value=8000.0)
    with col2: b = st.number_input("Area Coeff ($/m²)", value=800.0)
    with col3: c = st.number_input("Area Exponent", value=0.8)
    with col1: chu = st.number_input("Hot Utility ($/kW·yr)", value=80.0)
    with col2: ccu = st.number_input("Cold Utility ($/kW·yr)", value=20.0)
    econ = {"a": a, "b": b, "c": c, "c_hu": chu, "c_cu": ccu}

    if st.button("Calculate Economic Comparison"):
        # 1. NO INTEGRATION CASE
        total_q_hot_needed = proc_df[proc_df['Type']=='Cold'].apply(lambda x: x['mCp']*abs(x['Ts']-x['Tt']), axis=1).sum()
        total_q_cold_needed = proc_df[proc_df['Type']=='Hot'].apply(lambda x: x['mCp']*abs(x['Ts']-x['Tt']), axis=1).sum()
        opex_no = (total_q_hot_needed * chu) + (total_q_cold_needed * ccu)
        
        # 2. MER CASE (Target-based Energy Recovery)
        q_recovered_mer = total_q_hot_needed - qh
        opex_mer = (qh * chu) + (qc * ccu)
        # Simplified MER Area estimation for comparison
        u_proxy, lmtd_proxy = 0.5, 20.0 
        area_mer = q_recovered_mer / (u_proxy * lmtd_proxy)
        capex_mer = (a + b * (area_mer ** c)) * 0.2

        # 3. TAC OPTIMIZATION CASE
        hot_s, cold_s = prepare_optimizer_data(edited_df)
        f_matches = []
        for hs in hot_s:
            for cs in cold_s:
                qd = find_q_dep(hs, cs, econ)
                if qd: f_matches.append({"Hot Stream": hs['Stream'], "Cold Stream": cs['Stream'], "Recommended Load [kW]": qd})
        
        refined, opt_score = run_random_walk(f_matches, hot_s, cold_s, econ)
        q_rec_tac = sum(m['Recommended Load [kW]'] for m in refined)
        opex_tac = ((total_q_hot_needed - q_rec_tac) * chu) + ((total_q_cold_needed - q_rec_tac) * ccu)
        
        # Area estimation for TAC
        total_area_tac = 0
        for m in refined:
            hs = next(s for s in hot_s if s['Stream']==m['Hot Stream'])
            cs = next(s for s in cold_s if s['Stream']==m['Cold Stream'])
            tho, tco = hs['Ts'] - (m['Recommended Load [kW]'] / hs['mCp']), cs['Ts'] + (m['Recommended Load [kW]'] / cs['mCp'])
            total_area_tac += m['Recommended Load [kW]'] / (calculate_u(hs['h'], cs['h']) * lmtd_chen(hs['Ts'], tho, cs['Ts'], tco))
        capex_tac = (a + b * (total_area_tac ** c)) * 0.2

        # --- RESULTS TABLE ---
        st.markdown("---")
        st.subheader("5. Scenario Comparison Summary")
        comp_df = pd.DataFrame({
            "Scenario": ["No Integration", "MER Strategy", "TAC Optimized"],
            "Heat Recovery (kW)": [0, f"{q_recovered_mer:,.1f}", f"{q_rec_tac:,.1f}"],
            "Annual Operating Cost ($/yr)": [f"{opex_no:,.0f}", f"{opex_mer:,.0f}", f"{opex_tac:,.0f}"],
            "Annualized Capital ($/yr)": [0, f"{capex_mer:,.0f}", f"{capex_tac:,.0f}"],
            "Total Annual Cost (TAC)": [f"{opex_no:,.0f}", f"{opex_mer + capex_mer:,.0f}", f"{opex_tac + capex_tac:,.0f}"]
        })
        st.table(comp_df)
        st.success(f"Optimized TAC achieves a saving of **${(opex_no - (opex_tac + capex_tac)):,.2f}/yr** relative to the baseline.")

    # --- SECTION 6: EXPORT ---
    st.markdown("---")
    st.subheader("6. Export Results")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        edited_df.to_excel(writer, sheet_name='Input_Data', index=False)
        pd.DataFrame({"Metric": ["Qh", "Qc", "Pinch"], "Value": [qh, qc, pinch]}).to_excel(writer, sheet_name='Pinch_Summary', index=False)
    
    st.download_button(label="📥 Download HEN Report", data=output.getvalue(), file_name="HEN_Full_Report.xlsx")
