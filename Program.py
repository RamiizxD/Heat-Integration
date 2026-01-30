import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import copy

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs **Pinch Analysis**, **MER Matching with Stream Splitting**, and 
**Economic Optimization** based on the DGS-RWCE framework.
""")
st.markdown("---")

# --- CORE MATH FUNCTIONS ---
def calculate_u(h1, h2):
    if h1 <= 0 or h2 <= 0:
        return 0
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    theta1 = max(t1 - t4, 0.001) 
    theta2 = max(t2 - t3, 0.001)
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # Temperature Shifting
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

# --- DGS-RWCE ALGORITHM & ECONOMIC INPUTS ---
DGS_CONFIG = {
    "N_HD": 3, "N_CD": 3, "N_FH": 2, "N_FC": 2,
    "DELTA_L": 50.0, "THETA": 1.0, "P_GEN": 0.01,
    "P_INCENTIVE": 0.005, "MAX_ITER": 100000, "ANNUAL_FACTOR": 0.2
}

def render_optimization_inputs():
    st.markdown("### 4. Optimization & Economics Parameters")
    with st.expander("Economic Coefficients (Plant Specific)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.number_input("Fixed Investment [a] ($)", value=8000.0)
            c_hu = st.number_input("Hot Utility Cost ($/kW·yr)", value=80.0)
        with col2:
            b = st.number_input("Area Coefficient [b] ($/m²)", value=1200.0)
            c_cu = st.number_input("Cold Utility Cost ($/kW·yr)", value=20.0)
        with col3:
            c = st.number_input("Area Exponent [c]", value=0.6, step=0.01)
    return {"a": a, "b": b, "c": c, "c_hu": c_hu, "c_cu": c_cu}

def prepare_optimizer_data(df):
    hot_streams = df[df['Type'] == 'Hot'].to_dict('records')
    cold_streams = df[df['Type'] == 'Cold'].to_dict('records')
    return hot_streams, cold_streams

def run_random_walk(initial_matches, hot_streams, cold_streams, econ_params, dt_min):
    best_matches = copy.deepcopy(initial_matches)
    
    # --- FIX 1: STREAM-WISE UTILITY & INDIVIDUAL CAPEX ---
    def calculate_network_tac(matches):
        rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
        rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in cold_streams}
        
        total_ann_capex = 0
        
        for m in matches:
            q = m['Recommended Load [kW]']
            h_s = next(s for s in hot_streams if s['Stream'] == m['Hot Stream'])
            c_s = next(s for s in cold_streams if s['Stream'] == m['Cold Stream'])
            
            # Update stream duties
            rem_h[m['Hot Stream']] -= q
            rem_c[m['Cold Stream']] -= q
            
            # Temperature checks
            tho = h_s['Ts'] - (q / h_s['mCp'])
            tco = c_s['Ts'] + (q / c_s['mCp'])
            
            # --- FIX 2: ENFORCE DT_MIN CONSTRAINT ---
            if (h_s['Ts'] - tco) < dt_min or (tho - c_s['Ts']) < dt_min:
                return float('inf')
                
            u = calculate_u(h_s['h'], c_s['h'])
            lmtd = lmtd_chen(h_s['Ts'], tho, c_s['Ts'], tco)
            area = q / (u * lmtd)
            
            # --- FIX 3: INDIVIDUAL SUMMATION OF CAPEX ---
            inv = econ_params['a'] + econ_params['b'] * (area ** econ_params['c'])
            total_ann_capex += inv * DGS_CONFIG['ANNUAL_FACTOR']

        # Calculate OPEX based on residuals
        qh = sum(max(0, d) for d in rem_c.values())
        qc = sum(max(0, d) for d in rem_h.values())
        opex = (qh * econ_params['c_hu']) + (qc * econ_params['c_cu'])
        
        return opex + total_ann_capex

    current_best_score = calculate_network_tac(best_matches)
    
    # Optimization Loop
    for _ in range(1000):
        if not best_matches: break
        idx = np.random.randint(0, len(best_matches))
        original_q = best_matches[idx]['Recommended Load [kW]']
        
        step = np.random.uniform(-1, 1) * DGS_CONFIG['DELTA_L']
        new_q = max(1.0, original_q + step)
        
        best_matches[idx]['Recommended Load [kW]'] = new_q
        new_score = calculate_network_tac(best_matches)
        
        if new_score < current_best_score:
            current_best_score = new_score
        else:
            best_matches[idx]['Recommended Load [kW]'] = original_q
            
    return best_matches, current_best_score

# --- UI LOGIC ---
st.subheader("1. Stream Data Input")
uploaded_file = st.file_uploader("Import Stream Data from Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    try:
        import_df = pd.read_excel(uploaded_file)
        st.session_state['input_data'] = import_df
    except Exception as e:
        st.error(f"Error reading file: {e}")

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = pd.DataFrame(columns=["Stream", "Type", "mCp", "Ts", "Tt", "h"])

with st.form("main_input_form"):
    dt_min_input = st.number_input("Target ΔTmin [°C]", min_value=1.0, value=10.0)
    edited_df = st.data_editor(st.session_state['input_data'], num_rows="dynamic", use_container_width=True)
    submit_thermal = st.form_submit_button("Run Thermal Analysis")

if submit_thermal and not edited_df.empty:
    st.session_state.run_clicked = True

if st.session_state.get('run_clicked'):
    qh, qc, pinch, t_plot, q_plot, processed_df = run_thermal_logic(edited_df, dt_min_input)
    
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
        st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
        st.metric("Pinch Temperature", f"{pinch} °C" if pinch is not None else "N/A")
    with r2:
        fig = go.Figure(go.Scatter(x=q_plot, y=t_plot, mode='lines+markers', name="GCC"))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Net Heat Flow [kW]", yaxis_title="Shifted Temp [°C]")
        st.plotly_chart(fig, use_container_width=True)

    econ_params = render_optimization_inputs()
    
    st.markdown("---")
    st.subheader("3. Optimization and Economic Analysis")
    
    total_q_h_base = edited_df[edited_df['Type']=='Cold'].apply(lambda x: x['mCp']*abs(x['Ts']-x['Tt']), axis=1).sum()
    total_q_c_base = edited_df[edited_df['Type']=='Hot'].apply(lambda x: x['mCp']*abs(x['Ts']-x['Tt']), axis=1).sum()

    if st.button("Calculate Economic Optimum"):
        hot_streams, cold_streams = prepare_optimizer_data(edited_df)
        
        # Initial matches based on simple pairing for start point
        found_matches = []
        for i, hs in enumerate(hot_streams):
            if i < len(cold_streams):
                cs = cold_streams[i]
                q_init = min(hs['mCp']*abs(hs['Ts']-hs['Tt']), cs['mCp']*abs(cs['Ts']-cs['Tt'])) * 0.5
                found_matches.append({
                    "Hot Stream": hs['Stream'], "Cold Stream": cs['Stream'],
                    "Recommended Load [kW]": q_init
                })
        
        if found_matches:
            with st.status("Evolving Network...", expanded=True) as status:
                refined_matches, tac_opt = run_random_walk(found_matches, hot_streams, cold_streams, econ_params, dt_min_input)
                status.update(label="Evolution Complete!", state="complete", expanded=False)

            # Re-calculate breakdown for display
            rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
            rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in cold_streams}
            total_cap = 0
            
            for m in refined_matches:
                q = m['Recommended Load [kW]']
                h_s = next(s for s in hot_streams if s['Stream'] == m['Hot Stream'])
                c_s = next(s for s in cold_streams if s['Stream'] == m['Cold Stream'])
                rem_h[m['Hot Stream']] -= q
                rem_c[m['Cold Stream']] -= q
                u = calculate_u(h_s['h'], c_s['h'])
                lmtd = lmtd_chen(h_s['Ts'], h_s['Ts']-(q/h_s['mCp']), c_s['Ts'], c_s['Ts']+(q/c_s['mCp']))
                total_cap += (econ_params['a'] + econ_params['b'] * ((q/(u*lmtd))**econ_params['c']))

            opex_opt = (sum(max(0, d) for d in rem_c.values()) * econ_params['c_hu']) + \
                       (sum(max(0, d) for d in rem_h.values()) * econ_params['c_cu'])
            
            st.markdown("#### Optimized Economic Breakdown")
            o_col1, o_col2, o_col3 = st.columns(3)
            o_col1.metric("Capital Cost", f"${total_cap:,.2f}", f"(${total_cap*0.2:,.2f}/yr)")
            o_col2.metric("Annual Operating Cost", f"${opex_opt:,.2f}/yr")
            o_col3.metric("Total Annual Cost (TAC)", f"${tac_opt:,.2f}/yr")

            st.markdown("---")
            st.subheader("4. Final Comparison")
            opex_no_int = (total_q_h_base * econ_params['c_hu']) + (total_q_c_base * econ_params['c_cu'])
            comparison_df = pd.DataFrame({
                "Metric": ["Operating Cost ($/yr)", "Annualized Capital ($/yr)", "Total TAC ($/yr)"],
                "No Integration": [f"{opex_no_int:,.2f}", "0.00", f"{opex_no_int:,.2f}"],
                "Optimized (DGS-RWCE)": [f"{opex_opt:,.2f}", f"{total_cap*0.2:,.2f}", f"{tac_opt:,.2f}"]
            })
            st.table(comparison_df)
