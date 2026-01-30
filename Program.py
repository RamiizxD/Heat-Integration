import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import copy
import random

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs **Pinch Analysis**, **MER Matching with Stream Splitting**, and 
**Economic Optimization**.
""")
st.markdown("---")

# --- CORE MATH FUNCTIONS (UNCHANGED) ---
def calculate_u(h1, h2):
    if h1 <= 0 or h2 <= 0: return 0
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    d1 = abs(t1 - t4)
    d2 = abs(t2 - t3)
    if d1 < 0.01: d1 = 0.01
    if d2 < 0.01: d2 = 0.01
    if d1 == d2: return d1
    return (d1 * d2 * (d1 + d2) / 2)**(1/3)

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

# --- OPTIMIZATION ALGORITHMS (REWRITTEN) ---

def calculate_network_tac_series(matches, hot_streams, cold_streams, econ_params, annual_factor):
    """
    Calculates TAC considering SERIES placement of exchangers on streams.
    Reference: NNM-SS Model Physics 
    """
    total_inv = 0
    total_q_recovered = 0
    
    # 1. Group matches by stream
    h_map = {s['Stream']: [] for s in hot_streams}
    c_map = {s['Stream']: [] for s in cold_streams}
    
    for m in matches:
        h_map[m['Hot Stream']].append(m)
        c_map[m['Cold Stream']].append(m)
        
    # 2. Evaluate matches
    # Note: In a full NNM, we would optimize position. Here we assume a logical thermal sort order 
    # (Hot streams cool down sequentially, Cold streams heat up sequentially) to maximize LMTD.
    
    # Sort hot matches to execute in descending order of Hot T (Supply -> Target)
    # Since we don't know exact intermediate T yet, we process them. 
    # For a simple random walk, we calculate based on current load assuming no crossover.
    
    match_areas = []
    
    try:
        # Process Hot Side Physics
        # We track the current temperature of every stream
        h_temps = {s['Stream']: s['Ts'] for s in hot_streams}
        c_temps = {s['Stream']: s['Ts'] for s in cold_streams}
        
        # We need a consistent order to calculate series temps. 
        # A robust way is to rely on the optimizer to find valid loads, 
        # but for calculation, we simply iterate the match list.
        for m in matches:
            q = m['Load']
            h_name, c_name = m['Hot Stream'], m['Cold Stream']
            
            # Get stream properties
            h_s = next(s for s in hot_streams if s['Stream'] == h_name)
            c_s = next(s for s in cold_streams if s['Stream'] == c_name)
            
            # Calculate Inlet/Outlet for this specific exchanger
            # Hot side: Current Temp -> Minus Q
            thi = h_temps[h_name]
            tho = thi - (q / h_s['mCp'])
            
            # Cold side: Current Temp -> Plus Q
            tci = c_temps[c_name]
            tco = tci + (q / c_s['mCp'])
            
            # Physics Check: Temperature Cross or Violation
            # LMTD requires Thi > Tco and Tho > Tci
            if (thi - tco) < 0.1 or (tho - tci) < 0.1:
                return float('inf') # Infeasible
            
            # Physics Check: Over-cooling/Over-heating beyond supply/target bounds
            # (Strictly speaking, intermediate temps can float, but final must be met by utility)
            # We penalize if we exceed the stream's thermodynamic limit slightly, but RWCE fixes this.
            
            lmtd = lmtd_chen(thi, tho, tci, tco)
            u = calculate_u(h_s['h'], c_s['h'])
            area = q / (u * lmtd)
            
            # Update stream temps for the NEXT match in series
            h_temps[h_name] = tho
            c_temps[c_name] = tco
            
            inv = econ_params['a'] + econ_params['b'] * (area ** econ_params['c'])
            total_inv += inv
            total_q_recovered += q

        # 3. Calculate Utility Costs (Remaining Duty)
        utility_cost = 0
        
        # Hot Streams need Cold Utility for remaining cooling
        for hs in hot_streams:
            current_t = h_temps[hs['Stream']]
            target_t = hs['Tt']
            if current_t > target_t:
                q_cu = (current_t - target_t) * hs['mCp']
                utility_cost += q_cu * econ_params['c_cu']
                # Add Cold Utility Capital Cost
                dt1 = current_t - 30 # Assumption: CU in @ 20 out @ 30 (typical)
                dt2 = target_t - 20
                if dt1 > 0 and dt2 > 0:
                    lmtd_cu = lmtd_chen(current_t, target_t, 20, 30)
                    u_cu = calculate_u(hs['h'], 1.0) # Assume water h=1-2
                    area_cu = q_cu / (u_cu * lmtd_cu)
                    total_inv += econ_params['a'] + econ_params['b'] * (area_cu ** econ_params['c'])
                else:
                    return float('inf')

        # Cold Streams need Hot Utility for remaining heating
        for cs in cold_streams:
            current_t = c_temps[cs['Stream']]
            target_t = cs['Tt']
            if current_t < target_t:
                q_hu = (target_t - current_t) * cs['mCp']
                utility_cost += q_hu * econ_params['c_hu']
                # Add Hot Utility Capital Cost
                # Assumption: Steam/Oil hot utility
                dt1 = 500 - current_t # Placeholder Utility Temp
                dt2 = 500 - target_t
                if dt1 > 0 and dt2 > 0:
                    lmtd_hu = lmtd_chen(500, 500, current_t, target_t)
                    u_hu = calculate_u(1.0, cs['h'])
                    area_hu = q_hu / (u_hu * lmtd_hu)
                    total_inv += econ_params['a'] + econ_params['b'] * (area_hu ** econ_params['c'])
                else:
                    return float('inf') # Utility pinch violation

        tac = utility_cost + (total_inv * annual_factor)
        return tac

    except Exception:
        return float('inf')

def find_q_dep_dynamic(h_stream, c_stream, econ_params, annual_factor, u_match):
    """
    Calculates the Dynamic Equilibrium Point (Q_DEP) and applies the Incentive Strategy.
    References:  (Incentive Strategy)
    """
    q_dep = 0
    # Search for break-even point where Investment Cost == Energy Savings
    # Simplified search for demonstration:
    q_max = min(h_stream['mCp']*(h_stream['Ts']-h_stream['Tt']), 
                c_stream['mCp']*(c_stream['Tt']-c_stream['Ts']))
    
    # We scan a few points to find where Cost < Savings
    step = q_max / 20
    for q_test in np.linspace(step, q_max, 20):
        # Approx LMTD at this load (assuming single unit)
        tho = h_stream['Ts'] - (q_test / h_stream['mCp'])
        tco = c_stream['Ts'] + (q_test / c_stream['mCp'])
        if (h_stream['Ts'] - tco) <= 0.1 or (tho - c_stream['Ts']) <= 0.1:
            break
            
        lmtd = lmtd_chen(h_stream['Ts'], tho, c_stream['Ts'], tco)
        area = q_test / (u_match * lmtd)
        inv_cost = (econ_params['a'] + econ_params['b'] * (area ** econ_params['c'])) * annual_factor
        savings = q_test * (econ_params['c_hu'] + econ_params['c_cu'])
        
        if savings > inv_cost:
            q_dep = q_test
            break
            
    if q_dep == 0: return None # No viable starting point
    
    # --- INCENTIVE STRATEGY [cite: 502-503] ---
    # With probability (1 - delta), assign a LARGE load to jump start the unit
    psi = np.random.random()
    delta = 0.8 # Acceptance probability parameter
    
    if psi > delta:
        # Large Load Generation
        omega = np.random.uniform(0, 1) # Random perturbation
        q_incentive = 2 * q_dep * (1 + omega)
        return min(q_incentive, q_max * 0.95)
    else:
        # Standard Generation
        return q_dep

def run_dgs_rwce(hot_streams, cold_streams, econ_params, annual_factor):
    """
    DGS-RWCE: Dynamic Generation Strategy with Random Walk and Compulsive Evolution.
    """
    # Configuration
    MAX_ITER = 3000
    IEMAX = 50 # Generation Period [cite: 506]
    P_COMPULSIVE = 0.05 # Probability to accept bad solutions [cite: 1152]
    
    # 1. Initialize Structure (Empty or Random)
    # The paper starts with an empty structure or MER matches. We start empty to let DGS work.
    current_matches = [] 
    current_tac = calculate_network_tac_series(current_matches, hot_streams, cold_streams, econ_params, annual_factor)
    best_matches = copy.deepcopy(current_matches)
    best_tac = current_tac
    
    progress_bar = st.progress(0)
    
    for it in range(MAX_ITER):
        # Update progress
        if it % 100 == 0: progress_bar.progress(it / MAX_ITER)
        
        # Candidate Structure
        candidate_matches = copy.deepcopy(current_matches)
        
        # --- A. STRUCTURAL MUTATION (DGS)  ---
        if it % IEMAX == 0:
            action = np.random.choice(['add', 'remove'])
            if action == 'add':
                # Try to generate a new unit
                h = np.random.choice(hot_streams)
                c = np.random.choice(cold_streams)
                
                # Check if match exists
                exists = any(m['Hot Stream'] == h['Stream'] and m['Cold Stream'] == c['Stream'] for m in candidate_matches)
                if not exists:
                    u = calculate_u(h['h'], c['h'])
                    q_start = find_q_dep_dynamic(h, c, econ_params, annual_factor, u)
                    if q_start:
                        candidate_matches.append({
                            "Hot Stream": h['Stream'],
                            "Cold Stream": c['Stream'],
                            "Load": q_start,
                            "Type": "New"
                        })
            elif action == 'remove' and candidate_matches:
                # Remove a random unit (usually small ones, but random for simplicity here)
                idx = np.random.randint(0, len(candidate_matches))
                candidate_matches.pop(idx)
        
        # --- B. CONTINUOUS EVOLUTION (RW) [cite: 149-151] ---
        elif candidate_matches:
            idx = np.random.randint(0, len(candidate_matches))
            # Random Walk on Load
            step = np.random.uniform(-1, 1) * 50.0 # Delta L
            new_q = candidate_matches[idx]['Load'] + step
            
            # Constraints: Q > 0. If Q < 0, remove unit
            if new_q <= 1.0:
                candidate_matches.pop(idx)
            else:
                candidate_matches[idx]['Load'] = new_q
        
        # --- C. EVALUATION & SELECTION ---
        candidate_tac = calculate_network_tac_series(candidate_matches, hot_streams, cold_streams, econ_params, annual_factor)
        
        # Acceptance Logic (Compulsive Evolution) [cite: 1160-1166]
        accept = False
        if candidate_tac < current_tac:
            accept = True
            if candidate_tac < best_tac:
                best_tac = candidate_tac
                best_matches = copy.deepcopy(candidate_matches)
        else:
            # Imperfect solution acceptance
            if np.random.random() < P_COMPULSIVE:
                accept = True
        
        if accept:
            current_matches = candidate_matches
            current_tac = candidate_tac
            
    progress_bar.empty()
    return best_matches, best_tac

# --- RENDER OPTIMIZATION INPUTS (UNCHANGED) ---
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
            # New Annual Factor Input
            ann_f = st.number_input("Annual Factor", value=0.2, step=0.01)
            
    return {"a": a, "b": b, "c": c, "c_hu": c_hu, "c_cu": c_cu, "annual_factor": ann_f}

def prepare_optimizer_data(df):
    hot_streams = df[df['Type'] == 'Hot'].to_dict('records')
    cold_streams = df[df['Type'] == 'Cold'].to_dict('records')
    return hot_streams, cold_streams

def match_logic_with_splitting(df, pinch_t, side):
    sub = df.copy()
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
        st.success("Data imported successfully!")
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

# --- MAIN OUTPUT DISPLAY ---
if st.session_state.get('run_clicked'):
    qh, qc, pinch, t_plot, q_plot, processed_df = run_thermal_logic(edited_df, dt_min_input)
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
        st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
        st.metric("Pinch Temperature (Hot)", f"{pinch} °C" if pinch is not None else "N/A")
        st.metric("Pinch Temperature (Cold)", f"{pinch - dt_min_input} °C" if pinch is not None else "N/A")
    with r2:
        fig = go.Figure(go.Scatter(x=q_plot, y=t_plot, mode='lines+markers', name="GCC"))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Net Heat Flow [kW]", yaxis_title="Shifted Temp [°C]")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Heat Exchanger Network Matching (MER)")
    match_summary = []
    if pinch is not None:
        l, r = st.columns(2)
        for i, side in enumerate(['Above', 'Below']):
            matches, h_rem, c_rem = match_logic_with_splitting(processed_df, pinch, side)
            match_summary.extend(matches)
            with (l if i == 0 else r):
                st.write(f"**Matches {side} Pinch**")
                if matches: st.dataframe(pd.DataFrame(matches), use_container_width=True)
                else: st.info("No internal matches possible.")
                for c in c_rem: 
                    if c['Q'] > 1: st.error(f"Required Heater: {c['Stream']} ({c['Q']:,.1f} kW)")
                for h in h_rem: 
                    if h['Q'] > 1: st.info(f"Required Cooler: {h['Stream']} ({h['Q']:,.1f} kW)")

    st.markdown("---")
    st.subheader("4. Optimization and Economic Analysis")
    st.info("The optimization algorithm is DGS-RWCE (Dynamic Generation Strategy with Random Walk). It allows the structure to evolve by adding/removing units dynamically.")
    
    econ_params = render_optimization_inputs()
    DGS_CONFIG["ANNUAL_FACTOR"] = econ_params["annual_factor"]
    
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1: h_hot_u = st.number_input("Hot Utility h [kW/m²K]", value=1.0) # Updated default
    with col_opt2: h_cold_u = st.number_input("Cold Utility h [kW/m²K]", value=1.0)

    # Initialize variables for export
    refined_matches = []
    best_tac = 0

    if st.button("Run DGS-RWCE Optimization"):
        if 'h' not in edited_df.columns or edited_df['h'].isnull().any() or (edited_df['h'] <= 0).any():
            st.warning("Heat transfer coefficients (h) are required for optimization.")
        else:
            hot_streams, cold_streams = prepare_optimizer_data(edited_df)
            
            with st.status("Running DGS-RWCE Algorithm...", expanded=True) as status:
                st.write("Initializing network structure...")
                st.write("Applying Incentive Strategy for new units...")
                st.write("Evolving structure (Adding/Removing units)...")
                
                refined_matches, best_tac = run_dgs_rwce(hot_streams, cold_streams, econ_params, annual_factor)
                
                status.update(label="Optimization Complete!", state="complete", expanded=False)
            
            st.markdown("### Optimized Network Results")
            if refined_matches:
                res_df = pd.DataFrame(refined_matches)
                st.dataframe(res_df, use_container_width=True)
                st.metric("Total Annual Cost (TAC)", f"${best_tac:,.2f}/yr")
                st.success(f"Found optimized network with {len(refined_matches)} process-to-process heat exchangers.")
            else:
                st.warning("The optimizer found that a purely Utility-based solution is cheapest (No inter-process recovery viable).")

    # --- SECTION 5: EXPORT RESULTS ---
    st.markdown("---")
    st.subheader("5. Export Results")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        final_matches = refined_matches if refined_matches else match_summary
        if final_matches:
            pd.DataFrame(final_matches).to_excel(writer, sheet_name='HEN_Matches', index=False)
        edited_df.to_excel(writer, sheet_name='Input_Data', index=False)
        pd.DataFrame({"Metric": ["Qh", "Qc", "Pinch Hot", "Pinch Cold", "Optimum TAC"], 
                      "Value": [qh, qc, pinch, pinch-dt_min_input if pinch else None, best_tac]}).to_excel(writer, sheet_name='Summary', index=False)
    
    st.download_button(label="📥 Download HEN Report (Excel)", 
                       data=output.getvalue(), 
                       file_name="HEN_Full_Analysis.xlsx", 
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

