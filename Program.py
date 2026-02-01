import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import copy

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs **Pinch Analysis**, **MER Matching with Stream Splitting**, and 
**Economic Optimization** using the DGS-RWCE algorithm.
""")
st.markdown("---")

# --- CORE MATH FUNCTIONS ---
def calculate_u(h1, h2):
    """Calculate overall heat transfer coefficient"""
    if h1 <= 0 or h2 <= 0:
        return 0
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    """
    Chen approximation for LMTD - more accurate than simple LMTD
    Reference: Chen (1987)
    """
    theta1 = max(t1 - t4, 0.001) 
    theta2 = max(t2 - t3, 0.001)
    if abs(theta1 - theta2) < 0.01: 
        return theta1
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def run_thermal_logic(df, dt):
    """Perform pinch analysis to find minimum utility requirements"""
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # Shift temperatures
    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)
    
    # Create temperature intervals
    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)
    intervals = []
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_mcp = df[(df['Type'] == 'Hot') & (df['S_Ts'] >= hi) & (df['S_Tt'] <= lo)]['mCp'].sum()
        c_mcp = df[(df['Type'] == 'Cold') & (df['S_Ts'] <= lo) & (df['S_Tt'] >= hi)]['mCp'].sum()
        intervals.append({'hi': hi, 'lo': lo, 'net': (h_mcp - c_mcp) * (hi - lo)})
    
    # Find pinch point
    infeasible = [0] + list(pd.DataFrame(intervals)['net'].cumsum())
    qh_min = abs(min(min(infeasible), 0))
    feasible = [qh_min + val for val in infeasible]
    pinch_t = temps[feasible.index(0)] if 0 in feasible else None
    
    return qh_min, feasible[-1], pinch_t, temps, feasible, df

# --- DGS-RWCE ALGORITHM & ECONOMIC INPUTS ---
DGS_CONFIG = {
    "N_HD": 3, "N_CD": 3, "N_FH": 2, "N_FC": 2,
    "DELTA_L": 50.0, "THETA": 1.0, "P_GEN": 0.01,
    "P_INCENTIVE": 0.005, "MAX_ITER": 5000, "ANNUAL_FACTOR": 0.2,
    "MIN_SPLIT_RATIO": 0.10  # From paper: Section 3.2 - prune splits < 10%
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

def prune_and_normalize_matches(matches, streams_data):
    """
    Implements the paper's pruning logic (Section 3.2):
    1. Removes units with heat load < 10% of the total stream capacity.
    """
    pruned_matches = []
    for m in matches:
        hs = next((s for s in streams_data if s['Stream'] == m['Hot Stream']), None)
        if not hs:
            continue
            
        total_q = hs['mCp'] * abs(hs['Ts'] - hs['Tt'])
        if total_q <= 0:
            continue
            
        split_ratio = m['Recommended Load [kW]'] / total_q
        
        # Only keep matches with split ratio >= 10% (from paper)
        if split_ratio >= DGS_CONFIG["MIN_SPLIT_RATIO"]: 
            pruned_matches.append(m)
        
    return pruned_matches

def match_logic_with_splitting(df, pinch_t, side):
    """
    MER matching with improved split ratio handling
    Applies minimum split ratio threshold to prevent tiny splits
    """
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
        h = next((s for s in hot if s['Q'] > 1), None)
        if not h:
            break
            
        # Try to find suitable match based on CP matching
        c = next((s for s in cold if (s['mCp'] >= h['mCp'] if side=='Above' else h['mCp'] >= s['mCp']) and s['Q'] > 1), None)
        is_split = False
        
        if not c:
            c = next((s for s in cold if s['Q'] > 1), None)
            is_split = True
            
        if c:
            m_q = min(h['Q'], c['Q'])
            h_ratio = m_q / total_duties[h['Stream']] if total_duties[h['Stream']] > 0 else 0
            
            # Apply minimum split ratio threshold
            if h_ratio >= DGS_CONFIG["MIN_SPLIT_RATIO"] or h_ratio >= 0.99:
                ratio_text = f"{round(h_ratio, 2)} " if h_ratio < 0.99 else ""
                match_str = f"{ratio_text}Stream {h['Stream']} ↔ {c['Stream']}"
                h['Q'] -= m_q
                c['Q'] -= m_q
                matches.append({
                    "Match": match_str, 
                    "Duty [kW]": round(m_q, 2), 
                    "Type": "Split" if is_split or (0 < h_ratio < 0.99) else "Direct",
                    "Hot_Stream_ID": h['Stream'],
                    "Cold_Stream_ID": c['Stream']
                })
            else:
                # Skip this match if split ratio is too small
                break
        else:
            break
            
    return matches, hot, cold

def find_q_dep(h_stream, c_stream, econ_params, dt_min):
    """
    Find Dynamic Equilibrium Point (DEP) for heat load
    This is a critical function from the paper (Section 3.2.2)
    """
    q_ne = 1.0
    theta = DGS_CONFIG["THETA"]
    u_match = calculate_u(h_stream.get('h', 0), c_stream.get('h', 0))
    if u_match <= 0: 
        return None

    q_limit = min(
        h_stream['mCp'] * abs(h_stream['Ts'] - h_stream['Tt']), 
        c_stream['mCp'] * abs(c_stream['Tt'] - c_stream['Ts'])
    )

    max_iterations = 1000
    iteration = 0
    
    while q_ne < q_limit and iteration < max_iterations:
        iteration += 1
        
        # Calculate outlet temperatures
        tho = h_stream['Ts'] - (q_ne / h_stream['mCp'])
        tco = c_stream['Ts'] + (q_ne / c_stream['mCp'])
        
        # Check temperature feasibility
        if (h_stream['Ts'] - tco) < dt_min or (tho - c_stream['Ts']) < dt_min: 
            break
        
        # Calculate area and costs
        lmtd = lmtd_chen(h_stream['Ts'], tho, c_stream['Ts'], tco)
        if lmtd <= 0:
            break
            
        area = q_ne / (u_match * lmtd)
        ann_inv = (econ_params['a'] + econ_params['b'] * (area ** econ_params['c'])) * DGS_CONFIG['ANNUAL_FACTOR']
        util_savings = q_ne * (econ_params['c_hu'] + econ_params['c_cu'])
        
        # Correct DEP condition from paper
        if abs(ann_inv - util_savings) <= 0.01:  # Found DEP
            return round(q_ne, 2)
        elif ann_inv > util_savings:  # Haven't reached DEP yet
            q_ne += np.random.uniform(0.1, 0.5) * theta 
        else:  # Passed DEP
            return round(q_ne, 2)
    
    return None

def calculate_current_tac(matches, hot_streams, cold_streams, econ_params, dt_min):
    """Calculate Total Annual Cost for current network configuration"""
    rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
    rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in cold_streams}
    total_inv = 0
    
    for m in matches:
        q = m['Recommended Load [kW]']
        if q <= 0.001: 
            continue
            
        h_s = next((s for s in hot_streams if s['Stream'] == m['Hot Stream']), None)
        c_s = next((s for s in cold_streams if s['Stream'] == m['Cold Stream']), None)
        
        if not h_s or not c_s:
            continue
        
        # Check feasibility
        if q > rem_h.get(m['Hot Stream'], 0) + 0.1 or q > rem_c.get(m['Cold Stream'], 0) + 0.1:
            return float('inf')

        # Calculate temperatures
        tho = h_s['Ts'] - (q / h_s['mCp'])
        tco = c_s['Ts'] + (q / c_s['mCp'])
        
        # Check minimum temperature approach
        if (h_s['Ts'] - tco) < dt_min or (tho - c_s['Ts']) < dt_min: 
            return float('inf')
        
        # Update remaining duties
        rem_h[m['Hot Stream']] -= q
        rem_c[m['Cold Stream']] -= q
        
        # Calculate area and investment
        u = calculate_u(h_s['h'], c_s['h'])
        if u <= 0:
            return float('inf')
            
        lmtd = lmtd_chen(h_s['Ts'], tho, c_s['Ts'], tco)
        if lmtd <= 0:
            return float('inf')
            
        area = q / (u * lmtd)
        total_inv += (econ_params['a'] + econ_params['b'] * (area ** econ_params['c']))

    # Calculate utility costs
    actual_qh = sum(max(0, val) for val in rem_c.values())
    actual_qc = sum(max(0, val) for val in rem_h.values())
    opex = (actual_qh * econ_params['c_hu']) + (actual_qc * econ_params['c_cu'])
    
    return opex + (total_inv * DGS_CONFIG['ANNUAL_FACTOR'])

def run_random_walk(initial_matches, hot_streams, cold_streams, econ_params, dt_min):
    """
    Random walk with compulsive evolution (RWCE)
    """
    best_matches = copy.deepcopy(initial_matches)
    current_tac = calculate_current_tac(best_matches, hot_streams, cold_streams, econ_params, dt_min)
    
    if current_tac == float('inf'):
        return best_matches, current_tac
    
    no_improvement = 0
    delta_l = DGS_CONFIG['DELTA_L']
    
    for it in range(DGS_CONFIG['MAX_ITER']):
        if not best_matches: 
            break
            
        idx = np.random.randint(0, len(best_matches))
        old_q = best_matches[idx]['Recommended Load [kW]']
        
        # Adaptive step size
        if no_improvement > 500:
            delta_l = max(1.0, delta_l * 0.95)
            no_improvement = 0
            
        step = np.random.uniform(-1, 1) * delta_l
        best_matches[idx]['Recommended Load [kW]'] = max(0.0, old_q + step)
        
        new_tac = calculate_current_tac(best_matches, hot_streams, cold_streams, econ_params, dt_min)
        
        if new_tac < current_tac:
            current_tac = new_tac
            no_improvement = 0
        else:
            best_matches[idx]['Recommended Load [kW]'] = old_q
            no_improvement += 1
            
    return [m for m in best_matches if m['Recommended Load [kW]'] > 0.1], current_tac

def calculate_no_integration_costs(df, econ_params):
    """Calculate costs with no heat integration (all utilities)"""
    hot_streams = df[df['Type'] == 'Hot']
    cold_streams = df[df['Type'] == 'Cold']
    
    # All hot streams need cooling
    qc_total = sum(hot_streams['mCp'] * abs(hot_streams['Ts'] - hot_streams['Tt']))
    
    # All cold streams need heating
    qh_total = sum(cold_streams['mCp'] * abs(cold_streams['Ts'] - cold_streams['Tt']))
    
    opex = (qh_total * econ_params['c_hu']) + (qc_total * econ_params['c_cu'])
    
    return {
        'capital': 0.0,
        'ann_capital': 0.0,
        'opex': opex,
        'tac': opex,
        'qh': qh_total,
        'qc': qc_total
    }

def calculate_mer_capital_properly(match_summary, processed_df, econ_params):
    """
    Calculate MER capital cost using ACTUAL TEMPERATURES (not shifted)
    CRITICAL FIX: Must use Ts/Tt, NOT S_Ts/S_Tt for cost calculations
    """
    cap_mer = 0
    detailed_costs = []
    
    for m in match_summary:
        duty = m['Duty [kW]']
        if duty <= 0:
            continue
        
        # Get stream IDs from the match
        h_stream_id = m.get('Hot_Stream_ID')
        c_stream_id = m.get('Cold_Stream_ID')
        
        if h_stream_id is None or c_stream_id is None:
            continue
        
        # Find the streams using ACTUAL temperatures
        h_stream = None
        c_stream = None
        
        for _, row in processed_df.iterrows():
            if row['Stream'] == h_stream_id and row['Type'] == 'Hot':
                h_stream = row
            if row['Stream'] == c_stream_id and row['Type'] == 'Cold':
                c_stream = row
        
        if h_stream is not None and c_stream is not None:
            # CRITICAL: Use ACTUAL temperatures (Ts, Tt), NOT shifted (S_Ts, S_Tt)
            u = calculate_u(h_stream['h'], c_stream['h'])
            
            if u > 0:
                # Calculate outlet temperatures using ACTUAL inlet temperatures
                tho = h_stream['Ts'] - (duty / h_stream['mCp'])
                tco = c_stream['Ts'] + (duty / c_stream['mCp'])
                
                # Calculate LMTD using ACTUAL temperatures
                lmtd = lmtd_chen(h_stream['Ts'], tho, c_stream['Ts'], tco)
                
                if lmtd > 0:
                    area = duty / (u * lmtd)
                    cost = econ_params['a'] + econ_params['b'] * (area ** econ_params['c'])
                    cap_mer += cost
                    
                    detailed_costs.append({
                        'Match': m['Match'],
                        'Duty': duty,
                        'U': u,
                        'LMTD': lmtd,
                        'Area': area,
                        'Cost': cost
                    })
    
    return cap_mer, detailed_costs

# --- UI LOGIC ---
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
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), 
                         xaxis_title="Net Heat Flow [kW]", yaxis_title="Shifted Temp [°C]")
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
                if matches:
                    # Display matches table
                    display_df = pd.DataFrame(matches)[['Match', 'Duty [kW]', 'Type']]
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Calculate and display utilities for this side
                    total_duty = sum([m['Duty [kW]'] for m in matches])
                    st.caption(f"Total Heat Recovered: {total_duty:.2f} kW")
                else: 
                    st.info("No internal matches possible.")
        
        # Display overall utilities below the tables
        st.markdown("#### Utility Requirements")
        util_col1, util_col2 = st.columns(2)
        util_col1.metric("Hot Utility Required", f"{qh:,.2f} kW", help="Minimum hot utility from pinch analysis")
        util_col2.metric("Cold Utility Required", f"{qc:,.2f} kW", help="Minimum cold utility from pinch analysis")

    econ_params = render_optimization_inputs() 
    
    # MER Economics Calculation - FIXED VERSION
    # Calculate capital cost properly using ACTUAL temperatures
    cap_mer, mer_details = calculate_mer_capital_properly(match_summary, processed_df, econ_params)
    
    ann_cap_mer = cap_mer * DGS_CONFIG['ANNUAL_FACTOR']
    # Use qh and qc from pinch analysis (minimum utilities)
    opex_mer = (qh * econ_params['c_hu']) + (qc * econ_params['c_cu'])
    tac_mer = opex_mer + ann_cap_mer

    st.markdown("#### MER Economic Breakdown")
    
    # Display costs
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Capital Cost", f"${cap_mer:,.2f}", f"(${ann_cap_mer:,.2f}/yr)")
    m_col2.metric("Annual Operating Cost", f"${opex_mer:,.2f}/yr")
    m_col3.metric("Total Annual Cost (TAC)", f"${tac_mer:,.2f}/yr")
    
    # Optional: Show detailed breakdown
    if st.checkbox("Show detailed MER cost breakdown"):
        if mer_details:
            st.dataframe(pd.DataFrame(mer_details), use_container_width=True)

    st.markdown("---")
    st.subheader("4. Optimization and Economic Analysis")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1: 
        h_hot_u = st.number_input("Hot Utility h [kW/m²K]", value=5.0)
    with col_opt2: 
        h_cold_u = st.number_input("Cold Utility h [kW/m²K]", value=0.8)

    if st.button("Calculate Economic Optimum"):
        hot_streams, cold_streams = prepare_optimizer_data(edited_df)
        found_matches = []
        
        with st.status("Finding Dynamic Equilibrium Points...", expanded=True) as status:
            # Find DEP for all possible matches
            for hs in hot_streams:
                for cs in cold_streams:
                    q_dep = find_q_dep(hs, cs, econ_params, dt_min_input)
                    if q_dep and q_dep > 0.1:
                        found_matches.append({
                            "Hot Stream": hs['Stream'], 
                            "Cold Stream": cs['Stream'],
                            "Recommended Load [kW]": q_dep
                        })
            status.update(label=f"Found {len(found_matches)} potential matches", state="running")
        
        if found_matches:
            with st.status("Evolving Network via Random Walk...", expanded=True) as status:
                refined_matches, tac_opt = run_random_walk(
                    found_matches, hot_streams, cold_streams, econ_params, dt_min_input
                )
                status.update(label="Evolution Complete!", state="complete", expanded=False)
            
            # Apply Pruning from Paper Logic
            refined_matches = prune_and_normalize_matches(refined_matches, hot_streams)
            
            if refined_matches:
                display_matches = []
                final_cap = 0
                final_qh = 0
                final_qc = 0
                
                # Calculate remaining duties
                rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
                rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in cold_streams}
                
                for m in refined_matches:
                    q = m['Recommended Load [kW]']
                    hs = next(s for s in hot_streams if s['Stream'] == m['Hot Stream'])
                    cs = next(s for s in cold_streams if s['Stream'] == m['Cold Stream'])
                    
                    ratio = q / (hs['mCp'] * abs(hs['Ts'] - hs['Tt']))
                    ratio_text = f"{round(ratio, 2)} " if ratio < 0.99 else ""
                    
                    u = calculate_u(hs['h'], cs['h'])
                    tho = hs['Ts'] - (q / hs['mCp'])
                    tco = cs['Ts'] + (q / cs['mCp'])
                    l_val = lmtd_chen(hs['Ts'], tho, cs['Ts'], tco)
                    
                    if l_val > 0 and u > 0:
                        area = q / (u * l_val)
                        final_cap += (econ_params['a'] + econ_params['b'] * (area ** econ_params['c']))
                        
                        # Update remaining duties
                        rem_h[hs['Stream']] -= q
                        rem_c[cs['Stream']] -= q
                        
                        display_matches.append({
                            "Match": f"{ratio_text}Stream {hs['Stream']} ↔ {cs['Stream']}",
                            "Duty [kW]": round(q, 2),
                            "Area [m²]": round(area, 2)
                        })
                
                # Calculate final utilities
                final_qh = sum(max(0, val) for val in rem_c.values())
                final_qc = sum(max(0, val) for val in rem_h.values())
                
                ann_cap_opt = final_cap * DGS_CONFIG['ANNUAL_FACTOR']
                opex_opt = (final_qh * econ_params['c_hu']) + (final_qc * econ_params['c_cu'])
                tac_opt = ann_cap_opt + opex_opt

                st.dataframe(pd.DataFrame(display_matches), use_container_width=True)
                
                # Display utilities
                opt_u_col1, opt_u_col2 = st.columns(2)
                opt_u_col1.metric("Hot Utility (Qh)", f"{final_qh:,.2f} kW")
                opt_u_col2.metric("Cold Utility (Qc)", f"{final_qc:,.2f} kW")

                o_col1, o_col2, o_col3 = st.columns(3)
                o_col1.metric("Capital Cost", f"${final_cap:,.2f}", f"(${ann_cap_opt:,.2f}/yr)")
                o_col2.metric("Annual Operating Cost", f"${opex_opt:,.2f}/yr")
                o_col3.metric("Total Annual Cost (TAC)", f"${tac_opt:,.2f}/yr")

                st.markdown("---")
                st.subheader("5. Comprehensive Comparison")
                
                # Calculate no integration case
                no_int = calculate_no_integration_costs(edited_df, econ_params)
                
                # Important Note about utilities
                if final_qh > qh or final_qc > qc:
                    st.info("""
                    **Note on Utility Consumption:**
                    TAC optimization uses more utilities than MER to reduce capital cost.
                    This is expected - the algorithm trades energy recovery for lower investment.
                    """)
                
                # Create comparison table
                comparison_df = pd.DataFrame({
                    "Configuration": ["No Integration", "MER Setup", "Optimized (TAC)"],
                    "Heat Exchangers": [0, len(match_summary), len(refined_matches)],
                    "Capital Cost ($)": [
                        f"{no_int['capital']:,.2f}",
                        f"{cap_mer:,.2f}",
                        f"{final_cap:,.2f}"
                    ],
                    "Annual Capital ($/yr)": [
                        f"{no_int['ann_capital']:,.2f}",
                        f"{ann_cap_mer:,.2f}",
                        f"{ann_cap_opt:,.2f}"
                    ],
                    "Operating Cost ($/yr)": [
                        f"{no_int['opex']:,.2f}",
                        f"{opex_mer:,.2f}",
                        f"{opex_opt:,.2f}"
                    ],
                    "TAC ($/yr)": [
                        f"{no_int['tac']:,.2f}",
                        f"{tac_mer:,.2f}",
                        f"{tac_opt:,.2f}"
                    ],
                    "Hot Utility (kW)": [
                        f"{no_int['qh']:,.2f}",
                        f"{qh:,.2f}",
                        f"{final_qh:,.2f}"
                    ],
                    "Cold Utility (kW)": [
                        f"{no_int['qc']:,.2f}",
                        f"{qc:,.2f}",
                        f"{final_qc:,.2f}"
                    ]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Calculate savings
                st.markdown("#### Savings Analysis")
                s1, s2, s3 = st.columns(3)
                with s1:
                    mer_savings = ((no_int['tac'] - tac_mer) / no_int['tac'] * 100) if no_int['tac'] > 0 else 0
                    st.metric("MER vs No Integration", f"{mer_savings:.1f}% savings")
                with s2:
                    opt_savings = ((no_int['tac'] - tac_opt) / no_int['tac'] * 100) if no_int['tac'] > 0 else 0
                    st.metric("Optimized vs No Integration", f"{opt_savings:.1f}% savings")
                with s3:
                    mer_opt_diff = ((tac_mer - tac_opt) / tac_mer * 100) if tac_mer > 0 else 0
                    st.metric("Optimized vs MER", f"{mer_opt_diff:.1f}% improvement")
                    
            else:
                st.warning("No viable matches found after pruning. Try adjusting economic parameters.")
        else:
            st.warning("No viable matches found. Try adjusting ΔTmin or stream data.")
