import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pygad

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs **Pinch Analysis**, **MER Matching with Stream Splitting**, and 
**Economic Optimization** using **Genetic Algorithm (PyGAD)**.
""")
st.markdown("---")

# --- CORE MATH FUNCTIONS ---
def calculate_u(h1, h2, h_unit_factor=1.0):
    """
    Calculate overall heat transfer coefficient
    
    h_unit_factor: Conversion factor for h values
    - If h is in kW/m²K: use 1.0
    - If h is in W/m²K: use 0.001
    - If your h values seem too high (like 1.6): try 0.1
    """
    if h1 <= 0 or h2 <= 0:
        return 0
    h1_converted = h1 * h_unit_factor
    h2_converted = h2 * h_unit_factor
    return 1 / ((1/h1_converted) + (1/h2_converted))

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

def validate_dataframe(df):
    """Validate that the DataFrame has all required columns"""
    required_cols = ["Stream", "Type", "mCp", "Ts", "Tt", "h"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Check if DataFrame is empty
    if df.empty:
        return False, "DataFrame is empty. Please add stream data."
    
    # Check for valid types
    valid_types = df['Type'].isin(['Hot', 'Cold'])
    if not valid_types.all():
        return False, "Type column must contain only 'Hot' or 'Cold'"
    
    # Check for numeric columns
    numeric_cols = ['mCp', 'Ts', 'Tt', 'h']
    for col in numeric_cols:
        try:
            pd.to_numeric(df[col])
        except:
            return False, f"Column '{col}' must contain numeric values"
    
    return True, "Valid"

def run_thermal_logic(df, dt):
    """Perform pinch analysis to find minimum utility requirements"""
    # Validate input first
    is_valid, msg = validate_dataframe(df)
    if not is_valid:
        raise ValueError(msg)
    
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

# --- GENETIC ALGORITHM CONFIGURATION ---
GA_CONFIG = {
    "num_generations": 100,
    "num_parents_mating": 10,
    "sol_per_pop": 50,
    "mutation_percent_genes": 15,
    "min_split_ratio": 0.10  # Minimum split ratio threshold
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
    
    with st.expander("⚠️ Heat Transfer Coefficient Units Correction", expanded=True):
        st.warning("""
        **Important**: If your h values seem unusually high (like 1.6) and capital costs are too low,
        you may need to apply a unit conversion factor.
        """)
        h_factor = st.selectbox(
            "h value unit conversion factor",
            options=[1.0, 0.1, 0.01, 0.001],
            index=1,  # Default to 0.1
            help="""
            - 1.0 = h already in kW/m²K (typical range 0.1-0.5)
            - 0.1 = h needs scaling down by 10x (use if h values are 1-5)
            - 0.01 = h in 100× units
            - 0.001 = h in W/m²K (convert to kW/m²K)
            """
        )
        st.info(f"Current setting: h values will be multiplied by {h_factor}")
    
    with st.expander("Genetic Algorithm Settings", expanded=False):
        ga_col1, ga_col2 = st.columns(2)
        with ga_col1:
            num_gen = st.number_input("Number of Generations", value=100, min_value=10, max_value=1000)
            pop_size = st.number_input("Population Size", value=50, min_value=10, max_value=200)
        with ga_col2:
            num_parents = st.number_input("Parents Mating", value=10, min_value=2, max_value=50)
            mutation_rate = st.number_input("Mutation Rate (%)", value=15, min_value=1, max_value=50)
        
        GA_CONFIG["num_generations"] = num_gen
        GA_CONFIG["sol_per_pop"] = pop_size
        GA_CONFIG["num_parents_mating"] = num_parents
        GA_CONFIG["mutation_percent_genes"] = mutation_rate
    
    return {"a": a, "b": b, "c": c, "c_hu": c_hu, "c_cu": c_cu, "h_factor": h_factor}

def prepare_optimizer_data(df):
    hot_streams = df[df['Type'] == 'Hot'].to_dict('records')
    cold_streams = df[df['Type'] == 'Cold'].to_dict('records')
    return hot_streams, cold_streams

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
if h_ratio >= GA_CONFIG["min_split_ratio"] or h_ratio >= 0.99:
    ratio_text = f"{round(h_ratio, 2)} " if h_ratio < 0.99 else ""
    match_str = f"{ratio_text}Stream {h['Stream']} ↔ {c['Stream']}"
    
    h['Q'] -= m_q
    c['Q'] -= m_q

    matches.append({
        "Match": match_str, 
        "Duty [kW]": round(m_q, 2), 
        "Type": "Split" if is_split or (0 < h_ratio < 0.99) else "Direct",
        "Side": side
    })
else:
    # Skip this match if split ratio is too small
    continue


# --- GENETIC ALGORITHM IMPLEMENTATION ---

def create_match_pairs(hot_streams, cold_streams):
    """Create all possible hot-cold stream pairs"""
    pairs = []
    for hs in hot_streams:
        for cs in cold_streams:
            pairs.append({
                'hot_stream': hs,
                'cold_stream': cs,
                'max_duty': min(
                    hs['mCp'] * abs(hs['Ts'] - hs['Tt']),
                    cs['mCp'] * abs(cs['Ts'] - cs['Tt'])
                )
            })
    return pairs

def decode_solution(solution, match_pairs):
    """Convert GA solution (normalized 0-1) to actual heat loads"""
    matches = []
    for i, pair in enumerate(match_pairs):
        duty = solution[i] * pair['max_duty']
        if duty > 0.1:  # Minimum threshold
            matches.append({
                'Hot Stream': pair['hot_stream']['Stream'],
                'Cold Stream': pair['cold_stream']['Stream'],
                'Recommended Load [kW]': duty,
                'hot_stream_data': pair['hot_stream'],
                'cold_stream_data': pair['cold_stream']
            })
    return matches

def calculate_tac_for_matches(matches, hot_streams, cold_streams, econ_params, dt_min, annual_factor=0.2):
    """Calculate Total Annual Cost for a given network configuration"""
    rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
    rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in cold_streams}
    total_inv = 0
    h_factor = econ_params.get('h_factor', 1.0)
    
    for m in matches:
        q = m['Recommended Load [kW]']
        if q <= 0.001:
            continue
            
        h_s = m['hot_stream_data']
        c_s = m['cold_stream_data']
        
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
        u = calculate_u(h_s['h'], c_s['h'], h_factor)
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
    
    return opex + (total_inv * annual_factor)

def fitness_function(ga_instance, solution, solution_idx):
    """Fitness function for PyGAD - minimize TAC (return negative for maximization)"""
    # Get parameters from session state
    match_pairs = st.session_state.get('ga_match_pairs', [])
    hot_streams = st.session_state.get('ga_hot_streams', [])
    cold_streams = st.session_state.get('ga_cold_streams', [])
    econ_params = st.session_state.get('ga_econ_params', {})
    dt_min = st.session_state.get('ga_dt_min', 10.0)
    
    # Decode solution
    matches = decode_solution(solution, match_pairs)
    
    # Calculate TAC
    tac = calculate_tac_for_matches(matches, hot_streams, cold_streams, econ_params, dt_min)
    
    # Return negative TAC (PyGAD maximizes fitness)
    if tac == float('inf'):
        return -1e10
    return -tac

def run_genetic_algorithm(hot_streams, cold_streams, econ_params, dt_min):
    """Run genetic algorithm optimization"""
    # Create all possible match pairs
    match_pairs = create_match_pairs(hot_streams, cold_streams)
    num_genes = len(match_pairs)
    
    # Store in session state for fitness function
    st.session_state['ga_match_pairs'] = match_pairs
    st.session_state['ga_hot_streams'] = hot_streams
    st.session_state['ga_cold_streams'] = cold_streams
    st.session_state['ga_econ_params'] = econ_params
    st.session_state['ga_dt_min'] = dt_min
    
    # Define gene space (0 to 1 for each match, representing fraction of max duty)
    gene_space = [{'low': 0.0, 'high': 1.0} for _ in range(num_genes)]
    
    # Create GA instance
    ga_instance = pygad.GA(
        num_generations=GA_CONFIG['num_generations'],
        num_parents_mating=GA_CONFIG['num_parents_mating'],
        fitness_func=fitness_function,
        sol_per_pop=GA_CONFIG['sol_per_pop'],
        num_genes=num_genes,
        gene_space=gene_space,
        parent_selection_type="sss",  # Steady state selection
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=GA_CONFIG['mutation_percent_genes'],
        random_seed=42
    )
    
    # Run GA
    ga_instance.run()
    
    # Get best solution
    solution, solution_fitness, _ = ga_instance.best_solution()
    best_matches = decode_solution(solution, match_pairs)
    best_tac = -solution_fitness
    
    return best_matches, best_tac, ga_instance

def prune_matches(matches, min_ratio=0.10):
    """Remove matches with heat load < minimum ratio of stream capacity"""
    pruned = []
    for m in matches:
        h_s = m['hot_stream_data']
        total_q = h_s['mCp'] * abs(h_s['Ts'] - h_s['Tt'])
        
        if total_q <= 0:
            continue
            
        split_ratio = m['Recommended Load [kW]'] / total_q
        
        if split_ratio >= min_ratio:
            pruned.append(m)
    
    return pruned

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

def calculate_mer_capital_properly(match_summary, processed_df, econ_params, pinch_t, dt_min):
    """
    Calculate MER capital cost treating Above/Below pinch as separate systems.
    Fixes the "huge capital cost" bug by clamping supply temperatures.
    """
    cap_mer = 0
    h_factor = econ_params.get('h_factor', 1.0)
    
    # Based on your code's shift logic: Hot was unshifted, Cold was shifted +dt
    # Therefore, the 'pinch' variable equals the real Hot Pinch Temp.
    pinch_real_hot = pinch_t
    pinch_real_cold = pinch_t - dt_min
    
    for m in match_summary:
        duty = m['Duty [kW]']
        if duty <= 0:
            continue
            
        side = m.get('Side', 'Above') # Default to Above if tag missing
        
        # Parse stream IDs
        match_str = m['Match']
        if ' ' in match_str and match_str.split()[0].replace('.', '').isdigit():
            match_str = ' '.join(match_str.split()[1:])
            
        try:
            match_parts = match_str.replace('Stream ', '').split(' ↔ ')
            h_id = match_parts[0].strip()
            c_id = match_parts[1].strip()
            
            # Find original stream data
            h_stream = processed_df[(processed_df['Stream'].astype(str) == h_id) & (processed_df['Type'] == 'Hot')].iloc[0].to_dict()
            c_stream = processed_df[(processed_df['Stream'].astype(str) == c_id) & (processed_df['Type'] == 'Cold')].iloc[0].to_dict()
            
            # --- CLAMP START TEMPERATURES ---
            if side == 'Above':
                # ABOVE PINCH: Hot starts at Supply. Cold starts at Pinch.
                thi = h_stream['Ts'] 
                tci = max(c_stream['Ts'], pinch_real_cold) 
            else:
                # BELOW PINCH: Hot starts at Pinch. Cold starts at Supply.
                thi = min(h_stream['Ts'], pinch_real_hot)
                tci = c_stream['Ts']

            # Calculate Outlet Temperatures based on Duty
            tho = thi - (duty / h_stream['mCp'])
            tco = tci + (duty / c_stream['mCp'])
            
            # Calculate U and LMTD
            u = calculate_u(h_stream['h'], c_stream['h'], h_factor)
            lmtd = lmtd_chen(thi, tho, tci, tco)
            
            if u > 0 and lmtd > 0:
                area = duty / (u * lmtd)
                cap_mer += (econ_params['a'] + econ_params['b'] * (area ** econ_params['c']))
                
        except Exception:
            continue
            
    return cap_mer

# --- UI LOGIC ---
st.subheader("1. Stream Data Input")

# Add example data button
if st.button("Load Example Data"):
    example_data = pd.DataFrame({
        "Stream": [1, 2, 3, 4],
        "Type": ["Hot", "Hot", "Cold", "Cold"],
        "mCp": [10.0, 15.0, 13.0, 12.0],
        "Ts": [180, 150, 60, 90],
        "Tt": [60, 30, 160, 140],
        "h": [0.5, 0.6, 0.4, 0.5]
    })
    st.session_state['input_data'] = example_data
    st.success("Example data loaded!")

uploaded_file = st.file_uploader("Import Stream Data from Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    try:
        import_df = pd.read_excel(uploaded_file)
        is_valid, msg = validate_dataframe(import_df)
        if is_valid:
            st.session_state['input_data'] = import_df
            st.success("Data imported successfully!")
        else:
            st.error(f"Invalid data format: {msg}")
    except Exception as e:
        st.error(f"Error reading file: {e}")

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = pd.DataFrame(columns=["Stream", "Type", "mCp", "Ts", "Tt", "h"])

with st.form("main_input_form"):
    dt_min_input = st.number_input("Target ΔTmin [°C]", min_value=1.0, value=10.0)
    edited_df = st.data_editor(st.session_state['input_data'], num_rows="dynamic", use_container_width=True)
    submit_thermal = st.form_submit_button("Run Thermal Analysis")

if submit_thermal:
    # Validate before running
    is_valid, msg = validate_dataframe(edited_df)
    if is_valid:
        st.session_state.run_clicked = True
        st.session_state.edited_df = edited_df
        st.session_state.dt_min = dt_min_input
    else:
        st.error(f"❌ {msg}")
        st.session_state.run_clicked = False

if st.session_state.get('run_clicked'):
    try:
        edited_df = st.session_state.edited_df
        dt_min_input = st.session_state.dt_min
        
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
                        st.dataframe(pd.DataFrame(matches), use_container_width=True)
                    else: 
                        st.info("No internal matches possible.")

        econ_params = render_optimization_inputs() 
        
        # Add diagnostic information
        if edited_df is not None and len(edited_df) > 0:
            with st.expander("🔍 Heat Transfer Diagnostics", expanded=False):
                st.write("**Current h values and calculated U:**")
                h_factor = econ_params.get('h_factor', 1.0)
                
                diag_data = []
                for idx, row in edited_df.iterrows():
                    h_val = row['h']
                    h_corrected = h_val * h_factor
                    diag_data.append({
                        "Stream": row['Stream'],
                        "Original h": h_val,
                        "Corrected h": h_corrected,
                        "Unit": "kW/m²K"
                    })
                
                st.dataframe(pd.DataFrame(diag_data), use_container_width=True)
                
                # Calculate sample U values
                if len(edited_df) >= 2:
                    st.write("**Sample U values for stream pairs:**")
                    hot = edited_df[edited_df['Type'] == 'Hot'].iloc[0] if len(edited_df[edited_df['Type'] == 'Hot']) > 0 else None
                    cold = edited_df[edited_df['Type'] == 'Cold'].iloc[0] if len(edited_df[edited_df['Type'] == 'Cold']) > 0 else None
                    
                    if hot is not None and cold is not None:
                        u_sample = calculate_u(hot['h'], cold['h'], h_factor)
                        st.metric("Example U (process-to-process)", f"{u_sample:.4f} kW/m²K")
                        
                        if u_sample > 0.5:
                            st.error("⚠️ U value seems too HIGH! Typical process U values are 0.05-0.3 kW/m²K. Try a smaller h_factor!")
                        elif u_sample < 0.01:
                            st.warning("⚠️ U value seems too LOW! Try a larger h_factor.")
                        else:
                            st.success("✓ U value in reasonable range for industrial heat exchangers") 
        
        # MER Economics Calculation
        # Pass pinch and dt_min so the function knows where to split the streams
        cap_mer = calculate_mer_capital_properly(match_summary, processed_df, econ_params, pinch, dt_min_input)
        ann_cap_mer = cap_mer * 0.2
        opex_mer = (qh * econ_params['c_hu']) + (qc * econ_params['c_cu'])
        tac_mer = opex_mer + ann_cap_mer

        st.markdown("#### MER Economic Breakdown")
        u_col1, u_col2 = st.columns(2)
        u_col1.metric("Hot Utility (Qh)", f"{qh:,.2f} kW", help="Minimum hot utility from pinch analysis")
        u_col2.metric("Cold Utility (Qc)", f"{qc:,.2f} kW", help="Minimum cold utility from pinch analysis")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Capital Cost", f"${cap_mer:,.2f}", f"(${ann_cap_mer:,.2f}/yr)")
        m_col2.metric("Annual Operating Cost", f"${opex_mer:,.2f}/yr")
        m_col3.metric("Total Annual Cost (TAC)", f"${tac_mer:,.2f}/yr")

        st.markdown("---")
        st.subheader("4. Genetic Algorithm Optimization")

        if st.button("🧬 Run Genetic Algorithm Optimization"):
            hot_streams, cold_streams = prepare_optimizer_data(edited_df)
            
            with st.status("Running Genetic Algorithm...", expanded=True) as status:
                st.write(f"Evaluating {len(hot_streams) * len(cold_streams)} possible matches...")
                st.write(f"Population size: {GA_CONFIG['sol_per_pop']}")
                st.write(f"Generations: {GA_CONFIG['num_generations']}")
                
                optimized_matches, tac_opt, ga_instance = run_genetic_algorithm(
                    hot_streams, cold_streams, econ_params, dt_min_input
                )
                
                status.update(label="✅ Optimization Complete!", state="complete", expanded=False)
            
            # Apply pruning
            optimized_matches = prune_matches(optimized_matches, GA_CONFIG['min_split_ratio'])
            
            if optimized_matches:
                display_matches = []
                final_cap = 0
                
                # Calculate remaining duties
                rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
                rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in cold_streams}
                
                for m in optimized_matches:
                    q = m['Recommended Load [kW]']
                    hs = m['hot_stream_data']
                    cs = m['cold_stream_data']
                    
                    ratio = q / (hs['mCp'] * abs(hs['Ts'] - hs['Tt']))
                    ratio_text = f"{round(ratio, 2)} " if ratio < 0.99 else ""
                    
                    u = calculate_u(hs['h'], cs['h'], econ_params['h_factor'])
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
                
                ann_cap_opt = final_cap * 0.2
                opex_opt = (final_qh * econ_params['c_hu']) + (final_qc * econ_params['c_cu'])
                tac_opt_actual = ann_cap_opt + opex_opt

                st.markdown("#### Optimized Heat Exchanger Network")
                st.dataframe(pd.DataFrame(display_matches), use_container_width=True)
                
                # Display utilities
                opt_u_col1, opt_u_col2 = st.columns(2)
                opt_u_col1.metric("Hot Utility (Qh)", f"{final_qh:,.2f} kW")
                opt_u_col2.metric("Cold Utility (Qc)", f"{final_qc:,.2f} kW")

                o_col1, o_col2, o_col3 = st.columns(3)
                o_col1.metric("Capital Cost", f"${final_cap:,.2f}", f"(${ann_cap_opt:,.2f}/yr)")
                o_col2.metric("Annual Operating Cost", f"${opex_opt:,.2f}/yr")
                o_col3.metric("Total Annual Cost (TAC)", f"${tac_opt_actual:,.2f}/yr")

                # Show GA convergence plot
                st.markdown("#### GA Convergence")
                fitness_history = ga_instance.best_solutions_fitness
                fig_ga = go.Figure()
                fig_ga.add_trace(go.Scatter(
                    x=list(range(len(fitness_history))),
                    y=[-f for f in fitness_history],  # Convert back to positive TAC
                    mode='lines',
                    name='Best TAC'
                ))
                fig_ga.update_layout(
                    xaxis_title="Generation",
                    yaxis_title="Total Annual Cost ($/yr)",
                    height=300
                )
                st.plotly_chart(fig_ga, use_container_width=True)

                st.markdown("---")
                st.subheader("5. Comprehensive Comparison")
                
                # Calculate no integration case
                no_int = calculate_no_integration_costs(edited_df, econ_params)
                
                # Important Note about utilities
                st.info(f"""
                **Note on Utility Consumption:**
                - **MER Setup**: Uses minimum utilities from pinch analysis (Qh = {qh:.2f} kW, Qc = {qc:.2f} kW)
                - **Optimized TAC (GA)**: May use MORE utilities ({final_qh:.2f} kW, {final_qc:.2f} kW) to reduce capital cost
                - This is expected behavior - TAC optimization trades utility cost for lower capital investment
                """)
                
                # Create comparison table
                comparison_df = pd.DataFrame({
                    "Configuration": ["No Integration", "MER Setup", "Optimized (GA)"],
                    "Heat Exchangers": [0, len(match_summary), len(optimized_matches)],
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
                        f"{tac_opt_actual:,.2f}"
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
                    opt_savings = ((no_int['tac'] - tac_opt_actual) / no_int['tac'] * 100) if no_int['tac'] > 0 else 0
                    st.metric("GA-Optimized vs No Integration", f"{opt_savings:.1f}% savings")
                with s3:
                    mer_opt_diff = ((tac_mer - tac_opt_actual) / tac_mer * 100) if tac_mer > 0 else 0
                    st.metric("GA-Optimized vs MER", f"{mer_opt_diff:.1f}% improvement")
                    
            else:
                st.warning("No viable matches found after pruning. Try adjusting economic parameters or GA settings.")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Please check your input data and try again.")



