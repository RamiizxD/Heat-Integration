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
    
    with st.expander("⚠️ Cost Formula Selection", expanded=True):
        cost_formula = st.radio(
            "Select capital cost formula",
            options=["Benchmark (Linnhoff)", "Custom (a + b×A^c)"],
            index=0,
            help="""
            **Benchmark (Linnhoff)**: Annual cost = 1000 × A^0.6 ($/yr)
            - Simpler formula from literature
            - Directly gives annual cost (no annualization factor needed)
            - Use this to match benchmark problems
            
            **Custom**: Total capital = a + b × A^c, Annual = Total × factor
            - More flexible for different plants
            - Includes fixed cost (a) and custom exponent (c)
            """
        )
    
    if cost_formula == "Benchmark (Linnhoff)":
        st.info("Using benchmark formula: **Annual cost = 1000 × A^0.6** ($/yr)")
        econ_params = {
            "formula": "benchmark",
            "c_hu": 80.0,
            "c_cu": 20.0
        }
        
        # Still allow customization
        with st.expander("Customize benchmark parameters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                cost_coef = st.number_input("Cost Coefficient ($/m^0.6)", value=1000.0)
                c_hu = st.number_input("Hot Utility Cost ($/kW·yr)", value=80.0)
            with col2:
                cost_exp = st.number_input("Area Exponent", value=0.6, step=0.01)
                c_cu = st.number_input("Cold Utility Cost ($/kW·yr)", value=20.0)
            
            econ_params.update({
                "cost_coef": cost_coef,
                "cost_exp": cost_exp,
                "c_hu": c_hu,
                "c_cu": c_cu
            })
        
        # Set default values if not customized
        if "cost_coef" not in econ_params:
            econ_params.update({"cost_coef": 1000.0, "cost_exp": 0.6})
            
    else:  # Custom formula
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
                ann_factor = st.number_input("Annualization Factor", value=0.2, step=0.01)
        
        econ_params = {
            "formula": "custom",
            "a": a, "b": b, "c": c,
            "c_hu": c_hu, "c_cu": c_cu,
            "ann_factor": ann_factor
        }
    
    with st.expander("⚠️ Heat Transfer Coefficient Units Correction", expanded=True):
        st.warning("""
        **Important**: If your h values seem unusually high (like 1.6) and capital costs are too low,
        you may need to apply a unit conversion factor.
        """)
        h_factor = st.selectbox(
            "h value unit conversion factor",
            options=[1.0, 0.1, 0.01, 0.001],
            index=0,  # Default to 1.0 for benchmark (h=1.6 is correct)
            help="""
            - 1.0 = h already in kW/m²K (typical range 0.1-5)
            - 0.1 = h needs scaling down by 10x (use if h values are 10-50)
            - 0.01 = h in 100× units
            - 0.001 = h in W/m²K (convert to kW/m²K)
            """
        )
        st.info(f"Current setting: h values will be multiplied by {h_factor}")
    
    with st.expander("Utility Heat Transfer Coefficients", expanded=True):
        st.info("Enter h values for hot and cold utilities (for heaters/coolers)")
        util_col1, util_col2 = st.columns(2)
        with util_col1:
            h_hu = st.number_input("Hot Utility h (kW/m²K)", value=4.8, help="For steam condensers, etc.")
        with util_col2:
            h_cu = st.number_input("Cold Utility h (kW/m²K)", value=1.6, help="For cooling water, etc.")
        
        econ_params.update({"h_hu": h_hu, "h_cu": h_cu})
    
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
    
    econ_params["h_factor"] = h_factor
    return econ_params

def prepare_optimizer_data(df):
    hot_streams = df[df['Type'] == 'Hot'].to_dict('records')
    cold_streams = df[df['Type'] == 'Cold'].to_dict('records')
    return hot_streams, cold_streams

def calculate_hex_capital(area, econ_params):
    """
    Calculate capital cost for a single heat exchanger
    Returns annual capital cost in $/yr
    """
    if econ_params.get("formula") == "benchmark":
        # Benchmark formula: Annual cost = cost_coef × A^cost_exp
        cost_coef = econ_params.get("cost_coef", 1000.0)
        cost_exp = econ_params.get("cost_exp", 0.6)
        return cost_coef * (area ** cost_exp)
    else:
        # Custom formula: Total = a + b × A^c, Annual = Total × ann_factor
        a = econ_params.get("a", 8000.0)
        b = econ_params.get("b", 1200.0)
        c = econ_params.get("c", 0.6)
        ann_factor = econ_params.get("ann_factor", 0.2)
        total_capital = a + b * (area ** c)
        return total_capital * ann_factor

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
                    "Side": side  # Tag which side of pinch this match belongs to
                })
            else:
                # Skip this match if split ratio is too small
                break
        else:
            break
            
    return matches, hot, cold

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

def calculate_tac_for_matches(matches, hot_streams, cold_streams, econ_params, dt_min):
    """
    Calculate Total Annual Cost for a given network configuration.
    
    CRITICAL FIX: Now tracks current temperatures as streams pass through sequential exchangers.
    This prevents unrealistic LMTD calculations and ensures fair comparison with MER.
    """
    # Initialize remaining duties
    rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
    rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in cold_streams}
    
    # ===== NEW: Track current temperatures of each stream =====
    # Start with supply temperatures
    current_temp_hot = {s['Stream']: s['Ts'] for s in hot_streams}
    current_temp_cold = {s['Stream']: s['Ts'] for s in cold_streams}
    
    # Track target temperatures
    target_temp_hot = {s['Stream']: s['Tt'] for s in hot_streams}
    target_temp_cold = {s['Stream']: s['Tt'] for s in cold_streams}
    
    # Track mCp values for temperature calculations
    mcp_hot = {s['Stream']: s['mCp'] for s in hot_streams}
    mcp_cold = {s['Stream']: s['mCp'] for s in cold_streams}
    
    total_inv = 0
    h_factor = econ_params.get('h_factor', 1.0)
    
    # Sort matches to process them in a sensible order
    # This helps with convergence but the order doesn't affect the final result
    # since we're tracking temperatures
    matches = sorted(matches, key=lambda m: m['Recommended Load [kW]'], reverse=True)
    
    for m in matches:
        q = m['Recommended Load [kW]']
        if q <= 0.001:
            continue
            
        h_stream_id = m['Hot Stream']
        c_stream_id = m['Cold Stream']
        h_s = m['hot_stream_data']
        c_s = m['cold_stream_data']
        
        # Check feasibility
        if q > rem_h.get(h_stream_id, 0) + 0.1 or q > rem_c.get(c_stream_id, 0) + 0.1:
            return float('inf')

        # ===== CRITICAL FIX: Use CURRENT temperatures, not supply temperatures =====
        thi = current_temp_hot[h_stream_id]
        tci = current_temp_cold[c_stream_id]
        
        # Calculate outlet temperatures based on duty
        tho = thi - (q / mcp_hot[h_stream_id])
        tco = tci + (q / mcp_cold[c_stream_id])
        
        # Validate outlet temperatures don't cross stream targets
        # Hot stream should be cooling down (tho should be >= target)
        if tho < target_temp_hot[h_stream_id] - 0.1:
            return float('inf')
        # Cold stream should be heating up (tco should be <= target)
        if tco > target_temp_cold[c_stream_id] + 0.1:
            return float('inf')
        
        # Check minimum temperature approach
        if (thi - tco) < dt_min or (tho - tci) < dt_min:
            return float('inf')
        
        # Update remaining duties
        rem_h[h_stream_id] -= q
        rem_c[c_stream_id] -= q
        
        # ===== UPDATE current temperatures for next match =====
        current_temp_hot[h_stream_id] = tho
        current_temp_cold[c_stream_id] = tco
        
        # Calculate area and investment
        u = calculate_u(h_s['h'], c_s['h'], h_factor)
        if u <= 0:
            return float('inf')
            
        lmtd = lmtd_chen(thi, tho, tci, tco)
        if lmtd <= 0:
            return float('inf')
            
        area = q / (u * lmtd)
        total_inv += calculate_hex_capital(area, econ_params)

    # Calculate utility costs
    actual_qh = sum(max(0, val) for val in rem_c.values())
    actual_qc = sum(max(0, val) for val in rem_h.values())
    opex = (actual_qh * econ_params['c_hu']) + (actual_qc * econ_params['c_cu'])
    
    # Add utility heat exchangers to capital
    h_hu = econ_params.get('h_hu', 4.8)
    h_cu = econ_params.get('h_cu', 1.6)
    
    # Hot utility heaters
    if actual_qh > 0.1:
        lmtd_util = 50.0
        h_cold_avg = np.mean([s['h'] for s in cold_streams]) if cold_streams else 1.6
        u_hu = calculate_u(h_hu, h_cold_avg, h_factor)
        if u_hu > 0:
            area_hu = actual_qh / (u_hu * lmtd_util)
            total_inv += calculate_hex_capital(area_hu, econ_params)
    
    # Cold utility coolers
    if actual_qc > 0.1:
        lmtd_util = 30.0
        h_hot_avg = np.mean([s['h'] for s in hot_streams]) if hot_streams else 1.6
        u_cu = calculate_u(h_hot_avg, h_cu, h_factor)
        if u_cu > 0:
            area_cu = actual_qc / (u_cu * lmtd_util)
            total_inv += calculate_hex_capital(area_cu, econ_params)
    
    return opex + total_inv  # total_inv is already annual cost

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

def calculate_mer_capital_properly(match_summary, processed_df, econ_params, pinch_t, dt_min, qh_util, qc_util):
    """
    Calculate MER capital cost treating Above/Below pinch as separate systems.
    NOW INCLUDES UTILITY HEAT EXCHANGERS!
    
    CRITICAL FIX: This function now correctly handles stream splitting at the pinch.
    
    The Problem:
    - In Pinch Analysis, streams are split into "Above Pinch" and "Below Pinch" segments
    - A stream going from 500°C to 300°C with pinch at 400°C becomes TWO segments:
      * Above: 500°C → 400°C
      * Below: 400°C → 300°C
    
    The Bug (Before):
    - Used global supply temps (Ts) for ALL matches
    - For "Below Pinch" matches, this created temperature crosses
    - LMTD became tiny (0.001) → huge areas → 4x inflated capital cost
    - ALSO: Forgot to include heaters and coolers in capital cost!
    
    The Fix (Now):
    - Above Pinch: Hot uses Ts, Cold starts at Pinch
    - Below Pinch: Hot starts at Pinch, Cold uses Ts
    - This prevents temperature crosses and gives correct LMTD
    - PLUS: Adds capital cost for utility heat exchangers
    """
    cap_mer = 0
    h_factor = econ_params.get('h_factor', 1.0)
    
    # Define Real (Non-Shifted) Pinch Temperatures
    # Based on shift logic: Hot unshifted, Cold shifted +dt
    pinch_real_hot = pinch_t
    pinch_real_cold = pinch_t - dt_min
    
    # ===== PROCESS-TO-PROCESS HEAT EXCHANGERS =====
    for m in match_summary:
        duty = m['Duty [kW]']
        if duty <= 0:
            continue
            
        side = m.get('Side', 'Above')  # Default to Above if tag missing
        
        # Extract stream numbers from match string
        match_str = m['Match']
        # Remove ratio prefix if present
        if ' ' in match_str and match_str.split()[0].replace('.', '').isdigit():
            match_str = ' '.join(match_str.split()[1:])
        
        # Parse "Stream X ↔ Y"
        try:
            match_parts = match_str.replace('Stream ', '').split(' ↔ ')
            h_stream_id = match_parts[0].strip()
            c_stream_id = match_parts[1].strip()
            
            # Find the hot and cold streams
            h_stream = None
            c_stream = None
            
            for _, row in processed_df.iterrows():
                if str(row['Stream']) == h_stream_id and row['Type'] == 'Hot':
                    h_stream = row.to_dict()
                if str(row['Stream']) == c_stream_id and row['Type'] == 'Cold':
                    c_stream = row.to_dict()
            
            if h_stream is not None and c_stream is not None:
                # --- CRITICAL FIX: CLAMP START TEMPERATURES BASED ON PINCH SIDE ---
                # Determine effective Inlet Temperatures for THIS specific match
                
                if side == 'Above':
                    # ABOVE PINCH:
                    # Hot Stream: Comes from Supply, goes to Pinch. Start = Original Ts.
                    # Cold Stream: Comes from Pinch, goes to Target. Start = Pinch Temp.
                    thi = h_stream['Ts']
                    tci = max(c_stream['Ts'], pinch_real_cold)
                else:
                    # BELOW PINCH:
                    # Hot Stream: Comes from Pinch, goes to Target. Start = Pinch Temp.
                    # Cold Stream: Comes from Supply, goes to Pinch. Start = Original Ts.
                    thi = min(h_stream['Ts'], pinch_real_hot)
                    tci = c_stream['Ts']
                
                # Calculate Outlet Temperatures based on Duty and CORRECTED Inlets
                tho = thi - (duty / h_stream['mCp'])
                tco = tci + (duty / c_stream['mCp'])
                
                # Calculate U and LMTD
                u = calculate_u(h_stream['h'], c_stream['h'], h_factor)
                lmtd = lmtd_chen(thi, tho, tci, tco)
                
                if u > 0 and lmtd > 0:
                    area = duty / (u * lmtd)
                    cap_mer += calculate_hex_capital(area, econ_params)
                    
        except Exception as e:
            # Skip invalid matches
            continue
    
    # ===== UTILITY HEAT EXCHANGERS =====
    # Add capital cost for heaters (hot utility) and coolers (cold utility)
    
    h_hu = econ_params.get('h_hu', 4.8)
    h_cu = econ_params.get('h_cu', 1.6)
    
    # Hot Utility (Heaters) - if Qh > 0
    if qh_util > 0.1:
        # Find cold streams that need heating
        # Assume hot utility at high temperature (e.g., 450 K from benchmark)
        # and average cold stream needs heating at around 350 K
        # Use simplified LMTD calculation
        t_hu_in = 450  # Hot utility inlet (e.g., steam)
        t_hu_out = 450  # Isothermal for condensing steam
        
        # Average approach temperature for utility ~50°C is typical
        lmtd_util = 50.0  # Conservative estimate
        
        # Calculate U for hot utility
        # Use highest cold stream h value as representative
        h_cold_avg = processed_df[processed_df['Type'] == 'Cold']['h'].mean() if len(processed_df[processed_df['Type'] == 'Cold']) > 0 else 1.6
        u_hu = calculate_u(h_hu, h_cold_avg, h_factor)
        
        if u_hu > 0:
            area_hu = qh_util / (u_hu * lmtd_util)
            cap_mer += calculate_hex_capital(area_hu, econ_params)
    
    # Cold Utility (Coolers) - if Qc > 0
    if qc_util > 0.1:
        # Find hot streams that need cooling
        # Assume cold utility at low temperature (e.g., 293-313 K from benchmark)
        t_cu_in = 293  # Cold utility inlet
        t_cu_out = 313  # Cold utility outlet
        
        # Average approach temperature
        lmtd_util = 30.0  # Conservative estimate
        
        # Calculate U for cold utility
        h_hot_avg = processed_df[processed_df['Type'] == 'Hot']['h'].mean() if len(processed_df[processed_df['Type'] == 'Hot']) > 0 else 1.6
        u_cu = calculate_u(h_hot_avg, h_cu, h_factor)
        
        if u_cu > 0:
            area_cu = qc_util / (u_cu * lmtd_util)
            cap_mer += calculate_hex_capital(area_cu, econ_params)
    
    return cap_mer

# --- USER INTERFACE ---
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["Data Input", "Analysis & Optimization"])

if section == "Data Input":
    st.header("1. Process Stream Data")
    
    # Allow users to input data via table
    if 'stream_data' not in st.session_state:
        st.session_state.stream_data = pd.DataFrame({
            "Stream": ["H1", "H2", "C1", "C2"],
            "Type": ["Hot", "Hot", "Cold", "Cold"],
            "mCp": [10.0, 20.0, 15.0, 13.0],
            "Ts": [450.0, 270.0, 20.0, 140.0],
            "Tt": [50.0, 60.0, 350.0, 300.0],
            "h": [1.6, 1.6, 1.6, 1.6]
        })
    
    st.markdown("Enter your stream data below:")
    edited_df = st.data_editor(
        st.session_state.stream_data,
        num_rows="dynamic",
        use_container_width=True,
        key="stream_editor"
    )
    
    st.session_state.stream_data = edited_df
    
    # Validate data
    is_valid, msg = validate_dataframe(edited_df)
    if is_valid:
        st.success("✅ Data validation passed!")
    else:
        st.error(f"❌ {msg}")

elif section == "Analysis & Optimization":
    st.header("2. Pinch Analysis & Heat Exchanger Network Design")
    
    # Get data
    if 'stream_data' not in st.session_state:
        st.error("Please enter stream data in the 'Data Input' section first.")
        st.stop()
    
    edited_df = st.session_state.stream_data
    
    # Validate
    is_valid, msg = validate_dataframe(edited_df)
    if not is_valid:
        st.error(f"❌ {msg}")
        st.stop()
    
    # Input ΔTmin
    st.markdown("### 3. Pinch Analysis Parameters")
    dt_min_input = st.number_input("Minimum Temperature Approach (ΔTmin) [°C]", 
                                     value=10.0, min_value=1.0, step=1.0)
    
    try:
        # Run pinch analysis
        qh, qc, pinch_t, temps, feasible, processed_df = run_thermal_logic(edited_df, dt_min_input)
        
        st.success(f"✅ Pinch Analysis Complete!")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum Hot Utility (Qh)", f"{qh:,.2f} kW")
        col2.metric("Minimum Cold Utility (Qc)", f"{qc:,.2f} kW")
        col3.metric("Pinch Temperature", f"{pinch_t:,.1f} °C" if pinch_t else "N/A")
        
        # Composite Curves
        st.markdown("#### Composite Curves")
        fig = go.Figure()
        
        # Create composite curve data
        h_temps = []
        h_enthalpies = []
        c_temps = []
        c_enthalpies = []
        
        cumulative_h = 0
        cumulative_c = 0
        
        for i, temp in enumerate(temps):
            # Calculate enthalpy change in this interval
            if i < len(temps) - 1:
                next_temp = temps[i+1]
                interval_hot = processed_df[
                    (processed_df['Type'] == 'Hot') & 
                    (processed_df['S_Ts'] >= next_temp) & 
                    (processed_df['S_Tt'] <= temp)
                ]['mCp'].sum() * (temp - next_temp)
                
                interval_cold = processed_df[
                    (processed_df['Type'] == 'Cold') & 
                    (processed_df['S_Ts'] <= next_temp) & 
                    (processed_df['S_Tt'] >= temp)
                ]['mCp'].sum() * (temp - next_temp)
                
                h_temps.append(temp)
                h_enthalpies.append(cumulative_h)
                cumulative_h += interval_hot
                
                c_temps.append(temp - dt_min_input)
                c_enthalpies.append(cumulative_c)
                cumulative_c += interval_cold
        
        # Add final points
        if len(temps) > 0:
            h_temps.append(temps[-1])
            h_enthalpies.append(cumulative_h)
            c_temps.append(temps[-1] - dt_min_input)
            c_enthalpies.append(cumulative_c)
        
        fig.add_trace(go.Scatter(x=h_enthalpies, y=h_temps, mode='lines', 
                                  name='Hot Composite', line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=c_enthalpies, y=c_temps, mode='lines', 
                                  name='Cold Composite', line=dict(color='blue', width=2)))
        
        # Add pinch line
        if pinch_t:
            fig.add_hline(y=pinch_t, line_dash="dash", line_color="green", 
                          annotation_text=f"Pinch: {pinch_t:.1f}°C")
        
        fig.update_layout(
            xaxis_title="Enthalpy (kW)",
            yaxis_title="Temperature (°C)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # MER Network Design
        st.markdown("---")
        st.markdown("### MER Network Design (Stream Splitting Allowed)")
        
        # Above pinch matching
        matches_above, _, _ = match_logic_with_splitting(processed_df, pinch_t, 'Above')
        
        # Below pinch matching
        matches_below, _, _ = match_logic_with_splitting(processed_df, pinch_t, 'Below')
        
        # Combine matches
        match_summary = matches_above + matches_below
        
        if match_summary:
            st.markdown("#### Process-to-Process Heat Exchangers")
            st.dataframe(pd.DataFrame(match_summary), use_container_width=True)
            
            # Calculate MER capital cost with proper temperature tracking
            ann_cap_mer = calculate_mer_capital_properly(
                match_summary, processed_df, 
                render_optimization_inputs(), 
                pinch_t, dt_min_input, qh, qc
            )
            
            # Display MER economics
            econ_params = render_optimization_inputs()
            opex_mer = (qh * econ_params['c_hu']) + (qc * econ_params['c_cu'])
            tac_mer = ann_cap_mer + opex_mer
            
            num_mer_utility_hex = (1 if qh > 0.1 else 0) + (1 if qc > 0.1 else 0)
            
            st.markdown("#### MER Network Economics")
            mer_col1, mer_col2, mer_col3 = st.columns(3)
            mer_col1.metric("Annual Capital Cost", f"${ann_cap_mer:,.2f}/yr",
                           f"({len(match_summary)} process + {num_mer_utility_hex} utility HEX)")
            mer_col2.metric("Annual Operating Cost", f"${opex_mer:,.2f}/yr")
            mer_col3.metric("Total Annual Cost (TAC)", f"${tac_mer:,.2f}/yr")
            
            # GA Optimization
            st.markdown("---")
            st.markdown("### Genetic Algorithm Optimization")
            
            if st.button("🚀 Run GA Optimization", type="primary"):
                with st.status("Running Genetic Algorithm...", expanded=True) as status:
                    st.write("Preparing stream data...")
                    hot_streams, cold_streams = prepare_optimizer_data(edited_df)
                    
                    st.write("Initializing GA...")
                    optimized_matches, best_tac_from_ga, ga_instance = run_genetic_algorithm(
                        hot_streams, cold_streams, econ_params, dt_min_input
                    )
                    
                    status.update(label="✅ Optimization Complete!", state="complete", expanded=False)
                
                # Apply pruning
                optimized_matches = prune_matches(optimized_matches, GA_CONFIG['min_split_ratio'])
                
                if optimized_matches:
                    display_matches = []
                    final_cap_process = 0  # Capital for process-to-process HEX
                    
                    # Calculate remaining duties with temperature tracking
                    rem_h = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in hot_streams}
                    rem_c = {s['Stream']: s['mCp'] * abs(s['Ts'] - s['Tt']) for s in cold_streams}
                    
                    # Track current temperatures
                    current_temp_hot = {s['Stream']: s['Ts'] for s in hot_streams}
                    current_temp_cold = {s['Stream']: s['Ts'] for s in cold_streams}
                    mcp_hot = {s['Stream']: s['mCp'] for s in hot_streams}
                    mcp_cold = {s['Stream']: s['mCp'] for s in cold_streams}
                    
                    for m in optimized_matches:
                        q = m['Recommended Load [kW]']
                        hs = m['hot_stream_data']
                        cs = m['cold_stream_data']
                        
                        ratio = q / (hs['mCp'] * abs(hs['Ts'] - hs['Tt']))
                        ratio_text = f"{round(ratio, 2)} " if ratio < 0.99 else ""
                        
                        u = calculate_u(hs['h'], cs['h'], econ_params['h_factor'])
                        
                        # Use current temperatures
                        thi = current_temp_hot[hs['Stream']]
                        tci = current_temp_cold[cs['Stream']]
                        tho = thi - (q / mcp_hot[hs['Stream']])
                        tco = tci + (q / mcp_cold[cs['Stream']])
                        
                        l_val = lmtd_chen(thi, tho, tci, tco)
                        
                        if l_val > 0 and u > 0:
                            area = q / (u * l_val)
                            final_cap_process += calculate_hex_capital(area, econ_params)
                            
                            # Update remaining duties and temperatures
                            rem_h[hs['Stream']] -= q
                            rem_c[cs['Stream']] -= q
                            current_temp_hot[hs['Stream']] = tho
                            current_temp_cold[cs['Stream']] = tco
                            
                            display_matches.append({
                                "Match": f"{ratio_text}Stream {hs['Stream']} ↔ {cs['Stream']}",
                                "Duty [kW]": round(q, 2),
                                "Area [m²]": round(area, 2)
                            })
                    
                    # Calculate final utilities
                    final_qh = sum(max(0, val) for val in rem_c.values())
                    final_qc = sum(max(0, val) for val in rem_h.values())
                    
                    # Add utility heat exchangers to capital cost
                    final_cap_utilities = 0
                    h_hu = econ_params.get('h_hu', 4.8)
                    h_cu = econ_params.get('h_cu', 1.6)
                    
                    if final_qh > 0.1:
                        lmtd_util = 50.0
                        h_cold_avg = np.mean([s['h'] for s in cold_streams]) if cold_streams else 1.6
                        u_hu = calculate_u(h_hu, h_cold_avg, econ_params['h_factor'])
                        if u_hu > 0:
                            area_hu = final_qh / (u_hu * lmtd_util)
                            final_cap_utilities += calculate_hex_capital(area_hu, econ_params)
                    
                    if final_qc > 0.1:
                        lmtd_util = 30.0
                        h_hot_avg = np.mean([s['h'] for s in hot_streams]) if hot_streams else 1.6
                        u_cu = calculate_u(h_hot_avg, h_cu, econ_params['h_factor'])
                        if u_cu > 0:
                            area_cu = final_qc / (u_cu * lmtd_util)
                            final_cap_utilities += calculate_hex_capital(area_cu, econ_params)
                    
                    # Total capital (already annual cost)
                    ann_cap_opt = final_cap_process + final_cap_utilities
                    opex_opt = (final_qh * econ_params['c_hu']) + (final_qc * econ_params['c_cu'])
                    tac_opt_actual = ann_cap_opt + opex_opt
                    
                    num_utility_hex = (1 if final_qh > 0.1 else 0) + (1 if final_qc > 0.1 else 0)

                    st.markdown("#### Optimized Heat Exchanger Network")
                    st.dataframe(pd.DataFrame(display_matches), use_container_width=True)
                    
                    # Display utilities
                    opt_u_col1, opt_u_col2 = st.columns(2)
                    opt_u_col1.metric("Hot Utility (Qh)", f"{final_qh:,.2f} kW")
                    opt_u_col2.metric("Cold Utility (Qc)", f"{final_qc:,.2f} kW")

                    o_col1, o_col2, o_col3 = st.columns(3)
                    o_col1.metric("Annual Capital Cost", f"${ann_cap_opt:,.2f}/yr", 
                                 f"({len(optimized_matches)} process + {num_utility_hex} utility HEX)")
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
                        "Process HEX": [0, len(match_summary), len(optimized_matches)],
                        "Utility HEX": [0, num_mer_utility_hex, num_utility_hex],
                        "Total HEX": [0, len(match_summary) + num_mer_utility_hex, len(optimized_matches) + num_utility_hex],
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
