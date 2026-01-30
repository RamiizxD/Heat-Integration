import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import pygad
import outdoor

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Process Heat Integration Tool", layout="wide")

st.title("Process Integration & Heat Exchanger Network Analysis")
st.markdown("""
This application performs **Pinch Analysis**, **MER Matching with Stream Splitting**, and 
**Economic Optimization using a Genetic Algorithm**.
""")
st.markdown("---")

# --- CORE MATH FUNCTIONS ---
def calculate_u(h1, h2):
    return 1 / ((1/h1) + (1/h2))

def lmtd_chen(t1, t2, t3, t4):
    theta1 = max(abs(t1 - t4), 0.01)
    theta2 = max(abs(t2 - t3), 0.01)
    return (theta1 * theta2 * (theta1 + theta2) / 2)**(1/3)

def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)
    
    # Temperature Shifting: Cold is shifted UP by dt
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
    st.session_state['processed_df'] = processed_df # Store for Section 4
    
    st.markdown("---")
    st.subheader("2. Pinch Analysis Result")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Hot Utility (Qh)", f"{qh:,.2f} kW")
        st.metric("Cold Utility (Qc)", f"{qc:,.2f} kW")
        st.metric("Pinch Temperature (Hot)", f"{pinch} °C" if pinch is not None else "N/A")
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

    # --- SECTION 4: Optimization using Genetic Algorithm & OUTDOOR ---
    st.markdown("---")
    st.subheader("4. Optimization for TAC using Genetic Algorithm (SWS Model)")

    with st.expander("GA Optimization Settings"):
        c1, c2, c3 = st.columns(3)
        pop_size = c1.slider("Population Size", 10, 100, 30)
        generations = c2.slider("Max Generations", 10, 500, 100)
        mutation_p = c3.slider("Mutation Probability", 0.01, 0.2, 0.05)
        st.markdown("**Cost Coefficients (Investment = a + b * Area^c)**")
        cc1, cc2, cc3 = st.columns(3)
        fixed_cost = cc1.number_input("Fixed Cost ($/unit)", value=8000.0)
        area_coeff = cc2.number_input("Area Coefficient", value=433.3)
        exponent = cc3.number_input("Area Exponent", value=0.6)

    def fitness_func(ga_instance, solution, solution_idx):
        df = st.session_state['processed_df']
        n_hot = len(df[df['Type']=='Hot'])
        n_cold = len(df[df['Type']=='Cold'])
        n_stages = max(n_hot, n_cold)
        
        topology = solution.reshape((n_hot, n_cold, n_stages))
        sws_model = outdoor.Superstructure()
        
        for _, row in df.iterrows():
            sws_model.add_stream(name=row['Stream'], t_start=row['Ts'], t_target=row['Tt'], mcp=row['mCp'], h_coeff=row['h'], type=row['Type'])
        
        for i in range(n_hot):
            for j in range(n_cold):
                for k in range(n_stages):
                    if topology[i, j, k] == 1: sws_model.add_exchanger(hot_idx=i, cold_idx=j, stage=k)

        try:
            results = sws_model.solve() 
            investment = (np.sum(solution) * fixed_cost) + (area_coeff * (results.total_area ** exponent))
            tac = results.operating_cost + (investment / 5)
            return 1.0 / (tac + 1e-6)
        except:
            return 1e-9

    if st.button("Run SWS GA Optimization"):
        n_hot = len(processed_df[processed_df['Type']=='Hot'])
        n_cold = len(processed_df[processed_df['Type']=='Cold'])
        num_genes = n_hot * n_cold * max(n_hot, n_cold) 

        ga_instance = pygad.GA(
            num_generations=generations, num_parents_mating=int(pop_size/2), 
            fitness_func=fitness_func, sol_per_pop=pop_size, num_genes=num_genes,
            gene_type=int, init_range_low=0, init_range_high=2, mutation_probability=mutation_p
        )

        with st.spinner("OUTDOOR is solving superstructure nodes..."):
            ga_instance.run()

        sol, sol_fit, _ = ga_instance.best_solution()
        st.success(f"Optimization Complete! Optimal TAC: ${1.0/sol_fit:,.2f}")
        
        fig_conv = go.Figure(go.Scatter(y=ga_instance.best_solutions_fitness, mode='lines'))
        fig_conv.update_layout(title="GA Convergence", xaxis_title="Generation", yaxis_title="Fitness")
        st.plotly_chart(fig_conv, use_container_width=True)

    # --- FINAL EXPORT SECTION ---
    st.markdown("---")
    st.subheader("5. Export Results")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if match_summary: pd.DataFrame(match_summary).to_excel(writer, sheet_name='HEN_Matches', index=False)
        edited_df.to_excel(writer, sheet_name='Input_Data', index=False)
    st.download_button("📥 Download HEN Report", output.getvalue(), "HEN_Analysis.xlsx")
