# --- SECTION 1: PRIMARY DATA INPUT ---
st.subheader("1. Stream Data & System Parameters")

if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False

with st.form("main_input_form"):
    col_param, _ = st.columns([1, 2])
    with col_param:
        dt_min_input = st.number_input("Target ΔTmin", value=10.0, step=1.0)
    
    # Corrected Indentation for the 4-Stream Classic Benchmark
    init_data = pd.DataFrame([
        {"Stream": "H1", "Type": "Hot", "mCp": 30.0, "Ts": 170.0, "Tt": 60.0, "individual HTC (h)": 0.5},
        {"Stream": "H2", "Type": "Hot", "mCp": 15.0, "Ts": 150.0, "Tt": 30.0, "individual HTC (h)": 0.5},
        {"Stream": "C1", "Type": "Cold", "mCp": 20.0, "Ts": 20.0, "Tt": 135.0, "individual HTC (h)": 0.5},
        {"Stream": "C2", "Type": "Cold", "mCp": 40.0, "Ts": 80.0, "Tt": 140.0, "individual HTC (h)": 0.5}
    ])
    
    edited_df = st.data_editor(init_data, num_rows="dynamic", use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Hot", "Cold"], required=True),
            "individual HTC (h)": st.column_config.NumberColumn("(h)", min_value=0.01, format="%.3f")
        }
    )
    submit_thermal = st.form_submit_button("Run Pinch Analysis & MER HEN Synthesis")
