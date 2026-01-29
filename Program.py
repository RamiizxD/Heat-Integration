import pandas as pd
from hen_core import run_hen_optimization

# ============================================================
# USER INPUT
# ============================================================

dt_min = 10.0  # °C

econ = {
    'a': 8000.0,     # $
    'b': 1200.0,    # $/m²
    'c': 0.6,       # exponent
    'CRF': 0.2,     # capital recovery factor
    'c_hu': 80.0,   # $/kW·yr
    'c_cu': 20.0    # $/kW·yr
}

# ============================================================
# STREAM DATA (EXAMPLE – REPLACE WITH YOUR OWN)
# ============================================================

data = [
    ['H1', 'Hot', 2.5, 180, 60, 5.0],
    ['H2', 'Hot', 1.8, 150, 40, 5.0],
    ['C1', 'Cold', 3.0, 30, 140, 0.8],
    ['C2', 'Cold', 2.2, 50, 160, 0.8]
]

df = pd.DataFrame(
    data,
    columns=['Stream', 'Type', 'mCp', 'Ts', 'Tt', 'h']
)

# ============================================================
# RUN OPTIMIZATION
# ============================================================

results = run_hen_optimization(df, dt_min, econ)

# ============================================================
# OUTPUT
# ============================================================

print("\n===== HEN ECONOMIC RESULTS =====")
print(f"No Integration TAC : ${results['No Integration']:,.2f} /yr")

if results.get('Optimized TAC'):
    print(f"Optimized TAC      : ${results['Optimized TAC']:,.2f} /yr")
    print("\nOptimized Matches:")
    for m in results['Matches']:
        print(f"  {m['Hot']} → {m['Cold']} : {m['q']:.2f} kW")
else:
    print("No economically viable heat recovery found.")
