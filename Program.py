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
    if h1 <= 0 or h2 <= 0:
        return 0
    h1 *= h_unit_factor
    h2 *= h_unit_factor
    return 1 / ((1/h1) + (1/h2))


def lmtd_chen(t1, t2, t3, t4):
    theta1 = max(t1 - t4, 0.001)
    theta2 = max(t2 - t3, 0.001)
    if abs(theta1 - theta2) < 0.01:
        return theta1
    return (theta1 * theta2 * (theta1 + theta2) / 2) ** (1/3)


def validate_dataframe(df):
    required_cols = ["Stream", "Type", "mCp", "Ts", "Tt", "h"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    if df.empty:
        return False, "DataFrame is empty"
    if not df["Type"].isin(["Hot", "Cold"]).all():
        return False, "Type must be Hot or Cold"
    for col in ["mCp", "Ts", "Tt", "h"]:
        pd.to_numeric(df[col], errors="raise")
    return True, "Valid"


def run_thermal_logic(df, dt):
    df = df.copy()
    df[['mCp', 'Ts', 'Tt']] = df[['mCp', 'Ts', 'Tt']].apply(pd.to_numeric)

    df['S_Ts'] = np.where(df['Type'] == 'Hot', df['Ts'], df['Ts'] + dt)
    df['S_Tt'] = np.where(df['Type'] == 'Hot', df['Tt'], df['Tt'] + dt)

    temps = sorted(pd.concat([df['S_Ts'], df['S_Tt']]).unique(), reverse=True)

    intervals = []
    for i in range(len(temps)-1):
        hi, lo = temps[i], temps[i+1]
        h_mcp = df[(df.Type == "Hot") & (df.S_Ts >= hi) & (df.S_Tt <= lo)].mCp.sum()
        c_mcp = df[(df.Type == "Cold") & (df.S_Ts <= lo) & (df.S_Tt >= hi)].mCp.sum()
        intervals.append((h_mcp - c_mcp) * (hi - lo))

    cum = [0] + list(np.cumsum(intervals))
    qh_min = abs(min(cum))
    feasible = [x + qh_min for x in cum]
    pinch = temps[feasible.index(0)] if 0 in feasible else None

    return qh_min, feasible[-1], pinch, temps, feasible, df


# --- MER MATCHING (FIXED INDENTATION) ---
def match_logic_with_splitting(df, pinch_t, side):
    sub = df.copy()

    if side == "Above":
        sub["S_Ts"] = sub["S_Ts"].clip(lower=pinch_t)
        sub["S_Tt"] = sub["S_Tt"].clip(lower=pinch_t)
    else:
        sub["S_Ts"] = sub["S_Ts"].clip(upper=pinch_t)
        sub["S_Tt"] = sub["S_Tt"].clip(upper=pinch_t)

    sub["Q"] = sub.mCp * abs(sub.S_Ts - sub.S_Tt)
    total_duty = sub.set_index("Stream")["Q"].to_dict()

    hot = sub[sub.Type == "Hot"].to_dict("records")
    cold = sub[sub.Type == "Cold"].to_dict("records")

    matches = []

    while any(h["Q"] > 1 for h in hot) and any(c["Q"] > 1 for c in cold):
        h = next(x for x in hot if x["Q"] > 1)
        c = next((x for x in cold if x["Q"] > 1), None)

        if not c:
            break

        m_q = min(h["Q"], c["Q"])
        ratio = m_q / total_duty[h["Stream"]]

        if ratio >= 0.1 or ratio >= 0.99:
            label = f"{round(ratio,2)} " if ratio < 0.99 else ""
            h["Q"] -= m_q
            c["Q"] -= m_q

            matches.append({
                "Match": f"{label}Stream {h['Stream']} ↔ {c['Stream']}",
                "Duty [kW]": round(m_q, 2),
                "Type": "Split" if ratio < 0.99 else "Direct",
                "Side": side
            })
        else:
            break

    return matches, hot, cold


# --- ECONOMICS ---
def calculate_mer_capital_properly(matches, df, econ, pinch, dt):
    cap = 0
    h_factor = econ["h_factor"]

    for m in matches:
        h_id, c_id = m["Match"].replace("Stream ", "").split(" ↔ ")
        h = df[(df.Stream.astype(str) == h_id) & (df.Type == "Hot")].iloc[0]
        c = df[(df.Stream.astype(str) == c_id) & (df.Type == "Cold")].iloc[0]

        q = m["Duty [kW]"]

        thi = h.Ts if m["Side"] == "Above" else min(h.Ts, pinch)
        tci = max(c.Ts, pinch-dt) if m["Side"] == "Above" else c.Ts

        tho = thi - q / h.mCp
        tco = tci + q / c.mCp

        u = calculate_u(h.h, c.h, h_factor)
        lmtd = lmtd_chen(thi, tho, tci, tco)

        if u > 0 and lmtd > 0:
            area = q / (u * lmtd)
            cap += econ["a"] + econ["b"] * (area ** econ["c"])

    return cap
