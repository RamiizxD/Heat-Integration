import numpy as np
import pandas as pd
import copy

# ============================================================
# 1. HEAT TRANSFER & THERMODYNAMICS
# ============================================================

def calculate_u(h_hot, h_cold):
    return 1.0 / (1.0/h_hot + 1.0/h_cold)


def lmtd_chen(Th_in, Th_out, Tc_in, Tc_out):
    dT1 = max(Th_in - Tc_out, 1e-6)
    dT2 = max(Th_out - Tc_in, 1e-6)
    return ((dT1 * dT2 * (dT1 + dT2)) / 2.0) ** (1.0/3.0)


# ============================================================
# 2. SHIFTED STREAM REPRESENTATION (ΔTmin ENFORCED)
# ============================================================

def shifted_streams(df, dt_min):
    hot, cold = [], []

    for _, s in df.iterrows():
        if s['Type'] == 'Hot':
            hot.append({
                'Stream': s['Stream'],
                'mCp': s['mCp'],
                'Ts': s['Ts'] - dt_min / 2,
                'Tt': s['Tt'] - dt_min / 2,
                'h': s['h']
            })
        else:
            cold.append({
                'Stream': s['Stream'],
                'mCp': s['mCp'],
                'Ts': s['Ts'] + dt_min / 2,
                'Tt': s['Tt'] + dt_min / 2,
                'h': s['h']
            })
    return hot, cold


def max_feasible_q(hs, cs):
    q_hot = hs['mCp'] * (hs['Ts'] - hs['Tt'])
    q_cold = cs['mCp'] * (cs['Tt'] - cs['Ts'])
    return min(q_hot, q_cold)


# ============================================================
# 3. DGS SCREENING STEP (ΔTAC < 0 CONDITION)
# ============================================================

def find_q_dep(hs, cs, econ):
    u = calculate_u(hs['h'], cs['h'])
    q_max = max_feasible_q(hs, cs)

    q = 1.0
    step = 1.0

    while q < q_max:
        Th_out = hs['Ts'] - q / hs['mCp']
        Tc_out = cs['Ts'] + q / cs['mCp']

        if Th_out <= Tc_out:
            break

        lmtd = lmtd_chen(hs['Ts'], Th_out, cs['Ts'], Tc_out)
        A = q / (u * lmtd)

        cap = econ['a'] + econ['b'] * (A ** econ['c'])
        delta_TAC = econ['CRF'] * cap - q * (econ['c_hu'] + econ['c_cu'])

        if delta_TAC < 0:
            return q

        q += step

    return None


# ============================================================
# 4. STREAM-WISE UTILITY CALCULATION (CRITICAL FIX)
# ============================================================

def calculate_utilities(matches, hot, cold):
    hot_remaining = {
        s['Stream']: s['mCp'] * (s['Ts'] - s['Tt']) for s in hot
    }
    cold_remaining = {
        s['Stream']: s['mCp'] * (s['Tt'] - s['Ts']) for s in cold
    }

    for m in matches:
        hot_remaining[m['Hot']] -= m['q']
        cold_remaining[m['Cold']] -= m['q']

    Qh = sum(max(0, v) for v in cold_remaining.values())
    Qc = sum(max(0, v) for v in hot_remaining.values())

    return Qh, Qc


# ============================================================
# 5. TOTAL ANNUAL COST (PAPER-CORRECT FORM)
# ============================================================

def network_TAC(matches, hot, cold, econ):
    cap_total = 0.0

    for m in matches:
        hs = next(s for s in hot if s['Stream'] == m['Hot'])
        cs = next(s for s in cold if s['Stream'] == m['Cold'])

        u = calculate_u(hs['h'], cs['h'])
        Th_out = hs['Ts'] - m['q'] / hs['mCp']
        Tc_out = cs['Ts'] + m['q'] / cs['mCp']

        lmtd = lmtd_chen(hs['Ts'], Th_out, cs['Ts'], Tc_out)
        A = m['q'] / (u * lmtd)

        cap_total += econ['a'] + econ['b'] * (A ** econ['c'])

    Qh, Qc = calculate_utilities(matches, hot, cold)
    opex = Qh * econ['c_hu'] + Qc * econ['c_cu']

    return econ['CRF'] * cap_total + opex


# ============================================================
# 6. RANDOM WALK / RWCE OPTIMIZATION
# ============================================================

def random_walk(matches, hot, cold, econ, n_iter=500):
    best = copy.deepcopy(matches)
    best_TAC = network_TAC(best, hot, cold, econ)

    for _ in range(n_iter):
        i = np.random.randint(len(best))

        hs = next(s for s in hot if s['Stream'] == best[i]['Hot'])
        cs = next(s for s in cold if s['Stream'] == best[i]['Cold'])

        q_old = best[i]['q']
        q_new = max(1.0, q_old + np.random.uniform(-10, 10))

        if q_new > max_feasible_q(hs, cs):
            continue

        best[i]['q'] = q_new
        TAC_new = network_TAC(best, hot, cold, econ)

        if TAC_new < best_TAC:
            best_TAC = TAC_new
        else:
            best[i]['q'] = q_old

    return best, best_TAC


# ============================================================
# 7. MASTER DRIVER (NO INTEGRATION / MER / TAC)
# ============================================================

def run_hen_optimization(df, dt_min, econ):
    hot, cold = shifted_streams(df, dt_min)

    # --- NO INTEGRATION ---
    Qh_base = sum(s['mCp'] * (s['Tt'] - s['Ts']) for s in cold)
    Qc_base = sum(s['mCp'] * (s['Ts'] - s['Tt']) for s in hot)
    TAC_no = Qh_base * econ['c_hu'] + Qc_base * econ['c_cu']

    # --- DGS SCREENING ---
    matches = []
    for hs in hot:
        for cs in cold:
            q = find_q_dep(hs, cs, econ)
            if q:
                matches.append({
                    'Hot': hs['Stream'],
                    'Cold': cs['Stream'],
                    'q': q
                })

    if not matches:
        return {
            'No Integration': TAC_no,
            'MER': None,
            'Optimized': None
        }

    # --- RWCE OPTIMIZATION ---
    best_matches, TAC_opt = random_walk(matches, hot, cold, econ)

    return {
        'No Integration': TAC_no,
        'Optimized TAC': TAC_opt,
        'Matches': best_matches
    }
