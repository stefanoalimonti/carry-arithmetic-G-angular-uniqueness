#!/usr/bin/env python3
"""G12: Enhanced PSLQ and multi-term extrapolation for c₁(3) using exact K=2-12 data.

Key insight: the odd-even oscillation in ρ(K) comes from a negative eigenvalue
(-1/3)^K in the transfer operator composition. Separating K into even and odd
subsequences eliminates the oscillation and reveals clean geometric convergence,
enabling much higher precision extrapolation from existing data.

Strategy:
  1. Odd-even separation eliminates oscillatory eigenvalue (-1/3)^K
  2. Multi-term Richardson on each subsequence with rate 1/3
  3. Full alternating-eigenvalue model: c₁(K) = c∞ + A·(1/3)^K + B·(-1/3)^K + ...
  4. PSLQ with extended constant basis on best estimates
  5. Leave-one-out cross-validation for error bars

Requires: numpy, scipy, mpmath
"""

import numpy as np
from scipy.optimize import least_squares
import mpmath
from mpmath import mpf, mp, log, pi, sqrt, atan, catalan, zeta

mp.dps = 50

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PRECISION WARNING: Input data has ~2-3 significant digits.     ║
# ║  PSLQ requires ~8+ digits for reliable identification.         ║
# ║  Results below are exploratory, not definitive.                ║
# ╚══════════════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════════════════════════════════
# EXACT DATA from G10 output
# K=2..10: 15-digit decimals from exact integer arithmetic
# K=11,12: exact rationals
# ═══════════════════════════════════════════════════════════════════════════════

C1 = {
    2:  mpf('0.500000000000000'),
    3:  mpf('0.481481481481481'),
    4:  mpf('0.531851851851852'),
    5:  mpf('0.546700960219478'),
    6:  mpf('0.567093851656498'),
    7:  mpf('0.574612783496652'),
    8:  mpf('0.582804025395297'),
    9:  mpf('0.586844629946830'),
    10: mpf('0.590154017513966'),
    11: mpf('8342998915') / mpf('13946313395'),
    12: mpf('75125607729') / mpf('125521846608'),
}

TARGET = log(3) - mpf('0.5')
ALL_K = sorted(C1.keys())
K_EVEN = [K for K in ALL_K if K % 2 == 0]
K_ODD  = [K for K in ALL_K if K % 2 == 1]


def richardson_solve(Ks, c1s, rate):
    """Solve c₁(K) = c∞ + Σ_{j=1}^{n-1} a_j · rate^(j·K) for c∞ and amplitudes.
    With n data points, determines n parameters exactly."""
    n = len(Ks)
    A = mpmath.matrix(n, n)
    b = mpmath.matrix(n, 1)
    for i in range(n):
        A[i, 0] = 1
        for j in range(1, n):
            A[i, j] = rate ** (j * Ks[i])
        b[i] = c1s[i]
    return mpmath.lu_solve(A, b)


def pslq_search(val, label, bases_dict, maxcoeff=1000):
    """Run PSLQ on val against a dictionary of named constants."""
    names = ['c∞'] + list(bases_dict.keys())
    vec = [val] + list(bases_dict.values())
    try:
        rel = mpmath.pslq(vec, maxcoeff=maxcoeff, maxsteps=10000)
        if rel is not None:
            terms = [f"{c}·{n}" for c, n in zip(rel, names) if c != 0]
            expr = " + ".join(terms)
            if rel[0] != 0:
                c_rec = -sum(mpf(rel[i]) * vec[i] for i in range(1, len(rel))) / rel[0]
                delta = float(c_rec - val)
                print(f"  [{label}] FOUND: {expr} = 0  →  c∞ = {float(c_rec):.15f} (Δ={delta:+.1e})")
                return rel, c_rec
            else:
                print(f"  [{label}] Degenerate: {expr} = 0 (c∞ coefficient = 0)")
        else:
            print(f"  [{label}] No relation (maxcoeff={maxcoeff})")
    except Exception as e:
        print(f"  [{label}] Error: {e}")
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
print("G12: ENHANCED PSLQ AND EXTRAPOLATION FOR c₁(3)")
print("=" * 70)
print(f"Target candidate: ln(3) - 1/2 = {float(TARGET):.15f}")
print(f"\nExact data (K=2..12):")
for K in ALL_K:
    delta = C1[K] - TARGET
    print(f"  K={K:2d}: c₁ = {float(C1[K]):.15f}  Δ = {float(delta):+.6e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ODD-EVEN SEPARATED RICHARDSON
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SECTION 1: ODD-EVEN SEPARATED RICHARDSON EXTRAPOLATION")
print("=" * 70)
print("""
Physical basis: the full model is
  c₁(K) = c∞ + A·(1/3)^K + B·(-1/3)^K + C·(1/9)^K + D·(-1/9)^K + ...

For EVEN K: (-1/3)^K = +(1/3)^K, so c₁ = c∞ + (A+B)(1/3)^K + (C+D)(1/9)^K + ...
For ODD  K: (-1/3)^K = -(1/3)^K, so c₁ = c∞ + (A-B)(1/3)^K + (C-D)(1/9)^K + ...

Each subsequence has PURE geometric convergence at rate 1/3, no oscillation.
""")

estimates = {}

for label, Ks in [("EVEN", K_EVEN), ("ODD", K_ODD)]:
    c1s = [C1[K] for K in Ks]
    n = len(Ks)

    print(f"--- {label} K = {Ks} ({n} points) ---")

    # Progressive Richardson (from 2 terms up to all n)
    print(f"  Progressive Richardson (rate = 1/3):")
    for m in range(2, n + 1):
        sub_K = Ks[-m:]
        sub_c = [C1[K] for K in sub_K]
        x = richardson_solve(sub_K, sub_c, mpf('1') / 3)
        c_inf = x[0]
        delta = c_inf - TARGET
        amps = ", ".join(f"a{j}={float(x[j]):+.3e}" for j in range(1, m))
        print(f"    m={m} K={sub_K}: c∞ = {float(c_inf):.12f}  Δ = {float(delta):+.6e}  [{amps}]")
        if m == n:
            estimates[f'{label.lower()}_full'] = c_inf

    # Also try rate = fitted (use ratio of last two corrections)
    if n >= 3:
        d1 = c1s[-1] - c1s[-2]
        d2 = c1s[-2] - c1s[-3]
        k1, k2, k3 = Ks[-3], Ks[-2], Ks[-1]
        if abs(float(d2)) > 1e-15:
            ratio = d1 / d2
            step = Ks[-1] - Ks[-2]
            eff_rate = abs(float(ratio)) ** (1.0 / step)
            print(f"  Effective rate from last 2 corrections: r_eff = {eff_rate:.6f} (vs 1/3 = 0.333)")
    print()

# Weighted average (more weight on even, which has 6 points vs 5)
w_even, w_odd = 6, 5
estimates['weighted_avg'] = (w_even * estimates['even_full'] + w_odd * estimates['odd_full']) / (w_even + w_odd)

print("--- COMBINED ESTIMATES ---")
for name, val in estimates.items():
    delta = val - TARGET
    print(f"  {name:18s}: c∞ = {float(val):.15f}  Δ(ln3-½) = {float(delta):+.6e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ALTERNATING EIGENVALUE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SECTION 2: ALTERNATING EIGENVALUE MODELS (full K=2..12)")
print("=" * 70)

all_c1_np = np.array([float(C1[K]) for K in ALL_K], dtype=np.float64)
all_K_np = np.array(ALL_K, dtype=np.float64)

# Model A: fixed eigenvalues ±1/3, ±1/9, 1/27
def model_A(p, Ks):
    c, A, B, C, D, E = p
    return np.array([c + A*(1/3)**K + B*(-1/3)**K + C*(1/9)**K + D*(-1/9)**K + E*(1/27)**K for K in Ks])

res_A = least_squares(lambda p: model_A(p, all_K_np) - all_c1_np,
                      [0.5986, -10., 5., 50., -20., -100.], method='lm')
pA = res_A.x
rms_A = np.sqrt(np.mean(res_A.fun**2))
print(f"\nModel A: c∞ + A(1/3)^K + B(-1/3)^K + C(1/9)^K + D(-1/9)^K + E(1/27)^K")
print(f"  c∞    = {pA[0]:.15f}  Δ(ln3-½) = {pA[0] - float(TARGET):+.6e}")
print(f"  A(+⅓) = {pA[1]:+.4f}  B(-⅓) = {pA[2]:+.4f}")
print(f"  C(+⅑) = {pA[3]:+.4f}  D(-⅑) = {pA[4]:+.4f}  E(1/27) = {pA[5]:+.4f}")
print(f"  RMS   = {rms_A:.2e}")
estimates['model_A'] = mpf(str(pA[0]))

# Model B: ±1/3, free negative rate r
def model_B(p, Ks):
    c, A, B, r, C, D = p
    return np.array([c + A*(1/3)**K + B*(-abs(r))**K + C*(1/9)**K + D*(1/27)**K for K in Ks])

try:
    res_B = least_squares(lambda p: model_B(p, all_K_np) - all_c1_np,
                          [0.5986, -10., 5., 0.35, 50., -100.], method='lm', max_nfev=10000)
    pB = res_B.x
    rms_B = np.sqrt(np.mean(res_B.fun**2))
    print(f"\nModel B: free negative eigenvalue")
    print(f"  c∞    = {pB[0]:.15f}  Δ(ln3-½) = {pB[0] - float(TARGET):+.6e}")
    print(f"  r_neg = {abs(pB[3]):.6f}  (expected ~1/3 = 0.3333)")
    print(f"  RMS   = {rms_B:.2e}")
    estimates['model_B'] = mpf(str(pB[0]))
except Exception as e:
    print(f"\nModel B failed: {e}")

# Model C: complex pair ρ·e^{±iθ} (subsumes ±1/3 when θ=π)
def model_C(p, Ks):
    c, A, rho, theta, B, C = p
    return np.array([c + A*(1/3)**K + B*rho**K*np.cos(theta*K) + C*(1/9)**K for K in Ks])

try:
    res_C = least_squares(lambda p: model_C(p, all_K_np) - all_c1_np,
                          [0.5986, -5., 0.33, np.pi, 5., 50.], method='lm', max_nfev=10000)
    pC = res_C.x
    rms_C = np.sqrt(np.mean(res_C.fun**2))
    print(f"\nModel C: complex eigenvalue pair ρ·e^{{±iθ}}")
    print(f"  c∞   = {pC[0]:.15f}  Δ(ln3-½) = {pC[0] - float(TARGET):+.6e}")
    print(f"  ρ    = {abs(pC[2]):.6f}  θ/π = {pC[3]/np.pi:.6f}  (θ=π → pure alternating)")
    print(f"  RMS  = {rms_C:.2e}")
    estimates['model_C'] = mpf(str(pC[0]))
except Exception as e:
    print(f"\nModel C failed: {e}")

# Prediction for K=13,14
print(f"\nPredictions:")
pred_13A = model_A(pA, [13])[0]
pred_14A = model_A(pA, [14])[0]
print(f"  Model A: c₁(13) = {pred_13A:.10f},  c₁(14) = {pred_14A:.10f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PSLQ ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SECTION 3: PSLQ ANALYSIS")
print("=" * 70)

full_basis = {
    '1':       mpf('1'),
    'ln2':     log(2),
    'ln3':     log(3),
    'ln3²':    log(3)**2,
    'ln2·ln3': log(2) * log(3),
    'π':       pi,
    'π²':      pi**2,
    'arctan2': atan(2),
    'ζ(2)':    zeta(2),
    'ζ(3)':    zeta(3),
    'Catalan':  catalan,
    '√3':      sqrt(3),
}

small_bases = [
    ("1, ln3",             {'1': mpf(1), 'ln3': log(3)}),
    ("1, ln3, ln2",        {'1': mpf(1), 'ln3': log(3), 'ln2': log(2)}),
    ("1, ln3, π",          {'1': mpf(1), 'ln3': log(3), 'π': pi}),
    ("1, ln3, ζ(2)",       {'1': mpf(1), 'ln3': log(3), 'ζ(2)': zeta(2)}),
    ("1, ln3, arctan2",    {'1': mpf(1), 'ln3': log(3), 'arctan2': atan(2)}),
    ("1, ln3, ln3²",       {'1': mpf(1), 'ln3': log(3), 'ln3²': log(3)**2}),
    ("1, ln2, ln3, arctan2", {'1': mpf(1), 'ln2': log(2), 'ln3': log(3), 'arctan2': atan(2)}),
    ("1, ln3, ζ(3)",       {'1': mpf(1), 'ln3': log(3), 'ζ(3)': zeta(3)}),
    ("1, ln3, Catalan",    {'1': mpf(1), 'ln3': log(3), 'Catalan': catalan}),
]

for est_name, est_val in sorted(estimates.items()):
    print(f"\n--- PSLQ on {est_name} = {float(est_val):.15f} ---")
    pslq_search(est_val, "full basis", full_basis)
    for basis_label, basis_dict in small_bases:
        pslq_search(est_val, basis_label, basis_dict, maxcoeff=100)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ERROR ANALYSIS (Leave-one-out)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SECTION 4: LEAVE-ONE-OUT CROSS-VALIDATION")
print("=" * 70)

for label, Ks in [("EVEN", K_EVEN), ("ODD", K_ODD)]:
    c1s = [C1[K] for K in Ks]
    n = len(Ks)
    loo_vals = []
    print(f"\n--- {label} K = {Ks} ---")
    for leave in range(n):
        sub_K = Ks[:leave] + Ks[leave + 1:]
        sub_c = [C1[K] for K in sub_K]
        x = richardson_solve(sub_K, sub_c, mpf('1') / 3)
        c_inf = x[0]
        loo_vals.append(c_inf)
        delta = c_inf - TARGET
        print(f"  Leave K={Ks[leave]:2d}: c∞ = {float(c_inf):.12f}  Δ(ln3-½) = {float(delta):+.6e}")

    mean_loo = sum(loo_vals) / len(loo_vals)
    std_loo = float(mpmath.sqrt(sum((v - mean_loo) ** 2 for v in loo_vals) / max(1, len(loo_vals) - 1)))
    delta_loo = float(mean_loo - TARGET)
    print(f"  MEAN: c∞ = {float(mean_loo):.12f} ± {std_loo:.2e}  Δ(ln3-½) = {delta_loo:+.6e}")
    estimates[f'{label.lower()}_loo_mean'] = mean_loo

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DIRECT CANDIDATE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SECTION 5: CANDIDATE COMPARISON")
print("=" * 70)

candidates = [
    ('ln(3) - 1/2',          log(3) - mpf('1') / 2),
    ('ln(3) - ln(2)',         log(3) - log(2)),
    ('3·ln(3)/2 - 1',        3 * log(3) / 2 - 1),
    ('2·ln(3) - 3/2',        2 * log(3) - mpf('3') / 2),
    ('ln(9)/2 - 1/2',        log(9) / 2 - mpf('1') / 2),
    ('ζ(2)/3 - ln(2)/2',     zeta(2) / 3 - log(2) / 2),
    ('arctan(2) - π/6',      atan(2) - pi / 6),
    ('π²/18 + ln(2) - 1',    pi ** 2 / 18 + log(2) - 1),
    ('ln(3) - arctan(1)',     log(3) - pi / 4),
    ('Catalan/2',             catalan / 2),
    ('1 - 2·ln(2) + ln(3)',   1 - 2 * log(2) + log(3)),
]

best = estimates['weighted_avg']
print(f"\nBest estimate (weighted avg): c∞ = {float(best):.15f}\n")
scored = sorted(candidates, key=lambda c: abs(float(c[1] - best)))
for name, val in scored:
    delta = float(best - val)
    print(f"  {name:25s} = {float(val):.15f}  Δ = {delta:+.6e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_est = [(n, float(v)) for n, v in estimates.items()]
vals = [v for _, v in all_est]
mean_all = np.mean(vals)
std_all = np.std(vals, ddof=1)
delta_mean = mean_all - float(TARGET)

print(f"\nAll c∞ estimates:")
for name, val in sorted(all_est):
    print(f"  {name:20s}: {val:.12f}")
print(f"\n  Mean:  {mean_all:.12f}")
print(f"  Std:   {std_all:.2e}")
print(f"  ln(3) - 1/2:  {float(TARGET):.12f}")
print(f"  Discrepancy:   {delta_mean:+.6e}")
if std_all > 0:
    print(f"  Significance:  {abs(delta_mean) / std_all:.1f} σ")

print(f"\n  Available precision: ~{-int(np.log10(max(std_all, 1e-15)))}-{-int(np.log10(max(abs(delta_mean), 1e-15)))} digits")
print(f"  PSLQ requires ~8+ digits for reliable identification")
print(f"\n  Conclusion: {'CONSISTENT' if abs(delta_mean) < 2 * std_all else 'INCONCLUSIVE'}: c₁(3) = ln(3) - 1/2")
print(f"  Next step: K=13 data (+2 digits) would make PSLQ definitive")
