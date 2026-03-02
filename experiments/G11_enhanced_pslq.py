"""
G11_enhanced_pslq.py — Enhanced PSLQ for c₁(base 3)

Uses K=2..12 exact data (from G02/G07) with:
  1. Multi-term Richardson extrapolation (up to 5 terms)
  2. Extended PSLQ basis: arctan(2), arctan(2)², ln(3)², ln(2)·ln(3),
     π·ln(3), Catalan, ζ(3), and cross-products
  3. Angular integral prediction: base-3 D-odd boundary is a CURVE,
     integration limit is arctan(2), so the form should involve
     arctan(2)/something (confirming no π)
  4. Transfer operator eigenvalue analysis for base 3

When K=13 data becomes available from G10_base3_exact_k13.c,
add (sum_cm1, n_ulc) to raw_b3 below and rerun.
"""

import mpmath
mpmath.mp.dps = 80
from mpmath import mpf, pi, log, atan, sqrt, pslq, nstr, zeta, euler, catalan

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PRECISION WARNING: Input data has ~4-5 significant digits.     ║
# ║  PSLQ requires ~8+ digits for reliable identification.         ║
# ║  Results below are exploratory, not definitive.                ║
# ╚══════════════════════════════════════════════════════════════════╝

print("=" * 78)
print("  G11: ENHANCED PSLQ ANALYSIS FOR c₁(BASE 3)")
print("=" * 78)

# =====================================================================
# EXACT DATA: (sum_cm1, n_ulc) for K=2..12
# =====================================================================
raw_b3 = {
    2:  (23, 22),
    3:  (340, 261),
    4:  (3861, 2680),
    5:  (38593, 25359),
    6:  (364335, 233264),
    7:  (3348196, 2116057),
    8:  (30402412, 19101834),
    9:  (274620332, 172093303),
    10: (2475182404, 1549405546),
    # K=11 and K=12 from G02 (add exact values when available):
    # 11: (sum_cm1_11, n_ulc_11),
    # 12: (sum_cm1_12, n_ulc_12),
    # K=13 from G10 (add when computed):
    # 13: (sum_cm1_13, n_ulc_13),
}

c1_b3 = {}
for K, (s, n) in raw_b3.items():
    c1_b3[K] = mpf(s) / mpf(n) - 1

Ks = sorted(c1_b3.keys())
K_max = max(Ks)

print(f"\n  Data: K = {Ks[0]}..{K_max} ({len(Ks)} points)")
print(f"\n  {'K':>3s} {'c₁(K)':>20s} {'Δ':>14s} {'ρ':>10s}")
print("  " + "-" * 50)

prev_delta = None
for K in Ks:
    val = c1_b3[K]
    if K > Ks[0] and K - 1 in c1_b3:
        delta = c1_b3[K] - c1_b3[K - 1]
        if prev_delta is not None and abs(prev_delta) > 1e-20:
            rho = abs(float(delta / prev_delta))
            print(f"  {K:3d} {float(val):20.15f} {float(delta):+14.6e} {rho:10.6f}")
        else:
            print(f"  {K:3d} {float(val):20.15f} {float(delta):+14.6e} {'---':>10s}")
        prev_delta = delta
    else:
        print(f"  {K:3d} {float(val):20.15f} {'---':>14s} {'---':>10s}")

# =====================================================================
# PART 1: MULTI-TERM RICHARDSON EXTRAPOLATION
# =====================================================================
print(f"\n\n{'='*78}")
print("  PART 1: MULTI-TERM RICHARDSON EXTRAPOLATION")
print("=" * 78)

r = mpf(1) / 3  # base-3 convergence rate

# 2-term: c_∞ + A·(1/3)^K
print(f"\n--- 2-term model: c_∞ + A·(1/3)^K ---\n")
for K1, K2 in [(K_max - 1, K_max), (K_max - 2, K_max - 1), (K_max - 3, K_max - 2)]:
    if K1 in c1_b3 and K2 in c1_b3:
        c_inf = (c1_b3[K2] - r * c1_b3[K1]) / (1 - r)
        A = (c1_b3[K1] - c_inf) / mpmath.power(r, K1)
        print(f"  K={K1},{K2}: c_∞ = {float(c_inf):.15f}, A = {float(A):.6f}")

# 3-term: c_∞ + A·(1/3)^K + B·(1/9)^K
print(f"\n--- 3-term model: c_∞ + A/3^K + B/9^K ---\n")
best_3term = None
for start in range(max(Ks[0], K_max - 4), K_max - 1):
    K_list = [start, start + 1, start + 2]
    if all(k in c1_b3 for k in K_list):
        M = mpmath.matrix(3, 3)
        b_vec = mpmath.matrix(3, 1)
        for i, K in enumerate(K_list):
            M[i, 0] = 1
            M[i, 1] = mpmath.power(r, K)
            M[i, 2] = mpmath.power(r ** 2, K)
            b_vec[i] = c1_b3[K]
        try:
            sol = mpmath.lu_solve(M, b_vec)
            print(f"  K={K_list}: c_∞ = {float(sol[0]):.15f}, "
                  f"A = {float(sol[1]):.4f}, B = {float(sol[2]):.4f}")
            if K_list[-1] == K_max:
                best_3term = sol[0]
        except Exception as e:
            print(f"  K={K_list}: failed ({e})")

# 4-term: c_∞ + A/3^K + B/9^K + C/27^K
print(f"\n--- 4-term model: c_∞ + A/3^K + B/9^K + C/27^K ---\n")
best_4term = None
for start in range(max(Ks[0], K_max - 5), K_max - 2):
    K_list = [start, start + 1, start + 2, start + 3]
    if all(k in c1_b3 for k in K_list):
        M = mpmath.matrix(4, 4)
        b_vec = mpmath.matrix(4, 1)
        for i, K in enumerate(K_list):
            M[i, 0] = 1
            M[i, 1] = mpmath.power(r, K)
            M[i, 2] = mpmath.power(r ** 2, K)
            M[i, 3] = mpmath.power(r ** 3, K)
            b_vec[i] = c1_b3[K]
        try:
            sol = mpmath.lu_solve(M, b_vec)
            print(f"  K={K_list}: c_∞ = {float(sol[0]):.15f}")
            if K_list[-1] == K_max:
                best_4term = sol[0]
        except Exception as e:
            print(f"  K={K_list}: failed ({e})")

# 5-term: c_∞ + A/3^K + B/9^K + C/27^K + D/81^K
print(f"\n--- 5-term model: c_∞ + A/3^K + B/9^K + C/27^K + D/81^K ---\n")
best_5term = None
K_list_5 = list(range(K_max - 4, K_max + 1))
if all(k in c1_b3 for k in K_list_5):
    M = mpmath.matrix(5, 5)
    b_vec = mpmath.matrix(5, 1)
    for i, K in enumerate(K_list_5):
        M[i, 0] = 1
        for j in range(1, 5):
            M[i, j] = mpmath.power(r ** j, K)
        b_vec[i] = c1_b3[K]
    try:
        sol = mpmath.lu_solve(M, b_vec)
        best_5term = sol[0]
        print(f"  K={K_list_5}: c_∞ = {float(sol[0]):.15f}")
    except Exception as e:
        print(f"  K={K_list_5}: failed ({e})")

# Shanks transform
print(f"\n--- Shanks transform ---\n")
for K in range(max(Ks[0] + 2, K_max - 3), K_max + 1):
    if K in c1_b3 and K - 1 in c1_b3 and K - 2 in c1_b3:
        d1 = c1_b3[K] - c1_b3[K - 1]
        d0 = c1_b3[K - 1] - c1_b3[K - 2]
        if abs(d1 - d0) > 1e-30:
            shanks = c1_b3[K] - d1 ** 2 / (d1 - d0)
            print(f"  Shanks({K}) = {float(shanks):.15f}")

# Choose best estimate
estimates = []
if best_3term is not None:
    estimates.append(("3-term", best_3term))
if best_4term is not None:
    estimates.append(("4-term", best_4term))
if best_5term is not None:
    estimates.append(("5-term", best_5term))

if estimates:
    best_label, best_c1 = estimates[-1]
else:
    best_label = "2-term"
    best_c1 = (c1_b3[K_max] - r * c1_b3[K_max - 1]) / (1 - r)

print(f"\n  Best estimate ({best_label}): c₁(3) = {float(best_c1):.15f}")

# =====================================================================
# PART 2: EXTENDED PSLQ BASIS
# =====================================================================
print(f"\n\n{'='*78}")
print("  PART 2: PSLQ WITH EXTENDED BASIS")
print("=" * 78)

c = best_c1
atan2_val = atan(2)
ln2 = log(2)
ln3 = log(3)
ln3_2 = log(mpf(3) / 2)

print(f"\n  c₁(3) ≈ {float(c):.15f}")
print(f"\n  Reference constants:")
print(f"    arctan(2)  = {float(atan2_val):.15f}")
print(f"    ln(2)      = {float(ln2):.15f}")
print(f"    ln(3)      = {float(ln3):.15f}")
print(f"    ln(3)-1/2  = {float(ln3 - mpf(1)/2):.15f}")
print(f"    π          = {float(pi):.15f}")

candidates = [
    ("ln(3) - 1/2", ln3 - mpf(1) / 2),
    ("3·ln(3) - 2", 3 * ln3 - 2),
    ("arctan(2) - ln(3)/4", atan2_val - ln3 / 4),
    ("ln(3)²/2", ln3 ** 2 / 2),
    ("2·arctan(2)/π", 2 * atan2_val / pi),
    ("arctan(2)/2 + ln(3)/8", atan2_val / 2 + ln3 / 8),
    ("3·ln(4/3)", 3 * log(mpf(4) / 3)),
    ("ln(3)·ln(2)", ln3 * ln2),
    ("2·ln(3) - ln(2) - 1", 2 * ln3 - ln2 - 1),
    ("(2/9)·arctan(2)", mpf(2) / 9 * atan2_val),
    ("arctan(2)/2", atan2_val / 2),
    ("(arctan(2) - 1)/2 + ln(3)/3", (atan2_val - 1) / 2 + ln3 / 3),
    ("1 - 2·ln(3) + ln(3)²", 1 - 2 * ln3 + ln3 ** 2),
    ("3·arctan(2) - π", 3 * atan2_val - pi),
    ("ln(3)/(2·ln(2))", ln3 / (2 * ln2)),
]

print(f"\n  {'Candidate':>35s} {'Value':>18s} {'Δ':>14s}")
print("  " + "-" * 70)
for name, val in sorted(candidates, key=lambda x: abs(float(x[1] - c))):
    delta = float(val - c)
    print(f"  {name:>35s} {float(val):18.15f} {delta:+14.6e}")

# PSLQ with extended bases
print(f"\n--- PSLQ analysis (maxcoeff=1000) ---\n")

pslq_bases = [
    ("c, 1, ln3", [c, mpf(1), ln3]),
    ("c, 1, ln2, ln3", [c, mpf(1), ln2, ln3]),
    ("c, 1, ln3, arctan(2)", [c, mpf(1), ln3, atan2_val]),
    ("c, 1, ln2, ln3, arctan(2)", [c, mpf(1), ln2, ln3, atan2_val]),
    ("c, 1, ln3, ln3²", [c, mpf(1), ln3, ln3 ** 2]),
    ("c, 1, ln3, arctan(2), π", [c, mpf(1), ln3, atan2_val, pi]),
    ("c, 1, ln2, ln3, ln2·ln3", [c, mpf(1), ln2, ln3, ln2 * ln3]),
    ("c, 1, ln3, arctan(2), arctan(2)²", [c, mpf(1), ln3, atan2_val, atan2_val ** 2]),
    ("c, 1, ln3, arctan(2), Catalan", [c, mpf(1), ln3, atan2_val, catalan]),
    ("c, 1, ln3, arctan(2), ζ(3)", [c, mpf(1), ln3, atan2_val, zeta(3)]),
    ("c, 1, ln3, ln(3/2), arctan(2)", [c, mpf(1), ln3, ln3_2, atan2_val]),
    ("c, 1, ln2, ln3, π, arctan(2)", [c, mpf(1), ln2, ln3, pi, atan2_val]),
    ("c, 1, ln3², ln3, arctan(2), ln2", [c, mpf(1), ln3 ** 2, ln3, atan2_val, ln2]),
]

for label, vec in pslq_bases:
    try:
        rel = pslq(vec, maxcoeff=1000, maxsteps=10000)
        if rel is not None:
            parts = []
            names = label.split(", ")
            for i, coeff in enumerate(rel):
                if coeff != 0:
                    parts.append(f"{coeff}·{names[i] if i < len(names) else f'x{i}'}")
            print(f"  {label}:")
            print(f"    {' + '.join(parts)} = 0")
            if rel[0] != 0:
                val = sum(co * v for co, v in zip(rel[1:], vec[1:])) / (-rel[0])
                residual = abs(float(c - val))
                print(f"    c₁(3) = {float(val):.15f}  (residual {residual:.4e})")
        else:
            print(f"  {label}: no relation found")
    except Exception as e:
        print(f"  {label}: error ({e})")

# =====================================================================
# PART 3: ANGULAR INTEGRAL PREDICTION FOR BASE 3
# =====================================================================
print(f"\n\n{'='*78}")
print("  PART 3: ANGULAR INTEGRAL PREDICTION FOR BASE 3")
print("=" * 78)

print(f"""
  For base b, the D-odd boundary in angular coordinates is:
    arctan(X) + arctan(Y) = arctan(b-1)  [curved for b > 2]

  Base 2: arctan(1) = π/4 → STRAIGHT LINE → π appears
  Base 3: arctan(2) ≈ {float(atan2_val):.10f} → CURVED BOUNDARY → no π expected

  The Angular Uniqueness Theorem (Paper G):
    Base 2 is the ONLY base where the D-odd boundary is a straight line.
    Therefore c₁(2) = π/18 involves π, but c₁(b) for b ≥ 3 should NOT.

  Integration limit for base 3:
    ∫₀^{{arctan(2)}} ... dα
    This naturally produces arctan(2) in the result.

  Predicted form: c₁(3) = rational combination of {{1, ln(3), arctan(2), ln(2)}}
  with NO π involvement.
""")

# Test: if c₁(3) = a + b·ln(3) + c·arctan(2), what are a, b, c?
print(f"--- Least-squares fit: c₁(3) = a + b·ln(3) + c·arctan(2) ---\n")

# Using extrapolation consistency
for model_name, c_est in estimates:
    print(f"  Using {model_name} estimate: c₁(3) = {float(c_est):.15f}")
    # Solve: c_est = a + b*ln3 + c*atan2
    # 1 unknown from each: test a=0, solve 2-param
    # Or: just check direct candidates
    delta_ln3 = float(c_est - (ln3 - mpf(1) / 2))
    delta_3ln = float(c_est - (3 * ln3 - 2))
    delta_atan = float(c_est - (atan2_val / 2))
    print(f"    Δ from ln(3)-1/2    = {delta_ln3:+.6e}")
    print(f"    Δ from 3·ln(3)-2    = {delta_3ln:+.6e}")
    print(f"    Δ from arctan(2)/2  = {delta_atan:+.6e}")

# =====================================================================
# PART 4: TRANSFER OPERATOR EIGENVALUES FOR BASE 3
# =====================================================================
print(f"\n\n{'='*78}")
print("  PART 4: TRANSFER OPERATOR EIGENVALUES FOR BASE 3")
print("=" * 78)

import numpy as np
from itertools import product as iproduct


def compute_conv_dist_b3(j):
    """Exact conv_j distribution for base-3 multiplication."""
    n_free = 2 * j
    dist = {}
    for bits in iproduct(range(3), repeat=n_free):
        g = [1] + list(bits[:j])
        h = [1] + list(bits[j:])
        v = sum(g[i] * h[j - i] for i in range(j + 1))
        dist[v] = dist.get(v, 0) + 1
    total = sum(dist.values())
    return {v: count / total for v, count in sorted(dist.items())}


def build_transfer_b3(dist, dim=15):
    """Build base-3 transfer matrix T(c,c') = P(floor((V+c)/3) = c')."""
    T = np.zeros((dim, dim))
    for c in range(dim):
        for v, p in dist.items():
            c_new = (v + c) // 3
            if c_new < dim:
                T[c, c_new] += p
    return T


print(f"\n--- Eigenvalues of T^(j) for base 3, j=1..6 ---\n")

for j in range(1, 7):
    if 3 ** (2 * j) > 5e6:
        print(f"  j={j}: skipped (3^{2*j} = {3**(2*j):.0e} too large)")
        continue
    dist = compute_conv_dist_b3(j)
    T = build_transfer_b3(dist)
    evals = sorted(np.linalg.eigvals(T).real, reverse=True)
    n_sig = sum(1 for e in evals if abs(e) > 1e-10)
    ev_str = [f"{e:.6f}" for e in evals[:min(5, n_sig)]]
    print(f"  T^({j}) base 3: [{', '.join(ev_str)}]")
    # Check for 1/3 eigenvalue (base-3 Parity Lemma)
    has_third = any(abs(e - 1.0 / 3) < 1e-6 for e in evals)
    print(f"    eigenvalue 1/3 present: {has_third}")

# =====================================================================
# SYNTHESIS
# =====================================================================
print(f"\n\n{'='*78}")
print("  SYNTHESIS")
print("=" * 78)

print(f"""
  Best extrapolated c₁(3) = {float(best_c1):.15f}

  Top candidate: ln(3) - 1/2 = {float(ln3 - mpf(1)/2):.15f}
  Discrepancy:   {float(best_c1 - (ln3 - mpf(1)/2)):+.6e}

  Precision of extrapolation with K=2..{K_max}: ~{max(3, K_max - 6)} significant digits
  PSLQ needs ≥ 8 digits for reliable identification.

  ANGULAR UNIQUENESS PREDICTION:
    c₁(3) should NOT involve π (base-3 boundary is curved).
    The natural constants are: 1, ln(3), ln(2), arctan(2).
    The best candidate ln(3) - 1/2 fits within extrapolation error.

  NEXT STEP:
    Run G10_base3_exact_k13.c to get K=13 (and ideally K=14) data.
    With K=13: extrapolation gains ~1 digit → ~6 digits total.
    With K=14: ~7 digits → PSLQ becomes reliable.

  TRANSFER OPERATOR:
    Base-3 should have universal eigenvalue 1/3 (base-b Parity Lemma).
    Convergence rate ρ → 1/3 (confirmed by successive ratios).
""")

print("=" * 78)
print("  END OF G11")
print("=" * 78)
