"""
G09_base3_pslq.py — PSLQ analysis of c₁(base 3)

From G09_base3_exact.c:
  (sum_cm1, n_ulc) pairs for K=2..10

c₁ = sum_cm1/n_ulc - 1

Angular Uniqueness Theorem predicts:
  - c₁(3) should NOT involve π
  - Natural transcendentals: arctan(2), ln(3), ln(2)
  - Integration limit arctan(b-1) = arctan(2) for b=3
"""

import mpmath
mpmath.mp.dps = 50

print("G09: BASE-3 c₁(K) — PSLQ ANALYSIS")
print("=" * 70)

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
}

c1_b3 = {}
for K, (s, n) in raw_b3.items():
    c1_b3[K] = mpmath.mpf(s) / mpmath.mpf(n) - 1

Ks = sorted(c1_b3.keys())

print(f"\n{'K':>3s} {'c₁(K)':>18s} {'Δ':>14s} {'ρ':>10s}")
print("-" * 50)
prev_delta = None
for K in Ks:
    val = c1_b3[K]
    if K > Ks[0]:
        delta = c1_b3[K] - c1_b3[K-1]
        if prev_delta is not None and abs(prev_delta) > 1e-20:
            rho = abs(float(delta / prev_delta))
            print(f"{K:3d} {float(val):18.12f} {float(delta):+14.6e} {rho:10.6f}")
        else:
            print(f"{K:3d} {float(val):18.12f} {float(delta):+14.6e} {'---':>10s}")
        prev_delta = delta
    else:
        print(f"{K:3d} {float(val):18.12f} {'---':>14s} {'---':>10s}")

# ================================================================
# Extrapolation using ρ → 1/3
# ================================================================
print(f"\n{'='*70}")
print("EXTRAPOLATION (assuming ρ → 1/3)")
print("=" * 70)

target = mpmath.mpf('1') / 3

deltas = {}
for K in Ks:
    if K > Ks[0]:
        deltas[K] = c1_b3[K] - c1_b3[K-1]

print("\nShanks transform on c₁(K) with assumed rate 1/3:")
for K in range(6, max(Ks) + 1):
    if K in c1_b3 and K-1 in c1_b3 and K-2 in c1_b3:
        d1 = c1_b3[K] - c1_b3[K-1]
        d0 = c1_b3[K-1] - c1_b3[K-2]
        if abs(d1 - d0) > 1e-30:
            shanks = c1_b3[K] - d1**2 / (d1 - d0)
            print(f"  S({K}) = {float(shanks):.12f}")

print("\nModel: c₁(K) = c_∞ + A·(1/3)^K")
for K1, K2 in [(9, 10), (8, 9), (7, 8)]:
    if K1 in c1_b3 and K2 in c1_b3:
        r = mpmath.mpf(1) / 3
        c_inf = (c1_b3[K2] - r * c1_b3[K1]) / (1 - r)
        A = (c1_b3[K1] - c_inf) / mpmath.power(r, K1)
        print(f"  K={K1},{K2}: c_∞ = {float(c_inf):.12f}, A = {float(A):.6f}")

print("\nModel: c₁(K) = c_∞ + A·(1/3)^K + B·(1/9)^K")
K_list = [8, 9, 10]
M = mpmath.matrix(3, 3)
b_vec = mpmath.matrix(3, 1)
for i, K in enumerate(K_list):
    M[i, 0] = 1
    M[i, 1] = mpmath.power(mpmath.mpf(1)/3, K)
    M[i, 2] = mpmath.power(mpmath.mpf(1)/9, K)
    b_vec[i] = c1_b3[K]
try:
    sol = mpmath.lu_solve(M, b_vec)
    c_inf_3term = sol[0]
    A_3term = sol[1]
    B_3term = sol[2]
    print(f"  c_∞ = {float(c_inf_3term):.12f}")
    print(f"  A   = {float(A_3term):.6f}")
    print(f"  B   = {float(B_3term):.6f}")
except Exception as e:
    print(f"  3-term solve failed: {e}")
    c_inf_3term = mpmath.mpf('0.600')

print("\nUsing K=7..10 for 4-term model: c_∞ + A·(1/3)^K + B·(1/9)^K + C·(1/27)^K")
K_list = [7, 8, 9, 10]
M = mpmath.matrix(4, 4)
b_vec = mpmath.matrix(4, 1)
for i, K in enumerate(K_list):
    M[i, 0] = 1
    M[i, 1] = mpmath.power(mpmath.mpf(1)/3, K)
    M[i, 2] = mpmath.power(mpmath.mpf(1)/9, K)
    M[i, 3] = mpmath.power(mpmath.mpf(1)/27, K)
    b_vec[i] = c1_b3[K]
try:
    sol = mpmath.lu_solve(M, b_vec)
    c_inf_4term = sol[0]
    print(f"  c_∞ = {float(c_inf_4term):.12f}")
    print(f"  A   = {float(sol[1]):.6f}, B = {float(sol[2]):.6f}, C = {float(sol[3]):.6f}")
except Exception as e:
    print(f"  4-term solve failed: {e}")
    c_inf_4term = c_inf_3term

best_c1_b3 = c_inf_3term

# ================================================================
# PSLQ Analysis
# ================================================================
print(f"\n{'='*70}")
print(f"PSLQ ANALYSIS ON c₁(3) ≈ {float(best_c1_b3):.12f}")
print("=" * 70)

atan2 = mpmath.atan(2)
ln2 = mpmath.log(2)
ln3 = mpmath.log(3)
pi = mpmath.pi
phi = (1 + mpmath.sqrt(5)) / 2
euler = mpmath.euler
catalan = mpmath.catalan

c = best_c1_b3

print(f"\nReference constants:")
print(f"  arctan(2) = {float(atan2):.15f}")
print(f"  ln(2)     = {float(ln2):.15f}")
print(f"  ln(3)     = {float(ln3):.15f}")
print(f"  π         = {float(pi):.15f}")
print(f"  3·ln(3)-2 = {float(3*ln3-2):.15f}")
print(f"  arctan(2)/3 = {float(atan2/3):.15f}")
print(f"  ln(3)²/6  = {float(ln3**2/6):.15f}")

bases_pslq = [
    ("c, 1, ln2, ln3", [c, 1, ln2, ln3]),
    ("c, 1, ln2, ln3, arctan(2)", [c, 1, ln2, ln3, atan2]),
    ("c, 1, ln3, arctan(2)", [c, 1, ln3, atan2]),
    ("c, 1, ln2, ln3, π", [c, 1, ln2, ln3, pi]),
    ("c, 1, ln3, arctan(2), π", [c, 1, ln3, atan2, pi]),
    ("c, 1, ln2, ln3, ln3², arctan(2)", [c, 1, ln2, ln3, ln3**2, atan2]),
    ("c, 1, ln(3/2), ln3, arctan(2)", [c, 1, mpmath.log(mpmath.mpf(3)/2), ln3, atan2]),
    ("c, 1, ln2, ln3, arctan(2), ln2·ln3", [c, 1, ln2, ln3, atan2, ln2*ln3]),
    ("c, 1, ln2, ln3, arctan(2), arctan(2)²", [c, 1, ln2, ln3, atan2, atan2**2]),
    ("c, 1, ln3, arctan(2), arctan(√2)", [c, 1, ln3, atan2, mpmath.atan(mpmath.sqrt(2))]),
    ("c, 1, ln(4/3), ln3, arctan(2)", [c, 1, mpmath.log(mpmath.mpf(4)/3), ln3, atan2]),
    ("c, 1, ln2, ln3, ln3·arctan(2)", [c, 1, ln2, ln3, ln3*atan2]),
]

print(f"\nPSLQ results (maxcoeff=1000):")
for label, vec in bases_pslq:
    try:
        rel = mpmath.pslq(vec, maxcoeff=1000, maxsteps=10000)
        if rel is not None:
            parts = []
            for i, coeff in enumerate(rel):
                if coeff != 0:
                    parts.append(f"{coeff}·x{i}")
            print(f"  {label}:")
            print(f"    relation: {' + '.join(parts)} = 0")
            if rel[0] != 0:
                val = sum(coeff * v for coeff, v in zip(rel[1:], vec[1:])) / (-rel[0])
                print(f"    c₁(3) = {float(val):.15f}")
                residual = abs(float(c - val))
                print(f"    residual = {residual:.4e}")
        else:
            print(f"  {label}: no relation found")
    except Exception as e:
        print(f"  {label}: error ({e})")

# ================================================================
# Direct comparison with candidate forms
# ================================================================
print(f"\n{'='*70}")
print("DIRECT COMPARISON WITH CANDIDATE FORMS")
print("=" * 70)

candidates = [
    ("3·ln(3) - 2", 3*ln3 - 2),
    ("arctan(2) - ln(3)/4", atan2 - ln3/4),
    ("ln(3)²/2", ln3**2 / 2),
    ("2·arctan(2)/π", 2*atan2/pi),
    ("arctan(2)·ln(3)/π", atan2*ln3/pi),
    ("arctan(2)/2 + ln(3)/8", atan2/2 + ln3/8),
    ("(arctan(2))²/2", atan2**2 / 2),
    ("3·ln(4/3)", 3*mpmath.log(mpmath.mpf(4)/3)),
    ("ln(3)·ln(2)", ln3*ln2),
    ("2·ln(3) - ln(2) - 1", 2*ln3 - ln2 - 1),
    ("arctan(2)²/(2·ln3)", atan2**2/(2*ln3)),
    ("ln(3)²/(2·ln2)", ln3**2/(2*ln2)),
    ("3·arctan(2) - π", 3*atan2 - pi),
    ("3·arctan(2) - π + ln(3) - 1", 3*atan2 - pi + ln3 - 1),
]

print(f"\nc₁(3) ≈ {float(best_c1_b3):.12f}")
print(f"\n{'Candidate':>35s} {'Value':>16s} {'Δ':>14s}")
print("-" * 68)
for name, val in sorted(candidates, key=lambda x: abs(float(x[1] - best_c1_b3))):
    delta = float(val - best_c1_b3)
    print(f"  {name:>35s} {float(val):16.12f} {delta:+14.6e}")

print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print(f"""
Best extrapolated c₁(3) ≈ {float(best_c1_b3):.12f}
(from K=8,9,10 with 3-term model: c_∞ + A/3^K + B/9^K)

MC reference (L20b): c₁(3) = 0.5987 ± 0.0001

The Angular Uniqueness Theorem predicts no π involvement.
If PSLQ finds a relation involving only ln(3) and arctan(2),
this confirms that π in c₁(2) = π/18 is purely from base-2
angular geometry.
""")
